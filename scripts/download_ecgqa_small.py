"""Stage 1 -- controlled, reproducible PTB-XL ECG-QA subset download.

This builds a *small* slice of ECG-QA (PTB-XL, paraphrased split) suitable for a
preliminary paper experiment on a limited laptop. It never downloads the full
ECG-QA / PTB-XL / MIMIC-IV-ECG datasets:

1. clone (or reuse) the ECG-QA repo at a pinned tag -- this is just QA JSON, a
   few MB, no signals;
2. read the requested split(s) under ``ecgqa/ptbxl/paraphrased/<split>/`` and
   pick a deterministic subset (``--seed``) honouring ``--max_questions`` and
   ``--max_unique_ecgs``;
3. download *only* the PTB-XL WFDB records referenced by the chosen questions
   (each ``.hea``/``.dat`` is fetched once and skipped if already present);
4. write ``<output>/manifest.jsonl`` -- one row per question, carrying the
   ``ecg_path`` list (two paths for comparison questions).

The heavy ``prepare_ecg_signals.py`` stage reads only this manifest, so the raw
ECG-QA repo is never loaded as a whole at signal time.

Example::

    python scripts/download_ecgqa_small.py \\
      --subset train --max_questions 300 --max_unique_ecgs 100 \\
      --seed 42 --output data/ecgqa_small
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

ECGQA_REPO_URL = "https://github.com/Jwoo5/ecg-qa.git"
ECGQA_TAG = "v1.0.2"
PTBXL_BASE_URL = "https://physionet.org/files/ptb-xl/1.0.3"

SPLITS = ("train", "valid", "test")
# Budget split used when --subset all: train gets the bulk, valid/test a slice.
SUBSET_ALL_RATIOS = {"train": 0.70, "valid": 0.15, "test": 0.15}


def log(message: str) -> None:
    print(f"[download] {message}", flush=True)


# --------------------------------------------------------------------------- #
# ECG-QA repository                                                            #
# --------------------------------------------------------------------------- #
def ensure_repo(repo_dir: Path, allow_clone: bool) -> Path:
    """Return the ECG-QA repo root, cloning it at the pinned tag if missing."""
    paraphrased = repo_dir / "ecgqa" / "ptbxl" / "paraphrased"
    if paraphrased.is_dir():
        log(f"using existing ECG-QA repo at {repo_dir}")
        return repo_dir
    if not allow_clone:
        raise FileNotFoundError(
            f"ECG-QA repo not found at {repo_dir} and --no_clone was set. "
            f"Clone it manually: git clone --depth 1 --branch {ECGQA_TAG} {ECGQA_REPO_URL} {repo_dir}"
        )
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    log(f"cloning ECG-QA ({ECGQA_TAG}) into {repo_dir}")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", ECGQA_TAG, ECGQA_REPO_URL, str(repo_dir)],
            check=True,
        )
    except subprocess.CalledProcessError:
        log(f"tag {ECGQA_TAG} unavailable, cloning default branch")
        subprocess.run(["git", "clone", "--depth", "1", ECGQA_REPO_URL, str(repo_dir)], check=True)
    if not paraphrased.is_dir():
        raise FileNotFoundError(f"Cloned repo has no paraphrased split at {paraphrased}")
    return repo_dir


def iter_split_samples(repo_dir: Path, split: str) -> Iterator[dict[str, Any]]:
    """Yield raw ECG-QA samples for one paraphrased split, file by file.

    Each split directory holds many small JSON files (each a list of question
    dicts); they are streamed rather than concatenated so peak memory stays low.
    """
    split_dir = repo_dir / "ecgqa" / "ptbxl" / "paraphrased" / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing ECG-QA split directory: {split_dir}")
    for path in sorted(split_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rows = payload if isinstance(payload, list) else [payload]
        for row in rows:
            if isinstance(row, dict):
                yield row


# --------------------------------------------------------------------------- #
# Selection                                                                    #
# --------------------------------------------------------------------------- #
def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def select_subset(
    repo_dir: Path,
    split: str,
    max_questions: int,
    max_unique_ecgs: int,
    seed: int,
    question_types: set[str] | None,
) -> list[dict[str, Any]]:
    """Deterministically pick a subset of one split.

    Candidates (lightweight QA metadata only -- no signals) are shuffled with
    ``seed`` then greedily kept while respecting both the question budget and the
    unique-ECG budget. A candidate is admitted only if every ECG it needs is
    already selected or there is room left in the unique-ECG budget.
    """
    candidates: list[dict[str, Any]] = []
    for raw in iter_split_samples(repo_dir, split):
        qtype = str(raw.get("question_type", ""))
        if question_types and qtype not in question_types:
            continue
        ecg_ids = [int(e) for e in _as_list(raw.get("ecg_id"))]
        if not ecg_ids:
            continue
        candidates.append({"raw": raw, "ecg_ids": ecg_ids})

    random.Random(seed).shuffle(candidates)

    chosen: list[dict[str, Any]] = []
    used_ecgs: set[int] = set()
    for candidate in candidates:
        if len(chosen) >= max_questions:
            break
        needed = set(candidate["ecg_ids"])
        new_ecgs = needed - used_ecgs
        if len(used_ecgs) + len(new_ecgs) > max_unique_ecgs:
            continue
        used_ecgs |= new_ecgs
        chosen.append(candidate)
    log(f"split '{split}': selected {len(chosen)} questions over {len(used_ecgs)} unique ECGs "
        f"(from {len(candidates)} candidates)")
    return chosen


def split_budgets(subset: str, max_questions: int, max_unique_ecgs: int) -> dict[str, tuple[int, int]]:
    """Return {split: (question_budget, ecg_budget)} for the requested subset."""
    if subset != "all":
        return {subset: (max_questions, max_unique_ecgs)}
    budgets: dict[str, tuple[int, int]] = {}
    for name, ratio in SUBSET_ALL_RATIOS.items():
        budgets[name] = (max(1, round(max_questions * ratio)), max(1, round(max_unique_ecgs * ratio)))
    return budgets


# --------------------------------------------------------------------------- #
# PTB-XL signal download                                                       #
# --------------------------------------------------------------------------- #
def ptbxl_relpath(ecg_id: int) -> str:
    """records500 relative path (no extension) for a PTB-XL ecg_id."""
    folder = f"{(int(ecg_id) // 1000) * 1000:05d}"
    return f"records500/{folder}/{int(ecg_id):05d}_hr"


def download_record(ecg_id: int, signals_root: Path, base_url: str, retries: int) -> bool:
    """Download the ``.hea``/``.dat`` for one ecg_id; skip files already present.

    Returns True when both component files are available locally afterwards.
    """
    rel = ptbxl_relpath(ecg_id)
    ok = True
    for ext in (".hea", ".dat"):
        dest = signals_root / f"{rel}{ext}"
        if dest.exists() and dest.stat().st_size > 0:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"{base_url}/{rel}{ext}"
        downloaded = False
        for attempt in range(1, retries + 1):
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    data = response.read()
                tmp = dest.with_suffix(dest.suffix + ".part")
                tmp.write_bytes(data)
                tmp.replace(dest)
                downloaded = True
                break
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if attempt == retries:
                    log(f"[warn] failed to download {url} ({exc})")
                    ok = False
        if not downloaded and not (dest.exists() and dest.stat().st_size > 0):
            ok = False
    return ok


# --------------------------------------------------------------------------- #
# Manifest                                                                     #
# --------------------------------------------------------------------------- #
def build_manifest_row(candidate: dict[str, Any], split: str, signals_root: Path) -> dict[str, Any]:
    raw = candidate["raw"]
    ecg_ids = candidate["ecg_ids"]
    ecg_paths = [str(signals_root / ptbxl_relpath(eid)) for eid in ecg_ids]
    sample_id = raw.get("sample_id", raw.get("question_id"))
    identifier = f"ptbxl/{split}/{sample_id}"
    used = {"question", "answer", "question_type", "attribute_type", "ecg_id",
            "template_id", "sample_id", "question_id"}
    metadata = {k: v for k, v in raw.items() if k not in used}
    return {
        "id": identifier,
        "split": split,
        "question": str(raw.get("question", "")).strip(),
        "answer": _as_list(raw.get("answer")),
        "ecg_id": ecg_ids,
        "question_type": str(raw.get("question_type", "")),
        "attribute_type": str(raw.get("attribute_type", "")),
        "template_id": raw.get("template_id"),
        "sample_id": sample_id,
        "ecg_path": ecg_paths,
        "metadata": metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a small reproducible PTB-XL ECG-QA subset")
    parser.add_argument("--subset", choices=[*SPLITS, "all"], default="train",
                        help="Which paraphrased split to sample (or 'all' for train+valid+test)")
    parser.add_argument("--max_questions", type=int, default=300)
    parser.add_argument("--max_unique_ecgs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/ecgqa_small")
    parser.add_argument("--ecgqa_repo", type=Path, default=None,
                        help="ECG-QA repo dir (default: <output>/ecg-qa)")
    parser.add_argument("--ptbxl_signals", type=Path, default=None,
                        help="Existing PTB-XL root containing records500/ (default: <output>/signals/ptbxl)")
    parser.add_argument("--ptbxl_url", default=PTBXL_BASE_URL, help="PhysioNet PTB-XL base URL")
    parser.add_argument("--question_types", nargs="*", default=None,
                        help="Optional filter on question_type")
    parser.add_argument("--no_clone", action="store_true", help="Do not clone the ECG-QA repo")
    parser.add_argument("--no_download", action="store_true",
                        help="Do not fetch WFDB signals (manifest paths are still written)")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--force", action="store_true", help="Rebuild the manifest even if it exists")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    repo_dir = args.ecgqa_repo or (args.output / "ecg-qa")
    signals_root = args.ptbxl_signals or (args.output / "signals" / "ptbxl")
    manifest_path = args.output / "manifest.jsonl"

    if manifest_path.exists() and not args.force:
        existing = sum(1 for line in manifest_path.open(encoding="utf-8") if line.strip())
        log(f"manifest already exists with {existing} rows ({manifest_path}); use --force to rebuild")
        return

    ensure_repo(repo_dir, allow_clone=not args.no_clone)
    question_types = set(args.question_types) if args.question_types else None
    budgets = split_budgets(args.subset, args.max_questions, args.max_unique_ecgs)

    selected: list[tuple[dict[str, Any], str]] = []
    for split, (q_budget, ecg_budget) in budgets.items():
        for candidate in select_subset(repo_dir, split, q_budget, ecg_budget, args.seed, question_types):
            selected.append((candidate, split))

    # Download only the unique ECGs referenced by the selection.
    unique_ecgs = sorted({eid for candidate, _ in selected for eid in candidate["ecg_ids"]})
    downloaded_ok = 0
    if args.no_download:
        log(f"--no_download set; skipping {len(unique_ecgs)} WFDB records")
    else:
        log(f"downloading {len(unique_ecgs)} unique PTB-XL records into {signals_root}")
        for index, ecg_id in enumerate(unique_ecgs, 1):
            if download_record(ecg_id, signals_root, args.ptbxl_url, args.retries):
                downloaded_ok += 1
            if index % 25 == 0 or index == len(unique_ecgs):
                log(f"  signals {index}/{len(unique_ecgs)} ({downloaded_ok} ok)")

    by_split: dict[str, int] = {}
    by_qtype: dict[str, int] = {}
    with manifest_path.open("w", encoding="utf-8") as handle:
        for candidate, split in selected:
            row = build_manifest_row(candidate, split, signals_root)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            by_split[split] = by_split.get(split, 0) + 1
            by_qtype[row["question_type"]] = by_qtype.get(row["question_type"], 0) + 1

    summary = {
        "subset": args.subset,
        "seed": args.seed,
        "questions": len(selected),
        "unique_ecgs": len(unique_ecgs),
        "signals_downloaded_ok": downloaded_ok if not args.no_download else None,
        "by_split": by_split,
        "by_question_type": by_qtype,
        "manifest": str(manifest_path),
        "signals_root": str(signals_root),
    }
    (args.output / "download_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
