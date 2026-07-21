"""Rebuild the visualiser payload from a finished tracing run, without re-running it.

``xai/representational_tracing.py`` writes ``tracing_data.js`` at the end of a run,
so adding a new per-case field to the visualiser would otherwise mean spending the
inference again. This re-emits the payload from the tracing JSONL already on disk
and merges in the encoder attention exported separately by ``xai/attention_export.py``.

Cases are matched by ``case_id`` against the attention file's ``id``; cases without
attention simply keep the field absent, and the V6 panel hides itself for them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from xai.representational_tracing import build_viz_payload, stage_summary  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild tracing_data.js from disk")
    parser.add_argument("--tracing", type=Path, required=True)
    parser.add_argument("--encoder-attention", type=Path)
    parser.add_argument(
        "--viz-output", type=Path, default=PROJECT_ROOT / "visualizer/tracing_data.js"
    )
    args = parser.parse_args()

    cases = read_jsonl(args.tracing)
    if not cases:
        raise SystemExit(f"no cases in {args.tracing}")

    merged = 0
    if args.encoder_attention and args.encoder_attention.exists():
        attention = {
            str(record.get("id")): record for record in read_jsonl(args.encoder_attention)
        }
        for case in cases:
            record = attention.get(str(case.get("case_id")))
            if not record:
                continue
            case["encoder_attention"] = {
                "bins": record.get("bins"),
                "steps": record.get("steps"),
                "tokens": record.get("tokens"),
                "profile": record.get("mass_profile"),
                "per_token": record.get("attention_binned"),
                "entropy": record.get("token_entropy"),
            }
            merged += 1

    mode = "internal" if any(case.get("mode") == "internal" for case in cases) else "contrafactual_global"
    metric = cases[0].get("metric", "cosine")
    summary = stage_summary(cases, mode=mode, metric=metric)
    payload = build_viz_payload(cases, summary)

    args.viz_output.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, ensure_ascii=False)
    args.viz_output.write_text(f"window.TRACING_DATA = {body};\n", encoding="utf-8")
    args.viz_output.with_suffix(".json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"casos: {len(cases)}   con atencion del encoder: {merged}")
    if payload["meta"].get("attention"):
        meta = payload["meta"]["attention"]
        print(
            f"entropia media {meta['mean_entropy']:.4f} · minima {meta['min_entropy']:.4f} "
            f"· referencia uniforme {meta['uniform']:.4f}"
        )
    print(f"Saved to {args.viz_output} and {args.viz_output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
