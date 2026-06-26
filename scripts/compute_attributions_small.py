"""Compute REAL model attributions for the visualizer (V3 + V4 panels).

Runs the trained checkpoint over the test subset and, per sample, produces:

* ``ecg_patch_saliency`` -- gradient saliency |d logit / d series| summed over
  leads, aggregated into 20 temporal patches (drives V4's ECG heat strip).
* ``token_saliency``     -- gradient saliency over the question's input-token
  embeddings, one weight per token (drives V4's text relevance).
* ``embeddings``         -- the real LLM-space points (text-token embeddings +
  projected temporal/ECG tokens) for three conditions (completo / perturbado /
  sin proyector), PCA-reduced to 16 dims server-side so the browser only does
  16->2 (drives V3).

Output: ``outputs/ecgqa_small/attributions.jsonl`` (one row per sample), which
``export_visualizer_data.py`` merges into the visualizer data when present.

Example::

    python scripts/compute_attributions_small.py \\
      --model_path checkpoints/ecgqa_small_lora \\
      --data data/ecgqa_small/processed_test.jsonl \\
      --max_samples 30 \\
      --output outputs/ecgqa_small/attributions.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

N_PATCH = 20
EMB_DIM = 16


def pca_reduce(points: np.ndarray, dim: int = EMB_DIM) -> np.ndarray:
    """Center + SVD project ``[n, d]`` points to ``[n, dim]`` (zero-padded if d<dim)."""
    if points.shape[0] == 0:
        return points
    centered = points - points.mean(axis=0, keepdims=True)
    k = min(dim, centered.shape[0], centered.shape[1])
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        reduced = centered @ vt[:k].T
    except np.linalg.LinAlgError:
        reduced = centered[:, :k]
    if reduced.shape[1] < dim:
        reduced = np.pad(reduced, ((0, 0), (0, dim - reduced.shape[1])))
    return reduced


def patch_aggregate(values: np.ndarray, n_patch: int = N_PATCH) -> list[float]:
    """Mean-aggregate a 1D saliency vector into ``n_patch`` patches, min-max to 0..1."""
    if values.size == 0:
        return [0.3] * n_patch
    bounds = np.linspace(0, values.size, n_patch + 1, dtype=int)
    patches = np.array([
        values[bounds[p]:bounds[p + 1]].mean() if bounds[p + 1] > bounds[p] else 0.0
        for p in range(n_patch)
    ], dtype=np.float32)
    lo, hi = float(patches.min()), float(patches.max())
    norm = (patches - lo) / (hi - lo) if hi > lo else np.full(n_patch, 0.3, dtype=np.float32)
    return [round(float(0.08 + 0.9 * v), 4) for v in norm]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute real attributions for the ECG-QA visualizer")
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data/ecgqa_small/processed_test.jsonl")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/ecgqa_small/attributions.jsonl")
    parser.add_argument("--max_samples", type=int, default=30)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                        help="cpu = no GPU power draw (safe but slow)")
    parser.add_argument("--no_quantization", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # cuDNN no soporta el backward de una RNN (la GRU del encoder) con el modelo en
    # modo eval -> "cudnn RNN backward can only be called in training mode". Se
    # desactiva cuDNN para usar el kernel nativo (sí permite backward en eval) y así
    # calcular la saliencia por gradiente sin poner el modelo en modo entrenamiento.
    # Impacto de rendimiento despreciable para este subconjunto pequeño.
    torch.backends.cudnn.enabled = False

    from inference.runtime import load_checkpoint  # noqa: E402

    tokenizer, model, config = load_checkpoint(
        args.model_path, not args.no_quantization, device=args.device
    )
    device = model.input_device
    input_dim = config["input_dim"]
    embed = model.llm.get_input_embeddings()

    def build_prompt_tokens(question: str):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True
        )
        enc = tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")[0].tolist()
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        # Token indices that fall inside the question span (drop chat-template tokens).
        q_start = prompt.find(question)
        if q_start >= 0:
            q_end = q_start + len(question)
            q_idx = [i for i, (a, b) in enumerate(offsets) if b > a and a >= q_start and b <= q_end]
        else:
            q_idx = list(range(input_ids.shape[1]))
        if not q_idx:
            q_idx = list(range(input_ids.shape[1]))
        return input_ids, attention_mask, q_idx

    def signal_tensor(signal: np.ndarray) -> torch.Tensor:
        series = torch.zeros(1, signal.shape[0], input_dim, dtype=torch.float32, device=device)
        leads = min(signal.shape[1], input_dim)
        series[0, :, :leads] = torch.from_numpy(np.ascontiguousarray(signal[:, :leads]))
        return series

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8") as handle, args.data.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            if args.max_samples and count >= args.max_samples:
                break
            row = json.loads(line)
            signal = np.load(row["ecg_signal_path"]).astype(np.float32)
            input_ids, attention_mask, q_idx = build_prompt_tokens(row["question"])

            # ---- gradient saliency (ECG series + text-token embeddings) ----
            series = signal_tensor(signal).requires_grad_(True)
            text_embeds = embed(input_ids).detach().clone().requires_grad_(True)
            temporal_tokens, _ = model.time_series_encoder(series)
            temporal_embeds = model.temporal_projector(temporal_tokens).to(text_embeds.dtype)
            temporal_mask = torch.ones(temporal_embeds.shape[:2], dtype=attention_mask.dtype, device=device)
            inputs_embeds = torch.cat([temporal_embeds, text_embeds], dim=1)
            combined_mask = torch.cat([temporal_mask, attention_mask], dim=1)
            logits = model.llm(inputs_embeds=inputs_embeds, attention_mask=combined_mask).logits[:, -1]
            model.zero_grad(set_to_none=True)
            logits.max(dim=-1).values.sum().backward()

            ecg_sal = series.grad.detach().abs().sum(dim=-1)[0].float().cpu().numpy()
            ecg_patch = patch_aggregate(ecg_sal)
            tok_sal = text_embeds.grad.detach().abs().sum(dim=-1)[0].float().cpu().numpy()

            q_vals = np.array([tok_sal[i] for i in q_idx], dtype=np.float32)
            lo, hi = float(q_vals.min()), float(q_vals.max())
            q_norm = (q_vals - lo) / (hi - lo) if hi > lo else np.full(len(q_idx), 0.3, dtype=np.float32)
            token_saliency = []
            ids_list = input_ids[0].tolist()
            for pos, idx in enumerate(q_idx):
                word = tokenizer.decode([ids_list[idx]]).strip() or tokenizer.convert_ids_to_tokens(ids_list[idx])
                token_saliency.append({"w": word, "contrib": round(float(0.08 + 0.9 * q_norm[pos]), 4)})

            # ---- real LLM-space embeddings for V3 (3 conditions) ----
            with torch.no_grad():
                text_pts = text_embeds.detach()[0][q_idx].float().cpu().numpy()
                tok_words = [t["w"] for t in token_saliency]

                def projected(sig_tensor):
                    tt, _ = model.time_series_encoder(sig_tensor)
                    return model.temporal_projector(tt)[0].float().cpu().numpy(), tt[0].float().cpu().numpy()

                proj_full, pre_full = projected(series.detach())
                noise = 0.1 * series.detach().std() * torch.randn_like(series)
                proj_pert, _ = projected(series.detach() + noise)

            def make_condition(ecg_hi: np.ndarray, pad_to_text: bool) -> list[dict[str, Any]]:
                ecg = ecg_hi
                if pad_to_text and ecg.shape[1] < text_pts.shape[1]:
                    ecg = np.pad(ecg, ((0, 0), (0, text_pts.shape[1] - ecg.shape[1])))
                elif not pad_to_text and ecg.shape[1] != text_pts.shape[1]:
                    ecg = ecg[:, : text_pts.shape[1]] if ecg.shape[1] > text_pts.shape[1] else \
                        np.pad(ecg, ((0, 0), (0, text_pts.shape[1] - ecg.shape[1])))
                stacked = np.vstack([text_pts, ecg])
                reduced = pca_reduce(stacked)
                pts = []
                for i, w in enumerate(tok_words):
                    pts.append({"label": w, "kind": "texto", "emb": [round(float(v), 4) for v in reduced[i]]})
                for j in range(ecg.shape[0]):
                    pts.append({"label": f"ECG t{j}", "kind": "segmento ECG", "patch": j,
                                "feat": round(float(ecg_patch[min(j * (N_PATCH // max(ecg.shape[0], 1)), N_PATCH - 1)]), 3),
                                "emb": [round(float(v), 4) for v in reduced[len(tok_words) + j]]})
                return pts

            embeddings = {
                "completo": make_condition(proj_full, pad_to_text=False),
                "perturbado": make_condition(proj_pert, pad_to_text=False),
                "sinproy": make_condition(pre_full, pad_to_text=True),
            }

            handle.write(json.dumps({
                "id": row["id"],
                "ecg_patch_saliency": ecg_patch,
                "token_saliency": token_saliency,
                "embeddings": embeddings,
            }, ensure_ascii=False) + "\n")
            handle.flush()
            count += 1
            print(f"[{count}] {row['id']} tokens={len(token_saliency)}", flush=True)

    print(json.dumps({"attributions": count, "output": str(args.output)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
