"""M7-C7: measure CPU inference time for B5 and B3 on the same test matrices."""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dosesyn.models.baseline_models import build_dl_model  # noqa: E402
from dosesyn.models.m6b_models import build_m6b_model  # noqa: E402
from dosesyn.train.m6a_trainer import slice_graph_batch, vocab_graph_batch  # noqa: E402
from dosesyn.training.m4_data import load_graphs, load_matrix_table, table_batch  # noqa: E402

OUT = ROOT / "outputs/metrics/m7c"
M6B = ROOT / "configs/m6b_experiments.yaml"
TABLE = load_matrix_table("s1_random_condition", "test", seed=42)
N = len(TABLE.matrix_ids)


def time_model(name, model, forward):
    torch.set_num_threads(4)
    model.eval()
    runs = []
    with torch.no_grad():
        # warm-up
        for start in range(0, min(512, N), 128):
            idx = list(range(start, min(start + 128, N)))
            forward(idx)
        for _ in range(3):
            t0 = time.perf_counter()
            for start in range(0, N, 128):
                idx = list(range(start, min(start + 128, N)))
                forward(idx)
            runs.append((time.perf_counter() - t0) * 1000 / N)
    return {"model": name, "n_matrices": N, "ms_per_matrix_median": statistics.median(runs),
            "ms_per_matrix_runs": runs, "threads": 4, "device": "cpu"}


def main():
    cfg = yaml.safe_load(M6B.read_text())
    graphs, _ = load_graphs(cfg["inputs"]["m4_config"])
    device = torch.device("cpu")

    b5 = build_m6b_model(cfg["variants"]["b5_t3_film_reg"], cfg["model"])
    b5.load_state_dict(torch.load(ROOT / "outputs/runs/m6b/m6b__b5_t3_film_reg__s1_random_condition__seed_42/best.pt", map_location="cpu")["state_dict"])
    b5.eval()

    def fwd_b5(idx):
        b = table_batch(TABLE, idx, device=device)
        g = slice_graph_batch(vocab_graph_batch(TABLE, graphs, device), TABLE, idx, device)
        return b5.forward_all(b, g)["surface"]

    b3 = build_dl_model("b3_gine", n_drugs=36, cell_dim=50, hidden=128, drug_dim=32,
                        mlp_hidden=256, gine_layers=3, dropout=0.1)
    state = torch.load(ROOT / "outputs/runs/m4/b3_gine__s1_random_condition__seed_42/best.pt", map_location="cpu")
    b3.load_state_dict(state["model_state"])
    b3.eval()

    def fwd_b3(idx):
        b = table_batch(TABLE, idx, device=device)
        g = slice_graph_batch(vocab_graph_batch(TABLE, graphs, device), TABLE, idx, device)
        return b3.forward(b, g)

    rows = [time_model("B5 DoseSyn", b5, fwd_b5), time_model("B3 GINE", b3, fwd_b3)]
    (OUT / "inference_timing.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    for r in rows:
        print(f"{r['model']}: {r['ms_per_matrix_median']:.2f} ms/matrix ({r['n_matrices']} matrices, CPU, 4 threads)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
