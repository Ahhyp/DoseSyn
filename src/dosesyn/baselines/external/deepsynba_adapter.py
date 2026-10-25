"""DeepSynBa-style external baseline adapted to the frozen O'Neil splits.

This is NOT a byte-for-byte reproduction of the published NCI-ALMANAC model:

* the official NCI-60 MoLFormer embeddings are unavailable for O'Neil drugs, so
  drugs are represented by 2048-bit RDKit Morgan fingerprints projected to the
  official 768-d drug width;
* the official 976 landmark genes are used from CCLE expression (950/976 exact
  symbol matches; the remaining 26 are zero-filled and reported);
* the published objective requires SynBa parameter targets / synergy metadata,
  which O'Neil does not provide.  We keep the DeepSynBa two-branch drug-cell
  encoder and the nine SynBa parameter heads, and train the resulting surface
  directly against the 16 measured X/X0 values with a surface Huber loss.

Results are therefore reported as a feature/objective-adapted supplementary
baseline, never as a faithful reproduction.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from dosesyn.training.metrics import compute_metrics  # noqa: E402
from dosesyn.training.m4_data import load_matrix_table  # noqa: E402

NCI_GEX = Path("/home/kouzheng_2/AH_YeP/DoseSyn/data/external/m7a_data/deepsynba/features/NCI-60_landmark_gex.csv")
GENE_DIM = 976
DRUG_BITS = 2048
DRUG_DIM = 768


def landmark_genes():
    df = pd.read_csv(NCI_GEX, index_col=0)
    return [str(x) for x in df.index]


def cell_gene_matrix(primary, genes):
    expr = pd.read_parquet(ROOT / "data/processed/cell_expression.parquet")
    meta = {"depmap_id", "oneil_cell_name", "standard_cell_name", "lineage", "disease"}
    gene_cols = [c for c in expr.columns if c not in meta]
    # expression columns may carry "SYMBOL (entrez)" in the M2 source; here they
    # are already bare symbols, but handle both.
    bare = {c.split(" (")[0]: c for c in gene_cols}
    missing = [g for g in genes if g not in bare]
    present = [g for g in genes if g in bare]
    mat = expr[["depmap_id"] + [bare[g] for g in present]].copy()
    mat.columns = ["depmap_id"] + present
    for g in missing:
        mat[g] = 0.0
    mat = mat[["depmap_id"] + genes]
    lookup = {r.depmap_id: r for r in mat.itertuples(index=False)}
    cells = list(primary.depmap_id.unique())
    arr = np.stack([np.asarray(lookup[c][1:], dtype=np.float32) for c in cells])
    return cells, arr, len(present), missing


def morgan_fingerprints(drug_mapping):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    fps = {}
    for r in drug_mapping.itertuples():
        mol = Chem.MolFromSmiles(r.canonical_smiles)
        if mol is None:
            fp = np.zeros(DRUG_BITS, dtype=np.float32)
        else:
            gen = AllChem.GetMorganGenerator(radius=2, fpSize=DRUG_BITS)
            fp = gen.GetCountFingerprintAsNumPy(mol).astype(np.float32)
        fps[r.drug_id] = fp
    return fps


class DeepSynBaAdapted(nn.Module):
    def __init__(self, gene_channel=256, emb_size=1024, drug_dim=DRUG_DIM, dropout=0.2):
        super().__init__()
        self.drug_proj = nn.Linear(DRUG_BITS, drug_dim)
        self.encoder = nn.Sequential(
            nn.Linear(GENE_DIM + drug_dim, gene_channel),
            nn.BatchNorm1d(gene_channel),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        heads = []
        for _ in range(9):
            heads.append(nn.Sequential(
                nn.Linear(gene_channel * 2, emb_size),
                nn.LayerNorm(emb_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(emb_size, emb_size // 2),
                nn.LayerNorm(emb_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(emb_size // 2, 1),
            ))
        self.heads = nn.ModuleList(heads)

    def forward(self, genes, d1, d2, da, db):
        z1 = self.encoder(torch.cat([genes, self.drug_proj(d1)], dim=1))
        z2 = self.encoder(torch.cat([genes, self.drug_proj(d2)], dim=1))
        z = torch.cat([z1, z2], dim=1)
        raw = torch.cat([h(z) for h in self.heads], dim=1)  # [B,9]
        e1 = torch.sigmoid(raw[:, 0]).view(-1, 1, 1)
        e2 = torch.sigmoid(raw[:, 1]).view(-1, 1, 1)
        e3 = torch.sigmoid(raw[:, 2]).view(-1, 1, 1)
        logC1 = torch.relu(raw[:, 3]).view(-1, 1, 1) + 1e-6
        logC2 = torch.relu(raw[:, 4]).view(-1, 1, 1) + 1e-6
        h1 = torch.relu(raw[:, 5]).view(-1, 1, 1) + 1e-3
        h2 = torch.relu(raw[:, 6]).view(-1, 1, 1) + 1e-3
        alpha = torch.relu(raw[:, 7]).view(-1, 1, 1) + 1e-3
        # sigma head is unused for deterministic prediction
        da = da.clamp_min(1e-12)[:, :, None]
        db = db.clamp_min(1e-12)[:, None, :]
        e0 = 100.0
        A = logC1 * h1 + logC2 * h2
        B = torch.log(da) * h1 + logC2 * h2
        C = logC1 * h1 + torch.log(db) * h2
        D = torch.log(da) * h1 + torch.log(db) * h2
        mx = torch.maximum(torch.maximum(A, B), torch.maximum(C, D))
        ea, eb, ec, ed = [torch.exp(t - mx) for t in (A, B, C, D)]
        num = ea * e0 + eb * e1 * e0 + ec * e2 * e0 + ed * e3 * e0 * alpha
        den = ea + eb + ec + ed * alpha
        return ((num / den) / e0).reshape(num.shape[0], -1)


def cell_batch(cells, genes, cell_to_idx, idx):
    return torch.tensor(np.stack([genes[cell_to_idx[c]] for c in cells[idx]]), dtype=torch.float32)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--fold", type=int)
    ap.add_argument("--epochs", type=int, default=160)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    split = "test"
    train = load_matrix_table(args.scenario, split="train", seed=args.seed, fold=args.fold)
    val = load_matrix_table(args.scenario, split="val", seed=args.seed, fold=args.fold)
    test = load_matrix_table(args.scenario, split="test", seed=args.seed, fold=args.fold)
    primary = pd.read_parquet(ROOT / "data/processed/oneil_primary_v1_matrix_index.parquet")
    genes = landmark_genes()
    cells, gmat, n_match, missing = cell_gene_matrix(primary, genes)
    cell_to_idx = {c: i for i, c in enumerate(cells)}
    cell_depmap = dict(zip(primary.cell_id, primary.depmap_id))
    # standardize genes on TRAIN cells only
    train_cells = np.array(sorted({cell_depmap[c] for c in set(train.cell_ids)}))
    tr_idx = [cell_to_idx[c] for c in train_cells]
    mu = gmat[tr_idx].mean(0)
    sd = gmat[tr_idx].std(0)
    sd[sd < 1e-8] = 1.0
    gmat = (gmat - mu) / sd
    dm = pd.read_csv(ROOT / "data/processed/drug_mapping.csv")
    fps = morgan_fingerprints(dm)
    drug_ids = list(train.drug_vocab)
    drug_to_idx = {d: i for i, d in enumerate(drug_ids)}
    fp_mat = np.stack([fps.get(d, np.zeros(DRUG_BITS, dtype=np.float32)) for d in drug_ids])

    def tensors(tab):
        idx = np.arange(len(tab.matrix_ids))
        x = torch.tensor(gmat[[cell_to_idx[cell_depmap[c]] for c in tab.cell_ids]], dtype=torch.float32)
        d1 = torch.tensor(fp_mat[tab.drug_a_index], dtype=torch.float32)
        d2 = torch.tensor(fp_mat[tab.drug_b_index], dtype=torch.float32)
        da = torch.tensor(tab.dose_a_um / 1e6, dtype=torch.float32)
        db = torch.tensor(tab.dose_b_um / 1e6, dtype=torch.float32)
        y = torch.tensor(tab.y, dtype=torch.float32)
        return x, d1, d2, da, db, y

    model = DeepSynBaAdapted().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.HuberLoss(delta=1.0)
    Xtr = tensors(train); Xva = tensors(val); Xte = tensors(test)
    n = len(train.matrix_ids)
    best = float("inf"); best_ep = -1; best_state = None; hist = []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = np.random.default_rng(1000 * ep + 7).permutation(n)
        total = 0.0
        for s in range(0, n, args.batch_size):
            b = torch.tensor(perm[s:s + args.batch_size], dtype=torch.long)
            xb = [t[b].to(device) for t in Xtr]
            yb = xb[-1].reshape(len(b), 16)
            pred = model(*xb[:-1])
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            total += float(loss) * len(b)
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = []
            for s in range(0, len(val.matrix_ids), 512):
                b = torch.arange(s, min(s + 512, len(val.matrix_ids)))
                xb = [t[b].to(device) for t in Xva]
                pv.append(model(*xb[:-1]).cpu().numpy())
            pv = np.concatenate(pv)
        m = compute_metrics(val.y, pv, matrix_ids=val.matrix_ids, condition_ids=val.condition_ids)
        val_macro = m["matrix_macro_rmse"]; val_point = m["point_rmse"]
        hist.append(dict(epoch=ep, train_loss=total / n, val_macro=val_macro, val_point=val_point, lr=sched.get_last_lr()[0]))
        print(f"ep {ep} train {total/n:.5f} val_macro {val_macro:.5f} point {val_point:.5f}", flush=True)
        if val_macro < best:
            best, best_ep = val_macro, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif ep - best_ep >= args.patience:
            break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        pt = []
        for s in range(0, len(test.matrix_ids), 512):
            b = torch.arange(s, min(s + 512, len(test.matrix_ids)))
            xb = [t[b].to(device) for t in Xte]
            pt.append(model(*xb[:-1]).cpu().numpy())
        pt = np.concatenate(pt)
    mt = compute_metrics(test.y, pt, matrix_ids=test.matrix_ids, condition_ids=test.condition_ids)
    pred = pd.DataFrame(dict(
        matrix_id=np.repeat(test.matrix_ids, 16),
        row_idx=np.tile(np.repeat(np.arange(4), 4), len(test.matrix_ids)),
        col_idx=np.tile(np.tile(np.arange(4), 4), len(test.matrix_ids)),
        dose_a_um=test.dose_a_um[:, :, None].repeat(4, axis=2).reshape(-1),
        dose_b_um=test.dose_b_um[:, None, :].repeat(4, axis=1).reshape(-1),
        y_true=test.y.reshape(-1), y_pred=pt.reshape(-1),
    ))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(out, index=False)
    meta = dict(model="deepsynba_adapted", scenario=args.scenario, seed=args.seed, fold=args.fold,
                n_train=len(train.matrix_ids), n_val=len(val.matrix_ids), n_test=len(test.matrix_ids),
                best_epoch=best_ep, val_metrics=m, test_metrics=mt, history=hist,
                gene_dim=GENE_DIM, genes_matched=n_match, genes_missing=missing,
                drug_dim=DRUG_DIM, drug_features="morgan2048_projected",
                params=sum(p.numel() for p in model.parameters()),
                runtime_seconds=time.time() - t0, notes="feature/objective-adapted; not a faithful reproduction")
    (out.parent / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: mt[k] for k in ("matrix_macro_rmse", "point_rmse")}), flush=True)


if __name__ == "__main__":
    main()
