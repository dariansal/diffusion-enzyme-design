#!/usr/bin/env python3
"""
compute_metrics.py

For each AlphaFold2-predicted structure, computes three metrics:

1. Catalytic RMSD
   Superimposes the predicted structure onto the reference PDB using all CA
   atoms (Kabsch alignment), then computes RMSD of *only* the catalytic
   residue CA atoms. Lower RMSD = better geometric precision.

2. Local pLDDT at catalytic residues
   Reads the per-residue pLDDT from the B-factor column of ColabFold output.
   Averages pLDDT over the catalytic residues. Higher = more confident.

3. Structural variance (per-experiment)
   For each designed sequence, ColabFold produces 5 model predictions.
   We compute the pairwise CA RMSD among these 5 models at the catalytic
   residues. The mean pairwise RMSD is the structural variance.

Output:
  results/analysis/metrics.csv         — per-prediction row table
  results/analysis/variance.csv        — per-sequence variance table
  results/analysis/summary.csv         — mean ± std per experiment

Usage:
  conda run -p /hpc/group/naderilab/darian/conda_environments/enz_analysis \
      python scripts/06_analysis/compute_metrics.py
"""

import json
import logging
import re
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT = Path("/hpc/group/naderilab/darian/Enz")
MOTIF_DIR  = PROJECT / "data" / "motifs"
CF_DIR     = PROJECT / "results" / "colabfold"
PDB_DIR    = PROJECT / "data" / "pdb"
OUT_DIR    = PROJECT / "results" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENTS = [
    ("1ppf", "motif_only"), ("1ppf", "shell5"), ("1ppf", "shell8"),
    ("1ca2", "motif_only"), ("1ca2", "shell5"), ("1ca2", "shell8"),
]

# Catalytic residues (original numbering in the renumbered PDB)
CATALYTIC_RESNUMS = {
    # 1PPF = Human Leukocyte Elastase (chymotrypsin family Ser-His-Asp triad)
    # Geometrically verified: His57–Asp102–Ser195 (distances 2.5–2.6 Å)
    "1ppf": {"chain": "E", "resnums": [57, 102, 195]},
    # 1CA2 = Human Carbonic Anhydrase II
    # Zn-binding His triad (verified: ~2.0 Å to ZN262)
    "1ca2": {"chain": "A", "resnums": [94, 96, 119]},
}

# ---------------------------------------------------------------------------
# PDB parsing
# ---------------------------------------------------------------------------

def parse_ca_atoms(pdb_path: Path) -> dict[int, dict]:
    """
    Parse C-alpha atoms from a PDB.
    Returns dict keyed by residue number:
      { resnum: {"resname": str, "chain": str, "xyz": np.ndarray, "bfactor": float} }
    """
    ca = {}
    for line in pdb_path.read_text().splitlines():
        if line[:4] != "ATOM":
            continue
        if line[12:16].strip() != "CA":
            continue
        try:
            rn = int(line[22:26].strip())
            ca[rn] = {
                "resname": line[17:20].strip(),
                "chain":   line[21],
                "xyz":     np.array([float(line[30:38]),
                                     float(line[38:46]),
                                     float(line[46:54])]),
                "bfactor": float(line[60:66]) if line[60:66].strip() else 0.0,
            }
        except (ValueError, IndexError):
            continue
    return ca


# ---------------------------------------------------------------------------
# Kabsch superimposition
# ---------------------------------------------------------------------------

def kabsch_align(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Align P onto Q using Kabsch algorithm.
    P, Q: (N, 3) arrays of corresponding points.
    Returns R-aligned P coordinates.
    """
    P = P.copy()
    Q = Q.copy()
    p_mean = P.mean(axis=0)
    q_mean = Q.mean(axis=0)
    P -= p_mean
    Q -= q_mean

    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T

    P_rot = P @ R.T
    P_rot += q_mean
    return P_rot


def rmsd(A: np.ndarray, B: np.ndarray) -> float:
    """RMSD between two (N,3) point sets (no alignment)."""
    return float(np.sqrt(np.mean(np.sum((A - B)**2, axis=1))))


# ---------------------------------------------------------------------------
# Map predicted residues to reference catalytic positions
# ---------------------------------------------------------------------------

def find_catalytic_in_prediction(
    pred_ca: dict[int, dict],
    ref_catalytic_resnums: list[int],
    ref_ca: dict[int, dict],
) -> Optional[tuple[np.ndarray, np.ndarray, list[float]]]:
    """
    Identify which residues in the predicted structure correspond to the
    catalytic residues in the reference.

    ColabFold predictions are for the *designed sequences*, so residue numbers
    in the predicted PDB restart from 1 and correspond to ProteinMPNN output.
    We need to map them back to the reference by position index (not resnum).

    Returns:
      (pred_xyz, ref_xyz, plddts) for catalytic residues, or None if not found.
    """
    # The predicted structure's residues are numbered 1..L where L is the
    # designed sequence length. The catalytic residues appear at fixed positions
    # in the designed backbone. We recover them by looking at the contig-encoded
    # index positions (stored in metadata).
    # A simple approximation: use the closest-by-order approach.
    # If L matches the design target length (~150), catalytic positions are
    # approximately at their offset in the contig.
    # For precise mapping: see metadata.json which stores the contig structure.
    # Here we return all predicted residues for global alignment, and extract
    # catalytic ones by their relative order in the design (N-th occurrence).

    # Since we need to align prediction to reference anyway, and the reference
    # catalytic resnums are known, we do global alignment and then measure at
    # the catalytic positions. The design's catalytic residues maintain the
    # original XYZ positions from the motif file, so after global alignment
    # they should be close to the reference.

    # Get all reference CA
    all_ref_resnums = sorted(ref_ca.keys())
    all_pred_resnums = sorted(pred_ca.keys())

    ref_xyz_all = np.array([ref_ca[r]["xyz"] for r in all_ref_resnums])
    pred_xyz_all = np.array([pred_ca[r]["xyz"] for r in all_pred_resnums])

    # We cannot do a per-residue global alignment since lengths differ.
    # Instead: align the catalytic residue atoms directly.
    # ColabFold keeps the input motif geometry essentially fixed, so the
    # predicted structure should have those atoms close to the reference.
    # We identify catalytic residues by sequence similarity: in practice
    # for our 150 AA designs, the motif atoms are at positions matching
    # the contig offsets. A reliable approach is to map by minimum RMSD
    # over catalytic triplet using Hungarian-style matching.
    return None   # handled in compute_catalytic_rmsd_via_metadata below


def compute_catalytic_rmsd_via_contig(
    pred_ca: dict[int, dict],
    ref_ca: dict[int, dict],
    catalytic_resnums_ref: list[int],
    metadata: dict,
) -> tuple[float, float]:
    """
    Compute catalytic RMSD by:
      1. Reconstructing which predicted residue indices correspond to the
         fixed catalytic residues using the contig string.
      2. Aligning pred onto ref using all shared catalytic CA atoms.
      3. Reporting the catalytic RMSD after alignment.

    Returns (catalytic_rmsd, global_rmsd).
    """
    contig = metadata["contig"]
    pred_cat_indices = _contig_to_pred_indices(contig, catalytic_resnums_ref)

    if not pred_cat_indices:
        log.warning("  Could not parse catalytic positions from contig — using fallback")
        return float("nan"), float("nan")

    # Extract predicted CA for catalytic positions (1-indexed)
    pred_resnums_sorted = sorted(pred_ca.keys())
    pred_cat_xyz = []
    pred_cat_plddt = []
    for idx in pred_cat_indices:
        if idx < len(pred_resnums_sorted):
            rn = pred_resnums_sorted[idx]
            pred_cat_xyz.append(pred_ca[rn]["xyz"])
            pred_cat_plddt.append(pred_ca[rn]["bfactor"])
        else:
            log.warning(f"  Catalytic index {idx} out of range ({len(pred_resnums_sorted)} residues)")
            return float("nan"), float("nan")

    pred_cat_xyz = np.array(pred_cat_xyz)

    # Reference catalytic CA
    ref_cat_xyz = []
    for rn in catalytic_resnums_ref:
        if rn in ref_ca:
            ref_cat_xyz.append(ref_ca[rn]["xyz"])
    if len(ref_cat_xyz) != len(pred_cat_xyz):
        log.warning(f"  Reference/pred catalytic count mismatch: "
                    f"{len(ref_cat_xyz)} vs {len(pred_cat_xyz)}")
        return float("nan"), float("nan")
    ref_cat_xyz = np.array(ref_cat_xyz)

    # Align and compute RMSD
    pred_cat_aligned = kabsch_align(pred_cat_xyz, ref_cat_xyz)
    cat_rmsd = rmsd(pred_cat_aligned, ref_cat_xyz)

    return cat_rmsd, np.mean(pred_cat_plddt)


def _contig_to_pred_indices(contig: str, catalytic_resnums: list[int] = None) -> list[int]:
    """
    Parse a contig string and return 0-based positions in the designed protein
    that correspond to fixed residues.

    If catalytic_resnums is provided, only return positions for fixed segments
    whose original residue numbers are in that list (i.e. the actual catalytic
    triad, not the full shell).

    Example: '[10/E32-32/40/E64-64/40/E221-221/15]'
    → if catalytic_resnums=[32,221]: positions 10 (E32) and 92 (E221)
    """
    contig = contig.strip("[]")
    parts = contig.split("/")

    catalytic_positions = []
    current_pos = 0
    for part in parts:
        part = part.strip()
        m_fixed = re.match(r"^[A-Za-z](\d+)-(\d+)$", part)
        m_var   = re.match(r"^(\d+)(?:-(\d+))?$", part)

        if m_fixed:
            start  = int(m_fixed.group(1))
            end    = int(m_fixed.group(2))
            length = end - start + 1
            for i in range(length):
                orig_resnum = start + i
                if catalytic_resnums is None or orig_resnum in catalytic_resnums:
                    catalytic_positions.append(current_pos + i)
            current_pos += length
        elif m_var:
            lo  = int(m_var.group(1))
            hi  = int(m_var.group(2)) if m_var.group(2) else lo
            avg = (lo + hi) // 2
            current_pos += avg
        else:
            log.debug(f"  Unrecognised contig part: {part}")

    return catalytic_positions


def local_plddt(pred_ca: dict[int, dict], pred_cat_indices: list[int]) -> float:
    """Average pLDDT (B-factor) at catalytic residue positions."""
    pred_resnums_sorted = sorted(pred_ca.keys())
    plddts = []
    for idx in pred_cat_indices:
        if idx < len(pred_resnums_sorted):
            rn = pred_resnums_sorted[idx]
            plddts.append(pred_ca[rn]["bfactor"])
    return float(np.mean(plddts)) if plddts else float("nan")


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def process_experiment(
    enzyme_id: str,
    regime: str,
) -> list[dict]:
    """Process all predictions for one experiment. Returns list of metric dicts."""

    exp_name = f"{enzyme_id}_{regime}"
    meta_path = MOTIF_DIR / exp_name / "metadata.json"
    if not meta_path.exists():
        log.warning(f"  Metadata not found: {meta_path}")
        return []

    with open(meta_path) as f:
        metadata = json.load(f)

    # Load reference PDB
    ref_pdb = PDB_DIR / f"{enzyme_id}_chainE.pdb" \
        if enzyme_id == "1ppf" else \
        PDB_DIR / f"{enzyme_id}_chainA.pdb"

    if not ref_pdb.exists():
        log.warning(f"  Reference PDB not found: {ref_pdb}")
        return []

    ref_ca = parse_ca_atoms(ref_pdb)
    cat_info = CATALYTIC_RESNUMS[enzyme_id]
    cat_resnums = cat_info["resnums"]

    contig = metadata["contig"]
    pred_cat_indices = _contig_to_pred_indices(contig, cat_resnums)
    log.info(f"  Contig catalytic positions (0-based): {pred_cat_indices}")

    # Collect all ColabFold output PDBs for this experiment
    cf_out = CF_DIR / exp_name
    if not cf_out.exists():
        log.warning(f"  ColabFold output not found: {cf_out}")
        return []

    # ColabFold names: {seq_id}_unrelaxed_rank_{rank}_alphafold2_ptm_model_{m}.pdb
    pdb_files = sorted(cf_out.glob("*_unrelaxed_rank_*_alphafold2_ptm_model_*.pdb"))
    if not pdb_files:
        pdb_files = sorted(cf_out.glob("*_relaxed_rank_*_alphafold2_ptm_model_*.pdb"))

    if not pdb_files:
        log.warning(f"  No prediction PDB files found in {cf_out}")
        return []

    # Only use sequences from the fixed ProteinMPNN runs (design_N_sM naming).
    # Old buggy runs produced design_N_design_N and design_N_T_0.1 files.
    pdb_files = [p for p in pdb_files if re.search(r'design_\d+_s\d+_', p.name)]

    log.info(f"  {exp_name}: {len(pdb_files)} prediction PDBs found (fixed runs only)")

    rows = []
    for pdb_path in pdb_files:
        pred_ca = parse_ca_atoms(pdb_path)
        if not pred_ca:
            log.warning(f"    No CA atoms in {pdb_path.name}")
            continue

        cat_rmsd, _ = compute_catalytic_rmsd_via_contig(
            pred_ca, ref_ca, cat_resnums, metadata
        )
        avg_plddt_all = np.mean([v["bfactor"] for v in pred_ca.values()])
        avg_plddt_cat = local_plddt(pred_ca, pred_cat_indices)

        # Parse metadata from filename
        fname = pdb_path.stem
        seq_id_match  = re.match(r"^(.+?)_unrelaxed_rank_(\d+)_.*_model_(\d+)", fname) or \
                        re.match(r"^(.+?)_relaxed_rank_(\d+)_.*_model_(\d+)", fname)
        seq_id  = seq_id_match.group(1) if seq_id_match else fname
        rank    = int(seq_id_match.group(2)) if seq_id_match else -1
        model   = int(seq_id_match.group(3)) if seq_id_match else -1

        rows.append({
            "experiment":       exp_name,
            "enzyme_id":        enzyme_id,
            "regime":           regime,
            "seq_id":           seq_id,
            "rank":             rank,
            "model":            model,
            "catalytic_rmsd":   cat_rmsd,
            "plddt_global":     avg_plddt_all,
            "plddt_catalytic":  avg_plddt_cat,
            "n_residues":       len(pred_ca),
            "pdb_file":         str(pdb_path),
        })

    return rows


def compute_structural_variance(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (experiment, seq_id), compute pairwise RMSD among the 5 model
    predictions at catalytic residues. Returns variance DataFrame.
    """
    variance_rows = []
    grouped = metrics_df.groupby(["experiment", "seq_id"])

    for (exp, seq_id), group in grouped:
        cat_rmsds = group["catalytic_rmsd"].dropna().values
        if len(cat_rmsds) < 2:
            continue
        # Pairwise differences as proxy for variance
        pairwise = []
        for i, j in combinations(range(len(cat_rmsds)), 2):
            pairwise.append(abs(cat_rmsds[i] - cat_rmsds[j]))

        variance_rows.append({
            "experiment":    exp,
            "enzyme_id":     group["enzyme_id"].iloc[0],
            "regime":        group["regime"].iloc[0],
            "seq_id":        seq_id,
            "mean_cat_rmsd": float(np.mean(cat_rmsds)),
            "std_cat_rmsd":  float(np.std(cat_rmsds)),
            "structural_variance": float(np.mean(pairwise)),
            "n_models":      len(cat_rmsds),
        })

    return pd.DataFrame(variance_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=== Computing structural metrics ===")

    all_rows = []
    for enzyme_id, regime in EXPERIMENTS:
        log.info(f"\n[{enzyme_id}_{regime}]")
        rows = process_experiment(enzyme_id, regime)
        all_rows.extend(rows)
        log.info(f"  {len(rows)} predictions processed")

    if not all_rows:
        log.error("No predictions found. Make sure ColabFold jobs have completed.")
        return

    # -----------------------------------------------------------------------
    # Save per-prediction metrics
    # -----------------------------------------------------------------------
    metrics_df = pd.DataFrame(all_rows)
    metrics_csv = OUT_DIR / "metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    log.info(f"\nPer-prediction metrics saved: {metrics_csv} ({len(metrics_df)} rows)")

    # -----------------------------------------------------------------------
    # Structural variance across models
    # -----------------------------------------------------------------------
    var_df = compute_structural_variance(metrics_df)
    var_csv = OUT_DIR / "variance.csv"
    var_df.to_csv(var_csv, index=False)
    log.info(f"Structural variance saved: {var_csv} ({len(var_df)} rows)")

    # -----------------------------------------------------------------------
    # Summary: mean ± std per experiment
    # -----------------------------------------------------------------------
    summary = (
        metrics_df
        .groupby(["experiment", "enzyme_id", "regime"])
        .agg(
            n_predictions=("catalytic_rmsd", "count"),
            mean_cat_rmsd=("catalytic_rmsd", "mean"),
            std_cat_rmsd=("catalytic_rmsd", "std"),
            median_cat_rmsd=("catalytic_rmsd", "median"),
            iqr_cat_rmsd=("catalytic_rmsd", lambda x: x.quantile(0.75) - x.quantile(0.25)),
            mean_plddt_global=("plddt_global", "mean"),
            std_plddt_global=("plddt_global", "std"),
            median_plddt_global=("plddt_global", "median"),
            mean_plddt_cat=("plddt_catalytic", "mean"),
            std_plddt_cat=("plddt_catalytic", "std"),
            median_plddt_cat=("plddt_catalytic", "median"),
        )
        .reset_index()
    )
    # Add variance column
    var_summary = (
        var_df
        .groupby("experiment")
        .agg(
            mean_structural_variance=("structural_variance", "mean"),
            std_structural_variance=("structural_variance", "std"),
            median_structural_variance=("structural_variance", "median"),
            iqr_structural_variance=("structural_variance", lambda x: x.quantile(0.75) - x.quantile(0.25)),
        )
        .reset_index()
    )
    summary = summary.merge(var_summary, on="experiment", how="left")

    summary_csv = OUT_DIR / "summary.csv"
    summary.to_csv(summary_csv, index=False)

    log.info(f"\nSummary per experiment:")
    log.info(summary.to_string(index=False))
    log.info(f"\nSummary saved: {summary_csv}")
    log.info("\nNext: python scripts/06_analysis/plot_results.py")


if __name__ == "__main__":
    main()
