# Controlling Active-Site Geometric Precision with Diffusion-Based Protein Design

Enzyme catalysis depends on the precise three-dimensional positioning of a small set of active-site residues — even subtle deviations in distances or angles can reduce catalytic efficiency by orders of magnitude. Generative models such as RFdiffusion enable conditional backbone design around fixed structural motifs, ProteinMPNN designs sequences for those backbones, and AlphaFold2 provides high-accuracy structure prediction. This project investigates whether increasing structural constraints around a catalytic motif during diffusion-based design produces proteins that better maintain active-site geometry, and whether that effect generalizes across mechanistically distinct enzyme classes.

## Hypothesis

Tighter motif conditioning during RFdiffusion backbone generation will produce designs with lower catalytic RMSD relative to the reference geometry, higher local AlphaFold2 confidence (pLDDT) at active-site residues, and reduced structural variance across multiple AlphaFold2 predictions. I test this across two enzyme systems representing fundamentally different catalytic architectures — a Ser-His-Asp hydrogen-bond relay (1PPF) and a Zn²⁺-coordinated metal triad (1CA2) — to evaluate whether controllability generalizes across enzyme classes.

---

## Enzyme Systems

| System | PDB | Enzyme | Catalytic Residues | Motif Type |
|--------|-----|--------|-------------------|------------|
| 1ca2 | 1CA2 chain A | Human Carbonic Anhydrase II | His94–His96–His119 | Zn²⁺-binding triad |
| 1ppf | 1PPF chain E | Human Leukocyte Elastase | His57–Asp102–Ser195 | Ser-His-Asp charge relay |

## Conditioning Regimes

| Regime | Fixed Residues (1PPF) | Fixed Residues (1CA2) | Description |
|--------|----------------------|----------------------|-------------|
| motif_only | 3 | 3 | Catalytic triad only — maximum scaffold freedom |
| shell5 | 27 | 30 | All residues within 5 Å of any catalytic atom |
| shell8 | 61 | 61 | All residues within 8 Å of any catalytic atom |

6 experiments total (2 enzymes × 3 regimes), 20 RFdiffusion backbones each.

---

## Results Summary
A full write-up of the experimental design, results, and discussion is available in [`results/report.pdf`](results/report.pdf).


![Summary Panel](results/analysis/figures/fig5_summary_panel.png)
*8Å shell conditioning reduces catalytic RMSD by 50% and structural variance by 63% for the Zn²⁺-coordinated active site (1CA2), while the Ser-His-Asp catalytic triad (1PPF) worsens monotonically with tighter conditioning — revealing an enzyme-class-dependent response to motif conditioning.*

---

## Repository Structure

```
.
├── configs/                    # Per-experiment contig strings and residue lists
│   ├── 1ca2_motif_only/
│   ├── 1ca2_shell5/
│   ├── 1ca2_shell8/
│   ├── 1ppf_motif_only/
│   ├── 1ppf_shell5/
│   └── 1ppf_shell8/
├── data/
│   ├── pdb/                    # Raw PDB downloads (1PPF.pdb, 1CA2.pdb)
│   └── motifs/                 # Per-experiment motif PDBs and contig strings
├── results/
│   ├── analysis/
│   │   ├── figures/            # 5 publication figures (PDF)
│   │   └── summary.csv         # Aggregated metrics table
│   ├── report.tex              # LaTeX report
│   └── report.md               # Markdown report
├── scripts/
│   ├── 01_data/
│   │   └── download_pdbs.py
│   ├── 02_motifs/
│   │   └── extract_motifs.py
│   ├── 03_rfdiffusion/
│   │   ├── run_rfdiffusion.sh
│   │   └── submit_all.sh
│   ├── 04_proteinmpnn/
│   │   ├── run_proteinmpnn.sh
│   │   └── submit_all.sh
│   ├── 05_colabfold/
│   │   ├── run_colabfold.sh
│   │   └── submit_all.sh
│   └── 06_analysis/
│       ├── compute_metrics.py
│       └── plot_results.py
├── slurm/                      # Environment setup SLURM scripts
│   ├── fix_rfdiffusion_env.sh
│   ├── fix_proteinmpnn_env.sh
│   └── install_colabfold_env.sh
└── tools/                      # External tools (not committed — see Setup)
    ├── RFdiffusion/
    ├── ProteinMPNN/
    └── localcolabfold/
```

---

## Reproducibility


### 1. Clone and enter the repo

```bash
git clone git@github.com:dariansal/diffusion-enzyme-design.git
cd diffusion-enzyme-design
```

### 2. Install external tools

```bash
# RFdiffusion
git clone https://github.com/RosettaCommons/RFdiffusion.git tools/RFdiffusion

# ProteinMPNN
git clone https://github.com/dauparas/ProteinMPNN.git tools/ProteinMPNN

# LocalColabFold
cd tools && bash -c "$(curl -fsSL https://raw.githubusercontent.com/YoshitakaMo/localcolabfold/main/install_colabfold_linux.sh)"
cd ..
```

### 3. Create conda environments

**RFdiffusion environment (CUDA 12 / H200)**

```bash
conda create -n enz_rfdiffusion python=3.10 -y
conda activate enz_rfdiffusion

pip install numpy"<2.0" \
    torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121

pip install dgl -f https://data.dgl.ai/wheels/torch-2.1/cu121/repo.html

pip install pyrsistent iopath opt_einsum e3nn \
    hydra-core omegaconf biopython \
    tools/RFdiffusion/env/SE3Transformer

pip install -e tools/RFdiffusion
```

**ProteinMPNN environment (CUDA 12 / H200)**

```bash
conda create -n enz_proteinmpnn python=3.10 -y
conda activate enz_proteinmpnn

pip install numpy"<2.0" \
    torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121

pip install biopython "typing_extensions>=4.0"
```

**Analysis environment**

```bash
conda create -n enz_analysis python=3.10 -y
conda activate enz_analysis
pip install biopython numpy scipy matplotlib pandas
```

> **Note on CUDA:** If `module load CUDA/12.x` is unavailable on compute nodes, set the library path directly in each SLURM script:
> ```bash
> export LD_LIBRARY_PATH=/opt/apps/rhel9/cuda-12.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
> ```

### 4. Download model weights

**RFdiffusion ActiveSite checkpoint**

```bash
mkdir -p weights/rfdiffusion
# Download from RFdiffusion releases — ActiveSite_ckpt.pt
wget -P weights/rfdiffusion \
  https://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc6/ActiveSite_ckpt.pt
```

**ColabFold / AlphaFold2 weights**

```bash
# Run from inside the localcolabfold conda env
conda activate tools/localcolabfold/colabfold-conda
python -m colabfold.download
# Weights download to tools/localcolabfold/colabfold/params/ (~3.5 GB)
```

**ProteinMPNN weights** are included in the ProteinMPNN repo (`tools/ProteinMPNN/vanilla_model_weights/`).

### 5. Prepare data

```bash
conda activate enz_analysis
python scripts/01_data/download_pdbs.py        # downloads 1PPF.pdb, 1CA2.pdb
python scripts/02_motifs/extract_motifs.py     # writes configs/ and data/motifs/
```

### 6. Run the pipeline

Each stage is submitted as a SLURM array job. Jobs are chained with `--dependency=afterok`.

```bash
# Stage 1: RFdiffusion (6 jobs, ~4h each, 1 GPU)
bash scripts/03_rfdiffusion/submit_all.sh

# Stage 2: ProteinMPNN (6 jobs, ~30min each, 1 GPU)
# Submit after RFdiffusion completes, or chain with --dependency
bash scripts/04_proteinmpnn/submit_all.sh

# Stage 3: ColabFold (6 jobs, ~7h each, 1 GPU)
bash scripts/05_colabfold/submit_all.sh

# Stage 4: Analysis (CPU, ~5 min)
conda activate enz_analysis
python scripts/06_analysis/compute_metrics.py
python scripts/06_analysis/plot_results.py
```

### 7. Check outputs

```bash
# Per-experiment summary
cat results/analysis/summary.csv

# Figures
ls results/analysis/figures/
# fig1_catalytic_rmsd.pdf
# fig2_local_plddt.pdf
# fig3_structural_variance.pdf
# fig4_rmsd_vs_plddt.pdf
# fig5_summary_panel.pdf
```

## Metrics

| Metric | Definition | Better = |
|--------|-----------|----------|
| Catalytic RMSD | Cα RMSD of catalytic triad after Kabsch alignment to reference | Lower |
| Local pLDDT | AF2 confidence averaged at catalytic residues | Higher |
| Structural variance | Mean pairwise catalytic RMSD across 5 AF2 models | Lower |

---

## Citation

If you use this pipeline, please cite:
```
Watson et al. (2023). De novo design of protein structure and function with RFdiffusion. Nature, 620, 1089–1100.
Dauparas et al. (2022). Robust deep learning-based protein sequence design using ProteinMPNN. Science, 378, 49–56.
Mirdita et al. (2022). ColabFold: making protein folding accessible to all. Nature Methods, 19, 679–682.
```