# CAREPath

**CAREPath (Context-Aware REasoning Path)** is a KG–LLM framework for **drug repurposing** that predicts disease–drug associations by combining:

- **DFS-like constrained semantic path encoding** over short disease–gene–drug paths  
- **BFS-like mechanism context augmentation** from 1-hop gene neighborhoods  

It fuses these two complementary signals and scores pairs using an **XGBoost-based stacking ensemble**.

Across five biomedical knowledge graphs (MSI, PrimeKG, Hetionet, SuppKG, KEGG50k) and 18 baselines, CAREPath achieves the best overall **AUPRC**, including under the disease cold-start setting, with gains of up to **3.8%**.

This repository includes code to:
1) **Extract per-pair embeddings** (semantic path + mechanism context)  
2) **Run prediction and evaluation** (CV with random/drug/disease splits)

---

## What CAREPath does (high-level)
![CAREPath pipeline](carepath/pipeline.png)


Given a disease–drug pair *(s, d)*:

### 1) Constrained semantic path encoding (DFS-like)
- Enumerate short simple paths **s → gene(s) → d** with constraints (max hop=3, limited number of intermediate genes `k_max`).
- We set `k_max = 2` based on a coverage–redundancy trade-off analysis across BKGs.
- Convert each path into an NLI-style prompt:
  - `Premise: {disease} involves genes {g1, ..., gk}.`
  - `Hypothesis: {drug} can be repurposed to treat {disease}.`
  - `Label:`
- Encode each prompt with **BioLinkBERT (CLS)** and aggregate via **max pooling** to obtain a pair-specific semantic path embedding **Z_path(s,d)**.
- If no path exists, use a fallback prompt with `genes none`.

### 2) Mechanism context augmentation (BFS-like)
- Build entity-level context from **1-hop gene/protein neighbors only** (to reduce direct disease–drug leakage).
- Encode neighborhood sentences with BioLinkBERT and mean-pool into initial context embeddings.
- Apply similarity-guided pooling + residual mixing (with mixing weights α for diseases and β for drugs):
  - **Drugs:** pool within ATC-prefix–related drugs
  - **Diseases:** pool via gene-signature similarity (cosine kNN on weighted gene vectors)
- Produces robust context embeddings **Z_ctx^drug(d)** and **Z_ctx^dis(s)**, especially when paths are sparse/noisy.

### 3) Feature fusion + prediction
- Concatenate features:
  - `Z_path(s,d)`, `Z_ctx^dis(s)`, `Z_ctx^drug(d)`
- Score with an **XGBoost stacking ensemble** for final association probability.

---

## Setup

```bash
pip install -r requirements.txt
```

BioLinkBERT (`michiyasunaga/BioLinkBERT-base`) is downloaded automatically from Hugging Face on first run. To use a custom cache location, set the `BIOLINKBERT_CACHE_DIR` environment variable.

---
## Usage

> The examples below use the **MSI** knowledge graph. The same pipeline applies to the other four BKGs (PrimeKG, Hetionet, SuppKG, KEGG50k) once each is preprocessed into the same `graph.txt` / `nodetypes.tsv` / `dda_labels.tsv` format.

---
## Datasets

All five knowledge graphs used in this study are publicly available from their original sources. Due to their respective licenses, **we do not redistribute the datasets (raw or processed) in this repository**; please obtain them directly from the links below.

| Knowledge graph | Source | Reference |
|---|---|---|
| **MSI** | https://github.com/snap-stanford/multiscale-interactome | Ruiz et al., *Nat Commun* 2021 |
| **PrimeKG** | https://dataverse.harvard.edu/dataverse/primekg | Chandak et al., *Sci Data* 2023 |
| **Hetionet** | https://github.com/hetio/hetionet | Himmelstein et al., *eLife* 2017 |
| **SuppKG** | [https://github.com/rebeccawyq/SuppKG](https://github.com/biothings/SuppKG) | Schutte et al., *J Biomed Inform* 2022 |
| **KEGG50k** | [https://github.com/mdnunez/kegg50k](https://figshare.com/s/bbfc7b82d17e0b8b6a43) | Kanehisa et al., *Nucleic Acids Res* 2017 |

Each KG must be preprocessed into the common format used by CAREPath:

- `graph.txt` — edge list of the unified interaction graph
- `nodetypes.tsv` — node type annotations (drug / disease / gene)
- `dda_labels.tsv` — labeled drug–disease pairs (positives + sampled negatives)
- `7_drug_classification_df.tsv` — ATC drug classes (for drug context pooling)

Preprocessing scripts are provided in `scripts/`. The MSI pipeline (`scripts/preprocess.py`) is documented below as a worked example; the other four KGs follow the same target format.

---

### MSI (worked example)

We use the Multiscale Interactome (MSI) resources provided by the official repository:
https://github.com/snap-stanford/multiscale-interactome

We download MSI supplementary datasets **#1–#7**:
- **#1–#5** (interaction datasets) → unified interaction graph (`graph.txt`) and node types (`nodetypes.tsv`)
- **#6** (approved drug–disease pairs) → positive label set, converted into `dda_labels.tsv` and augmented with **random negative sampling**
- **#7** (ATC drug classes) → ATC-prefix–based drug context pooling

> The preprocessing step keeps the extracted MSI files under `data/raw/msi/extracted/`. The embedding step reads these raw files to map node IDs to human-readable drug/disease/gene names, so do not delete that folder after preprocessing.

---
## 0) Preprocess data
This step generates:
- `dataset/graph.txt`
- `dataset/nodetypes.tsv`
- `dataset/dda_labels.tsv`
- `dataset/7_drug_classification_df.tsv` (ATC drug classes)

```md
# Download MSI data.tar.gz and preprocess
python scripts/preprocess.py \
  --download \
  --out_dir "dataset" \
  --neg_ratio 1.0 \
  --seed 42
```
If you already have data.tar.gz, place it under data/raw/msi/data.tar.gz and run:
```md
python scripts/preprocess.py \
  --out_dir "dataset" \
  --neg_ratio 1.0 \
  --seed 42
```

## 1) Extract embeddings

This step creates a per-pair embedding dictionary (`.pkl`) keyed by `"{disease}__{drug}"`.

Passing `--dataset_dir` lets the script locate all dataset files (`graph.txt`, `nodetypes.tsv`, `dda_labels.tsv`, and the ATC file) automatically.

### Example

```md
python -m extract_embeddings.main \
  --dataset_dir "dataset" \
  --output_file "outputs/embeddings.pkl" \
  --seed 42 \
  --max_genes 2 \
  --workers 5 \
  --run_id 0
```

You can also point to each file explicitly instead of `--dataset_dir`:

```md
python -m extract_embeddings.main \
  --network_file "dataset/graph.txt" \
  --node_type_file "dataset/nodetypes.tsv" \
  --pair_file "dataset/dda_labels.tsv" \
  --atc_file "dataset/7_drug_classification_df.tsv" \
  --output_file "outputs/embeddings.pkl" \
  --seed 42 \
  --max_genes 2 \
  --workers 5 \
  --run_id 0
```

## 2) Train and Prediction
```md
mkdir -p outputs

python -m prediction.train_and_prediction \
  --embedding_file "outputs/embeddings_seed42.pkl" \
  --pair_file "dataset/dda_labels.tsv" \
  --seed 42 \
  --n_splits 5 \
  --splits "random,drug,disease" \
  --output_file "outputs/cv_results.tsv" \
  --pred_detail_file "outputs/cv_pred_details.tsv"
```
