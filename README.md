# TGCAE -- Temporal Graph Convolutional Autoencoder for Robust Anomaly Detection in Object-Centric Process Mining

A research project for **unsupervised anomaly detection in Object-Centric Event Logs (OCEL)**. Business processes involving multiple interacting objects are represented as process graphs, and deep autoencoders (flat, sequential, and graph-based) score each event by its reconstruction error. The project centers on **TGCAE (Temporal Graph Convolutional Autoencoder)**, which extends the GCN autoencoder baseline with explicit temporal modeling to improve detection of timestamp-related anomalies.

## Key Features

- **OCEL-to-graph pipeline**: converts object-centric event logs into process graphs (nodes = events, edges = object co-occurrence) using the [`ocpa`](https://github.com/corrni/ocpa) library.
- **Synthetic anomaly injection**: three anomaly types are injected at configurable contamination rates (5%–40%) to create labeled evaluation data.
- **Four detector architectures**: flat AE, LSTM-AE, GCN-AE baselines and the proposed TGCAE.
- **Temporal modeling in TGCAE**:
  - A deep non-linear time encoder operating on *elapsed time* (time since execution start) rather than absolute timestamps.
  - Per-node **adaptive gated residual fusion** that learns how much temporal information to blend into the structural embedding.
  - **Type-weighted anomaly scoring**: reconstruction errors are weighted differently per attribute group (attribute vs. timestamp vs. event anomalies).
  - **IQR-based thresholding** for converting scores into binary outlier labels.
- **Comprehensive evaluation**: AUC-ROC, AUC-PR, F1, Recall@k overall and per anomaly type, across multiple contamination rates, plus ablation studies and statistical significance tests.

## Models

| Model | Type | Input | Implementation |
|---|---|---|---|
| **AE** | Flat autoencoder | One-hot encoded event table | [`models/AE.py`](models/AE.py) |
| **LSTM-AE** | LSTM autoencoder | Event sequences per execution | [`models/LSTMAE.py`](models/LSTMAE.py) |
| **GCN-AE** | Graph convolutional autoencoder | Process graph | [`models/GCNAE.py`](models/GCNAE.py) |
| **TGCAE** (proposed) | Temporal GCN autoencoder | Process graph + temporal features | [`models/TGCAE.py`](models/TGCAE.py) |

### TGCAE Architecture

```
Node features X ──► GCN Encoder ──────────────► h_struct ──┐
                                                           ├──► concat ──► Gate g
EVENT_ELAPSED_TIME  ────► Time Encoder (MLP) ────► h_time ─┘         │
                                                                     ▼
                              h_fused ◄── Fusion(Linear+LayerNorm+ReLU)
                                       h_final = (1 - g) ⊙ h_struct + g ⊙ h_fused
                                       X̂ = GCN Decoder(h_final)
                                       score = weighted MSE(X, X̂) per attribute group
```

## Pipeline Overview

```
ocel/*.csv ──(0)──► data-prepro/ ──(1)──► outliers/ ──(2)──► process_graphs/ + process_tables/ ──(3/4/5)──► results/
```

| Stage                   | Script                                                                                                                        | Description                                                                                                 |
|-------------------------|-------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| 0. Preprocessing        | [`0_data_preprocessing.py`](0_data_preprocessing.py)                                                                          | Cleans raw OCEL CSVs (BPIC 2017, DS2, BPI 2013) into a normalized event table.                              |
| 1. Outlier injection    | [`1_outlier_injection_multi_rates.py`](1_outlier_injection_multi_rates.py) | Injects three anomaly types at one or several contamination rates (0.05–0.4) and saves ground-truth labels. |
| 2. Process encoding     | [`2_process_encoding_multi_rates.py`](2_process_encoding_multi_rates.py)    | Uses `ocpa` to build process graphs (`.sav`) and flat/sequential process tables.                            |
| 3. Graph model training | [`3_training_gcnae.py`](3_training_gcnae.py)                                                                                  | Trains GCN-AE and TGCAE on the process graphs.                                                              |
| 4. Other model training | [`4_training_others.py`](4_training_others.py)                                                                                | Trains the flat AE and LSTM-AE baselines on the process tables.                                             |
| 5. Graph model training | [`5_training_tgcae.py`](5_training_tgcae.py)                                                                                  | Trains TGCAE on the process graphs.                                                                         |

## Installation

Requires **Python 3.8**.

```bash
# 1. Create a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

Main dependencies:

- `torch==2.0.1+cu118`, `torch-geometric==2.3.0` (+ `torch-scatter`, `torch-sparse`, `torch-cluster`)
- `ocpa==1.3.3`, `pm4py==2.2.32` — process mining / OCEL processing
- `pygod==1.0.0`, `pyod==1.1.0` — outlier detection utilities
- `pandas==1.5.3`, `numpy==1.23.5`, `scikit-learn==1.3.0`
- `matplotlib==3.8.0`, `seaborn==0.12.2` — visualization

> Note: the PyG extension wheels must match your PyTorch/CUDA build (`pt20cu118` in this project). To run on CPU only, install the CPU build of PyTorch instead.

## Usage

Run the stages in order from the project root. Each stage skips datasets/contamination folders whose outputs already exist.

```bash
# Stage 0: preprocess raw OCEL data (edit the script to enable/disable datasets)
python 0_data_preprocessing.py

# Stage 1: inject outliers (adjust contamination_rates inside the script)
python 1_outlier_injection_multi_rates.py

# Stage 2: encode process graphs and tables
python 2_process_encoding_multi_rates.py

# Stage 3: train & evaluate GCNAE
python 3_training_gcnae.py

# Stage 4: train & evaluate AE and LSTM-AE baselines
python 4_training_others.py

# Stage 5: train & evaluate TGCAE
python 5_training_tgcae.py
```

Per-run results (metrics, hit rates, per-type Recall@k) are written to `result/` and `results/` as CSV files.

### Attribute Dimensions

Graph models require the per-attribute-group feature dimensions of each dataset, passed as `attribute_dims`:

| Dataset | Graph models (`process_graphs`) | Table models (`process_tables`) |
|---|---|---|
| BPIC 2017 | `[41, 1, 1, 1, 1, 1, 1, 2, 2, 4, 2, 3, 14, 145]` | `[2, 41, 4, 2, 1, 3, 1, 14, 1, 1, 1, 145, 1, 2]` |
| DS2 | `[21, 1, 1, 20, 17]` | `[17, 21, 1, 20, 1]` |

## Datasets

Raw OCEL files live in `ocel/`:

- **BPIC 2017** — loan application process (`application`, `offer` objects).
- **DS2** — order handling process (`orders`, `packages`, `items` objects).

## Evaluation

For every dataset × contamination-rate × model combination, the framework reports (mean ± std over multiple runs):

- **AUC-ROC** and **AUC Precision-Recall**
- **F1 score** (IQR-based thresholding)
- **Recall@k** (k = contamination rate in percent), overall and **per anomaly type** (T1/T2/T3)

## Project Structure

```
TGCAE/
├── ocel/                                               
├── functions/             
├── models/                
├── results/      
├── 0_data_preprocessing.py ... t_training_tgcae.py 
└── requirements.txt
```

## Acknowledgments

This project builds upon the original [DAEiOcBPvGNN](https://github.com/niro-a/DAEiOcBPvGNN) repository (Nolle et al., *DAEiOcBPvGNN: Deep Autoencoder-based Anomaly Detection in Object-Centric Business Process Data using Graph Neural Networks*), extended with the temporal GCN autoencoder (TGCAE), multi-rate outlier experiments, ablation studies, and statistical evaluation.

## License

This project is distributed under the **BSD 2-Clause License** (see [LICENSE](LICENSE)).
