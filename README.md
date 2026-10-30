# Distributed Matrix Factorization

Research code for experimenting with decentralized and federated matrix
factorization for recommendation systems. The project compares communication
topologies and learning strategies, and includes simulations of malicious
updates and lightweight defenses.

> **Research prototype:** Attack and defense components are experimental
> simulations, not production security controls. Results depend on random
> initialization and are not guaranteed to be reproducible across runs.

## What is included

- **Gossip learning** (`models/gossip_learning.py`): decentralized local
  training, compressed item-factor exchange, age-weighted merging, network
  churn, and complete, ring, star, or random topologies.
- **Distributed gradient descent** (`models/dgd_local.py`): distributed
  FedAvg, FedProx, and FedADMM experiment loops with configurable consensus
  weights and network topologies.
- **Attack and defense simulations** (`security/`): data poisoning, model
  manipulation, Byzantine-style updates, trust-based filtering, and
  statistical anomaly detection.
- **Data utilities** (`data/`): MovieLens ratings loading, train/test splitting,
  and horizontal or vertical partitioning.
- **Baseline matrix factorization** (`models/base_mf.py`).

The project is organized as follows:

```text
data/       Dataset loading and preprocessing
models/     Matrix-factorization and decentralized-learning algorithms
network/    Network topology, node state, and consensus weights
security/   Simulated attacks and defenses
test.py     Gossip-learning experiment runner
fl_test.py  Distributed-gradient-descent experiment runner
```

## Requirements

- Python 3.10 or later
- The packages listed in [`requirements.txt`](requirements.txt)
- A locally prepared MovieLens ratings file (see below)

## Setup

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

## Prepare the data

The experiment runners load `MovieLens100K` from
`dataset/MovieLens100K/ratings.csv`. The loader expects a **headerless,
comma-separated** file with four columns in this order:

```text
user_id,item_id,rating,timestamp
```

User and item IDs must be one-based. Each row represents one rating. Obtain the
MovieLens data from [GroupLens](https://grouplens.org/datasets/movielens/) and
prepare the file in this format before running an experiment. The dataset is
not included in this repository.

## Validation

GitHub Actions checks that the tracked Python modules compile on Python 3.10
through 3.13 for pushes and pull requests. This is a syntax check, not a
replacement for running the research experiments or validating their
scientific results.

Run the same check locally from the repository root:

```bash
python -m compileall -q data models network security test.py fl_test.py
```

## Run experiments

Run commands from the repository root:

```powershell
python test.py
```

Runs the gossip-learning experiment entry point. Its current default
configuration runs the defense experiments and writes plots and cached data
under `results/`.

```powershell
python fl_test.py
```

Runs the distributed-gradient-descent experiment suite and writes its outputs
under `dgd_results/`. This suite can take substantial time because it evaluates
multiple algorithms, topologies, and attack/defense scenarios.

Both runners use MovieLens100K. Configure the experiment parameters in the
respective script before running if you need a smaller or different experiment.

## Data, research materials, and licensing

Local datasets, PDFs, generated results, and Python cache files are excluded by
`.gitignore`. In particular, MovieLens data and the papers/reports in the
original working folder are not redistributed here; obtain datasets and
research papers from their respective sources and observe their terms.

This repository currently has no software license. Unless a license is added,
the source code remains under its default copyright protection; do not assume
permission to reuse or redistribute it.
