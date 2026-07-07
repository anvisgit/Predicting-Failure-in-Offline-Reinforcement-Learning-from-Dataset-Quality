# Offline Reinforcement Learning Research

This project implements pipeline for Offline Reinforcement Learning, focusing on Conservative Q-Learning (CQL) and Implicit Q-Learning (IQL) on the D4RL benchmark. The primary research question is: **How does dataset quality and coverage affect the performance and failure modes of conservative offline RL algorithms, and can we predict failure before deployment?**

## Running on Windows

This project does not require the `d4rl` Python package at runtime. The loader downloads the original Berkeley D4RL HDF5 files directly into `C:\Users\Anvi\.d4rl_cache`, which avoids the usual Windows install problems with `d4rl`.

Use the bundled runner so each new terminal uses the same venv that already has the packages installed:

```powershell
.\scripts\run.ps1 --algorithm cql --env hopper --dataset medium-replay --n_steps 5000 --device cpu
```

For the degradation study, use:

```powershell
.\scripts\degrade.ps1 --quick
```

The runner first uses:

```text
C:\Users\Anvi\.gemini\antigravity\scratch\offline-rl-research\venv\Scripts\python.exe
```

If that venv is not present, it falls back to:

```text
C:\Users\Anvi\Desktop\offline-rl-research\venv\Scripts\python.exe
```

To activate the known-good venv manually in PowerShell:

```powershell
& "C:\Users\Anvi\.gemini\antigravity\scratch\offline-rl-research\venv\Scripts\Activate.ps1"
```

Then run commands from `C:\Users\Anvi\Desktop\offline-rl-research`.

## Research Stages

1. **Baseline Reproduction**: Reproducing published CQL and IQL results on D4RL `medium-replay` datasets to establish a trusted baseline.
2. **Systematic Degradation Study**: Isolating the effects of dataset properties through three degradation protocols:
    *   **Coverage Reduction**: Subsampling trajectories to reduce state-space coverage.
    *   **Quality Degradation**: Mixing expert and random trajectories.
    *   **Gaussian Noise Injection**: Adding observation noise.
3. **Failure Prediction**: Building a predictive model to forecast algorithm failure based solely on pre-training dataset statistics.

## Directory Structure

```
offline-rl-research/
├── configs/                # YAML configuration files
│   ├── default.yaml
│   ├── cql.yaml
│   └── iql.yaml
├── scripts/                # Execution scripts for experiments
│   ├── train.py
│   ├── run_baseline.py
│   ├── run_degradation_study.py
│   ├── run_failure_prediction.py
│   └── analyze_results.py
├── src/                    # Source code
│   ├── algorithms/         # Algorithm wrappers (CQL, IQL)
│   ├── data/               # Data loading and degradation protocols
│   ├── metrics/            # Performance evaluation and dataset stats
│   ├── prediction/         # Failure predictor model
│   └── utils/              # Logging, visualization, helpers
├── README.md
└── requirements.txt
```

### 1. Train a Single Model

```bash
python scripts/train.py --algorithm cql --env hopper --dataset medium-replay
```

### 2. Run Baseline Reproduction (Stage 1)

```bash
python scripts/run_baseline.py --n_steps 200000 --seed 42
```

### 3. Run Degradation Study (Stage 2)

```bash
python scripts/run_degradation_study.py --quick
```

### 4. Run Failure Prediction (Stage 3)

```bash
python scripts/run_failure_prediction.py --train_envs halfcheetah hopper --test_env walker2d
```

### 5. Analyze Results

```bash
python scripts/analyze_results.py
```

## Citations

*   **CQL**: Kumar, A., Zhou, A., Tucker, G., & Levine, S. (2020). Conservative Q-Learning for Offline Reinforcement Learning.
*   **IQL**: Kostrikov, I., Nair, A., & Levine, S. (2021). Offline Reinforcement Learning with Implicit Q-Learning.
*   **D4RL**: Fu, J., Kumar, A., Nachum, O., Tucker, G., & Levine, S. (2020). D4RL: Datasets for Deep Data-Driven Reinforcement Learning.
