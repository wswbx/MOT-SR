# MOT-SR

Initial code release for the paper **“MOT-SR: Multi-Objective Tool-Augmented Symbolic Regression via LLMs”**.

Main Figure:

![Main Figure](./images/motsr.png)

This repository contains an initial research version of MOT-SR. It is provided for reference and early use, and may still be cleaned up or updated.

## Minimal Usage

- Environment: see [requirements.txt](./requirements.txt) or [environment.yml](./environment.yml)
- Main entry: [main.py](./main.py)
- Example script: [run_motsr.sh](./run_motsr.sh)

## Reproducibility

Prompts are in [specs/](specs/). Experiment settings are provided through
[main.py](main.py) and [run_motsr.sh](run_motsr.sh), and environment
specifications are in [requirements.txt](requirements.txt) and
[environment.yml](environment.yml). See [REPRODUCIBILITY.md](REPRODUCIBILITY.md)
for task-specific commands, random seeds, model and request settings,
failure handling, candidate-log availability and EMRI data-generation updates.

## Final equations

The four benchmark equation structures from MOT-SR (LLaMA-3.1) and the
fitted EMRI correction are available in
[results/final_equations](results/final_equations/README.md).

## Note

This release is intended to accompany the paper and share the core codebase.
Reproducibility resources, final equations and the coverage of the available
materials are documented in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
