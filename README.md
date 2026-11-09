# LearnLoop-Predict — Student Support Research Prototype

**Sector:** Education  
**Task:** Binary classification: synthetic at-risk indicator

A synthetic course-term exercise using attendance, assessment performance, prior GPA, study time, LMS activity, late submissions, employment, and aid status.

## Data and provenance

The offline demo uses a deterministic synthetic generator in `learnloop/data.py`. No public or private dataset is downloaded or redistributed. The generated values are illustrative and do not represent real people, organizations, or operational systems. Synthetic output is not evidence of real-world model performance.

## Method

- **Baseline:** majority-class prior classifier.
- **Stronger model:** histogram gradient-boosting classifier.
- **Split:** Stratified random holdout of independent synthetic student-term rows.. The seed is fixed by default and configurable.
- **Evaluation:** accuracy, balanced accuracy, macro F1, confusion matrix, and (for binary tasks) ROC-AUC, average precision, precision, and recall at threshold 0.5.
- **Error analysis:** confusion matrix and misclassification count are emitted with the metrics.

Research/education only. This is not a dropout decision system. Educational support requires contextual human review, student consent, and careful assessment of subgroup effects. The working-status diagnostic is descriptive.

## Run locally

    python3 -m venv .venv
    . .venv/bin/activate
    python -m pip install -e ".[test]"
    python -m learnloop.train --seed 42 --samples 2400
    python -m learnloop.train --seed 42 --samples 2400 --json
    python -m pytest -q

`make setup`, `make train`, and `make test` provide shortcuts. Python 3.9 or newer is required. No network access or data download is needed after dependencies are installed.

## Repository layout

- `learnloop/data.py` — validated domain-specific synthetic generator.
- `learnloop/split.py` — repeatable holdout split.
- `learnloop/model.py` — baseline, stronger model, metrics, and diagnostic summaries.
- `learnloop/train.py` — train/evaluate CLI.
- `tests/` — validation, split, model, and CLI tests.
- `.github/workflows/ci.yml` — automated test and CLI smoke run.

## Intended use

Educational and research prototype only. Do not use the synthetic benchmark to make decisions about real people or operations. Review the data assumptions, error patterns, and subgroup diagnostics before describing any findings.
