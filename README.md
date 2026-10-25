# Arabic IT Helpdesk Copilot

A learning-first NLP project: understanding Transformers from the inside by building an assistant that triages Arabic IT support tickets written in Egyptian Arabic, often mixed with English.

> **Status: work in progress.** Phase 1 (data audit) and Phase 2 (baseline) are complete. Phase 3 (Transformer from scratch) is in progress.

## The idea

A ticket arrives:

> "The VPN مش شغال من الصبح وعندي meeting بعد 10 دقايق."

The copilot **suggests**; a human agent **decides**:

- the issue category (16 classes, e.g. Network › VPN)
- which diagnostic information is missing (error message, device, location…)
- relevant troubleshooting passages from a knowledge base
- a reply that cites those passages, or **escalation to a human** when the evidence is weak or the ticket is high-risk

The system keeps three things apart: what the **model predicts** (with calibrated confidence), what the **retrieved evidence** says, and what a **human has verified**.

## Learning goals

- Build a Transformer encoder **from scratch** in PyTorch: tokenizer, embeddings, positional encoding, Q/K/V attention, multi-head attention, feed-forward network, residual connections, LayerNorm and a classification head.
- Compare it with a **pretrained** Arabic model (MARBERTv2) after fine-tuning.
- Evaluate honestly: leakage-free splits, Macro-F1, calibration, abstention and error analysis.

## Roadmap

| Phase | What | Status |
|---|---|---|
| 1 | Data audit: leakage, duplicates, label noise, group split | ✅ done |
| 2 | Baseline: TF-IDF + Logistic Regression | ✅ done |
| 3 | Transformer encoder from scratch | ⏳ in progress |
| 4 | Fine-tuning a pretrained Arabic model (MARBERTv2) | |
| 5 | Multi-task and gated-fusion experiments | |
| 6 | Calibration and abstention (escalate when unsure) | |
| 7 | Knowledge base and retrieval | |
| 8 | Grounded reply generation with citations | |
| 9 | Human-written challenge set | |
| 10 | FastAPI + Streamlit demo | |
| 11 | Technical report | |

## Key findings so far (Phase 1)

These were measured on the full dataset in [`01_explore_data.ipynb`](01_explore_data.ipynb). The complete list is in [`data/README.md`](data/README.md).

- **The label is written inside the input.** 1,187 of 10,000 descriptions (11.9%) contain the ticket's own label path ("فيه مشكلة مرتبطة بـ L1 / L2 / L3").
- **Shortcut phrases were injected by the generator.** "نسخة الأوفيس" means Office Apps in 525 of 525 tickets, and "Error 809" means VPN in 111 of 111.
- **Some labels are systematically wrong.** For example, all 14 copies of the "License Error" template are labeled L3 = Crash. Metrics cannot detect this, because test copies carry the same wrong label.
- **Urgency is not related to the text.** It is therefore not used as a training target.
- **The text is heavily templated.** The 9,549 unique tickets come from only 3,548 templates, so the data is split **by template** to keep copies of the same template out of both train and test.

## Baseline results so far (Phase 2)

TF-IDF (word 1–2-grams) + Logistic Regression on the L2 category (16 classes), trained on train and scored on **val** (Macro-F1). The test set has not been touched yet. Details are in [`01_explore_data.ipynb`](01_explore_data.ipynb).

| Input | val Macro-F1 |
|---|---|
| Raw title + description | 0.806 |
| Leak sentence removed (the fair baseline) | 0.785 |
| Title only | 0.773 |
| Description only | 0.668 |
| Fair baseline, noisy batches removed | 0.920 |

- **Urgency cannot be learned from the text.** Random guessing scores 0.192, shuffled labels 0.205 and real labels 0.206 (for L2 the same three numbers are 0.056, 0.065 and 0.806).
- **About 20% of the data has scrambled labels.** Error analysis by generation batch showed that four batches (TCKT-008/011/013/014, 1,899 tickets) have 61–77% val error, against at most 24% for every other batch. Reading their tickets confirmed that titles do not match labels. Without them the fair baseline reaches 0.920 instead of 0.785; part of that jump comes from a cleaner val set, and the shortcut phrases are still in the text.

## Repository layout

Files and folders are added one at a time, in the step that first needs them.

```
01_explore_data.ipynb   Phase 1: data audit (all checks and findings) + Phase 2: baseline
02_transformer.ipynb    Phase 3 (in progress): Transformer encoder built piece by piece
data/README.md          dataset source, license, how the split was made, all findings
data/raw/               downloaded data (git-ignored, except MANIFEST.json)
data/splits/            train / val / test ticket IDs
environment.yml         conda environment spec
requirements.lock.txt   exact installed package versions
```

## Setup

```bash
conda env create -f environment.yml
conda activate helpdesk
```

Then download the dataset file into `data/raw/` (see [`data/README.md`](data/README.md)).

## Data and license

The project uses a **synthetic** dataset (LLM-generated tickets) licensed **CC BY-NC 4.0**. Results on synthetic tickets do not prove real-world performance. Details are in [`data/README.md`](data/README.md).
