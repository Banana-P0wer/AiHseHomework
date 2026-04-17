# Practical Assignment 01: Toxic Comment Classification

## Problem Statement

The goal of this project is to develop a Python solution that classifies
comments on source code as **toxic** or **non-toxic**.

The assignment required:

1. Preparing labeled comments from the
   [ToxiCR](https://github.com/WSU-SEAL/ToxiCR/tree/master) dataset.
2. Training a classical machine learning model on textual features.
3. Training a transformer-based model.
4. Comparing the approaches using evaluation metrics and analyzing the results.

## Dataset

Each record contains a code review comment and a binary `is_toxic` label:

- `0` means a non-toxic comment;
- `1` means a toxic comment.

The cleaning step produced the following dataset sizes:

| Split | Raw rows | Cleaned rows | Non-toxic | Toxic |
| --- | ---: | ---: | ---: | ---: |
| Train | 19,571 | 12,707 | 10,215 | 2,492 |
| Test | 230 | 228 | 119 | 109 |

## Workflow

### 1. Data Preparation

Data preparation is implemented in
[`01_data_preparation.ipynb`](./01_data_preparation.ipynb).
The notebook loads the source tables, cleans the comments, displays processing
statistics, and saves `clean_train.csv` and `clean_test.csv`.

The processing step includes:

- removing links and email addresses;
- normalizing Unicode characters and non-standard apostrophes;
- expanding common English contractions, for example
  `doesn't` to `does not`;
- normalizing obfuscated profanity using `profane-words.txt`;
- reducing artificial character repetition in ordinary words;
- preserving meaningful technical notation and programming-related words;
- removing empty rows and duplicates.

### 2. Classical Model

The experiment is implemented in
[`02_classic_model.ipynb`](./02_classic_model.ipynb).

Text is represented with `TfidfVectorizer` using:

- unigrams and bigrams: `ngram_range=(1, 2)`;
- at most 20,000 features;
- rare-feature filtering: `min_df=2`.

The classifier is logistic regression with `class_weight="balanced"` to
account for class imbalance in the training split.

### 3. RoBERTa Model

The experiment is implemented in
[`03_transformer_model.ipynb`](./03_transformer_model.ipynb).

The pretrained `roberta-base` model is extended with a binary classification
head. Training parameters:

| Parameter | Value |
| --- | --- |
| Maximum sequence length | 128 tokens |
| Batch size | 8 |
| Epochs | 3 |
| Learning rate | `1e-5` |
| Optimizer | `AdamW` |

The trained model is saved to `models/roberta-toxic/`. This generated
directory is not tracked because it can be recreated by running the notebook.

## Results

### Model Comparison

| Model | Accuracy | Toxic precision | Toxic recall | Toxic F1 |
| --- | ---: | ---: | ---: | ---: |
| TF-IDF + Logistic Regression | 0.912 | 0.894 | 0.927 | 0.910 |
| RoBERTa | 0.925 | 0.897 | 0.954 | 0.924 |

Both models substantially exceeded the assignment reference level of
`F1 approximately 0.7`. RoBERTa achieved a slightly better result, primarily
through higher recall for toxic comments.

### Interpreting the Classical Model

The weights of logistic regression can be directly inspected. Terms most
strongly associated with the toxic class include `ugly`, `darn`, `sucks`,
`crap`, and `hate`.

Features associated with the non-toxic class include `remove`, `kill`, `pid`,
`question`, and `self`. This illustrates the importance of domain context:
for example, `kill` may refer to terminating a process rather than abuse.

### Interpreting RoBERTa

Unlike the linear model, RoBERTa's classification head operates on a hidden
representation of the entire text rather than on weights of individual words.
It is therefore not correct to derive a list of the "most toxic tokens" in the
same way as for TF-IDF. In this experiment, that reflects the main trade-off:
RoBERTa is more accurate but less directly interpretable.

## Conclusions

The project implements the full workflow: cleaning labeled comments, training
two models, and comparing their quality.

The classical TF-IDF and logistic regression approach achieved a strong and
readily interpretable result of `F1 = 0.910` for the toxic class. RoBERTa
increased this result to `F1 = 0.924` by capturing context more effectively,
at the cost of greater computational demand and less interpretable features.

The final comparison is also recorded in
[`04_conclusions.ipynb`](./04_conclusions.ipynb).

## Project Structure

```text
01-toxic-review-classification/
├── data/
│   ├── raw/                   # original data
│   ├── lexicons/              # dictionaries used during cleaning
│   └── preprocessed/          # cleaned data
├── 01_data_preparation.ipynb  # data preparation
├── 02_classic_model.ipynb     # TF-IDF + Logistic Regression
├── 03_transformer_model.ipynb # RoBERTa
├── 04_conclusions.ipynb       # final conclusions
└── requirements.txt           # dependencies
```

## Running the Project

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Execute the notebooks in order:

1. `01_data_preparation.ipynb`;
2. `02_classic_model.ipynb`;
3. `03_transformer_model.ipynb`;
4. `04_conclusions.ipynb`.
