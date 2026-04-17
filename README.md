# Applied Machine Learning for Text and Source Code

Two practical machine learning projects completed for an Artificial
Intelligence course at **HSE University** in 2025. The repository explores two
problems at the intersection of natural language processing and software
engineering:

- detecting toxic language in code review comments;
- generating meaningful Python function names from source code.

The projects are organized as reproducible notebook-based studies: data
preparation, model experiments, evaluation, and conclusions.

## Projects

| Project | Problem | Methods | Key result |
| --- | --- | --- | --- |
| [Toxic Comment Classification](./01-toxic-review-classification/) | Classify code review comments as toxic or non-toxic | Text normalization, TF-IDF + Logistic Regression, fine-tuned RoBERTa | RoBERTa achieved **0.924 F1** for toxic comments |
| [Function Name Suggestion](./02-func-name-suggestion/) | Generate a Python function name from its body | CodeSearchNet, `tree-sitter`, CodeT5+ generation | Documentation improved **Exact Match from 0.143 to 0.184** |

## 01. Toxic Comment Classification

Code reviews contain technical vocabulary, informal language, and occasional
abuse. This project investigates whether toxic review comments can be detected
reliably while preserving domain-specific context.

The experiment uses labeled comments from the
[ToxiCR](https://github.com/WSU-SEAL/ToxiCR/tree/master) dataset. The
preparation step normalizes text, removes noisy values and duplicates, handles
obfuscated profanity, and retains relevant programming terms.

Two classifiers were evaluated on the cleaned test data:

| Model | Accuracy | Precision, toxic | Recall, toxic | F1, toxic |
| --- | ---: | ---: | ---: | ---: |
| TF-IDF + Logistic Regression | 0.912 | 0.894 | 0.927 | 0.910 |
| RoBERTa | **0.925** | **0.897** | **0.954** | **0.924** |

The classical model provides an interpretable baseline, while RoBERTa improves
the final score by modeling context more effectively.

**Workflow:** [data preparation](./01-toxic-review-classification/01_data_preparation.ipynb) ·
[classical model](./01-toxic-review-classification/02_classic_model.ipynb) ·
[transformer model](./01-toxic-review-classification/03_transformer_model.ipynb) ·
[conclusions](./01-toxic-review-classification/04_conclusions.ipynb)

[Read the project documentation](./01-toxic-review-classification/README.md)

## 02. Function Name Suggestion

Good function names summarize intent, but existing code does not always make
that intent immediately visible. This project studies whether a pretrained
model for source code can infer useful Python function names and whether
documentation improves its predictions.

The experiment uses 1,000 Python functions from
[CodeSearchNet](https://huggingface.co/datasets/code-search-net/code_search_net).
Functions are parsed with `tree-sitter` to produce two input variants:
executable code only, and code with docstrings and comments. The pretrained
`Salesforce/codet5p-220m` model then generates function names without
additional training.

| Input given to the model | Exact Match | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | ---: | ---: | ---: | ---: |
| Function body only | 0.143 | 0.384 | 0.198 | 0.381 |
| Body with documentation and comments | **0.184** | **0.467** | **0.267** | **0.465** |

Natural-language context improved every measured metric. The result is also a
useful reminder that Exact Match is strict for naming tasks: multiple
different identifiers may describe the same function correctly.

**Workflow:** [data preparation](./02-func-name-suggestion/01_data_preparation.ipynb) ·
[model experiment](./02-func-name-suggestion/02_model_experiment.ipynb) ·
[conclusions](./02-func-name-suggestion/03_conclusions.ipynb)

[Read the project documentation](./02-func-name-suggestion/README.md)

## Repository Structure

```text
.
├── 01-toxic-review-classification/
│   ├── 01_data_preparation.ipynb
│   ├── 02_classic_model.ipynb
│   ├── 03_transformer_model.ipynb
│   ├── 04_conclusions.ipynb
│   └── README.md
├── 02-func-name-suggestion/
│   ├── 01_data_preparation.ipynb
│   ├── 02_model_experiment.ipynb
│   ├── 03_conclusions.ipynb
│   └── README.md
└── README.md
```

## Technology Stack

`Python` · `Jupyter Notebook` · `pandas` · `scikit-learn` · `PyTorch` ·
`Hugging Face Transformers` · `Hugging Face Datasets` · `tree-sitter` ·
`evaluate`

## Running the Projects

Each project has its own dependency file and detailed documentation:

```bash
cd 01-toxic-review-classification
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

or:

```bash
cd 02-func-name-suggestion
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Execute the notebooks in numeric order inside the selected project directory.
