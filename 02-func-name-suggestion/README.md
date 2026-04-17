# Practical Assignment 02: Function Name Suggestion

## Problem Statement

The goal of this project is to develop a Python solution for predicting a
function name from its implementation. The source data consists of real
functions from open-source repositories, and prediction is performed with a
pretrained model designed for source code.

For example, for the function:

```python
def some_function(x):
    """Increases each number in the given list by 1."""
    return [i + 1 for i in x]
```

a suitable predicted name might be `increment_list_elements`.

The assignment required:

1. Preparing a dataset of function bodies and names.
2. Producing input variants with and without documentation.
3. Running a pretrained model without additional training.
4. Comparing generation quality for the two input variants.

## Dataset

The project uses the Python test split of
[CodeSearchNet](https://huggingface.co/datasets/code-search-net/code_search_net).
The experiment evaluates its first 1,000 functions.

The original dataset includes:

- complete function source code;
- the function name;
- documentation;
- repository and file-location information.

The `func_name` field can contain a qualified method name such as
`ClassA.foo`, so the declared function name is additionally extracted from the
syntax tree.

## Workflow

### 1. Data Preparation

Data preparation is implemented in
[`01_data_preparation.ipynb`](./01_data_preparation.ipynb).

Source code is parsed using `tree-sitter`, which operates on the syntactic
structure of a Python function instead of relying on fragile string matching.

For each example, the notebook creates:

- `parsed_func_name`, the function name extracted from the syntax tree;
- `body_no_comments`, the function body without its docstring and comments;
- `body_with_comments`, the body retaining documentation and comments.

All 1,000 selected functions were processed successfully. Each contains a
docstring, while 368 functions also contain additional comments.

### 2. Function Name Generation

The model experiment is implemented in
[`02_model_experiment.ipynb`](./02_model_experiment.ipynb).

It uses the pretrained `Salesforce/codet5p-220m` model without additional
training. The function name is replaced with the special `<extra_id_0>` token,
and the model reconstructs a suitable identifier from the function body.

Two input variants are compared:

1. Executable function code only.
2. Executable code with the docstring and comments.

Generation parameters:

| Parameter | Value |
| --- | --- |
| Maximum input length | 512 tokens |
| Batch size | 8 |
| Beam search width | 5 |
| Maximum generated length | 16 tokens |

Quality is measured with:

- **Exact Match**, the share of names that exactly equal the reference;
- **ROUGE**, overlap between generated and reference identifiers.

## Results

| Input data | Exact Match | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | ---: | ---: | ---: | ---: |
| Code only | 0.143 | 0.384 | 0.198 | 0.381 |
| Code, documentation, and comments | 0.184 | 0.467 | 0.267 | 0.465 |

Adding documentation and comments improved every measured score. For example,
for the reference name `get_vid_from_url`, the model suggested
`parse_query_param` without documentation and `extract_video_id` when a
docstring was available, which better communicates the function's purpose.

Exact Match remains relatively low because one function can have several
semantically correct names. It should therefore be considered together with
the ROUGE scores.

## Conclusions

The project prepares a dataset of real Python functions, parses their syntax,
and generates names with CodeT5+.

The experiment shows that even without additional training, the model can
suggest meaningful function names. Documentation and comments substantially
improve performance: Exact Match increased from `0.143` to `0.184`, while
ROUGE-1 increased from `0.384` to `0.467`.

Limitations include the lack of task-specific model training, the 512-token
input limit, and evaluation against a single reference name.

## Project Structure

```text
02-func-name-suggestion/
├── 01_data_preparation.ipynb # data preparation
├── 02_model_experiment.ipynb # function-name generation and evaluation
├── 03_conclusions.ipynb      # final conclusions
└── requirements.txt          # dependencies
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
2. `02_model_experiment.ipynb`;
3. `03_conclusions.ipynb`.

After the first step, the prepared dataset is saved under `artifacts/`. This
directory is generated on demand and is not tracked in the repository.
