# Spell-Magic

---

## Description

Spell-magic is an auto spell-correction tool for `Bangla`. (\_under development)

## Installation

This project uses python `3.11.7` as it's `dev` environment.

This is the `dvc` bucket URL in google storage [`gs://spell-magic-dvc`](`gs://spell-magic-dvc`).
The bucket is publicly accessible. You simply need to place your `GOOGLE_APPLICATION_CREDENTIALS` in the root of the project, named `gcp-creds.json`.

#### Tools required

- Task (https://taskfile.dev/)
- Docker (https://docs.docker.com/get-docker/)
- Pyright (`vscode`)

### Docker

To install with docker

#### Build

> `docker build -t spell-magic .`

#### Run

> `docker run --rm -p 8080:8080 spell-magic`

### Local

You can install the `dev` requirements by running -

> `pip install -r requirements-dev.txt`

For running the production environment (with poetry) -

> `poetry install --no-root`

Finally, pull the weights

> `dvc pull`

Now, run

> `uvicorn app.main:app --reload --port 8080`

\_\_Note 1: It is recommended to use a separate `venv` for local installation.

## Usage

Test with the following curl request

```bash
curl --location 'localhost:8080/api/v1/correct' \
--header 'Content-Type: application/json' \
--data '{
"text": "আমি বংলায় গন গাঁই"
}'
```

Expected response:

```json
{
  "corrected": "আমি বাংলায় গান গাই।",
  "delay_in_seconds": 0.1145162582397461
}
```

## Project Structure

Here is the project structure

```
.
├── app/ --> contains the api
├── datasets/ --> contains the datasets
├── model_artifacts/ --> serialized model weights and tokenizers
├── models/ --> empty placeholder
├── notebooks/ --> contains training and evaluation pipelines
├── tests/ --> contains some unit tests
├── utils/ --> contains reusable code snippets
├── Dockerfile
├── gcp-creds.json
├── lefthook.yml
├── Taskfile.yml
├── poetry.lock
├── poetry.toml
├── pyproject.toml
├── pyrightconfig.json
├── README.md
└── requirements-dev.txt
```

## Dataset

For this experiment, the [`Potrika`](https://arxiv.org/abs/2210.09389) dataset was used to generate the synthetic data.

A subset of the entire dataset was used to train the model.

| Domain            | Sample Count |
| ----------------- | ------------ |
| Economy           | 5k           |
| Education         | 5k           |
| Entertainment     | 5k           |
| International     | 5k           |
| National          | 5k           |
| Politics          | 5k           |
| ScienceTechnology | 5k           |
| Sports            | 5k           |

Although, it's hard to simulate human error patterns, an attempt was made to generate some common errors -

| Function                                   | Action                                         |
| ------------------------------------------ | ---------------------------------------------- |
| swap_adjacent_characters                   | sometimes swaps consecutive characters         |
| insert_random_characters                   | randomly inserts characters in a sentence      |
| delete_random_characters                   | randomly deletes characters from a sentence    |
| substitute_phonetically_similar_characters | randomly swaps phonetically similar characters |
| repeat_characters                          | repeats random characters                      |
| remove_random_punctuation                  | deletes punctuation from random places         |
| delete_random_words                        | deletes random words from a sentence           |
| insert_random_spaces                       | inserts spaces at random positions             |

## Model

The problem of correcting spelling can be tackled in various ways. Some common ways are using statistical models, NER, hand picked rules, Seq2Seq, etc..
In this experiment, a Seq2Seq model was used [t5-small](https://huggingface.co/csebuetnlp/banglat5_small) to correct incorrect sentences. Using a transformer based model can essentially reduce the requirement of a lot of granular hand picked rules. With sufficient amount of diverse data, it can achieve a notable score at correcting spelling.

## Evaluation

The following tables contains the evaluation scores for the model

| Metric | Score  | Scale |
| ------ | ------ | ----- |
| Bleu   | 74.20  | 0-100 |
| WER    | 0.1296 | 0-1   |
| CER    | 0.0999 | 0-1   |

Note that the model was trained for only for 10 epochs on `40k` samples from the `Potrika` dataset.
For training and evaluation related details, please refer to the notebooks directory.
Here is a [`weights and bias`](https://wandb.ai/mdmmn378/spell-correction/?workspace=user-mdmmn378) dashboard containing the training logs and evaluation reports.

## Improvements

This project was done as a proof of concept. There are a few ways to improve the model.

- ML
  - Training the model with real world data, and adding more diverse synthetic data to the dataset should improve the scores..
  - Train a larger variation of `t5` should contribute to the improvement.
  - A pre-trained tokenizer was used instead of a custom trained one. The goal was to take full advantage of the pre-trained model embeddings. Training the model with a larger dataset along with a customized tokenizer should improve the performance.
  - Right now, the model has a smaller context length (128 tokens). It can be increased to handle longer contexts.
  - No heuristics are being applied to this project. In real world, there are a lot of edge cases, and heuristics must be applied in those scenarios.
  - (Unverified) In my opinion, a graph based model could be used along with the Seq2Seq to generalize token level dependency better.
  - (Unverified) A token classifier could be used to ensure the alignment of the source and target texts.
- Ops
  - Currently, the model is being trained from notebooks, without any DP/DDP pipeline. For larger models, it's better to include multi-node training capabilities.
  - The model is being loaded from `safetensors`. In production, an `onnx` serialized model should yield more tokens per second.
  - Early stopping and other regularization methods could be added to make the training pipeline robust.

## License

> MIT License
