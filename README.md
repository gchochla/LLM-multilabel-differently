# LLMs do Multi-Label Classification Differently

Link pending...
<!-- This repo contains the official implementation of [Large Language Models do Multi-Label Classification Differently](https://arxiv.org/abs/).  -->
Repo is still under construction, some scripts are missing.

## Abstract

> Multi-label classification is prevalent in real-world settings, but the behavior of Large Language Models (LLMs) in this setting is understudied. We investigate how autoregressive LLMs perform multi-label classification, with a focus on subjective tasks, by analyzing the output distributions of the models in each generation step. We find that their predictive behavior reflects the multiple steps in the underlying language modeling required to generate all relevant labels as they tend to suppress all but one label at each step. % owning to unrepresentative and spiky probability distributions at each step. We further observe that as model scale increases, their token distributions exhibit lower entropy, yet the internal ranking of the labels improves. Finetuning methods such as supervised finetuning and reinforcement learning amplify this phenomenon. To further study this issue, we introduce the task of distribution alignment for multi-label settings: aligning LLM-derived label distributions with empirical distributions estimated from annotator responses in subjective tasks. We propose both zero-shot and supervised methods which improve both alignment and predictive performance over existing approaches.

## Installation

This repo uses `Python 3.10` (type hints, for example, won't work with some previous versions). After you create and activate your virtual environment (with conda, venv, etc), install local dependencies with:

```bash
pip install -e .[dev]
```

## Data preparation


To run the GoEmotions experiments, we recommend using the emotion pooling we set up based on the hierarchical clustering (besides, the bash scripts are set up for it). To do so, create the file `emotion_clustering.json` under the root folder of the dataset with the following contents:

```JSON
{
    "joy": [
        "amusement",
        "excitement",
        "joy",
        "love"
    ],
    "optimism": [
        "desire",
        "optimism",
        "caring"
    ],
    "admiration": [
        "pride",
        "admiration",
        "gratitude",
        "relief",
        "approval",
        "realization"
    ],
    "surprise": [
        "surprise",
        "confusion",
        "curiosity"
    ],
    "fear": [
        "fear",
        "nervousness"
    ],
    "sadness": [
        "remorse",
        "embarrassment",
        "disappointment",
        "sadness",
        "grief"
    ],
    "anger": [
        "anger",
        "disgust",
        "annoyance",
        "disapproval"
    ]
}
```

For MFRC, please create a folder for the dataset (even though we use HuggingFace `datasets` for it), and copy the file `./configs/MFRC/splits.yaml` to that directory.

## Run experiments

Experiments are logged with [legm](https://github.com/gchochla/legm), so refer to the documentation there for an interpretation of the resulting `logs` folder, but navigating should be intuitive enough with some trial and error. Note that some bash scripts have arguments, which are self-explanatory. Make sure to run scripts from the root directory of this repo.

Also, you should create a `.env` file with your OpenAI key if you want to perform experiments with the GPTs.

```bash
OPENAI_API_KEY=<your-openai-key>
```

### Analysis

- `scripts/ml-distribution/linear-probing.sh` will run the scripts to, first, run the models on the entire dataset, and then run linear probing one them.
- `scripts/ml-distribution/plot_figures.sh` will plot their distributions.

More scripts pending...

### Distribution Alignment

- `scripts/ml-distribution/scores.sh` will calculate the scores of the linear probes to calculate alignment.
- `scripts/training/demux_ds.sh` followed by `demux_extract.sh` will calculate the scores of Demux (BERT-based) for alignment.

More scripts pending...
