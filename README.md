# LLMs do Multi-Label Classification Differently

This repo contains the official implementation of [Large Language Models do Multi-Label Classification Differently](https://arxiv.org/abs/2505.17510). 

If you have any questions, bugs, or comments, please contact mjma@usc.edu or chochlak@usc.edu!

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

### Main Experiments

The majority of the scripts are located in `scripts/prob_distr`. The main python file is `llm_prob_distr.py`, but the entrypoint for calling this python function is in all of the `pipeline-*.sh` bash scripts. All the `pipeline-*.sh` scripts take the following ordered arguments:
- Position 1: distribution type to evaluate (`baseline` for most experiments; `unary`/`binary` for results on distribution alignment; `multilabel_icl` for a sweep of multilabel prompts for Figure 6)
- Position 2: IDs to use for testing; most experiments will use `main_test_set`. See the folder `prob_distr_ids` for valid lists of testing IDs
- Position 3: Which GPUs to use (int based, i.e. to fit into `cuda:x`)
- Position 4: Model to use, should be exact Huggingface name
- Position 5: Whether to use `vllm` for efficiency (UNTESTED, might not work: leave blank to run with normal `transformers` library)

For example, to run the main experiments for MFRC, you could run:

```
bash pipeline-MFRC.sh baseline main_test_set 0 meta-llama/Llama-3.1-8B
```

After successfully running these scripts, a folder should appear under `logs/{dataset}/{test_id_set}/{distribution_type}/{model_name}_x`, where `x` is an integer that is usually 0 but sometimes 1 or higher. For example the above script would create the folder `logs/MFRC/main_test_set/baseline/meta-llama--Llama-3.1-8B_0`. There is a file in that folder, `indexed_metrics.yml`, which is the file that contains all of the relevant information for that experiment. It lists each individual test stimulus, along with its logits and probabilities for every generated label.

All of the `plot*.py` files are the files for plotting Figures 4, 5, and 6 and they use various `indexed_metrics.yml` files to process and analyze the data. If you want to run them, some of the initial settings might need to change according to which log files they point to.

### Linear Probing Analysis

- `scripts/ml-distribution/linear-probing.sh` will run the scripts to, first, run the models on the entire dataset, and then run linear probing one them.
- `scripts/ml-distribution/plot_figures.sh` will plot their distributions.

### Distribution Alignment

- `scripts/ml-distribution/scores.sh` will calculate the scores of the linear probes to calculate alignment.
- `scripts/training/demux_ds.sh` followed by `demux_extract.sh` will calculate the scores of Demux (BERT-based) for alignment.
