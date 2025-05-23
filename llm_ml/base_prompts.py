import os
import json
import warnings
import random
from string import Template
from typing import Any
from copy import deepcopy

import torch
import langcodes
import pandas as pd
import gensim.downloader as api
from ember.dataset import BaseDataset

from llm_ml.base_datasets import TextDataset
import pandas as pd


class ExampleSamplerMixin:
    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        return dict(
            retries=dict(
                type=int,
                default=3,
                help="number of retries for random conditional sampling",
            ),
            sentence_model_name_or_path=dict(
                type=str,
                # default="all-mpnet-base-v2",
                help="sentence embedding model name or path",
            ),
            sampling_strategy=dict(
                type=str,
                choices=["complete", "similarity", "uniform"],
                help="how to sample examples",
                searchable=True,
                metadata=dict(name=True, name_priority=1),
            ),
            label_mode=dict(
                type=str,
                help="how to handle labels",
                choices=[
                    "random",
                    "distribution",
                    "preds",
                    "constrastive",
                    "opposite",
                ],
                searchable=True,
                metadata=dict(name=True, name_priority=1),
            ),
            reasonableness_label_mode=dict(
                type=str,
                help="how to handle labels for reasonableness",
                choices=["random", "distribution"],
                default="distribution",
                searchable=True,
            ),
            query_label_mode=dict(
                type=str,
                help="how to handle query labels in prompt (ablation)",
                choices=["random", "distribution", "preds", "opposite"],
                searchable=True,
                metadata=dict(name=True, name_priority=1),
            ),
            cot_randomize=dict(
                type=bool,
                help="whether to sample chains of thought randomly",
                searchable=True,
            ),
            sentence_device=dict(
                type=str,
                default="args.device",
                help="device to use for sentence embeddings",
                metadata=dict(disable_comparison=True),
            ),
            label_randomization_seed=dict(
                type=int,
                searchable=True,
                help="random seed for label randomization",
                metadata=dict(disable_comparison=True),
            ),
            label_same_length=dict(
                action="store_true",
                help="whether to sample labels of the same length when randomizing",
            ),
        )

    def __init__(
        self,
        sentence_model_name_or_path: str | None = None,
        seed: int | None = None,
        label_randomization_seed: int | None = None,
        retries: int = 3,
        sentence_device: str = "cpu",
        sampling_strategy: str | None = None,
        label_mode: str | None = None,
        reasonableness_label_mode: str = "distribution",
        query_label_mode: str | None = None,
        cot_randomize: bool = False,
        label_same_length: bool = False,
        *args,
        **kwargs,
    ):
        assert retries > 0, "retries must be positive"
        assert (
            sampling_strategy != "similarity"
            or sentence_model_name_or_path is not None
        ), "sentence_model_name_or_path must be provided for similarity sampling"
        super().__init__(*args, **kwargs)
        self.sampling_strategy = sampling_strategy or "random"
        self._example_sampler_mixin_data = dict(
            seed=seed,
            retries=retries,
            label_randomization_seed=label_randomization_seed,
            label_same_length=label_same_length,
        )

        self._example_sampler_mixin_data["label_mode"] = (
            label_mode or "none"
        ).lower()
        self._example_sampler_mixin_data["reasonableness_label_mode"] = (
            reasonableness_label_mode
        ).lower()
        self._example_sampler_mixin_data["query_label_mode"] = (
            query_label_mode or "none"
        ).lower()
        self._example_sampler_mixin_data["cot_randomize"] = cot_randomize

        if self.sampling_strategy == "similarity":
            from sentence_transformers import SentenceTransformer

            self._example_sampler_mixin_data["sentence_embedding_model"] = (
                SentenceTransformer(
                    sentence_model_name_or_path, device=sentence_device
                )
            )

    def _handle_labels(
        self,
        samples: list[dict[str, str | torch.Tensor]],
        dataset: list[TextDataset],
        label_mode: str = "none",
        label_randomization_seed: int | None = None,
    ):
        """Handles label transformation of `samples` in-place
        according to `label_mode`.

        The basis for randomization is the paper
        `Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?`

        Args:
            samples: list of samples to handle labels for.
            dataset: dataset to sample from.
            label_mode: how to handle labels.
                - "random": randomize labels from dataset.
                - "distribution": sample from dataset.
                - "preds": use predicted labels.
                - "contrastive": not implemented yet.
                - "none": do not change labels.
            label_randomization_seed: random seed for label randomization.
        """

        if not isinstance(dataset, list):
            dataset = [dataset]

        if label_randomization_seed is None:
            label_randomization_seed = self._example_sampler_mixin_data[
                "label_randomization_seed"
            ]

        if (
            label_randomization_seed != self._example_sampler_mixin_data["seed"]
            and label_randomization_seed is not None
        ):
            # setting the same seed can result in the same samples twice
            # for distribution sampling
            # also don't set when None to avoid same seeds when user doesn't specify this one
            random.seed(
                self._example_sampler_mixin_data["label_randomization_seed"]
            )

        for sample in samples:

            if label_mode == "random":
                if dataset[0].multilabel:
                    new_labels = random.sample(
                        dataset[0].label_set,
                        random.randint(0, len(dataset[0].label_set)),
                    )
                else:
                    new_labels = random.choice(dataset.label_set)

                new_labels = dataset[0].get_label_from_str(new_labels)

            elif label_mode == "distribution":
                if (
                    hasattr(dataset[0], "label_lens")
                    and self._example_sampler_mixin_data["label_same_length"]
                ):
                    sample_token_len = None
                    for ds in dataset:
                        if sample["id"] in ds.label_lens:
                            sample_token_len = ds.label_lens[sample["id"]]
                    assert (
                        sample_token_len is not None
                    ), f"sample {sample['id']} not in datasets"

                    # sample from the same length as the original label
                    potential_labels = [
                        (k, j)
                        for k, ds in enumerate(dataset)
                        for j, e in enumerate(ds)
                        if ds.label_lens[e["id"]][0] == sample_token_len[0]
                    ]

                    inds = random.choice(potential_labels)
                    new_labels = dataset[inds[0]][inds[1]]["label"]

                else:
                    new_labels = random.choice(dataset[0])["label"]

            elif label_mode == "preds":
                if "pred_label" not in sample:
                    raise ValueError(
                        "label_mode is 'preds' but 'pred_label'"
                        f" not in sample {sample['id']}"
                    )
                new_labels = sample["pred_label"]

            elif label_mode == "contrastive":
                raise NotImplementedError(
                    "contrastive label mode not implemented"
                )

            elif label_mode == "opposite":
                assert (
                    not dataset[0].multilabel and len(dataset[0].label_set) == 2
                ), "`opposite` label mode only works for binary classification"
                label_0 = dataset[0].get_label_from_str(dataset[0].label_set[0])
                label_1 = dataset[0].get_label_from_str(dataset[0].label_set[1])

                new_labels = label_0 if sample["label"] == label_1 else label_1
            else:
                new_labels = sample["label"]

            sample["label"] = new_labels

    def handle_query_label(
        self,
        query: dict[str, str | torch.Tensor],
        is_demo: bool = False,
        dataset: TextDataset | list[TextDataset] = None,
    ) -> str:
        """Returns the label of `query` based on label mode.

        Args:
            query: query dictionary.
            is_demo: whether the query is a demonstration as well (ablation).
            dataset: dataset to use for label handling, e.g. sampling from.
        """

        if not is_demo:
            if (
                self._example_sampler_mixin_data["label_mode"] == "preds"
                and query["pred_label"] is not None
            ):
                return query["pred_label"]
            return query["label"]
        else:
            if (
                self._example_sampler_mixin_data["query_label_mode"] == "preds"
                and query["pred_label"] is not None
            ):
                return query["pred_label"]
            elif self._example_sampler_mixin_data["query_label_mode"] != "none":

                assert self._example_sampler_mixin_data[
                    "query_label_mode"
                ] != "distribution" or (
                    dataset is not None or dataset
                ), "query_label_mode is 'distribution' but dataset is None"

                # doesn't replicate across examples
                label_randomization_seed = None
                while True:
                    query_cp = deepcopy(query)
                    self._handle_labels(
                        [query_cp],
                        dataset,
                        self._example_sampler_mixin_data["query_label_mode"],
                        label_randomization_seed=label_randomization_seed,
                    )
                    any_ds = (
                        dataset[0] if isinstance(dataset, list) else dataset
                    )
                    if (
                        # if not multilabel, 0 is a distinct label, so we shouldnt avoid
                        not any_ds.multilabel
                        # otherwise, dont use None as random label
                        or not (query_cp["label"] == 0).all()
                    ):
                        break

                    # if all labels are zero, retry
                    label_randomization_seed = random.randint(0, 2**32 - 1)

                return query_cp["label"]
            return query["label"]

    def sample_cot(self, cots: dict[str, str], query_id: str) -> str | None:
        """Samples a chain of thought for `query_id` based on `cot_mode`."""
        if self._example_sampler_mixin_data["cot_randomize"] and cots:
            return random.choice(list(cots.values()))

        return cots.get(query_id, None)

    def _sample_examples(
        self,
        query: dict[str, str | torch.Tensor],
        dataset: TextDataset,
        shot: int,
    ):
        if shot == 0 or dataset is None:
            return []

        if self.sampling_strategy == "random" or self.sampling_strategy is None:
            samples = self._random_sample(
                query=query, dataset=dataset, shot=shot
            )
        elif self.sampling_strategy == "complete":
            samples = self._complete_sample(
                query=query, dataset=dataset, shot=shot
            )
        elif self.sampling_strategy == "similarity":
            samples = self._similarity_sample(
                query=query, dataset=dataset, shot=shot
            )
        elif self.sampling_strategy == "uniform":
            samples = self._uniform_sample(
                query=query, dataset=dataset, shot=shot
            )

        return samples

    def sample_with_strategy(
        self,
        query: dict[str, str | torch.Tensor],
        dataset: TextDataset,
        shot: int,
    ) -> list[dict[str, str | torch.Tensor]]:

        samples = self._sample_examples(query, dataset, shot)

        if query["id"].split(dataset.id_separator)[1] == "aggregate":
            self._handle_labels(
                samples, dataset, self._example_sampler_mixin_data["label_mode"]
            )

        return samples

    def sample_with_reasonableness(
        self,
        query: dict[str, str | torch.Tensor],
        dataset: TextDataset,
        shot: int,
        unreasonable_rate: float,
    ) -> tuple[list[dict[str, str | torch.Tensor]], list[int]]:

        samples = self._sample_examples(query, dataset, shot)
        r_labels = []
        unreasonable_ids = random.sample(
            range(len(samples)), int(unreasonable_rate * len(samples))
        )
        for i, sample in enumerate(samples):
            if i in unreasonable_ids:
                r_labels.append(0)
                self._handle_labels(
                    [sample],
                    dataset,
                    self._example_sampler_mixin_data[
                        "reasonableness_label_mode"
                    ],
                )
            else:
                r_labels.append(1)

        return samples, r_labels

    def _random_sample(
        self,
        query: dict[str, str | torch.Tensor],
        dataset: TextDataset,
        shot: int,
        **kwargs,
    ) -> list[dict[str, str | torch.Tensor]]:
        """Randomly samples `shot` examples from `dataset` to use in the prompt
        of `query`.

        Args:
            query: query dictionary.
            dataset: dataset to sample from.
            shot: number of examples to sample.

        Returns:
            List of sampled examples.
        """

        random.seed(self._example_sampler_mixin_data["seed"])

        annotator = query["id"].split(dataset.id_separator)[1]
        inds = dataset.annotator2inds[annotator]
        inds = random.sample(inds, shot)
        return [dataset[i] for i in inds]

    def _complete_sample(
        self,
        query: dict[str, str | torch.Tensor],
        dataset: TextDataset,
        shot: int,
    ) -> list[dict[str, str | torch.Tensor]]:
        """Samples `shot` examples from `dataset` to use in the prompt
        of `query` that contain all labels.

        This is not implemented optimally, only heuristically, because of
        the complexity of that for multilabel datasets.

        Args:
            query: query dictionary.
            shot: number of examples to sample.
            dataset: dataset to sample from.

        Returns:
            List of sampled examples.
        """

        def sample(shot, seed):
            random.seed(seed)

            annotator = query["id"].split(dataset.id_separator)[1]
            label_inds = dataset.annotator2label_inds[annotator]
            labels = set(dataset.label_set)

            samples = []
            inds = set()

            while labels and shot > 0:
                label = random.choice(list(labels))
                if shot == 1:
                    # make an effort to sample all remaining labels
                    for idx in label_inds[label]:
                        if (
                            set(dataset.index_label_set(dataset[idx]["label"]))
                            == labels
                        ):
                            break
                        # if not broken, we are going to get an assertion anyway
                        # so no need to handle it here
                else:
                    idx = random.choice(label_inds[label])

                # index cannot overlap because it comes from label
                # that has not been sampled yet
                inds.add(idx)
                sample = dataset[idx]
                labels.difference_update(
                    dataset.index_label_set(sample["label"])
                )
                samples.append(sample)
                shot -= 1

            assert shot > 0 or not labels

            extra_inds = random.sample(
                set(dataset.annotator2inds[annotator]).difference(inds), shot
            )
            samples.extend([dataset[i] for i in extra_inds])

            return samples

        seed = self._example_sampler_mixin_data["seed"]

        for _ in range(self._example_sampler_mixin_data["retries"]):
            try:
                return sample(shot, seed)
            except AssertionError:
                random.seed(seed)
                seed = random.randint(0, 2**32 - 1)

        raise RuntimeError(
            f"{shot}-shot not enough to sample all labels: "
            + ",".join(dataset.label_set)
        )

    def _similarity_sample(
        self,
        query: dict[str, str | torch.Tensor],
        dataset: TextDataset,
        shot: int,
    ) -> list[dict[str, str | torch.Tensor]]:
        """Samples `shot` examples from `dataset` to use in the prompt
        of `query` that are most similar to it w.r.t. sentence embeddings."""

        annotator = query["id"].split(dataset.id_separator)[1]
        inds = dataset.annotator2inds[annotator]

        if "sentence_embeddings" not in self._example_sampler_mixin_data:
            self._example_sampler_mixin_data[
                "sentence_embeddings"
            ] = self._example_sampler_mixin_data[
                "sentence_embedding_model"
            ].encode(
                [dataset[i]["text"] for i in inds],
                convert_to_tensor=True,
            )

        query_embedding = (
            self._example_sampler_mixin_data["sentence_embedding_model"]
            .encode([query["text"]], convert_to_tensor=True)
            .squeeze()
        )

        cossims = torch.cosine_similarity(
            query_embedding,
            self._example_sampler_mixin_data["sentence_embeddings"],
        )

        _, local_inds = torch.topk(cossims, k=shot)

        return [dataset[inds[i]] for i in local_inds]

    def _uniform_sample(
        self,
        query: dict[str, str | torch.Tensor],
        dataset: TextDataset,
        shot: int,
    ) -> list[dict[str, str | torch.Tensor]]:
        """Samples `shot` examples from `dataset` to use in the prompt
        of `query` that are uniformly distributed w.r.t. labels. Works only
        approximately for multilabel datasets (includes examples with no
        labels as an another label)."""

        random.seed(self._example_sampler_mixin_data["seed"])

        annotator = query["id"].split(dataset.id_separator)[1]

        label_set = deepcopy(dataset.label_set)
        if dataset.multilabel:
            label_set.append("none")
        shot_per_label = shot // len(label_set)

        inds = []
        for label in label_set:
            potential_inds = dataset.annotator2label_inds[annotator][label]
            # remove already sampled indices
            potential_inds = list(set(potential_inds).difference(inds))
            inds.extend(random.sample(potential_inds, shot_per_label))

        # in case labels do not divide shot evenly
        if len(inds) < shot:
            extra_inds = set(dataset.annotator2inds[annotator]).difference(inds)
            inds.extend(random.sample(extra_inds, shot - len(inds)))

        # shuffle to avoid bias
        random.shuffle(inds)
        return [dataset[i] for i in inds]


class LabelSimilarityMixin:
    """Mixin for computing closest label from model prediction based
    on word similarity between label set and predictions of model."""

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        return dict(
            word_embeddings_model=dict(
                type=str,
                help="word embeddings model to use for similarity",
                metadata=dict(name=True),
            )
        )

    def __init__(
        self,
        word_embeddings_model: str | None = None,
        similarity_threshold: int | None = None,
        *args,
        **kwargs,
    ) -> None:
        """Init.

        Args:
            word_embeddings_model: word embeddings model to use for similarity.
            threshold: similarity threshold for considering a label.
        """
        super().__init__(*args, **kwargs)

        self._label_similarity_mixin_data = dict(word_embeddings_model=None)

        if word_embeddings_model is not None:
            self._label_similarity_mixin_data["word_embeddings_model"] = (
                api.load(word_embeddings_model)
            )
            self._label_similarity_mixin_data["similarity_threshold"] = (
                similarity_threshold
            )

            if "conceptnet" in word_embeddings_model:
                lang = getattr(
                    self, "language", getattr(self, "lang", "english")
                ).lower()
                lang = langcodes.find(lang).language
                self._label_similarity_mixin_data["word_transformation"] = (
                    lambda x: f"/c/{lang}/{x.lower()}"
                )
            else:
                self._label_similarity_mixin_data["word_transformation"] = (
                    lambda x: x
                )

    def get_closest_label(self, prediction: str, label_set: list[str]) -> str:
        """Finds the closest label to `prediction` in `label_set` based on
        word similarity.

        Args:
            label: label to find the closest label to.
            label_set: set of labels to find the closest label in.

        Returns:
            Closest label to `label` in `label_set`.
        """

        wv = self._label_similarity_mixin_data["word_embeddings_model"]
        prediction = self._label_similarity_mixin_data["word_transformation"](
            prediction
        )

        if wv is None:
            return None

        if prediction not in wv.key_to_index:
            return None

        similarities = [
            wv.similarity(
                self._label_similarity_mixin_data["word_transformation"](label),
                prediction,
            )
            for label in label_set
        ]

        max_sim = max(similarities)

        if (
            self._label_similarity_mixin_data["similarity_threshold"]
            is not None
            and max_sim
            < self._label_similarity_mixin_data["similarity_threshold"]
        ):
            return None

        return label_set[similarities.index(max_sim)]


DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n\n"""


def _parse_prompt_fn(prompt):
    if prompt is not None and os.path.exists(prompt):
        with open(prompt) as fp:
            return fp.read()
    return prompt


class PromptBaseDataset(ExampleSamplerMixin, BaseDataset):
    """Prompt dataset for text classification, based on other TextDataset.
    Base class that doesn't use tokenization or `from_namespace`, so it can
    be inherited from."""

    name = "Prompt dataset"

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        args = dict(
            system_prompt=dict(
                type=str,
                help="prompt for system (use {labels} to insert labels)",
                searchable=True,
            ),
            instruction_prompt=dict(
                type=str,
                help="prompt for instructions (use {labels} to insert labels)",
                searchable=True,
            ),
            incontext_prompt=dict(
                type=str,
                help="prompt for in-context learning (use {text} and {label} "
                "to insert text and label of example)",
                searchable=True,
            ),
            query_prompt=dict(
                type=str,
                help="prompt for in-context query, if the default extension "
                "of the `incontext_prompt` is not desired (use {text} "
                "to insert text of example)",
                searchable=True,
            ),
            user_prompt=dict(
                type=str,
                help="prompt for user (use {text} to insert text of example)",
                searchable=True,
            ),
            assistant_prompt=dict(
                type=str,
                help="prompt for assistant (use {label} to insert label of example)",
                searchable=True,
            ),
            label_format=dict(
                type=str,
                help="formatting function for labels in the in-context prompt"
                " (default: join with commas)",
                searchable=True,
                metadata=dict(name=True),
            ),
            label_parser=dict(
                type=str,
                help="parsing function for labels in the response assuming "
                "`label_format` has been applied. Should return a list of strings.",
            ),
            shot=dict(
                type=int,
                required=True,
                help="number of examples to sample",
                metadata=dict(name=True, name_priority=1),
                searchable=True,
            ),
            cot_csv=dict(
                type=str,
                help="path to csv file with chains of thought",
            ),
        )
        return args | ExampleSamplerMixin.argparse_args()

    def __init__(
        self,
        train_dataset: TextDataset,
        test_dataset: TextDataset,
        shot: int,
        system_prompt: str | None = None,
        instruction_prompt: str | None = None,
        incontext_prompt: str | None = None,
        query_prompt: str | None = None,
        user_prompt: str | None = None,
        assistant_prompt: str | None = None,
        label_format: str | None = None,
        label_parser: str | None = None,
        cot_csv: str | None = None,
        *args,
        **kwargs,
    ):
        """Init.

        Args:
            train_dataset: train dataset.
            test_dataset: test dataset.
            shot: number of examples to sample.
            system_prompt: prompt for system.x
            instruction_prompt: prompt for instructions.
            incontext_prompt: prompt for in-context learning.
            query_prompt: prompt for in-context query.
            user_prompt: prompt for user. Provide if
                incontent_prompt is not used.
            assistant_prompt: prompt for assistant. Provide if
                incontent_prompt is not used.
            label_format: formatting function for labels in the
                in-context prompt.
            label_parser: parsing function for labels in the response
                assuming `label_format` has been applied. Should return
                a list of strings.
            args, kwargs: additional arguments, mostly for logging.
        """

        super().__init__(*args, **kwargs)

        system_prompt = _parse_prompt_fn(system_prompt)
        instruction_prompt = _parse_prompt_fn(instruction_prompt)
        incontext_prompt = _parse_prompt_fn(incontext_prompt)
        query_prompt = _parse_prompt_fn(query_prompt)
        user_prompt = _parse_prompt_fn(user_prompt)
        assistant_prompt = _parse_prompt_fn(assistant_prompt)

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.any_dataset = self.train_dataset or self.test_dataset

        assert self.test_dataset is not None

        self.shot = shot
        self.label_set = self.any_dataset.label_set

        assert self.label_set == self.test_dataset.label_set

        self.multilabel = self.any_dataset.multilabel

        if label_format and "json" in label_format:
            try:
                key = label_format.split("-")[1]
            except IndexError:
                key = "label"
            # ["none"] will only be used in multilabel setting
            self.label_formatter = lambda x: json.dumps(
                {key: x if x else ["none"]}
            )
            self.label_parser = lambda x: json.loads(
                # avoids instances where JSON is followed by explanation
                x[x.find("{") : x.rfind("}") + 1]
            )[key]
        elif label_format == "polysyndeton":
            self.label_formatter = lambda x: " and ".join(
                ["the " + e for e in (x if isinstance(x, list) else [x])]
            )
            self.label_parser = lambda x: [e[4:] for e in x.split(" and ")]
        elif not label_format:
            self.label_formatter = lambda x: ", ".join(
                (x if x else ["none"]) if isinstance(x, list) else [x]
            )
            self.label_parser = lambda x: [e.strip() for e in x.split(",")]
        else:
            self.label_formatter = eval(label_format)
            if label_parser:
                try:
                    self.label_parser = eval(label_parser)
                except Exception:
                    warnings.warn(
                        "Could not parse label_parser. Using default."
                    )
                    self.label_parser = None
            else:
                self.label_parser = None

        self.ids_per_query = {}

        self.cots = {}
        if cot_csv:
            cots = pd.read_csv(cot_csv, index_col=0)
            self.cots = {
                k: v["cot"] for k, v in cots.to_dict(orient="index").items()
            }

        ### Setting up system prompt
        # "is None" instead of "or" allows the user to specify an empty string
        if system_prompt is None:
            system_prompt = DEFAULT_SYSTEM_PROMPT
        elif system_prompt == " ":
            system_prompt = ""
        self.system_prompt = self._format_instruction_prompt(system_prompt)

        ### Setting up instruction prompt
        if instruction_prompt is None:
            instruction_prompt = (
                instruction_prompt
                or "Classify the following examples as {} of the following: ".format(
                    "none, one, or many" if self.multilabel else "one"
                )
                + "{labels}.\n\n"
            )
        elif instruction_prompt == " ":
            instruction_prompt = ""
        self.instruction_prompt = self._format_instruction_prompt(
            instruction_prompt
        )

        ### Setting up in-context prompts

        if user_prompt is None and assistant_prompt is not None:
            user_prompt = "{text}"
        elif user_prompt is not None and assistant_prompt is None:
            assistant_prompt = "{label}"

        if incontext_prompt is None:
            incontext_prompt = (
                (user_prompt or "{text}")
                + "\n"
                + (assistant_prompt or "{label}")
                + "\n"
            )

        self.incontext_prompt = incontext_prompt
        self.user_prompt = user_prompt
        self.assistant_prompt = assistant_prompt

        self.query_prompt = query_prompt or self.user_prompt
        if not self.query_prompt:
            idx = self.incontext_prompt.find("{cot}")
            if idx == -1:
                idx = self.incontext_prompt.find("{label}")
            self.query_prompt = self.incontext_prompt[:idx]

    def debug_message(self):
        s = ["Label set: " + ", ".join(self.label_set)]
        s.append("Example prompt: " + self[0]["text"])
        return "\n".join(s)

    def __len__(self):
        return len(self.test_dataset)

    def sample(
        self, query: dict[str, str | torch.Tensor]
    ) -> list[dict[str, str | torch.Tensor]]:
        """Samples `self.shot` examples from `self.train_dataset`
        to use in the prompt of `query`."""
        samples = self.sample_with_strategy(
            query, self.train_dataset, self.shot
        )
        self.ids_per_query[query["id"]] = [sample["id"] for sample in samples]

        return samples

    def _format_instruction_prompt(
        self, instruction_prompt_template: str
    ) -> str:
        """Formats the instruction prompt."""
        labels = ", ".join(self.label_set[:-1]) + " and " + self.label_set[-1]
        return Template(
            instruction_prompt_template.replace("{labels}", "$labels")
        ).safe_substitute(labels=labels)

    def _format_user_prompt(self, user_prompt_template: str, text: str) -> str:
        """Formats the user prompt."""
        return Template(
            user_prompt_template.replace("{text}", "$text")
        ).safe_substitute(text=text)

    def _format_assistant_prompt(
        self, assistant_prompt_template: str, label: str, cot: str | None = None
    ) -> str:
        """Formats the assistant prompt."""
        label = self.label_formatter(self.any_dataset.index_label_set(label))
        # if string is empty, use "none" as label
        # not if label is empty, because formatter could add other stuff
        # this is still necessary because of custom formatters
        if not label:
            label = "none"

        return self._format_cot(
            Template(
                assistant_prompt_template.replace("{label}", "$label")
            ).safe_substitute(label=label),
            cot,
        )

    def _format_cot(self, template: str, cot: str | None) -> str:
        """Formats the chain of thought."""
        if not cot:
            return template
        return Template(template.replace("{cot}", "$cot")).safe_substitute(
            cot=cot
        )

    def _format_incontext_prompt(
        self,
        incontext_prompt_template: str,
        text: str,
        label: str,
        cot: str | None = None,
    ) -> str:
        """Formats the in-context prompt."""
        return self._format_user_prompt(
            self._format_assistant_prompt(
                incontext_prompt_template, label, cot
            ),
            text,
        )

    def __getitem__(self, index) -> dict[str, str | torch.Tensor]:
        """Returns prompt dictionary for `index`-th example in
        `test_dataset`. The return dictionary contains the following keys:
            - `id`: ID of example;
            - `query`: query text;
            - `text`: prompt text;
            - `label`: tensor label of example.

        The prompt is constructed as follows:
            1. The system and the instruction prompt are added.
            2. For each support example, the in-context prompt is added.
            3. The query prompt is added.
        """
        query = self.test_dataset[index]
        support = self.sample(query)

        demo_query_label = None

        prompt = [self.system_prompt or "", self.instruction_prompt or ""]
        prompt.extend(
            [
                self._format_incontext_prompt(
                    self.incontext_prompt,
                    sample["text"],
                    sample["label"],
                    self.sample_cot(self.cots, sample["id"]),
                )
                for sample in support
            ]
        )
        prompt.append(self.query_prompt.format(text=query["text"]))

        # remove trailing newlines, spaces, etc
        # because some labels might be tokens with spaces in the beginning
        # and therefore a prompt without stripping would have a space
        # as a separate token
        prompt = "".join(prompt).strip()

        odict = dict(
            id=query["id"],
            query=query["text"],
            text=prompt,
            label=self.handle_query_label(query),
        )

        if demo_query_label is not None:
            odict["demo_label"] = demo_query_label

        return odict


class ReasonablenessPromptBaseDataset(PromptBaseDataset):
    """Prompt dataset for text-label pair binary classification as reasonable or not,
    based on other TextDataset."""

    name = "Base prompt reasonableness dataset"

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        return PromptBaseDataset.argparse_args() | dict(
            unreasonable_rate=dict(
                type=float,
                default=0.5,
                help="Rate of unreasonable samples",
                searchable=True,
            ),
            yn_labels=dict(
                type=bool,
                searchable=True,
                help="Whether to use yes/no labels",
            ),
        )

    def __init__(self, unreasonable_rate, yn_labels, *args, **kwargs):
        self.unreasonable_rate = unreasonable_rate
        self.yn_labels = yn_labels
        super().__init__(*args, **kwargs)
        self.multilabel = False
        self.label_set = (
            ["unreasonable", "reasonable"] if not yn_labels else ["no", "yes"]
        )

        instruction_prompt = _parse_prompt_fn(kwargs["instruction_prompt"])
        self.instruction_prompt = self._format_instruction_prompt(
            instruction_prompt
        )

        if "{label}" not in self.query_prompt:
            # bad manipulation by parent init
            idx = self.incontext_prompt.find("{cot}")
            if idx == -1:
                idx = self.incontext_prompt.find("{r}")
            self.query_prompt = self.incontext_prompt[:idx]

    def _format_r_incontext_prompt(
        self,
        incontext_prompt_template: str,
        text: str,
        label: str,
        cot: str | None,
        reasonable: bool,
    ) -> str:
        """Example string to be formatted:

        Input: {text}\nLabel: {label}\nReasoning: {cot}\nReasonable? {r}\n
        """

        return self._format_r_assistant_prompt(
            self._format_r_user_prompt(incontext_prompt_template, text, label),
            cot,
            reasonable,
        )

    def _format_r_user_prompt(
        self, user_prompt_template: str, text: str, label: str
    ) -> str:
        """Formats the user prompt for reasonableness."""
        return self._format_incontext_prompt(user_prompt_template, text, label)

    def _format_r_assistant_prompt(
        self, assistant_prompt_template: str, cot: str | None, reasonable: bool
    ) -> str:
        """Formats the assistant prompt for reasonableness."""
        r = self.label_set[int(reasonable)]

        return self._format_cot(
            Template(
                assistant_prompt_template.replace("{r}", "$r")
            ).safe_substitute(r=r),
            cot,
        )

    def sample(
        self, query: dict[str, str | torch.Tensor]
    ) -> tuple[list[dict[str, str | torch.Tensor]], list[int]]:
        samples, r_labels = self.sample_with_reasonableness(
            query, self.train_dataset, self.shot, self.unreasonable_rate
        )
        self.ids_per_query[query["id"]] = [sample["id"] for sample in samples]
        return samples, r_labels

    def __getitem__(self, index):
        query = self.test_dataset[index]
        support, r_labels = self.sample(query)

        query_label = self.handle_query_label(
            query,
            is_demo=True,
            dataset=[self.train_dataset, self.test_dataset],
        )

        prompt = [self.system_prompt or "", self.instruction_prompt or ""]
        prompt.extend(
            [
                self._format_r_incontext_prompt(
                    self.incontext_prompt,
                    sample["text"],
                    sample["label"],
                    self.sample_cot(self.cots, sample["id"]),
                    bool(r),
                )
                for sample, r in zip(support, r_labels)
            ]
        )
        prompt.append(
            self._format_r_user_prompt(
                self.query_prompt, query["text"], query_label
            )
        )

        # remove trailing newlines, spaces, etc
        # because some labels might be tokens with spaces in the beginning
        # and therefore a prompt without stripping would have a space
        # as a separate token
        prompt = "".join(prompt).strip()

        odict = dict(
            id=query["id"],
            query=query["text"],
            checked_label=query_label,
            text=prompt,
            label=torch.tensor(1),
            # label=torch.tensor(int(all(query["label"] == query_label))),
        )

        return odict
