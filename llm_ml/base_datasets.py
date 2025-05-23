from typing import Any, Sequence, Callable, Literal, Mapping
from abc import abstractmethod
from copy import deepcopy

import torch
from ember.dataset import BaseDataset
from transformers import PreTrainedTokenizer, AutoTokenizer


class TextDataset(BaseDataset):
    """Base dataset for text classification.

    Attributes:
        root_dir: path to root directory containing data.
        splits: splits to load.
        ids: list of effective IDs (used in __getitem__, might not correspond
            to dataset directly, e.g. augmented for annotators).
        real_ids: list of real IDs.
        multilabel: whether dataset is multilabel or not.
        annotator_labels: whether dataset has annotator labels or not.
        label_set: set of labels.
        source_domain: source domain of dataset.
        examples: dictionary containing examples indexed by IDs.
        annotations: dictionary containing annotator labels indexed
            by annotators and then IDs.
        preprocessor: function to preprocess text.
        annotation_mode: mode to load, one of "aggregate", "annotator",
            "both". If "aggregate", only the aggregated label is returned.
            If "annotator", only the annotator labels are returned. If
            "both", both the aggregated and annotator labels are returned.
        id_separator: separator for IDs of annotator and example.
        annotator2inds: mapping between annotator IDs and indices.
        annotator2label_inds: mapping between annotator IDs and per label indices.
    """

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        return dict(
            root_dir=dict(
                type=str,
                help="path to root directory containing data",
            ),
            splits=dict(
                type=str,
                nargs="+",
                splits=["train", "dev", "test"],
                help="train splits to load",
            ),
            ids_filename=dict(
                type=str,
                nargs="+",
                splits=["train", "dev", "test"],
                help="path to file containing subset of IDs to retain, "
                "must align with splits (use None for no filtering)",
            ),
            annotation_mode=dict(
                type=str,
                choices=["aggregate", "annotator", "both"],
                default="aggregate",
                help="mode to load, one of 'aggregate', 'annotator', 'both'. "
                "If 'aggregate', only the aggregated label is returned. "
                "If 'annotator', only the annotator labels are returned. "
                "If 'both', both the aggregated and annotator labels are returned. "
                "Error is raised if 'annotator' or 'both' is chosen but the dataset "
                "does not have annotator labels.",
                metadata=dict(name=True),
                searchable=True,
            ),
            debug_len=dict(
                type=int,
                nargs="+",
                splits=["train", "dev", "test"],
                help="number of examples to load for debugging",
            ),
            debug_ann=dict(
                type=int,
                nargs="+",
                splits=["train", "dev", "test"],
                help="annotators to load",
            ),
            text_preprocessor=dict(
                type=bool,
                help="whether to use text preprocessor",
                searchable=True,
            ),
            keep_same_examples=dict(
                action="store_true",
                help="whether to keep the same examples for all annotators",
                splits=["train", "dev", "test"],
            ),
            keep_one_after_filtering=dict(
                type=bool,
                help="whether to keep only one copy of each example after filtering with other flags "
                "(i.e. one random annotator); renamed to aggregate",
            ),
        )

    @property
    @abstractmethod
    def multilabel(self) -> bool:
        """Whether dataset is multilabel or not."""
        pass

    @property
    @abstractmethod
    def annotator_labels(self) -> bool:
        """Whether dataset has annotator labels or not."""
        pass

    @property
    @abstractmethod
    def source_domain(self) -> str:
        """Source domain of dataset."""
        pass

    def __init__(
        self,
        root_dir: str,
        splits: str | Sequence[str],
        text_preprocessor: Callable[[str], str] | None = None,
        annotation_mode: Literal[
            "aggregate", "annotator", "both"
        ] = "aggregate",
        debug_len: list[int] | int | None = None,
        debug_ann: int | list[int] | None = None,
        keep_same_examples: bool = False,
        keep_one_after_filtering: bool = False,
        ids_filename: str | None | list[str | None] = None,
        annotator_ids: list[str] | None = None,
        seed: int | None = None,
        *args,
        **kwargs,
    ):
        """Init.

        Args:
            root_dir: path to root directory containing data.
            splits: splits to load.
            text_preprocessor: function to preprocess text.
            annotation_mode: mode to load, one of "aggregate", "annotator",
                "both". If "aggregate", only the aggregated label is returned.
                If "annotator", only the annotator labels are returned. If
                "both", both the aggregated and annotator labels are returned.
            debug_len: number of examples to load for debugging.
            debug_ann: number of annotators to load for debugging if a single
                number (even if in a list), or a range of annotators to load
                (sorted by number of annotations) if a list.
            keep_same_examples: whether to keep the same examples for all
                annotators.
            keep_one_after_filtering: whether to keep only one copy of each
                example after filtering with other flags (i.e. one random
                annotator).
            ids_filename: path to file containing subset of IDs to retain.
            annotator_ids: list of annotator IDs to retain.
        """

        assert (
            annotation_mode == "aggregate" or self.annotator_labels
        ), f"No annotator labels, but annotation mode was set to {annotation_mode}."

        super().__init__()

        self.seed = seed

        self.id_separator = "__"

        if debug_len is None:
            pass  # just so that we dont have to check every time
        elif isinstance(debug_len, int):
            debug_len = range(0, debug_len)
        elif len(debug_len) == 1:
            debug_len = range(0, debug_len[0])
        elif len(debug_len) == 2:
            debug_len = range(debug_len[0], debug_len[1])

        self._debug_len = debug_len

        if debug_ann is None:
            pass  # just so that we dont have to check every time
        elif isinstance(debug_ann, int):
            debug_ann = range(0, debug_ann)
        elif len(debug_ann) == 1:
            debug_ann = range(0, debug_ann[0])
        elif len(debug_ann) == 2:
            debug_ann = range(debug_ann[0], debug_ann[1])

        self._debug_ann = debug_ann

        self.keep_same_examples = keep_same_examples
        self.keep_one_after_filtering = keep_one_after_filtering

        self.root_dir = root_dir
        self.splits = [splits] if isinstance(splits, str) else splits
        self.ids_filename = (
            [ids_filename] if isinstance(ids_filename, str) else ids_filename
        )
        self.preprocessor = text_preprocessor or (lambda x: x)
        self.annotation_mode = annotation_mode

        data = []
        for split in self.splits:
            split_data, label_set = self._load_data(split)
            label_set = [l.lower() for l in label_set]
            data.append(split_data)
            self.label_set = label_set

        self.examples = {
            example_id: {
                k: d[example_id][k] for k in d[example_id] if k != "label"
            }
            for d in data
            for example_id in d
        }
        self.annotations, self.annotators = self._convert_data(
            data, annotator_ids
        )
        self.real_ids = list(self.examples)

        (
            self.ids,
            self.annotator2inds,
        ) = self._extend_dataset_for_annotators()

        self.annotator2label_inds = self._extend_dataset_for_sampling()

    def _extend_dataset_for_annotators(
        self,
    ) -> tuple[list[tuple[Any, str]], dict[str, list[int]]]:
        """Creates IDs for dataset according to annotation mode,
        mapping between annotator IDs and indices."""

        ids = []
        annotator2inds = {}

        for worker_id, annotator_data in self.annotations.items():
            if (
                worker_id == "aggregate" and self.annotation_mode == "annotator"
            ) or (
                worker_id != "aggregate" and self.annotation_mode == "aggregate"
            ):
                continue
            for example_id in annotator_data:
                annotator2inds.setdefault(worker_id, []).append(len(ids))
                ids.append((example_id, worker_id))

        return ids, annotator2inds

    def _extend_dataset_for_sampling(self) -> dict[str, dict[str, list[int]]]:
        """Creates mapping between annotator IDs and per label indices
        for easier sampling."""

        annotator2label_inds = {}
        for annotator, inds in self.annotator2inds.items():
            annotator2label_inds[annotator] = {}
            for i in inds:
                if self.multilabel:
                    labels = self[i]["label"]
                    labels = [self.label_set[l] for l in labels.nonzero()[:, 0]]
                    for label in labels:
                        annotator2label_inds[annotator].setdefault(
                            label, []
                        ).append(i)
                    # need to include no labels for multilabel
                    # for wont happen for no labels so no need for if-else
                    if not labels:
                        annotator2label_inds[annotator].setdefault(
                            "none", []
                        ).append(i)
                else:
                    label = self.label_set[int(self[i]["label"])]
                    annotator2label_inds[annotator].setdefault(
                        label, []
                    ).append(i)

        return annotator2label_inds

    def _convert_data(
        self,
        data: list[dict[Any, dict[str, str | torch.Tensor]]],
        annotator_ids: list[str] | None = None,
    ) -> tuple[dict[str, dict[str, torch.Tensor]], set[str]]:
        """Converts data from loading format to annotator dict format (i.e.
        {"ann1": {"id1": label1, "id2": label2, ...}, "ann2": ...,}).
        Adds "aggregate" key for aggregate/default label.
        Makes all ID strings. If `annotator_ids` is provided, only keeps
        those annotators."""

        annotations = {}

        if self.ids_filename:
            ids_to_keep = set()
            for fn, split_data in zip(self.ids_filename, data):
                if fn:
                    with open(fn) as fp:
                        ids_to_keep.update(
                            set([l.strip() for l in fp.readlines()])
                        )
                else:
                    ids_to_keep.update(split_data.keys())

        else:
            # all IDs
            ids_to_keep = [k for d in data for k in d]

        # filter data and turn into single dict
        data = {k: v for d in data for k, v in d.items() if k in ids_to_keep}

        for example_id in list(data):
            datum = data[example_id]
            example_id = str(example_id)
            if torch.is_tensor(datum["label"]):
                # then no worker key -> only aggregate
                annotations.setdefault("aggregate", {})[example_id] = datum[
                    "label"
                ]
                data[example_id]["label"] = {"aggregate": datum["label"]}
            else:
                # we have aggregate and annotators, make sure everything a str
                worker_ids = list(datum["label"])
                for worker_id in worker_ids:
                    if not annotator_ids or str(worker_id) in annotator_ids:
                        annotations.setdefault(str(worker_id), {})[
                            example_id
                        ] = datum["label"][worker_id]

        aggregate_annotations = annotations.get("aggregate", None)

        if self._debug_ann:
            # keep annotators with most examples
            annotations.pop("aggregate")
            annotation_items = sorted(
                annotations.items(),
                # x is tuple of key, value, so [1] is value
                key=lambda x: len(x[1]),
                reverse=True,
            )
            annotations = dict([annotation_items[i] for i in self._debug_ann])

        if self.keep_same_examples:
            # keep only examples that all annotators have
            common_examples = set.intersection(
                *[
                    set(v.keys())
                    for v in annotations.values()
                    if isinstance(v, dict)
                ]
            )

            assert (
                common_examples
            ), "No common examples found, consider limiting annotators with `debug_ann`."

            # annotations are aligned as well across annotators
            annotations = {
                worker_id: {
                    example_id: label
                    for example_id, label in v.items()
                    if example_id in common_examples
                }
                for worker_id, v in annotations.items()
            }

        if self._debug_len:

            # if self.annotator_labels:
            annotators_per_example = {
                example_id: [
                    str(k)
                    for k in v["label"]
                    if str(k) in set(annotations.keys())
                ]
                for example_id, v in data.items()
            }

            # examples with most annotators on top
            annotators_per_example = dict(
                sorted(
                    annotators_per_example.items(),
                    key=lambda x: len(x[1]),
                    reverse=True,
                )
            )

            # keep examples with most annotators
            new_annotations = {}
            for worker_id, v in annotations.items():
                sorted_items = sorted(
                    v.items(),
                    key=lambda x: len(annotators_per_example[x[0]]),
                    reverse=True,
                )
                new_annotations[worker_id] = dict(
                    [
                        sorted_items[i]
                        for i in self._debug_len
                        if i < len(sorted_items)
                    ]
                )
            annotations = new_annotations

            # else:
            #     # keep "random" examples
            #     # need to be same across seeds
            #     # (mostly for testing purposes)
            #     random.seed(0)

            #     permuted_example_ids = random.sample(
            #         list(annotations["aggregate"]),
            #         len(annotations["aggregate"]),
            #     )
            #     example_ids = [permuted_example_ids[i] for i in self._debug_len]

            #     annotations = {
            #         "aggregate": {
            #             example_id: annotations["aggregate"][example_id]
            #             for example_id in example_ids
            #         }
            #     }

        if self.keep_one_after_filtering:
            # keep original annotators for filtering of other splits
            annotators = set(annotations)

            aggregate_annotations = (
                aggregate_annotations or list(annotations.values())[0]
            )

            # keep only one copy of each example
            annotations = {
                "aggregate": {
                    example_id: aggregate_annotations[example_id]
                    # use keys from filtered annotations
                    for example_id in list(annotations.values())[0]
                },
            }

            return annotations, annotators

        if self.annotation_mode == "annotator":
            annotations.pop("aggregate", None)

        return annotations, set(annotations)

    def __len__(self):
        """Returns length of dataset."""
        return len(self.ids)

    def __getitem__(self, idx) -> tuple[Any, str, str]:
        """Returns text and label at index `idx`."""
        example_id, worker_id = self.ids[idx]
        data = deepcopy(self.examples[example_id])
        data["label"] = self.annotations[worker_id][example_id]
        return dict(id=example_id + self.id_separator + worker_id, **data)

    def index_label_set(
        self,
        label: torch.Tensor | int | list[int],
        alternative_label_set: list[str] | None = None,
    ) -> str | list[str]:
        """Returns label names given numerical `label`."""

        sampling_label_set = (
            alternative_label_set if alternative_label_set else self.label_set
        )

        if not self.multilabel:
            if torch.is_tensor(label):
                label = int(label.item())
            if isinstance(label, list):
                assert len(label) == 1
                label = int(label[0])
            return sampling_label_set[int(label)]

        if torch.is_tensor(label):
            label = label.tolist()
        return [sampling_label_set[i] for i, l in enumerate(label) if l == 1]

    def get_label_from_str(self, label: str | list[str]) -> torch.Tensor:
        """Returns label index given string `label`."""

        multilabel_no_label = torch.zeros(len(self.label_set))

        if isinstance(label, str):
            try:
                label = self.label_set.index(label)
            except ValueError:
                assert self.multilabel, (
                    f"Label {label} not found in label set {self.label_set}. "
                    "Only multilabel datasets can have no label"
                )
                return multilabel_no_label

            if self.multilabel:
                return multilabel_no_label.scatter(0, torch.tensor(label), 1)
            return torch.tensor(label)

        assert (
            self.multilabel
        ), "Cannot convert list of labels to single label for non-multilabel dataset."

        return (
            torch.stack(
                [self.get_label_from_str(l) for l in label]
                or [multilabel_no_label]  # in case of empty list
            )
            .sum(0)
            .clamp_max(1)
        )

    def getitem_by_id(self, example_id: Any) -> dict[str, str | torch.Tensor]:
        """Returns item with ID `example_id`."""
        if self.id_separator in example_id:
            example_id = example_id.split(self.id_separator)[0]

        odict = self.examples[example_id]
        if "aggregate" in self.annotations:
            odict["label"] = self.annotations["aggregate"][example_id]
        else:
            odict["label"] = None

        return odict

    @abstractmethod
    def _load_data(self, split: str) -> tuple[
        dict[Any, dict[str, str | torch.Tensor | dict[str, torch.Tensor]]],
        list[str],
    ]:
        """Loads data from `split` and returns IDs, texts, labels in a dict
        indexed by IDs that contains another dictionary with `"text"`
        (after preprocessing), `"original_text"` (before preprocessing),
        `"label"`, and potentially other keys, e.g.,
        {
            id1: {"text": "lorem ipsum", "label": torch.tensor([4]), ...},
            ...
        },
        and the label set. Alternatively, if the dataset contains annotator
        labels, the `"label"` key can be a dictionary of annotator labels,
        along with the aggregated label, e.g.,
        {
            id1: {
                "text": "lorem ipsum",
                "label": {
                    "ann1": torch.tensor([4]),
                    "ann2": torch.tensor([3, 4]),
                    "aggregate": torch.tensor([4]),
                },
                ...
            },
            ...
        }
        """


class TokenizationMixin:
    """Mixin for tokenizing text for the `transformers` library.
    MUST be inherited before any other class because it requires
    init arguments.

    Attributes:
        _tokenization_mixin_data: dictionary containing the tokenizer and
            the maximum tokenization length.
    """

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        return dict(
            max_length=dict(
                type=int,
                help="maximum length of tokenized text",
            )
        )

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer | None = None,
        model_name_or_path: str | None = None,
        max_length: int | None = None,
        cache_dir: str | None = None,
        trust_remote_code: bool = False,
        for_llm: bool = True,
        *args,
        **kwargs,
    ):
        """Init.

        Args:
            tokenizer: tokenizer to use for tokenizing utterances,
                otherwise provide `model_name_or_path`.
            model_name_or_path: model name or path to load tokenizer from,
                otherwise provide `tokenizer`.
            max_length: maximum length of utterance.
            cache_dir: path to `transformers` cache directory.
            trust_remote_code: whether to trust remote code.

        Raises:
            AssertionError: if neither `tokenizer` nor `model_name_or_path`
                are provided.
        """

        assert (
            tokenizer is not None or model_name_or_path is not None
        ), "Either tokenizer or model_name_or_path must be provided."

        self._tokenization_mixin_data = dict(
            tokenizer=tokenizer
            or AutoTokenizer.from_pretrained(
                model_name_or_path,
                cache_dir=cache_dir,
                trust_remote_code=trust_remote_code,
            ),
            max_length=max_length,
        )
        if self._tokenization_mixin_data["tokenizer"].pad_token is None:
            self._tokenization_mixin_data["tokenizer"].pad_token = getattr(
                self._tokenization_mixin_data["tokenizer"], "unk_token", "[PAD]"
            )

        if for_llm:
            self._tokenization_mixin_data["tokenizer"].padding_side = "left"
        self.adds_special_token_in_first_position()

        super().__init__(*args, **kwargs)

    def adds_special_token_in_first_position(self) -> bool:
        if not hasattr(
            self._tokenization_mixin_data["tokenizer"], "special_first"
        ):
            wo_space = self._tokenization_mixin_data["tokenizer"].tokenize(
                "test"
            )
            with_space = self._tokenization_mixin_data["tokenizer"].tokenize(
                " test"
            )
            self._tokenization_mixin_data["tokenizer"].special_first = len(
                wo_space
            ) != len(with_space)
        return self._tokenization_mixin_data["tokenizer"].special_first

    def dict_tokenize(
        self,
        data: dict[Any, dict[str, str | torch.Tensor]],
    ):
        """Tokenizes text in `data` in-place.

        Args:
            data: dictionary containing text to tokenize.
            text_preprocessor: function to preprocess text.
        """
        for k in data:
            data[k]["encoding"] = self.tokenize(data[k]["text"])

    def batch_tokenize(self, texts: list[str | list[str]]):
        """Tokenizes text in `data` in-place.

        Args:
            data: list of dictionaries containing text to tokenize.
            text_preprocessor: function to preprocess text.
        """

        return self._tokenization_mixin_data["tokenizer"].batch_encode_plus(
            texts,
            max_length=self._tokenization_mixin_data["max_length"],
            truncation="only_second",
            padding="max_length",
            return_tensors="pt",
        )

    def tokenize(self, text: str, **kwargs) -> Mapping[str, torch.Tensor]:
        """Tokenizes text.

        Args:
            text: text to tokenize.
            kwargs: other arguments to pass to tokenizer.

        Returns:
            Tensor of token ids.
        """
        if self._tokenization_mixin_data["max_length"] is None:
            return self._tokenization_mixin_data["tokenizer"](
                text, return_tensors="pt", return_token_type_ids=False, **kwargs
            )

        return self._tokenization_mixin_data["tokenizer"](
            text,
            max_length=self._tokenization_mixin_data["max_length"],
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            return_token_type_ids=False,
            **kwargs,
        )

    def decode(self, input_ids: torch.Tensor) -> str:
        return self._tokenization_mixin_data["tokenizer"].decode(
            input_ids.squeeze()
        )

    def get_tokenizer(self) -> PreTrainedTokenizer:
        return self._tokenization_mixin_data["tokenizer"]

    def tokenize_conversation(self, conversation: list[dict[str, str]]):
        """Tokenizes a conversation.

        Args:
            conversation: list of dictionaries containing text to tokenize.
                Keys are "role" and "content". "role" is either "user", "system",
                or "assistant".

        Returns:
            List of tokenized texts.
        """

        return self._tokenization_mixin_data["tokenizer"].apply_chat_template(
            conversation, return_tensors="pt"
        )
