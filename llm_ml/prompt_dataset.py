import warnings
import random
import re
from typing import Any
from copy import deepcopy
from string import Template

import torch
from legm import from_namespace
from llm_ml.base_datasets import TokenizationMixin
from llm_ml.base_prompts import (
    PromptBaseDataset,
    ReasonablenessPromptBaseDataset,
)
from llm_ml.utils import string_overlap_idx_in_token_space


class OpenAIPromptTextDataset(PromptBaseDataset):
    """Prompt dataset for text classification, based on other TextDatasetWithPriors.
    Uses `from_namespace`, so it cannot be inherited from.
    """

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        args = PromptBaseDataset.argparse_args()
        args.update(
            dict(
                use_system_prompt=dict(
                    type=bool,
                    default=False,
                    help="whether to use role in prompt",
                    searchable=True,
                ),
            )
        )
        return args

    @from_namespace
    def __init__(self, use_system_prompt: bool, *args, **kwargs):
        self.use_system_prompt = use_system_prompt
        if self.use_system_prompt:
            kwargs["system_prompt"] = kwargs["instruction_prompt"]
            kwargs["instruction_prompt"] = ""
            self.log("Using instruction as system prompt", "debug")
        else:
            kwargs["system_prompt"] = ""
        super().__init__(*args, **kwargs)


class OpenAIReasonablenessPromptTextDataset(ReasonablenessPromptBaseDataset):
    """Prompt dataset for text classification, based on other TextDatasetWithPriors.
    Uses `from_namespace`, so it cannot be inherited from.
    """

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        args = ReasonablenessPromptBaseDataset.argparse_args()
        args.update(
            dict(
                use_system_prompt=dict(
                    type=bool,
                    default=False,
                    help="whether to use role in prompt",
                    searchable=True,
                ),
            )
        )
        return args

    @from_namespace
    def __init__(self, use_system_prompt: bool, *args, **kwargs):
        self.use_system_prompt = use_system_prompt
        if self.use_system_prompt:
            kwargs["system_prompt"] = kwargs["instruction_prompt"]
            kwargs["instruction_prompt"] = ""
            self.log("Using instruction as system prompt", "debug")
        else:
            kwargs["system_prompt"] = ""
        super().__init__(*args, **kwargs)


class PromptTextDataset(PromptBaseDataset):
    """Prompt dataset for text classification, based on other TextDatasetWithPriors.
    Uses `from_namespace`, so it shouldn't be inherited from."""

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PromptDataset(TokenizationMixin, PromptBaseDataset):
    """Prompt dataset for text classification, based on other TextDatasetWithPriors."""

    name = "Prompt tokenized dataset"

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        return (
            PromptBaseDataset.argparse_args()
            | TokenizationMixin.argparse_args()
        )

    @from_namespace
    def __init__(self, seed: int | None = None, *args, **kwargs):
        # Store the user‑selected flag **before** the parent initialises any
        # prompts so that downstream formatting can already act on it
        self.seed = seed
        super().__init__(*args, **kwargs)
        example_prompt = self[0]
        self.log(
            "Example tokenization: "
            + self.decode(example_prompt["encoding"]["input_ids"]),
            "debug",
        )
        self.train_dataset.label_lens = {
            e["id"]: self._get_label_tokenization_length(e["label"])
            for e in self.train_dataset
        }
        self.test_dataset.label_lens = {
            e["id"]: self._get_label_tokenization_length(e["label"])
            for e in self.test_dataset
        }

    def debug_message(self):
        return "\n".join(
            [
                super().debug_message(),
                "Example tokenization: "
                + self.decode(self[0]["encoding"]["input_ids"]),
            ]
        )

    def _get_label_tokenization_length(self, label: torch.Tensor) -> int:
        """Returns the length of the label tokenization."""
        # Tokenize the label without adding special tokens
        label = self.any_dataset.index_label_set(label)
        label = self.label_formatter(label)
        label_tokens = self.tokenize(label, add_special_tokens=False)[
            "input_ids"
        ][0]
        return len(label_tokens), label

    def _find_strings_with_equal_token_len(
        self, labels: list[str]
    ) -> list[str]:
        """
        For every label, return **one** random space-free string that tokenizes
        (after being formatted) to the exact same length as the label itself.
        Only `labels` is an argument.
        """
        tokenizer = self.get_tokenizer()

        fmt = lambda x: (
            self.label_formatter([x])
            if self.multilabel
            else self.label_formatter(x)
        )
        tok_len = lambda x: len(self.tokenize(fmt(x)).input_ids[0])

        # accept ONLY pure alphabetic, lowercase chunks, no blanks at all
        _ALPHA_RE = re.compile(r"^[a-z]+$")

        def is_word_start(tok: str) -> bool:
            if tok in tokenizer.all_special_tokens:
                return False
            if tok.startswith(("▁", "Ġ")):  # SentencePiece / GPT-2 BPE
                return True
            if tok.startswith("##"):  # WordPiece suffix
                return False
            return True

        id2tok = tokenizer.convert_ids_to_tokens(range(tokenizer.vocab_size))

        start_ids, sub_ids = [], []
        for tid, tok in enumerate(id2tok):
            if tid in tokenizer.all_special_ids:
                continue
            decoded = tokenizer.decode([tid], skip_special_tokens=True).strip()
            if not _ALPHA_RE.fullmatch(
                decoded
            ):  # drop tokens with spaces or symbols
                continue
            (start_ids if is_word_start(tok) else sub_ids).append(tid)

        if not sub_ids:  # rare – fall back to starts
            sub_ids = start_ids

        empty_len = tok_len("")  # template + empty slot
        used = set()
        results: dict[str, str] = {}

        random.seed(self.seed)
        for label in labels:
            target_len = tok_len(label)
            needed = target_len - empty_len
            if needed <= 0:
                raise ValueError("Template leaves no room for inserted text.")

            while True:
                cand_ids = [random.choice(start_ids)]
                if needed > 1:
                    cand_ids.extend(random.choices(sub_ids, k=needed - 1))

                candidate = tokenizer.decode(
                    cand_ids, skip_special_tokens=True
                ).strip()

                if (
                    candidate
                    and " " not in candidate  # no spaces in the final string
                    and candidate.lower() != label.lower()
                    and candidate not in used
                    and tok_len(candidate) == target_len
                ):
                    results[label] = candidate
                    used.add(candidate)
                    break

        return results

    def _format_instruction_prompt(self, instruction_prompt_template):
        """Formats the instruction prompt."""
        label_set = self.label_set

        labels = ", ".join(label_set[:-1]) + " and " + label_set[-1]
        return Template(
            instruction_prompt_template.replace("{labels}", "$labels")
        ).safe_substitute(labels=labels)

    def _format_assistant_prompt(
        self, assistant_prompt_template, label, cot=None
    ):
        """Formats the assistant prompt."""
        label_set = None
        label = self.label_formatter(
            self.any_dataset.index_label_set(
                label, alternative_label_set=label_set
            )
        )
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

    def __getitem__(self, index) -> dict[str, str | torch.Tensor]:
        """Returns prompt dictionary for `index`-th example in
        `test_dataset`. The return dictionary contains the following keys:
            - `id`: ID of example;
            - `query`: query text;
            - `text`: prompt text;
            - `encoding`: tokenized prompt;
            - `label`: tensor label of example.

        The prompt is constructed as follows:
            1. The instruction prompt is added.
            2. For each support example, the in-context prompt is added.
            3. The query prompt is added.
        """

        if self.user_prompt is None or self.assistant_prompt is None:
            warnings.warn(
                "Not using conversation template because user_prompt "
                "and/or assistant_prompt is not set"
            )
            item = super().__getitem__(index)
            item["encoding"] = self.tokenize(item["text"])
            return item

        query = self.test_dataset[index]
        support = self.sample(query)

        prompt = []

        if self.system_prompt:
            prompt.append(dict(role="system", content=self.system_prompt))

        for i, sample in enumerate(support):
            prompt.extend(
                [
                    dict(
                        role="user",
                        content=(self.instruction_prompt if i == 0 else "")
                        + self._format_user_prompt(
                            self.user_prompt, sample["text"]
                        ),
                    ),
                    dict(
                        role="assistant",
                        content=self._format_assistant_prompt(
                            self.assistant_prompt, sample["label"]
                        ),
                    ),
                ]
            )

        query_text = self._format_user_prompt(self.query_prompt, query["text"])
        if (
            not prompt or prompt[-1]["role"] == "system"
        ) and self.instruction_prompt:
            query_text = self.instruction_prompt + query_text

        prompt.append(dict(role="user", content=query_text))

        encoding = self.tokenize_conversation(prompt)

        return dict(
            id=query["id"],
            query=query["text"],
            text=self.decode(encoding[0]),
            encoding=dict(input_ids=encoding),
            label=self.handle_query_label(query),
        )

    def get_initial_label_tokens(self):
        """Returns the initial token for each label as it is
        going to appear in the prompt, i.e., if the tokens appears
        as a subword, the first token will differ from that of
        just tokenizing the labels."""

        def _equal_dict(d1, d2):
            """Checks if two dictionaries are equal."""
            if len(d1) != len(d2):
                return False
            for k, v in d1.items():
                if k not in d2 or d2[k] != v:
                    return False
            return True

        label_tokens = []

        # perform this loop in case something tokenizes weirdly
        # in some examples. Don't stop until two consecutive
        # examples have the same tokenization
        # (or we run out of examples)
        # this is a bit of a hack, but it works
        for random_example in self.test_dataset:

            # make a prompt with a dummy label to find the
            # length of the prompt before the label

            # add dummy text
            prompt_wo_label = self._format_user_prompt(
                self.incontext_prompt, random_example["text"]
            )

            # add dummy label so we know what to look for
            dummy_label = self.label_formatter(["{label}"])
            prompt_wo_label = Template(
                prompt_wo_label.replace("{label}", "$label")
            ).safe_substitute(label=dummy_label)

            # add cot just in case
            prompt_wo_label = self._format_cot(
                prompt_wo_label,
                self.sample_cot(self.cots, random_example["id"]),
            )

            # find overlap with dummy label
            idx = string_overlap_idx_in_token_space(
                self.get_tokenizer(), prompt_wo_label, "{label}"
            )

            label_tokens.append({})

            label_set_with_empty = [l for l in self.label_set]
            if self.multilabel:
                label_set_with_empty.append([])

            for label in label_set_with_empty:
                # create the prompt with the label
                prompt = self._format_incontext_prompt(
                    self.incontext_prompt,
                    random_example["text"],
                    self.any_dataset.get_label_from_str(label),
                    self.sample_cot(self.cots, random_example["id"]),
                ).strip()

                # find the token that corresponds to the label
                label_tokens[-1][label if label else "none"] = self.tokenize(
                    prompt, add_special_tokens=False
                )["input_ids"][0, idx]

            if len(label_tokens) > 1:
                # check if the label tokens are the same
                # if they are, we can just use the first one
                if not _equal_dict(label_tokens[-1], label_tokens[-2]):
                    label_tokens.pop(-2)
                else:
                    return label_tokens[-1]

        raise ValueError(
            "Check tokenizer, something is wrong with the tokenization"
        )


class ReasonablenessPromptDataset(
    TokenizationMixin, ReasonablenessPromptBaseDataset
):
    """Prompt dataset for text-label reasonableness classification,
    based on other TextDatasetWithPriors."""

    name = "Reasonableness prompt tokenized dataset"

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        return (
            ReasonablenessPromptBaseDataset.argparse_args()
            | TokenizationMixin.argparse_args()
        )

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        example_prompt = self[0]
        self.log(
            "Example tokenization: "
            + self.decode(example_prompt["encoding"]["input_ids"]),
            "debug",
        )

    def debug_message(self):
        return "\n".join(
            [
                super().debug_message(),
                "Example tokenization: "
                + self.decode(self[0]["encoding"]["input_ids"]),
            ]
        )

    def __getitem__(self, index) -> dict[str, str | torch.Tensor]:
        """Returns prompt dictionary for `index`-th example in
        `test_dataset`. The return dictionary contains the following keys:
            - `id`: ID of example;
            - `query`: query text;
            - `text`: prompt text;
            - `encoding`: tokenized prompt;
            - `label`: tensor label of example.

        The prompt is constructed as follows:
            1. The instruction prompt is added.
            2. For each support example, the in-context prompt is added.
            3. The query prompt is added.
        """

        if self.user_prompt is None or self.assistant_prompt is None:
            warnings.warn(
                "Not using conversation template because user_prompt "
                "and/or assistant_prompt is not set"
            )
            item = super().__getitem__(index)
            item["encoding"] = self.tokenize(item["text"])
            return item

        query = self.test_dataset[index]
        support, r_labels = self.sample(query)

        prompt = []

        if self.system_prompt:
            prompt.append(dict(role="system", content=self.system_prompt))

        for i, sample in enumerate(support):
            prompt.extend(
                [
                    dict(
                        role="user",
                        content=(self.instruction_prompt if i == 0 else "")
                        + self._format_r_user_prompt(
                            self.user_prompt, sample["text"], sample["label"]
                        ),
                    ),
                    dict(
                        role="assistant",
                        content=self._format_r_assistant_prompt(
                            self.assistant_prompt,
                            self.sample_cot(self.cots, sample["id"]),
                            r_labels[i],
                        ),
                    ),
                ]
            )

        query_text = self._format_r_user_prompt(
            self.query_prompt, query["text"], query["label"]
        )
        if (
            not prompt or prompt[-1]["role"] == "system"
        ) and self.instruction_prompt:
            query_text = self.instruction_prompt + query_text

        prompt.append(dict(role="user", content=query_text))

        encoding = self.tokenize_conversation(prompt)

        return dict(
            id=query["id"],
            query=query["text"],
            text=self.decode(encoding[0]),
            encoding=dict(input_ids=encoding),
            label=torch.tensor(1),
            checked_label=query["label"],
        )

    def get_initial_label_tokens(self):
        """Returns the initial token for each label as it is
        going to appear in the prompt, i.e., if the tokens appears
        as a subword, the first token will differ from that of
        just tokenizing the labels."""

        random_example = self.test_dataset[0]

        # make a prompt with a dummy label to find the
        # length of the prompt before the label

        # add dummy text
        prompt_wo_label = self._format_r_user_prompt(
            self.incontext_prompt,
            random_example["text"],
            random_example["label"],
        )

        # add dummy label so we know what to look for
        dummy_label = "{label}"
        prompt_wo_label = Template(
            prompt_wo_label.replace("{r}", "$r")
        ).safe_substitute(r=dummy_label)

        # add cot just in case
        prompt_wo_label = self._format_cot(
            prompt_wo_label,
            self.sample_cot(self.cots, random_example["id"]),
        )

        # find overlap with dummy label
        idx = string_overlap_idx_in_token_space(
            self.get_tokenizer(), prompt_wo_label, "{label}"
        )

        label_tokens = {}
        for i, label in enumerate(self.label_set):
            # create the prompt with the label
            prompt = self._format_r_incontext_prompt(
                self.incontext_prompt,
                random_example["text"],
                random_example["label"],
                self.sample_cot(self.cots, random_example["id"]),
                i,
            ).strip()

            # find the token that corresponds to the label
            label_tokens[label] = self.tokenize(
                prompt, add_special_tokens=False
            )["input_ids"][0, idx]

        return label_tokens


class PromptDatasetWithQueryLabels(TokenizationMixin, PromptBaseDataset):
    """Prompt dataset for text classification, based on other TextDatasetWithPriors.
    This includes the query labels in the prompt, used for finetuning."""

    name = "Prompt tokenized dataset with query"

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        return (
            PromptBaseDataset.argparse_args()
            | TokenizationMixin.argparse_args()
        )

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        example_prompt = self[0]
        self.log(
            "Example tokenization: "
            + self.decode(example_prompt["encoding"]["input_ids"]),
            "debug",
        )

    def __getitem__(self, index) -> dict[str, str | torch.Tensor]:
        """Returns prompt dictionary for `index`-th example in
        `test_dataset`. The return dictionary contains the following keys:
            - `id`: ID of example;
            - `query`: query text;
            - `input`: prompt text
            - `text`: prompt text including labels;
            - `encoding`: tokenized prompt;
            - `label`: tensor label of example.

        The prompt is constructed as follows:
            1. The instruction text is added (system prompt + instruction prompt)
            2. Adding the formatted support prompts (demonstration examples)
            3. Adding the query prompt formatted with the query text
            4. Adding the formatted query label using the label_formatter
        """
        query = self.test_dataset[index]
        support = self.sample(query)
        if self.include_query_in_demos:
            demo_query = deepcopy(query)
            demo_query_label = self.handle_query_label(
                query,
                is_demo=True,
                dataset=self.train_dataset,
            )
            demo_query["label"] = demo_query_label
            support = [demo_query] + support
        else:
            demo_query_label = None

        instruction_text = f"{self.system_prompt}" f"{self.instruction_prompt}"

        support_prompts = [
            self._format_incontext_prompt(
                self.incontext_prompt, sample["text"], sample["label"]
            )
            for sample in support
        ]

        prompt = (
            instruction_text
            + "".join(support_prompts)
            + self.query_prompt.format(text=query["text"])
            + self.label_formatter(
                self.any_dataset.index_label_set(query['label'])
            )
        )

        ## For eval purposes
        input = instruction_text + self.query_prompt.format(text=query["text"])

        item = dict(
            id=query["id"],
            query=query["text"],
            input=input,
            text=prompt,
            label=self.handle_query_label(query),
            demo_label=demo_query_label,
        )

        if self.user_prompt is None or self.assistant_prompt is None:
            warnings.warn(
                "Not using conversation template because user_prompt "
                "and/or assistant_prompt is not set"
            )

        item["encoding"] = self.tokenize(item["text"])
        return item

    def get_list(self):
        """Converts the dataset to List."""
        test_data = [self[item] for item in range(len(self))]
        return test_data
