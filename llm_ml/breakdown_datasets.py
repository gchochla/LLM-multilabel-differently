import warnings
from typing import Any
from copy import deepcopy
from string import Template
import math
import itertools
import random

import torch
from legm import from_namespace
from llm_ml.base_datasets import TokenizationMixin
from llm_ml.base_prompts import (
    PromptBaseDataset,
    ReasonablenessPromptBaseDataset,
)
from llm_ml.utils import string_overlap_idx_in_token_space

class UnaryBreakdownDataset(TokenizationMixin, PromptBaseDataset):
    """Prompt dataset for unary breakdown classification, based on other TextDatasetWithPriors."""

    name = "Unary breakdown dataset"

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        return (
            PromptBaseDataset.argparse_args()
            | TokenizationMixin.argparse_args()
        )

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.multilabel:
            self.original_label_set = self.label_set + ['none']
        else:
            self.original_label_set = self.label_set
        self.label_set = ['reasonable', 'unreasonable']
        
        """
        self.index converts from a single digit to the corresponding question
        and single label to evaluate, i.e. self.index[35] = (4, 4)
        """
        
        self.test_questions = []
        self.test_answers = []
        self.index = []
        
        for i in range(len(self.test_dataset)):
            question = self.test_dataset[i]['text']
            self.test_questions.append(question)
            self.index += [(i, j) for j in range(len(self.original_label_set))]
            
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
        
    def __len__(self):
        return len(self.index)
    
    def multilabel_to_list(self, labels):
        """
        Converts multilabel tensor to a list of label indices. Adds -1 if no labels.
        """
        indices = torch.nonzero(labels, as_tuple=True)[0].tolist()
        if len(indices) == 0: # no labels
            indices = [-1]
        return indices
        
    def __getitem__(self, index) -> dict[str, str | torch.Tensor]:
        
        # unary index is index of answer we are investigating
        question_index, unary_index = self.index[index]
        
        query = self.test_dataset[question_index] # original question text
        
        original_question_text = self.test_questions[question_index]

        support = self.sample(query)
        
        """
        Not sure how to properly sample this using random... leaving as this for now.
        We want to sample half "Reasonable" and half "Unreasonable" (splitting ties randomly).
        For the "Unreasonable", randomly sample from all other labels.
        Again, not sure if this will work for multi-label.
        """
        
        num_reasonable = len(support) // 2
        if num_reasonable < len(support) / 2:
            if random.random() < 0.5:
                num_reasonable += 1 # split ties randomly
        num_unreasonable = len(support) - num_reasonable
        
        support_labels = [self.label_set[0]] * num_reasonable + [self.label_set[1]] * num_unreasonable
        random.shuffle(support_labels)
        
        prompt = 'Classify the following question-label pairs as either "reasonable" or "unreasonable". ' + \
            'Output either "reasonable" or "unreasonable" and nothing else.\n\n'
        
        for example, support_label in zip(support, support_labels):
            # example_text, example_label_texts = self.test_dataset.separate_question_labels(example['text'])
            example_text = example['text']
            prompt += f'Question: {example_text}\n'
            
            if support_label == self.label_set[0]: # reasonable label
                if not self.multilabel:
                    example_label = self.original_label_set[int(example['label'].item())]
                else:
                    example_labels = self.multilabel_to_list(example['label'])
                    if -1 in example_labels:
                        example_label = 'none'
                    else:
                        example_label = self.original_label_set[random.choice(example_labels)]
                    
            elif support_label == self.label_set[1]: # unreasonable label
                if not self.multilabel:
                    # remove correct label, then randomly sample
                    unreasonable_labels = [text for i, text in enumerate(self.original_label_set) if i != int(example['label'].item())]
                    example_label = random.choice(unreasonable_labels)
                else:
                    example_labels = self.multilabel_to_list(example['label'])
                    unreasonable_labels = [text for i, text in enumerate(self.original_label_set) if i not in example_labels]
                    if unreasonable_labels == []:
                        example_label = 'none'
                    else:
                        example_label = random.choice(unreasonable_labels)
            
            prompt += f'Label: {example_label}\n'
            prompt += f'Is the label reasonable: {support_label}\n\n'
            
        prompt += f'Question: {original_question_text}\n'
        prompt += f'Label: {self.original_label_set[unary_index]}\n'
        prompt += f'Is the label reasonable: '
        
         # we don't use the label in odict, but need it for code compatability
        if self.multilabel:
            dummy_label = torch.zeros(len(self.original_label_set) - 1) # we added "none" as a label, but need to remove that
        else:
            dummy_label = torch.tensor(0)
        
        odict = dict(
            id=query['id'] + '_unary_' + self.original_label_set[unary_index],
            query=query['text'],
            text=prompt,
            label=dummy_label, # label here doesn't matter
            original_label=query['label'],
            encoding=self.tokenize(prompt),
        )
        
        return odict

    def get_initial_label_tokens(self):
        """Returns the initial token for each label as it is
        going to appear in the prompt, i.e., if the tokens appears
        as a subword, the first token will differ from that of
        just tokenizing the labels."""

        # random_example = self.test_dataset[0]

        # # make a prompt with a dummy label to find the
        # # length of the prompt before the label

        # # add dummy text
        # prompt_wo_label = self._format_user_prompt(
        #     self.incontext_prompt, random_example["text"]
        # )

        # # add dummy label so we know what to look for
        # dummy_label = self.label_formatter(["{label}"])
        # prompt_wo_label = Template(
        #     prompt_wo_label.replace("{label}", "$label")
        # ).safe_substitute(label=dummy_label)

        # # add cot just in case
        # prompt_wo_label = self._format_cot(
        #     prompt_wo_label,
        #     self.sample_cot(self.cots, random_example["id"]),
        # )

        # # find overlap with dummy label
        # idx = string_overlap_idx_in_token_space(
        #     self.get_tokenizer(), prompt_wo_label, "{label}"
        # )

        # label_tokens = {}
        
        # for label in self.label_set:
        #     # create the prompt with the label
        #     prompt = self._format_incontext_prompt(
        #         self.incontext_prompt,
        #         random_example["text"],
        #         # self.any_dataset.get_label_from_str(label),
        #         torch.tensor(self.label_set.index(label)),
        #         self.sample_cot(self.cots, random_example["id"]),
        #     )

        #     # find the token that corresponds to the label
        #     label_tokens[label] = self.tokenize(
        #         prompt, add_special_tokens=False
        #     )["input_ids"][0, idx]
        
        label_tokens = {}
        
        for label in self.label_set:
            label_tokens[label] = self.tokenize(label, add_special_tokens=False)['input_ids'][0, 0]
        
        return label_tokens
    
    
class BinaryBreakdownDataset(TokenizationMixin, PromptBaseDataset):
    
    """Prompt dataset for binary breakdown classification, based on other TextDatasetWithPriors."""

    name = "Binary breakdown dataset"

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        return (
            PromptBaseDataset.argparse_args()
            | TokenizationMixin.argparse_args()
        )

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.multilabel:
            self.original_label_set = self.label_set + ['none'] # add none as a label
        else:
            self.original_label_set = self.label_set
        
        self.label_set = ['a', 'b']
        # self.label_set = ['A', 'B']
        
        """
        self.index converts from a single digit to the corresponding question
        and the two labels to compare, i.e. self.index[35] = (4, 0, 2)
        """
        
        self.test_questions = []
        self.index = []
        
        for i in range(len(self.test_dataset)):
            question = self.test_dataset[i]['text']
            self.test_questions.append(question)
            
            for pair in list(itertools.combinations(range(len(self.original_label_set)), 2)):
                self.index.append((i, pair[0], pair[1]))
        
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
        
    def multilabel_to_list(self, labels):
        """
        Converts multilabel tensor to a list of label indices. Adds -1 if no labels.
        """
        indices = torch.nonzero(labels, as_tuple=True)[0].tolist()
        if len(indices) == 0: # no labels
            indices = [-1]
        return indices
        
    def __len__(self):
        return len(self.index)
        
    def __getitem__(self, index) -> dict[str, str | torch.Tensor]:
        
        question_index, a_index, b_index = self.index[index]
        
        query = self.test_dataset[question_index] # original question text
        
        original_question_text = self.test_questions[question_index]
        
        prompt = 'Classify the following question into one of the two following labels. ' + \
            'Output either "a" or "b" depending on which label is better and nothing else.\n\n'
        
        """
        Not sure how to properly have in-context examples for binary breakdown, as we don't
        really have a good ground truth for "comparison." So, just just comparing correct label to
        another random label for in-context.
        """
        
        support = self.sample(query)
        
        for example in support:
            # example_text, example_label_texts = self.test_dataset.separate_question_labels(example['text'])
            example_text = example['text']
            prompt += f'Question: {example_text}\n'
            
            if not self.multilabel:
                correct_label = self.original_label_set[int(example['label'].item())]
                incorrect_labels = [text for i, text in enumerate(self.original_label_set) if i != int(example['label'].item())]
                incorrect_label = random.choice(incorrect_labels)
            else:
                correct_labels = self.multilabel_to_list(example['label'])
                incorrect_labels = [text for i, text in enumerate(self.original_label_set) if i not in correct_labels]
                
                # either (or both) correct or incorrect labels is guaranteed to be not empty
                if -1 in correct_labels:
                    correct_label = 'none'
                else:
                    correct_label = self.original_label_set[random.choice(correct_labels)]
                
                if len(incorrect_labels) == 0:
                    incorrect_label = 'none'
                else:
                    incorrect_label = random.choice(incorrect_labels)
            
            
            if random.random() < 0.5:
                prompt += f'a. {correct_label}\n'
                prompt += f'b. {incorrect_label}\n'
                prompt += 'Correct Label: a\n\n'
            else:
                prompt += f'a. {incorrect_label}\n'
                prompt += f'b. {correct_label}\n'
                prompt += f'Correct Label: b\n\n'
            
        prompt += f'Question: {original_question_text}\n'
        prompt += f'a. {self.original_label_set[a_index]}\n'
        prompt += f'b. {self.original_label_set[b_index]}\n'
        prompt += f'Correct Label: '
        
        # we don't use the label in odict, but need it for code compatability
        if self.multilabel:
            dummy_label = torch.zeros(len(self.original_label_set) - 1) # we added "none" as a label, but need to remove that
        else:
            dummy_label = torch.tensor(0)
        
        odict = dict(
            id=query['id'] + '_binary_' + self.original_label_set[a_index] + '_' + self.original_label_set[b_index],
            query=query['text'],
            text=prompt,
            label=dummy_label,
            original_label=query['label'],
            encoding=self.tokenize(prompt),
        )
        
        return odict

    def get_initial_label_tokens(self):
        """Returns the initial token for each label as it is
        going to appear in the prompt, i.e., if the tokens appears
        as a subword, the first token will differ from that of
        just tokenizing the labels."""

        # random_example = self.test_dataset[0]

        # # make a prompt with a dummy label to find the
        # # length of the prompt before the label

        # # add dummy text
        # prompt_wo_label = self._format_user_prompt(
        #     self.incontext_prompt, random_example["text"]
        # )

        # # add dummy label so we know what to look for
        # dummy_label = self.label_formatter(["{label}"])
        # prompt_wo_label = Template(
        #     prompt_wo_label.replace("{label}", "$label")
        # ).safe_substitute(label=dummy_label)

        # # add cot just in case
        # prompt_wo_label = self._format_cot(
        #     prompt_wo_label,
        #     self.sample_cot(self.cots, random_example["id"]),
        # )

        # # find overlap with dummy label
        # idx = string_overlap_idx_in_token_space(
        #     self.get_tokenizer(), prompt_wo_label, "{label}"
        # )

        # label_tokens = {}
        
        # for label in self.label_set:
        #     # create the prompt with the label
        #     prompt = self._format_incontext_prompt(
        #         self.incontext_prompt,
        #         random_example["text"],
        #         # self.any_dataset.get_label_from_str(label),
        #         torch.tensor(self.label_set.index(label)),
        #         self.sample_cot(self.cots, random_example["id"]),
        #     )

        #     # find the token that corresponds to the label
        #     label_tokens[label] = self.tokenize(
        #         prompt, add_special_tokens=False
        #     )["input_ids"][0, idx]
        #     import pdb; pdb.set_trace()
        
        label_tokens = {}
        
        for label in self.label_set:
            label_tokens[label] = self.tokenize(label, add_special_tokens=False)['input_ids'][0, 0]
        
        return label_tokens
    
class ICLMultiLabelRatioDataset(TokenizationMixin, PromptBaseDataset):
    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        return (
            PromptBaseDataset.argparse_args()
            | TokenizationMixin.argparse_args()
        )
    
    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        for i in range(6):
            example_prompt = self[i]
    
    def __len__(self):
        return len(self.test_dataset) * 6
    
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
        query = self.test_dataset[index // 6]
        
        ratio = (index % 6) / 5.0
        
        support = self._multilabel_sample(query=query, dataset=self.train_dataset, shot=self.shot, ratio=ratio)
        
        if query["id"].split(self.train_dataset.id_separator)[1] == "aggregate":
            self._handle_labels(
                support, self.train_dataset, self._example_sampler_mixin_data["label_mode"]
            )
        
        self.ids_per_query[query["id"]] = [sample["id"] for sample in support]

        if self.include_query_in_demos:
            demo_query = deepcopy(query)
            demo_query_label = self.handle_query_label(
                query,
                is_demo=True,
                dataset=self.train_dataset,
            )
            demo_query["label"] = demo_query_label

            support = (
                support[: self.query_order]
                + [demo_query]
                + support[self.query_order :]
            )
        else:
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
            id=query["id"] + f'_ratio_{round(ratio, 1)}',
            query=query["text"],
            text=prompt,
            label=self.handle_query_label(query),
        )
        
        odict['encoding'] = self.tokenize(odict['text'])

        if demo_query_label is not None:
            odict["demo_label"] = demo_query_label

        return odict
    
    def get_initial_label_tokens(self):
        """Returns the initial token for each label as it is
        going to appear in the prompt, i.e., if the tokens appears
        as a subword, the first token will differ from that of
        just tokenizing the labels."""

        random_example = self.test_dataset[0]

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

        label_tokens = {}

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
            label_tokens[label if label else "none"] = self.tokenize(
                prompt, add_special_tokens=False
            )["input_ids"][0, idx]

        return label_tokens