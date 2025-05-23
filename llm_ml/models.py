from typing import Any, Literal, Callable
import os
import warnings
import json
import logging

import torch
import torch.nn as nn
from legm import from_namespace
from vllm import LLM, SamplingParams
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
from dotenv import load_dotenv
from trl import SFTTrainer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

from llm_ml.base_prompts import LabelSimilarityMixin
from llm_ml.utils import string_overlap_idx_in_token_space, tensor_overlap


class vLMForGeneration(nn.Module):
    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        return dict(
            model_name_or_path=dict(
                type=str,
                required=True,
                help="model name or path to load tokenizer and model from",
                metadata=dict(name=True, name_priority=2),
                searchable=True,
            ),
            max_new_tokens=dict(
                type=int,
                help="maximum number of new tokens to generate",
            ),
            trust_remote_code=dict(
                action="store_true",
                help="whether to trust remote code for model",
            ),
            quantization=dict(
                choices=["gptq", "awq", "fp8"],
                default="fp8",
                type=str,
                help="quantization to use for model",
            ),
        )

    def __init__(
        self,
        model_name_or_path: str,
        max_new_tokens: int | None = None,
        trust_remote_code: bool = False,
        quantization: Literal["gptq", "awq", "fp8"] = "fp8",
        logprobs: int | None = None,
        *args,
        **kwargs,
    ):
        """Init.

        Args:
            model_name_or_path: model name or path to load tokenizer and model from.
            cache_dir: path to `transformers` cache directory.
            trust_remote_code: whether to trust remote code for model.
            quantization: quantization to use for model.
            max_new_tokens: maximum number of new tokens to generate.
            logprobs: maximum number of logprobs to return.
            args, kwargs: arguments to pass to the model.
        """

        super().__init__()

        self.lm = LLM(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            quantization=quantization,
            max_logprobs=logprobs,
        )
        self.sampling_params = SamplingParams(
            temperature=0,
            max_tokens=max_new_tokens,
            logprobs=logprobs,
        )

    def forward(
        self,
        text: str | list[str],
        cutoff_str: str | None = None,
        prefix_cutoff_str: str | None = None,
    ):
        """Generates text from the model.

        Args:
            text: prompt to generate text from.
            cutoff_str: string to stop generation at.
            prefix_cutoff_str: string to stop reasoning at.

        Returns:
            Generated text.
        """

        if isinstance(text, str):
            text = [text]

        if cutoff_str is not None:
            self.sampling_params.stop = [cutoff_str]

        out, *_ = self.lm.generate(text, self.sampling_params, use_tqdm=False)

        out = {
            "ids": [o.token_ids for o in out.outputs],
            "text": [o.text.strip() for o in out.outputs],
            "scores": [o.logprobs for o in out.outputs],
        }

        if prefix_cutoff_str is not None:
            prefix_cutoff_inds = []
            for o in out["text"]:
                i = o.find(prefix_cutoff_str)
                if i == -1:
                    prefix_cutoff_inds.append(len(o))
                else:
                    prefix_cutoff_inds.append(i)

            out["prefix_text"] = [
                o[:i] for o, i in zip(out["text"], prefix_cutoff_inds)
            ]
            out["text"] = [
                o[i:].strip() for o, i in zip(out["text"], prefix_cutoff_inds)
            ]

            prefix_cutoff_token_inds = [
                string_overlap_idx_in_token_space(
                    self.lm.get_tokenizer(), o, prefix_cutoff_str
                )
                for o in out["text"]
            ]

            out["ids"] = [
                o[i:] for o, i in zip(out["ids"], prefix_cutoff_token_inds)
            ]
            out["scores"] = [
                o[i:] for o, i in zip(out["scores"], prefix_cutoff_token_inds)
            ]

        return out


class vLMForClassification(LabelSimilarityMixin, vLMForGeneration):

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        return (
            vLMForGeneration.argparse_args()
            | LabelSimilarityMixin.argparse_args()
        )

    @from_namespace
    def __init__(
        self, labels: list[str] | dict[str, torch.Tensor], *args, **kwargs
    ):
        """Init.

        Args:
            labels: labels to use for predictions, either a list of string
                labels, or a dictionary where string labels are keys and
                their tokenization the values.
            args, kwargs: arguments to pass to vLMForGeneration.
        """
        super().__init__(*args, **kwargs)
        self.labels = labels
        self.label_first_token_ids = None

    def set_label_decoding_utils(self, label_tokens: dict[str, int]):
        self.label_first_token_ids = {
            k: v.item() for k, v in label_tokens.items()
        }

    def forward(
        self, label_parser: Callable[[str], list[str]] | None = None, **kwargs
    ):
        out = super().forward(**kwargs)

        labels = list(self.labels)  # list or keys to list
        out["preds"] = [
            [label for label in labels if label in o.lower()]
            for o in out["text"]
        ]

        if label_parser is not None:
            try:  # make this try internal to each loop
                preds = [label_parser(o) for o in out["text"]]
                preds = [
                    [pred.lower() for pred in example_preds]
                    for example_preds in preds
                ]
                preds = [
                    [
                        (
                            pred
                            if pred in labels
                            else self.get_closest_label(pred, labels)
                        )
                        for pred in example_preds
                    ]
                    for example_preds in preds
                ]
                out["preds"] = [
                    list(
                        # in case some preds are the same
                        # because of similarity matching
                        set([pred for pred in example_pred if pred is not None])
                    )
                    for example_pred in preds
                ]
            except:
                pass

        if self.label_first_token_ids is not None:

            # find the index of the first token that is a label
            all_first_label_inds = [
                [
                    tensor_overlap(torch.tensor(o), torch.tensor([i]))
                    for i in self.label_first_token_ids.values()
                ]
                for o in out["ids"]
            ]
            first_label_inds = [min(i) for i in all_first_label_inds]

            # get the scores for all labels from the first token that is a label
            out["scores"] = [
                (
                    torch.tensor(
                        [
                            o[i][j].logprob if j in o[i] else float("-inf")
                            for j in self.label_first_token_ids.values()
                        ]
                    )
                    if i < len(o)
                    # in case of multilabel, label_first_token_ids also has None
                    else torch.zeros(
                        max(len(self.labels), len(self.label_first_token_ids))
                    )
                )
                for o, i in zip(out["scores"], first_label_inds)
            ]

            out["scores"] = [
                {
                    k: v.item()
                    for k, v in zip(self.label_first_token_ids, o.softmax(0))
                }
                for o in out["scores"]
            ]

        return out


class LMForGeneration(nn.Module):
    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        return dict(
            model_name_or_path=dict(
                type=str,
                required=True,
                help="model name or path to load tokenizer and model from",
                metadata=dict(name=True, name_priority=2),
                searchable=True,
            ),
            max_new_tokens=dict(
                type=int,
                help="maximum number of new tokens to generate",
            ),
            generation_max_length=dict(
                type=int,
                help="maximum length of generation (including prompt)",
            ),
            model_dtype=dict(
                type=str,
                default="float",
                help="dtype of model",
            ),
            load_in_8bit=dict(
                action="store_true",
                help="whether to load model in 8bit",
            ),
            load_in_4bit=dict(
                action="store_true",
                help="whether to load model in 4bit",
            ),
            trust_remote_code=dict(
                action="store_true",
                help="whether to trust remote code for model",
            ),
            device=dict(
                type=str,
                help="device to load model on",
            ),
        )

    def __init__(
        self,
        model_name_or_path: str,
        max_new_tokens: int | None = None,
        generation_max_length: int | None = None,
        model_dtype: str = "float",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        device: str | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        trust_remote_code: bool = False,
        *args,
        **kwargs,
    ):
        """Init.

        Args:
            model_name_or_path: model name or path to load tokenizer and model from.
            max_new_tokens: maximum number of new tokens to generate.
            generation_max_length: maximum length of generation (including prompt).
            model_dtype: dtype of model.
            load_in_8bit: whether to load model in 8bit.
            load_in_4bit: whether to load model in 4bit.
            device: device to load model on.
            tokenizer: tokenizer to use.
            trust_remote_code: whether to trust remote code for model.
        """

        assert (
            generation_max_length is not None or max_new_tokens is not None
        ), "Either generation_max_length or max_new_tokens must be provided"

        assert not (
            load_in_8bit and load_in_4bit
        ), "Only one of load_in_8bit and load_in_4bit can be provided"

        super().__init__()

        load_kwargs = dict(
            torch_dtype=(
                getattr(torch, model_dtype)
                if isinstance(model_dtype, str)
                else model_dtype
            ),
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            device_map=device,
            trust_remote_code=trust_remote_code,
        )

        try:
            self.lm = AutoLigerKernelForCausalLM.from_pretrained(
                model_name_or_path, **load_kwargs
            ).eval()
            self.causal = True
        except ValueError:  # not a causal LM but a seq2seq LM
            self.lm = AutoModelForSeq2SeqLM.from_pretrained(
                model_name_or_path, **load_kwargs
            ).eval()
            self.causal = False

        if self.lm.generation_config.pad_token_id is None:
            if not hasattr(self.lm.generation_config.eos_token_id, "__len__"):
                self.lm.generation_config.pad_token_id = (
                    self.lm.generation_config.eos_token_id
                )
            else:
                self.lm.generation_config.pad_token_id = (
                    self.lm.generation_config.eos_token_id[0]
                )
        if max_new_tokens is not None:
            self.lm.generation_config.max_new_tokens = max_new_tokens
        else:
            self.lm.generation_config.max_length = generation_max_length

        self.tokenizer = tokenizer

    def _process_cutoff_args(
        self,
        cutoff_ids: list[torch.Tensor] | None,
        cutoff_str: list[str] | None,
    ) -> tuple[torch.Tensor | None, str | None]:
        """Check that the cutoff arguments are valid,
        and computes both IDs and str."""

        def aux(cids, cstr):
            if not cstr:
                cstr = None
            if cids is None or not cids.tolist():
                cids = None

            if cstr is not None:
                if cids is None:
                    assert (
                        self.tokenizer is not None
                    ), f"Tokenizer required to encode new example string \"{cstr}\""

                    # shape is 1 x tokenization_length
                    cids = self.tokenizer(
                        cstr, return_tensors="pt", add_special_tokens=False
                    )["input_ids"]
                else:
                    if self.tokenizer is not None:
                        assert torch.all(
                            cids
                            == self.tokenizer(
                                cstr,
                                return_tensors="pt",
                                add_special_tokens=False,
                            )["input_ids"]
                        ), (
                            f"Provided cutoff string `{cstr}` "
                            f"and ids `{cids.tolist()}` do not match"
                        )
            elif cids is not None and self.tokenizer is not None:
                cstr = self.tokenizer.decode(cids)

            return cids, cstr

        if cutoff_ids is not None and not isinstance(cutoff_ids, list):
            cutoff_ids = [cutoff_ids]
        if cutoff_str is not None and not isinstance(cutoff_str, list):
            cutoff_str = [cutoff_str]

        if cutoff_ids is None and cutoff_str is None:
            return None, None
        elif cutoff_ids is None and cutoff_str is not None:
            vals = [aux(None, cstr) for cstr in cutoff_str]
        elif cutoff_str is None:
            vals = [aux(cids, None) for cids in cutoff_ids]
        else:
            vals = [
                aux(cids, cstr) for cids, cstr in zip(cutoff_ids, cutoff_str)
            ]

        return [v[0] for v in vals], [v[1] for v in vals]

    @torch.no_grad()
    def forward(
        self,
        cutoff_ids: list[torch.Tensor] | None = None,
        cutoff_str: list[str] | None = None,
        prefix_cutoff_ids: torch.Tensor | None = None,
        prefix_cutoff_str: str | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor | list[str]]:
        """Generates text from the model.

        Args:
            cutoff_ids: ids to stop generation at.
            cutoff_str: string to stop generation at.
            prefix_cutoff_ids: ids to stop reasoning at.
            prefix_cutoff_str: string to stop reasoning at.
            **kwargs: keyword arguments to pass to the model.

        Returns:
            A dictionary of the form
            {
                "ids":             list of generated ids before cutoff,
                "text":            list of generated text before cutoff,
                "residual_ids":    list of residual ids after cutoff,
                "residual_text":   list of residual text after cutoff,
                "last_hidden_state": torch.FloatTensor of shape (batch_size, hidden_size)
                                     containing the last‑layer hidden representation of the
                                     **first generated token**,
            }
        """

        assert (
            not self.causal or "input_ids" in kwargs
        ), "input_ids required for causal LM to decode predictions"

        cutoff_ids, cutoff_str = self._process_cutoff_args(
            cutoff_ids, cutoff_str
        )

        self.lm.generation_config.stop_strings = cutoff_str

        out_dict = self.lm.generate(
            **kwargs,
            return_dict_in_generate=True,
            output_hidden_states=True,
            output_attentions=True,
            output_scores=True,
            tokenizer=self.tokenizer,
        )
        out = out_dict.sequences

        orig_ids = out
        # causal models "generate" the input as well
        if self.causal:
            out = out[:, kwargs["input_ids"].shape[-1] :]

        out = {"ids": out, "scores": torch.cat(out_dict.scores).detach()}
        # add dummy batch dimension to scores -- if necessary -- for consistency
        if out["scores"].ndim == 2:
            out["scores"] = out["scores"].unsqueeze(0)

        if self.tokenizer is not None:
            out["text"] = [
                self.tokenizer.decode(o, skip_special_tokens=True).strip()
                for o in out["ids"]
            ]

        prefix_cutoff_ids, prefix_cutoff_str = self._process_cutoff_args(
            prefix_cutoff_ids, prefix_cutoff_str
        )
        prefix_cutoff_ids = prefix_cutoff_ids[0] if prefix_cutoff_ids else None
        prefix_cutoff_str = prefix_cutoff_str[0] if prefix_cutoff_str else None

        # remove hallucinated examples
        cutoff_str = cutoff_str[0] if cutoff_str else None
        cutoff_ids = cutoff_ids[0] if cutoff_ids else None
        if cutoff_str is not None and self.tokenizer is not None:
            cutoff_inds = []
            for o in out["text"]:
                i = o.find(cutoff_str)
                # need the loop because of this
                # ow i = -1 will keep the last character in o[:i]
                if i == -1:
                    cutoff_inds.append(len(o))
                else:
                    cutoff_inds.append(i)
            out["residual_text"] = [
                o[i:] for o, i in zip(out["text"], cutoff_inds)
            ]
            out["text"] = [
                o[:i].strip() for o, i in zip(out["text"], cutoff_inds)
            ]

            # NOTE: this can cause mismatches when the first token is a subword,
            # because here the tokenization begins with it,
            # and therefore it cannot be a subword token by default.
            if self.causal:
                orig_text = [
                    self.tokenizer.decode(o, skip_special_tokens=True)
                    for o in orig_ids
                ]
                orig_lens = [
                    len(self.tokenizer.decode(ii, skip_special_tokens=True))
                    for ii in kwargs["input_ids"]
                ]
                orig_text_wo_residual = [
                    o[: l + i + 1]
                    for o, i, l in zip(orig_text, cutoff_inds, orig_lens)
                ]

                out["ids"] = [
                    self.tokenizer(
                        ot, return_tensors="pt", add_special_tokens=False
                    ).input_ids[0, len(i) :]
                    for ot, i in zip(
                        orig_text_wo_residual,
                        [
                            [
                                tid
                                for tid in inp
                                if tid not in self.tokenizer.all_special_ids
                            ]
                            for inp in kwargs["input_ids"]
                        ],
                    )
                ]
            else:
                out["ids"] = [
                    self.tokenizer(
                        o, return_tensors="pt", add_special_tokens=False
                    ).input_ids[0]
                    for o in out["text"]
                ]

            out["residual_ids"] = [
                self.tokenizer(
                    o, return_tensors="pt", add_special_tokens=False
                )["input_ids"][0]
                for o in out["residual_text"]
            ]

            # NOTE: doesnt work because of o[:i].strip() in text
            # ids_cutoff_inds = [
            #     tensor_overlap(o, c.to(o.device))
            #     for o, c in zip(orig_ids, cutoff_ids)
            # ]

            out["scores"] = [
                s[: len(i)] for s, i in zip(out["scores"], out["ids"])
            ]

        elif cutoff_ids is not None:
            warnings.warn(
                "Provided `cutoff_ids` and no tokenizer, there's a "
                "chance tokenization doesn't yield identical IDs."
            )
            cutoff_ids = cutoff_ids.squeeze()
            cutoff_inds = [
                tensor_overlap(o, c.to(o.device))
                for o, c in zip(out["ids"], cutoff_ids)
            ]
            out["residual_ids"] = [
                o[i:] for o, i in zip(out["ids"], cutoff_inds)
            ]
            out["ids"] = [o[:i] for o, i in zip(out["ids"], cutoff_inds)]
            out["scores"] = [o[:i] for o, i in zip(out["scores"], cutoff_inds)]

        try:
            cutoff_inds = [
                tensor_overlap(o, c.to(o.device))
                for o, c in zip(out["ids"], cutoff_ids)
            ]
        except:
            cutoff_inds = None

        # remove reasoning
        if prefix_cutoff_str is not None and self.tokenizer is not None:
            prefix_cutoff_inds = []
            for o in out["text"]:
                i = o.find(prefix_cutoff_str)
                if i == -1:
                    prefix_cutoff_inds.append(len(o))
                else:
                    prefix_cutoff_inds.append(i)

            out["prefix_text"] = [
                o[:i] for o, i in zip(out["text"], prefix_cutoff_inds)
            ]
            out["text"] = [
                o[i:].strip() for o, i in zip(out["text"], prefix_cutoff_inds)
            ]
            out["ids"] = [
                self.tokenizer(
                    o, return_tensors="pt", add_special_tokens=False
                )["input_ids"][0]
                for o in out["text"]
            ]
            out["prefix_ids"] = [
                self.tokenizer(
                    o, return_tensors="pt", add_special_tokens=False
                )["input_ids"][0]
                for o in out["prefix_text"]
            ]

            out["scores"] = [
                s[: len(i)] for s, i in zip(out["scores"], out["ids"])
            ]

        elif prefix_cutoff_ids is not None:
            warnings.warn(
                "Provided `prefix_cutoff_ids` and no tokenizer, there's a "
                "chance tokenization doesn't yield identical IDs."
            )
            prefix_cutoff_ids = prefix_cutoff_ids.squeeze()
            prefix_cutoff_inds = [
                tensor_overlap(o, c.to(o.device))
                for o, c in zip(out["ids"], prefix_cutoff_ids)
            ]
            out["prefix_ids"] = [
                o[:i] for o, i in zip(out["ids"], prefix_cutoff_inds)
            ]
            out["ids"] = [o[i:] for o, i in zip(out["ids"], prefix_cutoff_inds)]
            out["scores"] = [
                o[i:] for o, i in zip(out["scores"], prefix_cutoff_inds)
            ]
            assert all(
                len(ids) == len(scores)
                for ids, scores in zip(out["ids"], out["scores"])
            )

        try:
            prefix_cutoff_inds = [
                tensor_overlap(o, c.to(o.device))
                for o, c in zip(out["ids"], prefix_cutoff_ids)
            ]
        except:
            prefix_cutoff_inds = None

        if (
            self.causal
            and hasattr(out_dict, "hidden_states")
            and out_dict.hidden_states
        ):
            # hidden_states dim:
            #   tuple(generated_tokens) x tuple(layers) x bs x {input len if first token else 1} x hidden_size
            # tokens dim:
            # generated_tokens x bs x hidden_size

            tokens = torch.stack(
                [hs[-1][:, -1, :] for hs in out_dict.hidden_states]
            )
            # bs x generated_tokens x hidden_size
            tokens = tokens.permute(1, 0, 2)

            if cutoff_inds is not None:
                tokens = [
                    t[:i] if i < len(t) else torch.zeros_like(t)
                    for t, i in zip(tokens, cutoff_inds)
                ]
            if prefix_cutoff_inds is not None:
                tokens = [
                    t[i:] if i < len(t) else torch.zeros_like(t)
                    for t, i in zip(tokens, prefix_cutoff_inds)
                ]
        elif (
            (not self.causal)
            and hasattr(out_dict, "decoder_hidden_states")
            and out_dict.decoder_hidden_states
        ):
            # hidden_states dim:
            #   tuple(generated_tokens) x tuple(layers) x bs x {input len if first token else 1} x hidden_size
            # tokens dim:
            # generated_tokens x bs x hidden_size
            tokens = torch.stack(
                [hs[-1][:, -1, :] for hs in out_dict.decoder_hidden_states]
            )
            # bs x generated_tokens x hidden_size
            tokens = tokens.permute(1, 0, 2)

            if cutoff_inds is not None:
                tokens = [
                    t[:i] if i < len(t) else torch.zeros_like(t)
                    for t, i in zip(tokens, cutoff_inds)
                ]
            if prefix_cutoff_inds is not None:
                tokens = [
                    t[i:] if i < len(t) else torch.zeros_like(t)
                    for t, i in zip(tokens, prefix_cutoff_inds)
                ]
        else:
            tokens = None

        out["last_hidden_state"] = tokens

        # attention dim:
        # tuple(generated_tokens) x tuple(layers) x bs x attention heads x {input_len if first token else 1} x {input_len if first token else input_len + gen tokens up to then}
        out["attentions"] = [
            [layer_att[:, :, -1, :] for layer_att in gen_tokens]
            for gen_tokens in out_dict.attentions
        ]

        return out


class LMForClassification(LabelSimilarityMixin, LMForGeneration):

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        return (
            LMForGeneration.argparse_args()
            | LabelSimilarityMixin.argparse_args()
        )

    @from_namespace
    def __init__(
        self, labels: list[str] | dict[str, torch.Tensor], *args, **kwargs
    ):
        """Init.

        Args:
            labels: labels to use for predictions, either a list of string
                labels, or a dictionary where string labels are keys and
                their tokenization the values.
            args, kwargs: arguments to pass to LMForGeneration.
        """
        super().__init__(*args, **kwargs)
        self.lm.generation_config.top_k = None
        self.lm.generation_config.do_sample = False
        self.lm.generation_config.top_p = None
        self.lm.generation_config.temperature = None
        if isinstance(labels, list):
            labels = [label.lower() for label in labels]
        else:
            labels = {label.lower(): value for label, value in labels.items()}
        self.labels = labels

        self.label_first_token_ids = None

    def set_label_decoding_utils(self, label_tokens: dict[str, int]):
        self.label_first_token_ids = label_tokens

    @torch.no_grad()
    def forward(
        self,
        label_parser: (
            Callable[
                [
                    str,
                ],
                list[str],
            ]
            | None
        ) = None,
        **kwargs,
    ) -> dict[str, torch.Tensor | list[str]]:
        """Generates predictions from the model.

        Args:
            label_parser: function to parser response into a list of strings.
            **kwargs: keyword arguments to pass to the model and `LMForGeneration`.

        Returns:
            A dictionary from `LMForGeneration.forward` with an additional
            "preds" key containing the predictions as a list of strings.
        """

        out = super().forward(**kwargs)

        if "text" in out:  # almost means tokenizer
            labels = list(self.labels)  # list or keys to list

            # good if label_parser is not provided or if it fails
            out["preds"] = [
                [label for label in labels if label in o.lower()]
                for o in out["text"]
            ]

            if label_parser is not None:
                try:
                    preds = [label_parser(o) for o in out["text"]]
                    preds = [
                        [pred.lower() for pred in example_preds]
                        for example_preds in preds
                    ]
                    preds = [
                        [
                            (
                                pred
                                if pred in labels
                                else self.get_closest_label(pred, labels)
                            )
                            for pred in example_preds
                        ]
                        for example_preds in preds
                    ]
                    out["preds"] = [
                        list(
                            # in case some preds are the same
                            # because of similarity matching
                            set(
                                [
                                    pred
                                    for pred in example_pred
                                    if pred is not None
                                ]
                            )
                        )
                        for example_pred in preds
                    ]
                except:
                    pass

        else:
            assert isinstance(
                self.labels, dict
            ), "Labels must be a dict if no tokenizer is provided"
            out["preds"] = [
                [
                    label
                    for label, label_ids in self.labels.items()
                    if tensor_overlap(o, label_ids) < len(o)
                ]
                for o in out["ids"]
            ]

        if self.label_first_token_ids is not None:

            # find the index of the first token that is a label
            all_first_label_inds = [
                [
                    tensor_overlap(torch.tensor(o), torch.tensor([i]))
                    for i in self.label_first_token_ids.values()
                ]
                for o in out["ids"]
            ]
            first_label_inds = [min(i) for i in all_first_label_inds]
            all_label_inds = [
                sorted([i for i in inds if i < len(o)])
                for o, inds in zip(out["ids"], all_first_label_inds)
            ]

            out["all_scores"] = [
                [
                    (
                        o[i, list(self.label_first_token_ids.values())]
                        if i < len(o)
                        else torch.zeros(len(self.labels))
                    )
                    for i in inds
                ]
                # use scores before it is indexed
                for o, inds in zip(out["scores"], all_label_inds)
            ]

            # get the scores for all labels from the first token that is a label
            out["scores"] = [
                (
                    o[i, list(self.label_first_token_ids.values())]
                    if i < len(o)
                    else torch.zeros(len(self.labels))
                )
                for o, i in zip(out["scores"], first_label_inds)
            ]

            out["scores"] = [
                {
                    k: v.item()
                    for k, v in zip(self.label_first_token_ids, o.softmax(0))
                }
                for o in out["scores"]
            ]
            out["all_scores"] = [
                [
                    {
                        k: v.item()
                        for k, v in zip(
                            self.label_first_token_ids, o.softmax(0)
                        )
                    }
                    for o in all_scores
                ]
                for all_scores in out["all_scores"]
            ]

            ### HANDLE ATTN

            try:
                # attention dim:
                # tuple(generated_tokens) x tuple(layers) x bs x attention heads x (input_len + gen tokens up to then)

                # list(bs) x list(label_tokens) x tuple(layers) x attention heads x (input_len + gen tokens up to then)
                label_token_attentions = [
                    [
                        [layer_attn[j] for layer_attn in out["attentions"][i]]
                        for i in inds[
                            1:
                        ]  # only the second+ label token can attend to the previous ones
                    ]
                    for j, inds in enumerate(all_label_inds)
                ]
                input_lens = kwargs["input_ids"].shape[-1] if self.causal else 0

                input_attns = []
                prev_label_attns = []
                only_label_token_attns = []
                for attn, inds in zip(label_token_attentions, all_label_inds):
                    # attn: tuple(label_tokens) x tuple(layers) x attention heads x (input_len + gen tokens up to then)
                    for label_attn in attn:
                        # label_attn: tuple(layers) x attention heads x (input_len + gen tokens up to then)
                        input_attns.append(
                            [
                                torch.stack(
                                    [
                                        layer_attn[:, :input_lens]
                                        .sum(-1)
                                        .mean(-1)
                                        for layer_attn in label_attn
                                    ]
                                )
                                .mean()
                                .item(),
                                input_lens,
                            ]
                        )

                        prev_label_attns.append(
                            [
                                torch.stack(
                                    [
                                        layer_attn[:, input_lens:]
                                        .sum(-1)
                                        .mean(-1)
                                        for layer_attn in label_attn
                                    ]
                                )
                                .mean()
                                .item(),
                                label_attn[0].shape[-1] - input_lens,
                            ]
                        )

                        only_labels = []
                        positions = [
                            input_lens + k - 1
                            for k in inds
                            if input_lens + k - 1 < label_attn[0].shape[-1]
                        ]
                        for layer_attn in label_attn:
                            if positions:
                                only_labels.append(
                                    layer_attn[:, positions].sum(-1).mean(-1)
                                )

                        if only_labels:
                            only_label_token_attns.append(
                                [
                                    torch.stack(only_labels).mean().item(),
                                    len(positions),
                                ]
                            )
                        else:
                            only_label_token_attns.append([None, None])

                out["label_attn"] = {
                    "input": input_attns,
                    "prev_label": prev_label_attns,
                    "only_label_token": only_label_token_attns,
                }
                del out["attentions"]
            except:
                # if the model does not have attention weights
                out["label_attn"] = {
                    "input": [(None, None) for _ in range(len(all_label_inds))],
                    "prev_label": [
                        (None, None) for _ in range(len(all_label_inds))
                    ],
                    "only_label_token": [
                        (None, None) for _ in range(len(all_label_inds))
                    ],
                }
                del out["attentions"]

            ### END HANDLE ATTN

            if out["last_hidden_state"] is not None:
                # sanity check
                assert [
                    len(o) == len(hs)
                    for o, hs in zip(out["ids"], out["last_hidden_state"])
                ], (
                    "Last hidden state and ids must have the same length"
                    f" but got {len(out['ids'])} and {len(out['last_hidden_state'])}"
                )
                out["last_hidden_state"] = [
                    o[i] if i < len(o) else torch.zeros_like(o)
                    for o, i in zip(out["last_hidden_state"], first_label_inds)
                ]

        return out


# inherit from nn.Module because of the way the trainer works
# e.g. it calls model.to(device) and model.train()
class OpenAIModel(nn.Module):
    """Wrapper for OpenAI API.

    Attributes:
        model_name: OpenAI model name to use.
        max_tokens: maximum number of tokens to generate.
        temperature: temperature to use for sampling.
        mode: mode to use for generation, can be "chat" or None.
        completion_tokens: number of tokens generated.
        prompt_tokens: number of prompt tokens.
    """

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        return dict(
            model_name=dict(
                type=str,
                required=True,
                help="OpenAI model to use",
                metadata=dict(name=True, name_priority=2),
                searchable=True,
            ),
            max_new_tokens=dict(
                type=int,
                default=128,
                help="maximum number of tokens to generate",
            ),
            temperature=dict(
                type=float,
                default=0.0,
                help="temperature to use for sampling",
                searchable=True,
            ),
            mode=dict(
                type=str,
                default="chat",
                help="mode to use for generation",
            ),
        )

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int,
        temperature: float = 0.0,
        mode: Literal["chat"] | None = "chat",
    ):
        """Init.

        Args:
            model_name: OpenAI model name to use.
            max_new_tokens: maximum number of tokens to generate.
            temperature: temperature to use for sampling.
            mode: mode to use for generation, can be "chat" or None.
        """

        super().__init__()

        load_dotenv()

        self.model_name = model_name
        self.max_tokens = max_new_tokens
        self.temperature = temperature
        self.mode = mode
        self.completion_tokens = 0
        self.prompt_tokens = 0

        self.client = OpenAI()

    @retry(
        wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6)
    )
    def completion_with_backoff(self, **kwargs):
        """Retries completion with exponential backoff."""
        kwargs |= dict(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        if self.mode == "chat":
            # input has to be messages=[dict(role="...", content="..."), ...]
            # use models like gpt-3.5-turbo
            resp = self.client.chat.completions.create(**kwargs)
            self.completion_tokens += resp.usage.completion_tokens
            self.prompt_tokens += resp.usage.prompt_tokens
            return resp

        # input has to be prompt="..."
        # use models like gtp-3.5-turbo-instruct
        resp = self.client.completions.create(**kwargs)
        self.completion_tokens += resp.usage.completion_tokens
        self.prompt_tokens += resp.usage.prompt_tokens
        return resp

    def __call__(
        self, user_prompt: str, system_prompt: str | None = None, **kwargs
    ):
        """Generates completion for prompt.

        Args:
            prompt: prompt to generate completion for.
            role: role to use for chat mode.

        Returns:
            Generated completion.

        Raises:
            AssertionError: if role is None in chat mode.
        """
        if self.mode == "chat":
            messages = [
                dict(role="user", content=user_prompt),
            ]
            if system_prompt is not None:
                messages.append(dict(role="system", content=system_prompt))
            resp = self.completion_with_backoff(messages=messages)
            return resp.choices[0].message.content

        assert (
            system_prompt is None
        ), "system_prompt must be None in NON-chat mode"
        resp = self.completion_with_backoff(prompt=user_prompt)
        return resp.choices[0].text


class OpenAIClassifier(LabelSimilarityMixin, OpenAIModel):
    """Wrapper for OpenAI API for classification."""

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        args = (
            OpenAIModel.argparse_args() | LabelSimilarityMixin.argparse_args()
        )
        args.pop("temperature")
        return args

    @from_namespace
    def __init__(self, labels: list[str], **kwargs):
        """Init.

        Args:
            labels: labels to use for predictions.
            **kwargs: keyword arguments to pass to OpenAIModel.
        """
        self.labels = labels
        kwargs["temperature"] = 0
        super().__init__(**kwargs)

    def __call__(
        self,
        label_parser: (
            Callable[
                [
                    str,
                ],
                list[str],
            ]
            | None
        ) = None,
        prefix_cutoff_str: str | None = None,
        **kwargs,
    ):
        """Generates completion for prompt.

        Args:
            **kwargs: keyword arguments to pass to OpenAIModel.

        Returns:
            Generated completion and predictions based on labels.
        """
        out = {}
        # make 2d for compatibility with other models
        out["text"] = [super().__call__(**kwargs)]

        if prefix_cutoff_str is not None:
            prefix_cutoff_inds = []
            for o in out["text"]:
                i = o.find(prefix_cutoff_str)
                if i == -1:
                    prefix_cutoff_inds.append(len(o))
                else:
                    prefix_cutoff_inds.append(i)

            out["prefix_text"] = [
                o[:i] for o, i in zip(out["text"], prefix_cutoff_inds)
            ]
            out["text"] = [
                o[i:].strip() for o, i in zip(out["text"], prefix_cutoff_inds)
            ]

        # good if label_parser is not provided or if it fails
        out["preds"] = [
            [label for label in self.labels if label in o.lower()]
            for o in out["text"]
        ]
        if label_parser is not None:
            try:
                preds = [label_parser(o) for o in out["text"]]
                preds = [
                    [pred.lower() for pred in example_preds]
                    for example_preds in preds
                ]
                preds = [
                    [
                        (
                            pred
                            if pred in self.labels
                            else self.get_closest_label(pred, self.labels)
                        )
                        for pred in example_preds
                    ]
                    for example_preds in preds
                ]
                out["preds"] = [
                    list(
                        # in case some preds are the same
                        # because of similarity matching
                        set([pred for pred in example_pred if pred is not None])
                    )
                    for example_pred in preds
                ]
            except:
                pass

        return out


class LMForFinetuning:
    """Model to handle InstructionTuning of LLMs using 4bit QLoRa"""

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        return dict(
            model_name_or_path=dict(
                type=str,
                required=True,
                help="model name or path to load tokenizer and model from",
            ),
            lora_r=dict(
                type=int,
                default=64,
                help="Lora R dimension",
            ),
            lora_alpha=dict(
                type=int,
                default=16,
                help="Lora Alpha",
            ),
            lora_dropout=dict(
                type=float,
                default=0.1,
                help="Lora Dropout",
            ),
            per_device_train_batch_size=dict(
                type=int,
                default=8,
                help="Training batch size per device",
            ),
            learning_rate=dict(
                type=float,
                default=2e-4,
                help="Learning rate for the optimizer",
            ),
            num_train_epochs=dict(
                type=int,
                default=3,
                help="Number of training epochs",
            ),
            output_dir=dict(
                type=str,
                help="Directory to save the model output",
            ),
            log_dir=dict(
                type=str,
                help="Directory to save the training logs",
            ),
            cache_dir=dict(
                type=str,
                help="directory where models are cached, if any",
            ),
        )

    def __init__(
        self,
        model_name_or_path: str,
        output_dir: str,
        log_dir: str,
        num_train_epochs: int,
        cache_dir: str | None = None,
        lora_r: int = 64,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        per_device_train_batch_size: int = 8,
        learning_rate: float = 2e-4,
        *args,
        **kwargs,
    ):

        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        # Configs as per QLoRa paper
        self.lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.lm = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=self.bnb_config,
            # cache_dir="/project/shrikann_35/llm-shared",
            cache_dir=cache_dir,
            use_cache=True,
            device_map="auto",
            token=os.getenv('ACCESS_TOKEN'),
        )
        self.lm.config.pretraining_tp = 1

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, cache_dir=cache_dir, use_cache=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.training_args = TrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
            # per_device_eval_batch_size=4,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            output_dir=output_dir,
            logging_dir=log_dir,
            gradient_accumulation_steps=2,
            gradient_checkpointing=True,
            optim="paged_adamw_32bit",
            logging_steps=10,
            # save_strategy="epoch",
            evaluation_strategy='steps',
            eval_steps=50,
            # eval_accumulation_steps=10,
            bf16=True,
            fp16=False,
            # tf32=True,
            tf32=False,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            lr_scheduler_type="constant",
            dataloader_num_workers=8,
            save_steps=10,
            save_total_limit=2,
        )

        self.logger = logging.getLogger()
        self.logger.setLevel(
            logging.INFO
        )  # You can set this to the desired level
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_dir + '/training.log')
        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)

    def prepare_for_finetuning(self):
        self.lm = prepare_model_for_kbit_training(self.lm)
        self.lm = get_peft_model(self.lm, self.lora_config)

    def log(self, log_history):
        for entry in log_history:
            json_entry = json.dumps(entry, indent=4)
            self.logger.info(json_entry)
        print("Logs written")

    def finetune(self, train_data, val_data, format_instruction):

        self.prepare_for_finetuning()
        trainer = SFTTrainer(
            model=self.lm,
            train_dataset=train_data,
            eval_dataset=val_data,
            peft_config=self.lora_config,
            max_seq_length=256,
            tokenizer=self.tokenizer,
            packing=True,
            formatting_func=format_instruction,
            args=self.training_args,
        )
        trainer.train()
        trainer.save_model()
        self.log(trainer.state.log_history)
