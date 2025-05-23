import os
import logging
from typing import Any

import torch
from torch.utils.data import DataLoader
from ember.trainer import BaseTrainer
from sklearn.metrics import (
    f1_score,
    jaccard_score,
    accuracy_score,
    roc_auc_score,
)
from sklearn.preprocessing import MultiLabelBinarizer


class PromptEvaluator(BaseTrainer):
    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        args = BaseTrainer.argparse_args()
        args_discard = [
            "save_model",
            "discard_classifier",
            "classifier_layer_name",
            "lr",
            "adam_beta1",
            "adam_beta2",
            "adam_epsilon",
            "weight_decay",
            "eval_steps",
            "max_steps",
            "num_train_epochs",
            "train_batch_size",
            "eval_batch_size",
            "warmup_ratio",
            "early_stopping_patience",
            "early_stopping_metric",
            "early_stopping_delta",
            "early_stopping_lower_better",
        ]
        for arg in args_discard:
            args.pop(arg)
        args["linear_probing"] = dict(
            action="store_true",
            help="store hidden states for linear probing",
        )
        return args

    def __init__(self, experiment_manager, *args, **kwargs):
        setattr(experiment_manager, "eval_batch_size", 1)
        kwargs["experiment_manager"] = experiment_manager

        super().__init__(*args, **kwargs)

    def run_init(self):
        initial_label_tokens = self.test_dataset.get_initial_label_tokens()
        self.model.set_label_decoding_utils(initial_label_tokens)

    def get_logits_from_model(
        self, return_vals: Any, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:

        device = (
            "cuda"
            if self.exp_manager.device == "auto"
            else self.exp_manager.device
        )

        batch_preds: list[list[str]] = return_vals["preds"]
        if self.any_dataset.multilabel:
            preds = torch.tensor(
                [
                    [
                        int(label in preds)
                        for label in self.any_dataset.outward_label_set
                    ]
                    for preds in batch_preds
                ],
                device=device,
            )
        else:
            preds = []
            for p in batch_preds:
                try:
                    preds.append(self.any_dataset.outward_label_set.index(p[0]))
                except IndexError:
                    preds.append(0)
            preds = torch.tensor(preds, device=device)

        return preds, return_vals.get("all_scores", None)

    def get_extra_data_from_model(
        self, return_vals: dict[str, str | torch.Tensor], batch: dict[str, Any]
    ) -> dict[str, list[Any]]:
        if "text" in return_vals:
            odict = dict(
                outs=return_vals["text"],
                residual_outs=return_vals.get(
                    "residual_text", [None] * len(return_vals["text"])
                ),
                prefix_outs=return_vals.get(
                    "prefix_text", [None] * len(return_vals["text"])
                ),
            )
        else:
            odict = dict(outs=return_vals["ids"])
            residual_outs = return_vals.get("residual_ids", None)
            if residual_outs is not None:
                odict["residual_outs"] = residual_outs.tolist()
            else:
                odict["residual_outs"] = [None] * len(odict["outs"])

        for k in return_vals:
            if "attention" in k:
                # list so that is is not perceived
                # as a random list of arguments but a vector
                odict[k] = [return_vals[k].tolist()]

        odict["label_v_input_attn"] = return_vals.get("label_attn", None)

        if self.exp_manager.linear_probing:
            # list so that is is not perceived
            # as a random list of arguments but a vector
            odict["last_hidden_state"] = [return_vals["last_hidden_state"]]

        return odict

    def calculate_cls_loss(
        self,
        logits: tuple[torch.Tensor],
        labels: torch.Tensor,
        train: bool,
        aggregate: bool = True,
        epoch: int = 0,
    ) -> torch.Tensor:
        device = (
            "cuda"
            if self.exp_manager.device == "auto"
            else self.exp_manager.device
        )
        if aggregate:
            return torch.tensor(0.0, device=device)
        return torch.zeros(len(labels), device=device)

    def input_batch_args(self, batch: dict[str, Any]) -> dict[str, Any]:
        encoding = {k: v[0] for k, v in batch["encoding"].items()}

        if "{cot}" in self.any_dataset.incontext_prompt:
            sidx = self.any_dataset.incontext_prompt.index("{cot}") + len(
                "{cot}"
            )
            eidx = self.any_dataset.incontext_prompt.index("{label}")
            prefix_cutoff_str = self.any_dataset.incontext_prompt[
                sidx:eidx
            ].strip()
        else:
            prefix_cutoff_str = None

        cutoff_str = self.any_dataset.incontext_prompt.split("{text}")[0]
        cutoff_str = [cutoff_str.strip(), cutoff_str]

        return encoding | dict(
            # assumes that the text appears first, which is reasonable for causal LMs
            cutoff_str=cutoff_str,
            prefix_cutoff_str=prefix_cutoff_str,
            label_parser=self.any_dataset.label_parser,
        )

    def batch_labels(self, batch: dict[str, Any]):
        if "demo_label" in batch and batch["demo_label"] is not None:
            return batch["demo_label"]
        return batch["label"]

    def batch_ids(self, batch: dict[str, Any]):
        return batch["id"]

    def get_eval_preds_from_batch(
        self, logits: tuple[torch.Tensor, torch.Tensor | None]
    ) -> list[list[int]]:
        return logits[0].tolist()

    def get_eval_scores_from_batch(
        self, logits: tuple[torch.Tensor, torch.Tensor | None]
    ) -> list[dict[str, float]] | None:
        if logits[1] is not None:
            return logits[1]
        return None

    def run_end(self):
        self.exp_manager.log_metrics()
        # self._save_best_model()
        self.exp_manager.aggregate_results()
        self.exp_manager.plot(
            groups=(
                [
                    [f"{clss}_f1" for clss in self.dev_dataset.label_set],
                    [f"{clss}_auc" for clss in self.dev_dataset.label_set],
                ]
                if self.do_eval
                else None
            )
        )

    def evaluation_metrics(
        self,
        eval_outs: dict[str, Any],
        eval_outs_id: dict[str, Any],
        eval_extras: dict[str, Any],
        data_loader: DataLoader | None = None,
    ) -> dict[str, Any]:
        sep = self.any_dataset.any_dataset.id_separator

        annotator_info = {
            _id.split(sep)[1]: {
                "true": [],
                "pred": [],
            }
            for _id in eval_outs_id["ids"]
        }
        for _id, true, pred in zip(
            eval_outs_id["ids"],
            eval_outs_id["gt"] or [None] * len(eval_outs_id["ids"]),
            eval_outs_id["preds"] or [None] * len(eval_outs_id["ids"]),
        ):
            annotator_id = _id.split(sep)[1]
            annotator_info[annotator_id]["true"].append(true)
            annotator_info[annotator_id]["pred"].append(pred)

        results = {}

        for annotator_id, info in annotator_info.items():
            if self.test_dataset.multilabel:
                # zero division should 1 for all matching and constant labels and predictions
                # we set it to 0 because that is unlikely to happen
                macro_f1 = f1_score(
                    info["true"], info["pred"], average="macro", zero_division=0
                )
                micro_f1 = f1_score(
                    info["true"], info["pred"], average="micro", zero_division=0
                )

                js = jaccard_score(
                    info["true"],
                    info["pred"],
                    average="samples",
                    zero_division=1,
                )

                f1_scores = f1_score(
                    info["true"], info["pred"], average=None, zero_division=0
                )

                results[annotator_id] = {
                    "jaccard_score": js,
                    "micro_f1": micro_f1,
                    "macro_f1": macro_f1,
                } | {
                    f"{clss}_f1": f1
                    for clss, f1 in zip(
                        data_loader.dataset.label_set, f1_scores
                    )
                }

            else:
                acc = accuracy_score(info["true"], info["pred"])
                macro_f1 = f1_score(
                    info["true"], info["pred"], average="macro", zero_division=0
                )
                results[annotator_id] = {
                    "accuracy": acc,
                    "macro_f1": macro_f1,
                }
                try:
                    macro_auc = roc_auc_score(
                        info["true"], info["pred"], average="macro"
                    )
                    micro_auc = roc_auc_score(
                        info["true"], info["pred"], average="micro"
                    )
                    results[annotator_id] |= {
                        "macro_auc": macro_auc,
                        "micro_auc": micro_auc,
                    }
                except:
                    results[annotator_id] |= {
                        "macro_auc": 0.0,
                        "micro_auc": 0.0,
                    }

        return results

    def index_label_set(
        self, labels: torch.Tensor | int | list[int]
    ) -> str | list[str]:
        return self.any_dataset.any_dataset.index_label_set(labels)

    def evaluate(self, *args, **kwargs):
        annotator_results, example_info = super().evaluate(*args, **kwargs)

        # annotator_results:
        #   per annotator results, IDs are annotator IDs or "aggregate"
        #   + some aggregate metrics
        # example_info:
        #   per example info IDs are going to be example_id - annotator_id
        #   the actual example results have annotator_id == "aggregate"

        aggregate_results = annotator_results.pop("aggregate", {})

        # remove dummy losses
        for k in list(annotator_results):
            if not isinstance(annotator_results[k], dict):
                annotator_results.pop(k)

        sep = self.any_dataset.any_dataset.id_separator

        aggregate_info = {}
        annotator_info = {}
        attentions = {}
        embeddings = {}
        for k, v in example_info.items():
            example_id, annotator_id = k.split(sep)

            # add text for easier debugging
            example = self.any_dataset.test_dataset.getitem_by_id(example_id)
            v["text"] = example["text"]
            for kk in list(v):
                if kk in ("preds", "gt", "pred"):
                    v[kk] = self.index_label_set(v[kk])
                if kk == "checked_label":
                    v[kk] = self.any_dataset.any_dataset.index_label_set(v[kk])
                elif "attention" in kk:
                    attentions.setdefault(annotator_id, {}).setdefault(
                        example_id, {}
                    )[kk] = v.pop(kk)
                elif kk == "last_hidden_state":
                    embeddings.setdefault(annotator_id, {})[example_id] = v.pop(
                        kk
                    )[0]

            if annotator_id == "aggregate":
                aggregate_info[example_id] = v
            else:
                annotator_info.setdefault(annotator_id, {})[example_id] = v

        # we have to make numpy scalars into floats manually
        # because we are using manual logging
        annotator_results = {
            k: {kk: float(vv) for kk, vv in v.items()}
            for k, v in annotator_results.items()
        }

        self.exp_manager.set_custom_data(
            annotator_results, "annotator_metrics.yml"
        )
        self.exp_manager.set_custom_data(annotator_info, "annotator_preds.yml")
        if attentions:
            self.exp_manager.set_custom_data(attentions, "attentions.pt")
        if embeddings and self.exp_manager.linear_probing:
            self.exp_manager.set_custom_data(embeddings, "last_hidden_state.pt")

        return aggregate_results, aggregate_info


class vPromptEvaluator(PromptEvaluator):
    def input_batch_args(self, batch):
        inp = super().input_batch_args(batch)
        return dict(
            text=batch["text"][0],
            cutoff_str=inp["cutoff_str"][0],
            prefix_cutoff_str=inp["prefix_cutoff_str"],
            label_parser=inp["label_parser"],
        )


class APIPromptEvaluator(PromptEvaluator):

    def run_init(self): ...

    def input_batch_args(self, batch: dict[str, Any]) -> dict[str, Any]:
        if "{cot}" in self.any_dataset.incontext_prompt:
            sidx = self.any_dataset.incontext_prompt.index("{cot}") + len(
                "{cot}"
            )
            eidx = self.any_dataset.incontext_prompt.index("{label}")
            prefix_cutoff_str = self.any_dataset.incontext_prompt[
                sidx:eidx
            ].strip()
        else:
            prefix_cutoff_str = None

        return dict(
            user_prompt=batch["text"][0],
            system_prompt=self.test_dataset.system_prompt,
            prefix_cutoff_str=prefix_cutoff_str,
        )

    def run_end(self):
        self.exp_manager.set_custom_data(
            dict(
                completion_tokens=self.model.completion_tokens,
                prompt_tokens=self.model.prompt_tokens,
            ),
            "tokens.yml",
        )
        return super().run_end()


class ReasonablenessEvaluator(PromptEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluation_metrics(
        self,
        eval_outs: dict[str, Any],
        eval_outs_id: dict[str, Any],
        eval_extras: dict[str, Any],
        data_loader: DataLoader | None = None,
    ) -> dict[str, Any]:
        sep = self.any_dataset.any_dataset.id_separator

        annotator_info = {
            _id.split(sep)[1]: {
                "true": [],
                "pred": [],
            }
            for _id in eval_outs_id["ids"]
        }
        for _id, true, pred in zip(
            eval_outs_id["ids"],
            eval_outs_id["gt"] or [None] * len(eval_outs_id["ids"]),
            eval_outs_id["preds"] or [None] * len(eval_outs_id["ids"]),
        ):
            annotator_id = _id.split(sep)[1]
            annotator_info[annotator_id]["true"].append(true)
            annotator_info[annotator_id]["pred"].append(pred)

        results = {}

        for annotator_id, info in annotator_info.items():
            acc = accuracy_score(info["true"], info["pred"])
            f1 = f1_score(info["true"], info["pred"], zero_division=0)
            results[annotator_id] = {"accuracy": acc, "f1": f1}
            try:
                auc = roc_auc_score(info["true"], info["pred"])
                results[annotator_id] |= {"auc": auc}
            except:
                results[annotator_id] |= {"auc": 0.0}

        return results

    # def get_eval_preds_from_batch(
    #     self, logits: tuple[torch.Tensor, torch.Tensor | None]
    # ) -> list[list[int]]:
    #     try:
    #         return [l.index(1) for l in logits[0].tolist()]
    #     except ValueError:
    #         return [0]

    def index_label_set(self, labels: int):
        return self.any_dataset.label_set[labels]

    def get_extra_data_from_model(self, return_vals, batch):
        odict = super().get_extra_data_from_model(return_vals, batch)
        if "checked_label" in batch:
            odict["checked_label"] = batch["checked_label"]
        return odict


class vReasonablenessEvaluator(ReasonablenessEvaluator):
    def input_batch_args(self, batch):
        inp = super().input_batch_args(batch)
        return dict(
            text=batch["text"][0],
            cutoff_str=inp["cutoff_str"][0],
            prefix_cutoff_str=inp["prefix_cutoff_str"],
            label_parser=inp["label_parser"],
        )


class APIReasonablenessEvaluator(APIPromptEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluation_metrics(
        self,
        eval_outs: dict[str, Any],
        eval_outs_id: dict[str, Any],
        eval_extras: dict[str, Any],
        data_loader: DataLoader | None = None,
    ) -> dict[str, Any]:
        sep = self.any_dataset.any_dataset.id_separator

        annotator_info = {
            _id.split(sep)[1]: {
                "true": [],
                "pred": [],
            }
            for _id in eval_outs_id["ids"]
        }
        for _id, true, pred in zip(
            eval_outs_id["ids"],
            eval_outs_id["gt"] or [None] * len(eval_outs_id["ids"]),
            eval_outs_id["preds"] or [None] * len(eval_outs_id["ids"]),
        ):
            annotator_id = _id.split(sep)[1]
            annotator_info[annotator_id]["true"].append(true)
            annotator_info[annotator_id]["pred"].append(pred)

        results = {}

        for annotator_id, info in annotator_info.items():
            acc = accuracy_score(info["true"], info["pred"])
            f1 = f1_score(info["true"], info["pred"], zero_division=0)

            results[annotator_id] = {"accuracy": acc, "f1": f1}

        return results

    def index_label_set(self, labels: int):
        return self.any_dataset.label_set[labels]

    # def get_eval_preds_from_batch(
    #     self, logits: tuple[torch.Tensor, torch.Tensor | None]
    # ) -> list[list[int]]:
    #     try:
    #         return [l.index(1) for l in logits[0].tolist()]
    #     except ValueError:
    #         # return no if not proper response
    #         return [0]

    def get_extra_data_from_model(self, return_vals, batch):
        odict = super().get_extra_data_from_model(return_vals, batch)
        if "checked_label" in batch:
            odict["checked_label"] = batch["checked_label"].tolist()
        return odict


class FinetuneEvaluator:
    def __init__(self, model, test_dataset, labels, log_dir):
        self.model = model
        self.tokenizer = model.tokenizer
        self.labels = labels
        self.dataset = test_dataset
        self.log_dir = log_dir
        self.generated = []
        self.preds = []
        self.og_labels = []
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_dir + '/training.log')
        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)

    def evaluate(self):
        ds = self.dataset.get_dataframe()
        print(f'{len(ds)} : Examples')
        for i, example in enumerate(ds):
            input_ids = self.tokenizer(example['input'], return_tensors='pt')[
                "input_ids"
            ].to('cuda')
            output = self.model.forward(input_ids=input_ids)
            self.generated.append(output['text'])
            self.preds.append(output['preds'][0])
            self.og_labels.append(
                self.dataset.any_dataset.index_label_set(example['label'])
            )

        return self._calculate_metrics()

    def _calculate_metrics(self):
        mlb = MultiLabelBinarizer(classes=self.labels)
        labels = mlb.fit_transform(self.og_labels)
        predictions = mlb.transform(self.preds)

        jaccard = jaccard_score(
            labels, predictions, average="samples", zero_division=1
        )
        micro_f1 = f1_score(
            labels, predictions, average='micro', zero_division=0
        )
        macro_f1 = f1_score(
            labels, predictions, average='macro', zero_division=0
        )

        self.logger.info(f'Evaluation Complete:')
        self.logger.info(f'Jaccard Score:{jaccard}')
        self.logger.info(f'Micro F1 Score:{micro_f1}')
        self.logger.info(f'Macro F1 Score:{macro_f1}')
        return {'jaccard': jaccard, 'micro_f1': micro_f1, 'macro_f1': macro_f1}
