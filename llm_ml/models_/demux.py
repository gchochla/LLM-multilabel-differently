import torch
import torch.nn as nn
import os
import yaml
from transformers import AutoModel, PretrainedConfig, AutoConfig
from ember.trainer import BaseTrainer
from sklearn.metrics import jaccard_score, f1_score, accuracy_score
from legm import from_namespace

from llm_ml.base_datasets import TextDatasetWithPriors, TokenizationMixin
from llm_ml.benchmarks import (
    SemEval2018Task1Ec,
    GoEmotions,
    MFRC,
    Hatexplain,
    MSPPodcast,
)


class Demux(nn.Module):
    """Demux.

    Attributes:
        bert: large LM with a BertModel-like 'interface'.
        classifier: shared FFN on top of contextual representations.
        class_inds: indices of each emotion/label in the tokenized
            input, expected to be constant in training and testing
            (because prompt goes first).
    """

    @staticmethod
    def argparse_args():
        return dict(
            model_name_or_path=dict(
                type=str,
                help="model to load into BERT parts of model",
            ),
            dropout_prob=dict(
                default=0.1,
                type=float,
                help="dropout before final linear layer",
            ),
        )

    def __init__(
        self,
        config: PretrainedConfig,
        class_inds: list[torch.Tensor],
        dropout_prob: float = 0.1,
    ):
        """Init.

        Args:
            config: LM configuration from `AutoConfig`.
            class_inds: indices (`torch.long`) of each emotion/label
                in the tokenized input.
            dropout_prob: dropout before final linear layer.
        """
        super().__init__()

        self.class_inds = class_inds

        config.hidden_dropout_prob = dropout_prob
        config.attention_probs_dropout_prob = dropout_prob

        try:
            self.bert = AutoModel.from_config(config, add_pooling_layer=False)
        except TypeError:
            self.bert = AutoModel.from_config(config)

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout_prob),
            nn.Linear(config.hidden_size, 1),
        )

    @classmethod
    def from_pretrained(cls, pretrained_lm: str, *args, **kwargs) -> "Demux":
        config = AutoConfig.from_pretrained(pretrained_lm)
        model = cls(config, *args, **kwargs)
        try:
            model.bert.load_state_dict(
                AutoModel.from_pretrained(
                    pretrained_lm, add_pooling_layer=False
                ).state_dict()
            )
        except TypeError:
            model.bert.load_state_dict(
                AutoModel.from_pretrained(pretrained_lm).state_dict()
            )
        return model

    def forward(self, class_inds=None, *args, **kwargs) -> torch.Tensor:
        """Forward propagation.

        Args:
            `transformers`-style LM inputs.
            class_inds: different `class_inds` can be specified
                from the one at initialization only for this
                forward pass.

        Returns:
            Logits whose number is equal to `len(self.class_inds)`,
            contextual representation used for each is average
            of outputs specified by each list.
        """

        if class_inds is None:
            class_inds = self.class_inds

        last_hidden_state = self.bert(*args, **kwargs).last_hidden_state

        # if asked not to aggregate at logit level,
        # or class_inds is List of Tensors (i.e. single emotion setting)
        if isinstance(class_inds, torch.Tensor):
            last_emotion_state = torch.stack(
                [
                    last_hidden_state.index_select(
                        dim=1,
                        index=(
                            torch.cat(inds) if isinstance(inds, list) else inds
                        ).to(last_hidden_state.device),
                    ).mean(1)
                    for inds in class_inds
                ],
                dim=1,
            )

            preds = self.classifier(last_emotion_state).squeeze(-1)
            return preds

        last_emotion_state = [
            torch.stack(
                [
                    last_hidden_state.index_select(
                        dim=1, index=inds.to(last_hidden_state.device)
                    ).mean(1)
                    for inds in emo_inds
                ],
                dim=1,
            )
            for emo_inds in class_inds
        ]
        preds = torch.stack(
            [
                self.classifier(cluster_stack).max(1)[0]
                for cluster_stack in last_emotion_state
            ],
            dim=1,
        ).squeeze(-1)

        return preds


class DemuxDatasetforTransformers(TokenizationMixin, TextDatasetWithPriors):
    @staticmethod
    def argparse_args():
        return (
            TokenizationMixin.argparse_args()
            | TextDatasetWithPriors.argparse_args()
        )

    def __init__(self, *args, **kwargs):
        self.preds_are_reasonableness = False
        super().__init__(*args, **kwargs, for_llm=False)
        self.prompt = ", ".join(self.label_set)
        for k in self.examples:
            self.examples[k]["text"] = (self.prompt, self.examples[k]["text"])
        self.class_inds = self.get_class_inds()

    def _load_labels_from_logs(
        self, experiment_dir: str, pred_log_index: int | str | None = None
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Loads annotator labels from logs in `experiment_dir` and returns
        a dictionary indexed by IDs that contains another dictionary with
        annotator IDs as keys and their labels as values, e.g.,
        {
            id1: {
                "ann1": torch.tensor([4]),
                "ann2": torch.tensor([3, 4]),
            },
            ...
        }
        """

        labels = {}

        label_fn = os.path.join(experiment_dir, "indexed_metrics.yml")
        with open(label_fn) as fp:
            logs = yaml.safe_load(fp)[f"experiment_{pred_log_index or 0}"]

        if "description" in logs:
            del logs["description"]

        for example_id, example_metrics in logs.items():
            labels.setdefault("aggregate", {})[example_id] = (
                self.get_label_from_str(
                    example_metrics["test_preds"],
                    reasonableness=bool(
                        example_metrics["test_preds"].lower()
                        in ("reasonable", "yes", "unreasonable", "no")
                    ),
                )
            )

        ann_label_fn = os.path.join(experiment_dir, "annotator_preds.yml")
        with open(ann_label_fn) as f:
            ann_logs = yaml.safe_load(f)["experiment_0"]

        for worker_id, annotator_metrics in ann_logs.items():
            if worker_id == "description":
                continue
            for example_id, example_metrics in annotator_metrics.items():
                labels.setdefault(worker_id, {})[example_id] = (
                    self.get_label_from_str(
                        example_metrics["test_preds"],
                        reasonableness=bool(
                            example_metrics["test_preds"].lower()
                            in ("reasonable", "yes", "unreasonable", "no")
                        ),
                    )
                )

        return labels

    def get_label_from_str(self, label, reasonableness=False):
        if not reasonableness:
            return super().get_label_from_str(label)

        self.preds_are_reasonableness = True
        return torch.tensor(float(label.lower() in ("reasonable", "yes")))

    def get_class_inds(self):
        tokenizer = self.get_tokenizer()
        class_ids = [
            tokenizer.convert_tokens_to_ids(tokenizer.tokenize(l))
            for l in self.label_set
        ]
        prompt_ids = tokenizer(self.prompt)["input_ids"]
        class_inds = []
        for ids in class_ids:
            inds = []
            for _id in ids:
                id_idx = prompt_ids.index(_id)
                prompt_ids[id_idx] = None
                inds.append(id_idx)

            class_inds.append(torch.tensor(inds, dtype=torch.long))

        return class_inds

    def collate_fn(self, batch):
        """Collate function for `transformers`."""
        batch = {k: [ex[k] for ex in batch] for k in batch[0].keys()}
        batch["encoding"] = self.batch_tokenize(batch["text"])
        batch["label"] = torch.stack(batch["label"])
        if "pred_label" in batch:
            batch["pred_label"] = torch.stack(batch["pred_label"])
        return batch


class DemuxSemEval2018Task1EcDataset(
    DemuxDatasetforTransformers, SemEval2018Task1Ec
):

    @staticmethod
    def argparse_args():
        return (
            DemuxDatasetforTransformers.argparse_args()
            | SemEval2018Task1Ec.argparse_args()
        )

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DemuxGoEmotionsDataset(DemuxDatasetforTransformers, GoEmotions):

    @staticmethod
    def argparse_args():
        return (
            DemuxDatasetforTransformers.argparse_args()
            | GoEmotions.argparse_args()
        )

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DemuxMFRCDataset(DemuxDatasetforTransformers, MFRC):

    @staticmethod
    def argparse_args():
        return (
            DemuxDatasetforTransformers.argparse_args() | MFRC.argparse_args()
        )

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DemuxHatexplainDataset(DemuxDatasetforTransformers, Hatexplain):
    @staticmethod
    def argparse_args():
        args = (
            DemuxDatasetforTransformers.argparse_args()
            | Hatexplain.argparse_args()
        )
        del args["root_dir"]
        return args

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DemuxMSPPodcastDataset(DemuxDatasetforTransformers, MSPPodcast):
    @staticmethod
    def argparse_args():
        return (
            DemuxDatasetforTransformers.argparse_args()
            | MSPPodcast.argparse_args()
        )

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


DATASETS = {
    "SemEval": DemuxSemEval2018Task1EcDataset,
    "GoEmotions": DemuxGoEmotionsDataset,
    "MFRC": DemuxMFRCDataset,
    "Hatexplain": DemuxHatexplainDataset,
    "MSPPodcast": DemuxMSPPodcastDataset,
}


class DemuxTrainer(BaseTrainer):

    @staticmethod
    def argparse_args():
        return BaseTrainer.argparse_args() | dict(
            intra_loss_coef=dict(
                default=0.0,
                type=float,
                help="coefficient for intra regularization loss",
                searchable=True,
            ),
            filter_predictions=dict(
                action="store_true",
                help="whether to filter predictions when `pred_label` is different "
                "from `label`",
            ),
        )

    def input_batch_args(self, batch):
        return batch["encoding"]

    def batch_labels(self, batch):

        # for reasonableness
        if self.any_dataset.preds_are_reasonableness:
            if not self.exp_manager.filter_predictions:
                return batch["label"]

            if "pred_label" in batch:
                batch["label"][(1 - batch["pred_label"]).bool()] = -1

            return batch["label"]

        # for liahr
        if not self.exp_manager.filter_predictions:
            return (
                batch["label"]
                if "pred_label" not in batch
                else batch["pred_label"]
            )

        if "pred_label" in batch:
            batch["label"][batch["label"] != batch["pred_label"]] = -1

        return batch["label"]

    def batch_ids(self, batch):
        return batch["id"]

    def calculate_cls_loss(
        self, logits, labels, train, aggregate=True, epoch=None
    ):

        if self.any_dataset.multilabel:
            criterion = nn.BCEWithLogitsLoss(
                reduction="mean" if aggregate else "none"
            )
        else:
            criterion = nn.CrossEntropyLoss(
                reduction="mean" if aggregate else "none"
            )
        logits = logits[labels != -1]
        labels = labels[labels != -1]
        if logits.shape[0] == 0:
            if not aggregate:
                return torch.tensor(0.0, device=logits.device).expand(
                    logits.shape[0]
                )
            return torch.tensor(0.0, device=logits.device)

        labels = (
            labels.long() if not self.any_dataset.multilabel else labels.float()
        )

        loss = criterion(logits, labels)
        if not aggregate:
            loss = loss.mean(-1)
        return loss

    def calculate_regularization_loss(
        self,
        intermediate_representations,
        logits,
        batch,
        train,
        aggregate=True,
        epoch=None,
    ):
        def _intra_correlation(
            vals: torch.Tensor,
            trues: torch.Tensor,
        ) -> torch.Tensor:
            """Calculates local correlation loss loss but for same group predictions for one example."""
            intra_exp_diff = (
                lambda x: (x + x.unsqueeze(-1)).triu(diagonal=1).exp().mean()
            )
            example_loss = torch.tensor(0.0, device=vals.device)

            absent_inds = trues < 0.5
            present_inds = trues >= 0.5

            if any(absent_inds):
                absent = vals[absent_inds]
                example_loss = example_loss + intra_exp_diff(absent)

            if any(present_inds):
                present = vals[present_inds]
                ## - for exp_diff loss
                example_loss = example_loss + intra_exp_diff(-present)

            if any(present_inds) and any(absent_inds):
                example_loss = example_loss / 2

            return example_loss

        if self.any_dataset.multilabel:

            return (
                torch.stack(
                    [
                        _intra_correlation(y_pred, y_true)
                        for y_pred, y_true in zip(
                            logits,
                            batch["label"],
                        )
                    ]
                ).mean(),
                self.exp_manager.intra_loss_coef,
            )

        return torch.tensor(0.0, device=logits.device)

    def get_eval_scores_from_batch(self, logits):
        if self.any_dataset.multilabel:
            return logits.sigmoid().cpu().tolist()
        return logits.softmax(dim=-1).cpu().tolist()

    def get_eval_labels_from_batch(self, labels):
        return labels.cpu().tolist()

    def get_eval_preds_from_batch(self, logits):
        if self.any_dataset.multilabel:
            return (logits >= 0.5).int().cpu().tolist()
        return logits.argmax(dim=-1).int().cpu().tolist()

    def evaluation_metrics(
        self, eval_outs, eval_outs_id, eval_extras, data_loader=None
    ):
        sep = self.any_dataset.id_separator

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

            if self.any_dataset.multilabel:
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
                results[annotator_id] = dict(
                    accuracy=accuracy_score(info["true"], info["pred"]),
                    f1=f1_score(
                        info["true"],
                        info["pred"],
                        zero_division=0,
                        average="macro",
                    ),
                )

        return results

    def evaluate(self, *args, **kwargs):
        eval_outs, eval_outs_id = super().evaluate(*args, **kwargs)
        eval_outs |= eval_outs.pop("aggregate", None)
        return eval_outs, eval_outs_id
