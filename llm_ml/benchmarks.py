import os
import langcodes
import json
import re
from typing import Any
from collections import Counter

import torch
import yaml
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

from llm_ml.base_datasets import TextDatasetWithPriors


class SemEval2018Task1Ec(TextDatasetWithPriors):
    """Plain text dataset for `SemEval 2018 Task 1: Affect in Tweets`
    (https://competitions.codalab.org/competitions/17751). Class doesn't
    use from_namespace decorator, so it can be used for inheritance.

    Attributes:
        Check `TextDatasetWithPriors` for attributes.
        language: language to load.
    """

    multilabel = True
    annotator_labels = False
    name = "SemEval 2018 Task 1"
    source_domain = "Twitter"

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        args = TextDatasetWithPriors.argparse_args()
        args.update(
            dict(
                language=dict(
                    type=str, default="english", help="language to load"
                )
            )
        )
        return args

    def __init__(self, language, *args, **kwargs):
        """Initializes dataset.

        Args:
            language: language to load.
            Check `TextDataset` for other arguments.
        """
        self.language = language
        super().__init__(*args, **kwargs)

    def _load_data(
        self, split: str
    ) -> tuple[list[str], list[str], torch.Tensor, list[str]]:
        split_mapping = dict(
            train="train", dev="dev", test="test-gold", smalldev="smalldev"
        )
        filename = os.path.join(
            self.root_dir,
            self.language.title(),
            "E-c",
            f"2018-E-c-{langcodes.find(self.language.lower()).language.title()}-{split_mapping[split]}.txt",
        )
        df = pd.read_csv(filename, sep="\t")

        emotions = list(df.columns[2:])
        sorted_emotions = sorted(emotions)
        emotion_inds = [emotions.index(e) for e in sorted_emotions]
        texts = df.Tweet.values.tolist()
        ids = df.ID.values.tolist()

        labels = torch.tensor(df.iloc[:, 2:].values[:, emotion_inds]).float()

        return {
            _id: dict(text=self.preprocessor(t), original_text=t, label=l)
            for _id, t, l in zip(ids, texts, labels)
        }, sorted_emotions


class GoEmotions(TextDatasetWithPriors):
    """Plain text dataset for `GoEmotions`. Class doesn't
    use from_namespace decorator, so it can be used for inheritance.

    Attributes:
        Check `TextDatasetWithPriors` for attributes.
    """

    multilabel = True
    annotator_labels = True
    name = "GoEmotions"
    source_domain = "Reddit"

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        args = TextDatasetWithPriors.argparse_args() | dict(
            emotion_clustering_json=dict(
                type=str,
                help="JSON file with clustering of emotions",
            )
        )
        return args

    def __init__(self, emotion_clustering_json, *args, **kwargs):
        """Initializes dataset.

        Args:
            emotion_clustering_json: JSON file with clustering of emotions.
            Check `TextDatasetWithPriors` for other arguments.
        """
        self.emotion_clustering_json = emotion_clustering_json
        super().__init__(*args, **kwargs)

    def _multilabel_one_hot(
        self, labels: "np.ndarray", n_classes: int = 27
    ) -> torch.Tensor:
        """GoEmotions-specific label transformer to multilable one-hot,
        neutral emotion is discarded (represented as 0s)."""

        labels = [
            list(filter(lambda x: x < n_classes, map(int, lbl.split(","))))
            for lbl in labels
        ]
        new_labels = [
            torch.nn.functional.one_hot(
                torch.tensor(lbl, dtype=int), n_classes
            ).sum(0)
            for lbl in labels
        ]
        return torch.stack(new_labels)

    def _subset_emotions(
        self,
        annotations: dict[Any, dict[str, str | torch.Tensor]],
        emotions: list[str],
    ) -> list[str]:
        """Transforms emotions to a subset of emotions based on clustering
        in `emotion_clustering_json`. Each new label is union of old labels."""

        if not self.emotion_clustering_json:
            return emotions

        with open(self.emotion_clustering_json) as fp:
            clustering = json.load(fp)

        new_emotions = list(clustering)

        for annotation in annotations.values():
            for worker_id, label in annotation["label"].items():
                new_label = torch.zeros(len(new_emotions))

                for i, emotion in enumerate(new_emotions):
                    for old_emotion in clustering[emotion]:
                        new_label[i] += label[emotions.index(old_emotion)]

                annotation["label"][worker_id] = new_label.clamp(0, 1)

        return new_emotions

    def _load_data(
        self, split: str
    ) -> tuple[dict[Any, dict[str, str | torch.Tensor]], list[str]]:
        ## read emotions from file
        emotion_fn = os.path.join(self.root_dir, "emotions.txt")
        emotions = pd.read_csv(emotion_fn, header=None)[0].values.tolist()[
            :-1
        ]  # gets rid of neutral emotion

        ## read aggregated labels from file
        filename = os.path.join(self.root_dir, f"{split}.tsv")
        df = pd.read_csv(filename, sep="\t", header=None)

        ids = df.iloc[:, -1].values.tolist()
        aggr_labels = {
            _id: y
            for _id, y in zip(
                ids,
                self._multilabel_one_hot(
                    df.iloc[:, 1].values, len(emotions)
                ).float(),
            )
        }

        if self.annotation_mode == "aggregate":
            annotations = {
                _id: dict(
                    text=self.preprocessor(text),
                    original_text=text,
                    label={"aggregate": aggr_labels[_id]},
                )
                for _id, text in zip(ids, df.iloc[:, 0].values)
            }
            self.annotators = set()

        else:
            ## read annotator labels from file
            filenames = [
                os.path.join(self.root_dir, f"goemotions_{i}.csv")
                for i in range(1, 4)
            ]
            df = pd.concat([pd.read_csv(fn) for fn in filenames])
            df = df[df["id"].isin(set(ids))]
            df["labels"] = [
                [row[lbl] for lbl in emotions] for _, row in df.iterrows()
            ]

            groupby = df[["text", "rater_id", "id", "labels"]].groupby("id")
            annotations = groupby.agg(
                {
                    "text": lambda x: x.iloc[0],
                    "rater_id": lambda x: x.tolist(),
                    "labels": lambda x: x.tolist(),
                }
            )

            annotations = {
                _id: dict(
                    text=self.preprocessor(text),
                    original_text=text,
                    label={
                        worker_id: torch.tensor(labels).float()
                        for worker_id, labels in zip(rater_ids, label_list)
                    }
                    | {"aggregate": aggr_labels[_id]},
                )
                for _id, text, rater_ids, label_list in annotations.itertuples()
            }

            self.annotators = set(df["rater_id"].unique())

        emotions = self._subset_emotions(annotations, emotions)

        return annotations, emotions


class MFRC(TextDatasetWithPriors):
    """Plain text dataset for `MFRC`. Class doesn't
    use from_namespace decorator, so it can be used for inheritance.

    Attributes:
        Check `TextDatasetWithPriors` for attributes.
    """

    multilabel = True
    annotator_labels = True
    name = "MFRC"
    source_domain = "Reddit"

    def _load_data(self, split: str) -> tuple[
        dict[Any, dict[str, str | torch.Tensor | dict[str, torch.Tensor]]],
        list[str],
    ]:
        # { id1: { "text": "lorem ipsum", "label": {
        #   "ann1": torch.tensor([4]), "ann2": torch.tensor([3, 4]), "aggregate": torch.tensor([4]),
        # }, ... }, ... }

        # only train available in the dataset, contains entire dataset
        dataset = load_dataset("USC-MOLA-Lab/MFRC", split="train")

        with open(os.path.join(self.root_dir, "splits.yaml"), "r") as fp:
            text2id = yaml.safe_load(fp)[split]

        label_set = set()
        annotations = {}
        for e in dataset:
            id = text2id.get(e["text"], None)
            if id is None:
                # from another split
                continue

            labels = e["annotation"].split(",")
            if len(labels) > 1 and (
                "Non-Moral" in labels or "Thin Morality" in labels
            ):
                # https://arxiv.org/pdf/2208.05545v2 Appendix A.2.1:
                # Thin Morality only if no other label,
                # Non-Moral if no other label and not Thin Morality
                # so it cannot be that either is present and more than one modality
                continue
            elif labels[0] == "Non-Moral" or labels[0] == "Thin Morality":
                labels = []

            label_set.update(labels)

            if id not in annotations:
                annotations[id] = {
                    "text": self.preprocessor(e["text"]),
                    "original_text": e["text"],
                    "label": {e["annotator"]: labels},
                }
            else:
                annotations[id]["label"][e["annotator"]] = labels

        label_set = sorted(label_set)
        mlb = MultiLabelBinarizer().fit([label_set])
        for id in annotations:
            for annotator, label in annotations[id]["label"].items():
                annotations[id]["label"][annotator] = torch.tensor(
                    mlb.transform([label])[0]
                ).float()

            annotations[id]["label"]["aggregate"] = (
                (
                    sum(annotations[id]["label"].values())
                    / len(annotations[id]["label"])
                )
                >= 0.5
            ).float()

        return annotations, label_set


class MMLUPro(TextDatasetWithPriors):
    """Plain text dataset for `MMLU-Pro`. Class doesn't
    use from_namespace decorator, so it can be used for inheritance.

    Attributes:
        Check `TextDatasetWithPriors` for attributes.
    """

    multilabel = False
    annotator_labels = False
    name = "MMLU-Pro"
    source_domain = "Many"

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        args = TextDatasetWithPriors.argparse_args() | dict(
            delimiter=dict(
                type=str,
                default="\n",
                help="delimiter for options",
            )
        )
        del args["root_dir"]
        return args

    def __init__(self, delimiter: str = "\n", *args, **kwargs):
        self.delimiter = delimiter
        super().__init__(root_dir=None, *args, **kwargs)

    def _load_data(self, split: str) -> tuple[
        dict[Any, dict[str, str | torch.Tensor]],
        list[str],
    ]:
        # { id1: { "text": "lorem ipsum", "label": "label1", ... }, ... }
        # 12k in test, 70 in dev, no train

        if split in ("train", "dev"):
            split = "validation"

        dataset = load_dataset("TIGER-Lab/MMLU-Pro")[split]

        annotations = {}
        max_answers = 0
        for e in dataset:
            text = [e["question"]]
            text = [
                f"{chr(ord('A') + i)}. {o}" for i, o in enumerate(e["options"])
            ]
            text = e["question"] + "\n" + self.delimiter.join(text)

            max_answers = max(max_answers, len(e["options"]))

            annotations[str(e["question_id"])] = {
                "text": self.preprocessor(text),
                "original_text": text,
                "label": torch.tensor(e["answer_index"]),
            }

        return annotations, [chr(ord('A') + i) for i in range(max_answers)]


class Boxes(TextDatasetWithPriors):
    """Plain text dataset for `Boxes`. Class doesn't
    use from_namespace decorator, so it can be used for inheritance."""

    multilabel = True
    annotator_labels = False
    name = "Boxes Dataset"
    source_domain = "None"

    @staticmethod
    def argparse_args():
        return TextDatasetWithPriors.argparse_args() | dict(
            subset=dict(
                type=str,
                choices=[
                    "few_shot_boxes_nso_exp2_max3",
                    "few_shot_boxes_nso_exp2_max3_move_contents",
                    # "few_shot_boxes_nso_exp2_max3_ambiref",
                ],
                default="few_shot_boxes_nso_exp2_max3_move_contents",
                help="subset to load",
            )
        )

    def __init__(self, subset, *args, **kwargs):
        self.subset = subset
        super().__init__(*args, **kwargs)

    def _load_data(self, split: str) -> tuple[
        dict[Any, dict[str, str | torch.Tensor]],
        list[str],
    ]:
        # { id1: { "text": "lorem ipsum", "label": "label1", ... }, ... }

        if split == "train":
            # use smaller subset for prompting
            split = "train-subset"
        elif split == "test":
            split = "test-subsample-states"

        df = pd.read_json(
            os.path.join(
                self.root_dir,
                self.subset,
                f"{split}-t5.jsonl",
            ),
            lines=True,
        )

        with open(os.path.join(self.root_dir, "objects.txt")) as fp:
            all_items = fp.readline().strip().split(",")

        annotations = {}
        same_id_cnt = 0
        prev_id = None
        for _, row in df.iterrows():
            special_token_idx = row["sentence_masked"].find("<extra_id_0>")
            sentence = row["sentence_masked"][:special_token_idx].strip()

            items = row["masked_content"][len("<extra_id_0>") :]
            items = [l.strip() for l in items.replace("the", "").split("and")]

            if row["sample_id"] == prev_id:
                same_id_cnt += 1
            else:
                same_id_cnt = 0
                prev_id = row["sample_id"]

            # a lot of examples share id
            annotations[f"{row['sample_id']}-{same_id_cnt}"] = {
                "text": self.preprocessor(sentence),
                "original_text": sentence,
                "label": items,
            }

        mlb = MultiLabelBinarizer().fit([all_items])
        for id in annotations:
            annotations[id]["label"] = torch.tensor(
                mlb.transform([annotations[id]["label"]])[0]
            ).float()

        return annotations, all_items


class MSPPodcast(TextDatasetWithPriors):
    """Plain text dataset for `MSP Podcast`. Class doesn't
    use from_namespace decorator, so it can be used for inheritance."""

    @staticmethod
    def argparse_args():
        return TextDatasetWithPriors.argparse_args() | dict(
            multilabel=dict(
                action="store_true",
                help="whether to load as multilabel",
            ),
            annotator_labels=dict(
                action="store_true",
                help="whether to load annotator labels",
            ),
        )

    name = "MSP-Podcast-v1.11"
    source_domain = "Audio"
    # set here to avoid "can't instantiate abstract class" error
    annotator_labels = True
    multilabel = False

    def __init__(
        self, multilabel=False, annotator_labels=False, *args, **kwargs
    ):
        self.multilabel = multilabel
        self.annotator_labels = annotator_labels
        super().__init__(*args, **kwargs)

    def _load_data(self, split):
        # { id1: { "text": "lorem ipsum", "label": {
        #   "ann1": torch.tensor([4]), "ann2": torch.tensor([3, 4]), "aggregate": torch.tensor([4]),
        # }, ... }, ... }

        transcript_fn = os.path.join(self.root_dir, "transcripts.csv")
        ann_label_fn = os.path.join(
            self.root_dir, "Labels", "labels_detailed.csv"
        )
        split_label_fn = os.path.join(
            self.root_dir, "Labels", "labels_consensus.csv"
        )

        ann_df = pd.read_csv(ann_label_fn, index_col=0)
        # remove .wav from ID
        ann_df.index = ann_df.index.str.replace(".wav", "")

        split_df = pd.read_csv(
            split_label_fn,
            index_col="FileName",
            usecols=["FileName", "Split_Set"],
        )
        # remove .wav from ID
        split_df.index = split_df.index.str.replace(".wav", "")
        split = {"train": "Train", "dev": "Development", "test": "Test1"}[split]
        split_ids = set(split_df.index[split_df["Split_Set"] == split])

        # drop ids that are not in split
        ann_df = ann_df[ann_df.index.isin(split_ids)]
        cols = [
            "worker_id",
            "single-label",
            "multiple-labels",
            "a",
            "v",
            "d",
            "null",
        ]
        # break EmoDetail into columns at ";"
        ann_df[cols] = ann_df.EmoDetail.str.split(";", expand=True)
        # keep only relevant columns
        ann_df = ann_df[cols[:3]]

        unique_ids = ann_df.index.unique()
        annotations = {}
        transcripts_df = pd.read_csv(transcript_fn, index_col="id")
        for _id in unique_ids:
            text = transcripts_df.loc[_id, "text"]
            if text and text is not np.nan:
                annotations[_id] = {
                    "text": self.preprocessor(text),
                    "original_text": text,
                    "label": {},
                }

        label_set = set()

        for _id, row in tqdm(
            ann_df.iterrows(), desc="Processing labels", total=len(ann_df)
        ):
            if _id not in annotations:
                continue

            if self.multilabel:
                other_idx = row["multiple-labels"].find("Other")
                if other_idx == -1:
                    label = row["multiple-labels"].split(",")
                else:
                    label = row["multiple-labels"][:other_idx].split(",")
                label = [l.strip() for l in label if l.strip()]
                label_set.update(label)
            else:
                label = row["single-label"].strip()
                if not label or "other" in label.lower():
                    continue
                label_set.add(label)

            annotations[_id]["label"][row["worker_id"]] = label

        ids_to_remove = []
        if self.multilabel:
            label_set = sorted(label_set.difference(["Neutral"]))
            mlb = MultiLabelBinarizer().fit([label_set])
            for _id in annotations:
                if "label" not in annotations[_id]:
                    ids_to_remove.append(_id)
                    continue
                for worker_id, label in annotations[_id]["label"].items():
                    annotations[_id]["label"][worker_id] = torch.tensor(
                        mlb.transform([label])[0]
                    ).float()

                annotations[_id]["label"]["aggregate"] = (
                    (
                        sum(annotations[_id]["label"].values())
                        / len(annotations[_id]["label"])
                    )
                    >= 0.5
                ).float()
        else:
            label_set = sorted(label_set)
            for _id in annotations:
                if not annotations[_id]["label"]:
                    ids_to_remove.append(_id)
                    continue

                labels = []
                for worker_id, label in annotations[_id]["label"].items():
                    label = label_set.index(label)
                    annotations[_id]["label"][worker_id] = torch.tensor(
                        label
                    ).float()
                    labels.append(label)

                # use most frequent label as aggregate
                cnt = Counter(labels)
                annotations[_id]["label"]["aggregate"] = torch.tensor(
                    int(cnt.most_common(1)[0][0])
                ).float()

        for _id in ids_to_remove:
            del annotations[_id]

        if not self.annotator_labels:
            for _id in annotations:
                annotations[_id]["label"] = annotations[_id]["label"][
                    "aggregate"
                ]

        return annotations, label_set


class QueerReclaimLex(TextDatasetWithPriors):
    """Plain text dataset for `QueerReclaimLex`. Class doesn't
    use from_namespace decorator, so it can be used for inheritance."""

    # Instances are labeled for two definitions of harm depending on speaker identity. Scores are one of {0, 0.5, 1} where 0 means no harm, .5 means uncertain and 1 means harmful.
    # - `HARMFUL_IN`: Whether the post is harmful, given that the author is an *ingroup* member
    # - `HARMFUL_OUT`: Whether the post is harmful, given that the author is an *outgroup* member.

    # Each type of harm has variables for 4 different values. The same can be extended for `HARMFUL_OUT`.
    # - `HARMFUL_IN_1` denotes annotator 1's score, `HARMFUL_IN_2` for annotator 2's score
    # - `HARMFUL_IN_mu` for the mean of the two annotator's harm scores
    # - `HARMFUL_IN_gold` is a binary variable reflecting whether the harm score's mean is above a threshold of 0.5.

    annotator_labels = True
    name = "QueerReclaimLex"
    source_domain = "Twitter"

    @staticmethod
    def argparse_args():
        return TextDatasetWithPriors.argparse_args() | dict(
            type=dict(
                type=str,
                choices=["in", "out", "both"],
                default="both",
                help="type of harm to load",
                searchable=True,
            ),
            discard_ambiguous=dict(
                action="store_true",
                help="discard ambiguous examples (label == 0.5)",
            ),
        )

    @property
    def multilabel(self):
        return self.type == "both"

    def __init__(
        self,
        type: str = "both",
        discard_ambiguous: bool = False,
        *args,
        **kwargs,
    ):
        assert type in (
            "in",
            "out",
            "both",
        ), "type must be 'in', 'out', or 'both'"
        self.type = type
        self.discard_ambiguous = discard_ambiguous
        super().__init__(*args, **kwargs)

    def _load_data(self, split: str) -> tuple[
        dict[Any, dict[str, str | torch.Tensor]],
        list[str],
    ]:
        # read split
        with open(os.path.join(self.root_dir, "balanced-splits.json")) as fp:
            split_ids = json.load(fp)[split]

        df = pd.read_csv(
            os.path.join(self.root_dir, "QueerReclaimLex.csv"), index_col=0
        )
        # id is combination of col "template_idx" and col "term", e.g. 1-queer.
        # to access, we will create a dict from col "template" to this id
        df["id"] = df["template_idx"].astype(str) + "-" + df["term"]
        # filter by split
        df = df[df["id"].astype(str).isin(split_ids)]
        ids = {row["template"]: row["id"] for _, row in df.iterrows()}

        # read all Annotator#.xlsx from "Annotations" folder
        files = os.listdir(os.path.join(self.root_dir, "Annotations"))
        # use regex to match Annotator#.xlsx
        files = sorted([f for f in files if re.match(r"Annotator\d+.xlsx", f)])
        # read all files
        dfs = [
            pd.read_excel(
                os.path.join(self.root_dir, "Annotations", f),
                sheet_name="slurs",
            )
            for f in files
        ] + [
            pd.read_excel(
                os.path.join(self.root_dir, "Annotations", f),
                sheet_name="identity terms",
            )
            for f in files
        ]
        # remove columns where HARMFUL_IF_IN, HARMFUL_IF_OUT is not 0, 0.5, 1
        for i, df in enumerate(dfs):
            df = df[
                df["HARMFUL_IF_IN"].isin([0, 0.5, 1])
                & df["HARMFUL_IF_OUT"].isin([0, 0.5, 1])
            ]
            dfs[i] = df

        # add annotator id to each dataframe
        for i, df in enumerate(dfs):
            df["annotator_id"] = os.path.splitext(files[i % len(files)])[0]

        # concatenate all dataframes
        df = pd.concat(dfs)

        annotations = {}
        for i, row in df.iterrows():
            if self.type == "in":
                label = torch.tensor(row["HARMFUL_IF_IN"]).float()
            elif self.type == "out":
                label = torch.tensor(row["HARMFUL_IF_OUT"]).float()
            else:
                label = torch.tensor(
                    [row["HARMFUL_IF_IN"], row["HARMFUL_IF_OUT"]]
                ).float()

            _id = ids.get(row["POST"], None)
            if _id is None:
                continue

            if _id not in annotations:
                annotations[_id] = {
                    "text": self.preprocessor(row["POST"]),
                    "original_text": row["POST"],
                    "label": {row["annotator_id"]: label},
                }
            else:
                annotations[_id]["label"][row["annotator_id"]] = label

        for _id in list(annotations):
            aggregate = sum(annotations[_id]["label"].values()) / len(
                annotations[_id]["label"]
            )
            # round aggregate to 0 or 1
            if self.discard_ambiguous and aggregate == 0.5:
                annotations.pop(_id)
            else:
                annotations[_id]["label"]["aggregate"] = (
                    aggregate > 0.5
                ).float()

        label_set = (
            ["ingroup harm", "outgroup harm"]
            if self.multilabel
            else ["no harm", "harm"]
        )

        return annotations, label_set


class Hatexplain(TextDatasetWithPriors):
    """Plain text dataset for `Hatexplain`. Class doesn't
    use from_namespace decorator, so it can be used for inheritance."""

    multilabel = False
    annotator_labels = True
    name = "Hatexplain"
    source_domain = "None"

    @staticmethod
    def argparse_args():
        args = TextDatasetWithPriors.argparse_args()
        del args["root_dir"]
        return args

    def __init__(self, *args, **kwargs):
        super().__init__(root_dir=None, *args, **kwargs)

    def _load_data(self, split):
        split = {
            "train": "train",
            "dev": "validation",
            "test": "test",
        }[split]
        dataset = load_dataset("Hate-speech-CNERG/hatexplain", split=split)

        annotations = {}

        for e in dataset:
            text = " ".join(e["post_tokens"])
            annotations[e["id"]] = {
                "text": self.preprocessor(text),
                "original_text": text,
                "label": {},
            }
            for ann_id, label in zip(
                e["annotators"]["annotator_id"], e["annotators"]["label"]
            ):
                annotations[e["id"]]["label"][ann_id] = torch.tensor(
                    label
                ).float()

        # use most frequent label as aggregate
        for _id in annotations:
            cnt = Counter(
                [x.item() for x in annotations[_id]["label"].values()]
            )
            annotations[_id]["label"]["aggregate"] = torch.tensor(
                int(cnt.most_common(1)[0][0])
            ).float()

        return annotations, ["hate", "normal", "offensive"]


class TREC(TextDatasetWithPriors):
    multilabel = False
    annotator_labels = False
    name = "Text REtrieval Conference"
    source_domain = "None"

    @staticmethod
    def argparse_args():
        args = TextDatasetWithPriors.argparse_args()
        del args["root_dir"]
        return args

    def __init__(self, *args, **kwargs):
        super().__init__(root_dir=None, *args, **kwargs)

    def _load_data(self, split):
        # 'ABBR' (0): Abbreviation.
        # 'ENTY' (1): Entity.
        # 'DESC' (2): Description and abstract concept.
        # 'HUM' (3): Human being.
        # 'LOC' (4): Location.
        # 'NUM' (5): Numeric value.

        split = {
            "train": "train",
            "dev": "test",
            "test": "test",
        }[split]

        dataset = load_dataset("trec", split=split)
        annotations = {}
        for i, e in enumerate(dataset):
            text = e["text"]
            annotations[str(i)] = {
                "text": self.preprocessor(text),
                "original_text": text,
                "label": torch.tensor(e["coarse_label"]).float(),
            }

        return annotations, [
            "Abbreviation",
            "Entity",
            "Description",
            "Human",
            "Location",
            "Numeric",
        ]
