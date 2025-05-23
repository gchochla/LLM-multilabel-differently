"""Add ROC AUC to existing aggregated metrics YAML."""

import yaml
import os
import pickle

import gridparse
import torch
import numpy as np
from tqdm import tqdm
from legm import splitify_namespace
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.metrics import jaccard_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", module="sklearn.*")

from llm_ml import DATASETS


def parse_args():
    parser = gridparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        nargs="+",
        help="Path to experiment to perform linear probes on.",
        required=True,
    )
    parser.add_argument(
        "--eval-folder",
        type=str,
        nargs="+",
        help="Path to experiment to perform linear probes on (eval sets to get scores)",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        nargs="+",
        help="Path to output folder to save the results.",
    )

    args = parser.parse_args()

    if args.eval_folder is None:
        args.eval_folder = [None] * len(args.folder)

    assert len(args.eval_folder) == len(
        args.folder
    ), "If eval-folder is provided, it must be the same length as folder."

    if args.output_folder is None:
        args.output_folder = [None] * len(args.folder)

    assert len(args.output_folder) == len(
        args.folder
    ), "If output-folder is provided, it must be the same length as folder."

    return args


def main():
    args = parse_args()

    for f, ef, of in zip(args.folder, args.eval_folder, args.output_folder):

        print(f"Linear probes on {f}")
        exp_str = "experiment_0"

        param_file_path = os.path.join(f, "params.yml")
        with open(param_file_path) as file:
            params = yaml.safe_load(file)[exp_str]

        model_metrics_path = os.path.join(f, "indexed_metrics.yml")
        with open(model_metrics_path) as file:
            model_metrics = yaml.safe_load(file)[exp_str]

        if ef is not None:
            features_eval = torch.load(
                os.path.join(ef, "last_hidden_state.pt"),
                map_location=torch.device("cpu"),
            )[exp_str]["aggregate"]
            features_eval = {
                k: f for k, f in features_eval.items() if f.ndim == 1
            }
            ids_eval = list(features_eval)
            features_eval = np.stack(
                [f.numpy() for f in features_eval.values()]
            )
            ids_eval = np.array(ids_eval)
            # remove features that are all zeros
            ids_eval = ids_eval[~np.all(features_eval == 0, axis=1)]
            features_eval = features_eval[~np.all(features_eval == 0, axis=1)]

        print("\tLoaded eval features")

        for k, v in params.items():
            if isinstance(v, str):
                params[k] = v.replace("/project/shrikann_35/", "/data1/")

        ds = DATASETS[params["task"]](
            init__namespace=splitify_namespace(
                gridparse.Namespace(**params), "test"
            )
        )

        features = torch.load(
            os.path.join(f, "last_hidden_state.pt"),
            map_location=torch.device("cpu"),
        )[exp_str]["aggregate"]
        print("\tLoaded features")
        features = {k: f for k, f in features.items() if f.ndim == 1}
        ids = list(features)

        features = np.stack([f.numpy() for f in features.values()])
        labels = np.array(
            [ds.getitem_by_id(_id)["label"].numpy() for _id in ids]
        )

        # remove features that are all zeros
        labels = labels[~np.all(features == 0, axis=1)]
        ids = np.array(ids)[~np.all(features == 0, axis=1)]
        features = features[~np.all(features == 0, axis=1)]

        # Split the data and the IDs into training and testing sets
        X_train, X_test, y_train, y_test, ids_train, ids_test = (
            train_test_split(
                features, labels, ids, test_size=0.2, random_state=42
            )
        )

        # normalization and dimensionality reduction
        scaler = make_pipeline(
            StandardScaler(with_mean=False),
            TruncatedSVD(
                n_components=max(features.shape[-1] // 4, 512), random_state=42
            ),
        )
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        if ef is not None:
            features_eval = scaler.transform(features_eval)

        model_preds = {
            _id: ds.get_label_from_str(model_metrics[_id]["test_preds"])
            for _id in ids
        }

        model_first_preds = {}
        for _id in ids:
            # get the first output of the model
            label_inds = []
            for l in ds.label_set:
                idx = model_metrics[_id]["test_outs"].find(l)
                if idx == -1:
                    idx = float("inf")
                label_inds.append(idx)
            # get argmin to find the label
            model_first_preds[_id] = ds.get_label_from_str(
                ds.label_set[np.argmin(label_inds)]
            ).numpy()

        # get labels of model outputs to train to predict what the model output
        # based on the features of the first token
        y_model_train = np.array(
            [
                ds.get_label_from_str(model_metrics[_id]["test_preds"])
                for _id in ids_train
            ]
        )
        y_model_test = np.array(
            [
                ds.get_label_from_str(model_metrics[_id]["test_preds"])
                for _id in ids_test
            ]
        )

        # Create a Logistic Regression model
        # take into account multilabel vs single label

        if ds.multilabel:
            # labels are [1, 0, 1, 0, 1], not integer
            # sklearn does not support that
            # so we use a list of models
            model = [
                LogisticRegression(
                    max_iter=1000,
                    multi_class="auto",
                    solver="saga",
                    penalty=None,
                )
                for _ in range(labels.shape[1])
            ]
            pred_model = [
                LogisticRegression(
                    max_iter=1000,
                    multi_class="auto",
                    solver="saga",
                    penalty=None,
                )
                for _ in range(labels.shape[1])
            ]
            # Fit the model
            preds = []
            eval_preds = []
            for i in tqdm(range(labels.shape[1]), desc="Training models"):
                model[i].fit(X_train, y_train[:, i])
                # Make predictions
                preds.append(model[i].predict(X_test))
                if ef is not None:
                    eval_preds.append(
                        model[i].predict_proba(features_eval)[:, -1]
                    )

            preds_for_model = []
            for i in tqdm(
                range(labels.shape[1]), desc="Training models on model"
            ):
                pred_model[i].fit(X_train, y_model_train[:, i])
                # Make predictions
                preds_for_model.append(pred_model[i].predict(X_test))

            preds = np.array(preds).T
            preds_for_model = np.array(preds_for_model).T

            # remove the first token from the model predictions
            y_toremove = np.array([model_first_preds[_id] for _id in ids_test])
            # find which one is 1 and get the index
            y_toremove = np.array(
                [
                    np.where(y == 1)[0][0] if np.any(y == 1) else -1
                    for y in y_toremove
                ]
            )
            # remove that index from the y_model_test
            y_model_test_first = np.array(
                [
                    np.delete(y_model_test[i], y_toremove[i])
                    for i in range(len(y_model_test))
                    if y_toremove[i] != -1
                ]
            )
            preds_for_model_first = np.array(
                [
                    np.delete(preds_for_model[i], y_toremove[i])
                    for i in range(len(preds_for_model))
                    if y_toremove[i] != -1
                ]
            )

            print("\rGot predictions")

            if ef is not None:
                eval_preds = np.array(eval_preds).T
                scores_eval = {
                    str(_id): {
                        str(l): float(prob)
                        for l, prob in zip(ds.label_set, probs.tolist())
                    }
                    for _id, probs in zip(ids_eval, eval_preds)
                }

            model_preds = np.array(
                [
                    ds.get_label_from_str(model_metrics[_id]["test_preds"])
                    for _id in ids_test
                ]
            )
            results = {
                "js": float(
                    jaccard_score(
                        y_test, preds, average="samples", zero_division=1
                    )
                ),
                "macro_f1": float(
                    f1_score(y_test, preds, average="macro", zero_division=0)
                ),
                "micro_f1": float(
                    f1_score(y_test, preds, average="micro", zero_division=0)
                ),
                "llm_js": float(
                    jaccard_score(
                        y_test, model_preds, average="samples", zero_division=1
                    )
                ),
                "llm_macro_f1": float(
                    f1_score(
                        y_test, model_preds, average="macro", zero_division=0
                    )
                ),
                "llm_micro_f1": float(
                    f1_score(
                        y_test, model_preds, average="micro", zero_division=0
                    )
                ),
            }

            # measure for the performance of the regression on the model predictions
            # but also for the non-first token
            results_model = {
                "js": float(
                    jaccard_score(
                        y_model_test,
                        preds_for_model,
                        average="samples",
                        zero_division=1,
                    )
                ),
                "macro_f1": float(
                    f1_score(
                        y_model_test,
                        preds_for_model,
                        average="macro",
                        zero_division=0,
                    )
                ),
                "micro_f1": float(
                    f1_score(
                        y_model_test,
                        preds_for_model,
                        average="micro",
                        zero_division=0,
                    )
                ),
                "js_nonfirst": float(
                    jaccard_score(
                        y_model_test_first,
                        preds_for_model_first,
                        average="samples",
                        zero_division=1,
                    )
                ),
                "macro_f1_nonfirst": float(
                    f1_score(
                        y_model_test_first,
                        preds_for_model_first,
                        average="macro",
                        zero_division=0,
                    )
                ),
                "micro_f1_nonfirst": float(
                    f1_score(
                        y_model_test_first,
                        preds_for_model_first,
                        average="micro",
                        zero_division=0,
                    )
                ),
                "js_nonzero": float(
                    jaccard_score(
                        y_model_test[~np.all(y_model_test == 0, axis=1)],
                        preds_for_model[~np.all(y_model_test == 0, axis=1)],
                        average="samples",
                        zero_division=1,
                    )
                ),
                "macro_f1_nonzero": float(
                    f1_score(
                        y_model_test[~np.all(y_model_test == 0, axis=1)],
                        preds_for_model[~np.all(y_model_test == 0, axis=1)],
                        average="macro",
                        zero_division=0,
                    )
                ),
                "micro_f1_nonzero": float(
                    f1_score(
                        y_model_test[~np.all(y_model_test == 0, axis=1)],
                        preds_for_model[~np.all(y_model_test == 0, axis=1)],
                        average="micro",
                        zero_division=0,
                    )
                ),
            }

        else:
            model = LogisticRegression(
                max_iter=1000,
                multi_class="auto",
                solver="saga",
                penalty=None,
                n_jobs=16,
            )
            pred_model = LogisticRegression(
                max_iter=1000,
                multi_class="auto",
                solver="saga",
                penalty=None,
                n_jobs=16,
            )
            # Fit the model
            model.fit(X_train, y_train)
            pred_model.fit(X_train, y_model_train)
            # Make predictions
            preds = model.predict(X_test)
            preds_for_model = model.predict(X_test)
            if ef is not None:
                eval_preds = model.predict_proba(features_eval)[:, -1]
                scores_eval = {
                    str(_id): {
                        str(l): float(prob)
                        for l, prob in zip(ds.label_set, probs)
                    }
                    for _id, probs in zip(ids_eval, eval_preds)
                }

            model_preds = np.array(
                [
                    ds.get_label_from_str(model_metrics[_id]["test_preds"])
                    for _id in ids_test
                ]
            )
            results = {
                "accuracy": float(accuracy_score(y_test, preds)),
                "f1": float(
                    f1_score(y_test, preds, average="macro", zero_division=0)
                ),
                "llm_accuracy": float(accuracy_score(y_test, model_preds)),
                "llm_f1": float(
                    f1_score(
                        y_test, model_preds, average="macro", zero_division=0
                    )
                ),
            }
            # measure for the performance of the regression on the model predictions
            results_model = {
                "accuracy": float(
                    accuracy_score(y_model_test, preds_for_model)
                ),
                "f1": float(
                    f1_score(y_model_test, preds_for_model, average="macro")
                ),
            }

        print(f"Results: {results}")
        print(f"Results model: {results_model}")
        os.makedirs(os.path.join(f, "linear_probes"), exist_ok=True)

        # Save the results to a YAML file
        results_file_path = os.path.join(f, "linear_probes", "metrics.yml")
        with open(results_file_path, "w") as fp:
            yaml.dump(results, fp)

        # Save the results model to a YAML file
        results_model_file_path = os.path.join(
            f, "linear_probes", "metrics_model.yml"
        )
        with open(results_model_file_path, "w") as fp:
            yaml.dump(results_model, fp)

        # save preds per ID
        preds_file_path = os.path.join(
            f, "linear_probes", "indexed_metrics.yml"
        )
        preds = {
            str(ids_test[i]): {
                "preds": ds.index_label_set(preds[i]),
                "label": ds.index_label_set(y_test[i]),
                "model_preds": model_metrics[ids_test[i]]["test_preds"],
                "model_outs": model_metrics[ids_test[i]]["test_outs"],
                "gt": model_metrics[ids_test[i]]["test_outs"],
            }
            for i in range(len(ids_test))
        }
        with open(preds_file_path, "w") as fp:
            yaml.dump(preds, fp)

        # Save the model to a file
        model_file_path = os.path.join(f, "linear_probes", "model.pkl")
        with open(model_file_path, "wb") as fp:
            pickle.dump(model, fp)

        if ef is not None:
            os.makedirs(os.path.join(ef, "linear_probes"), exist_ok=True)
            # Save the eval preds to a YAML file
            eval_scores_file_path = os.path.join(
                ef, "linear_probes", "scores.yml"
            )
            with open(eval_scores_file_path, "w") as fp:
                yaml.dump(scores_eval, fp)

            # Save the model to a file
            model_file_path = os.path.join(ef, "linear_probes", "model.pkl")
            with open(model_file_path, "wb") as fp:
                pickle.dump(model, fp)

        if of is not None:
            os.makedirs(os.path.join(of), exist_ok=True)
            # Save the model to a file
            model_file_path = os.path.join(of, "model.pkl")
            with open(model_file_path, "wb") as fp:
                pickle.dump(model, fp)

            # Save the results to a YAML file
            results_file_path = os.path.join(of, "metrics.yml")
            with open(results_file_path, "w") as fp:
                yaml.dump(results, fp)

            # Save the results model to a YAML file
            results_model_file_path = os.path.join(of, "metrics_model.yml")
            with open(results_model_file_path, "w") as fp:
                yaml.dump(results_model, fp)

            # save preds per ID
            preds_file_path = os.path.join(of, "indexed_metrics.yml")
            with open(preds_file_path, "w") as fp:
                yaml.dump(preds, fp)

            if ef is not None:
                # Save the eval preds to a YAML file
                eval_scores_file_path = os.path.join(of, "eval_scores.yml")
                with open(eval_scores_file_path, "w") as fp:
                    yaml.dump(scores_eval, fp)

                # Save the model to a file
                model_file_path = os.path.join(of, "eval_model.pkl")
                with open(model_file_path, "wb") as fp:
                    pickle.dump(model, fp)


if __name__ == "__main__":
    main()
