import yaml

from legm import splitify_namespace
from legm.argparse_utils import parse_args_and_metadata

from llm_ml.models_.demux import DATASETS


def main():
    grid_args, _ = parse_args_and_metadata(
        modules=[],
        args=dict(
            experiment_name=dict(
                type=str,
                required=True,
                help="Name of the experiment to extract scores from.",
            )
        ),
        subparsers=DATASETS.keys(),
        dest="task",
        conditional_modules=DATASETS,
    )

    for args in grid_args:
        print(f"Extracting scores for {args.experiment_name}...\n\n")

        ds = DATASETS[args.task](
            init__namespace=splitify_namespace(args, "train"),
            model_name_or_path="bert-base-uncased",
        )

        # Load the YAML file
        with open(
            f"./logs/DEMUX/{args.task}/{args.experiment_name}/indexed_metrics.yml"
        ) as fp:
            data = yaml.safe_load(fp)["experiment_0"]

        data.pop("description")
        scores = {
            k: {l: lv for l, lv in zip(ds.label_set, v["test_scores"])}
            for k, v in data.items()
        }

        print(f"Saving scores for {args.experiment_name}...\n\n")
        # Save the scores to a new YAML file
        with open(
            f"./logs/DEMUX/{args.task}/{args.experiment_name}/scores.yml", "w"
        ) as fp:
            yaml.dump(scores, fp)


if __name__ == "__main__":
    main()
