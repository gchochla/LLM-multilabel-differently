from legm import splitify_namespace, ExperimentManager
from legm.argparse_utils import parse_args_and_metadata

from llm_ml.models_.demux import Demux, DATASETS, DemuxTrainer
from llm_ml import CONSTANT_ARGS
from llm_ml.utils import preprocessor


def main():
    grid_args, metadata = parse_args_and_metadata(
        modules=[Demux, DemuxTrainer, ExperimentManager],
        args=CONSTANT_ARGS,
        subparsers=DATASETS.keys(),
        dest="task",
        conditional_modules=DATASETS,
    )
    for i, args in enumerate(grid_args):
        print(f"Running {i + 1}/{len(grid_args)}: {args}\n\n")

        exp_manager = ExperimentManager(
            "./logs/DEMUX",
            args.task,
            logging_level=args.logging_level,
            description=args.description,
            alternative_experiment_name=args.alternative_experiment_name,
        )
        exp_manager.set_namespace_params(args)
        exp_manager.set_param_metadata(metadata[args.task], args)
        exp_manager.start()

        # this is done after exp_manager.set_namespace_params
        # so as not to log the actual preprocessing function
        if args.text_preprocessor:
            args.text_preprocessor = preprocessor(
                DATASETS[args.task].source_domain
            )
        else:
            args.text_preprocessor = None

        train_dataset = DATASETS[args.task](
            init__namespace=splitify_namespace(args, "train")
        )
        dev_dataset = DATASETS[args.task](
            init__namespace=splitify_namespace(args, "dev"),
            annotator_ids=train_dataset.annotators,
        )
        test_dataset = DATASETS[args.task](
            init__namespace=splitify_namespace(args, "test"),
            annotator_ids=train_dataset.annotators,
        )
        model = Demux.from_pretrained(
            args.model_name_or_path,
            dropout_prob=args.dropout_prob,
            class_inds=train_dataset.class_inds,
        )

        trainer = DemuxTrainer(
            model=model,
            train_dataset=train_dataset,
            dev_dataset=dev_dataset,
            test_dataset=test_dataset,
            experiment_manager=exp_manager,
        )

        trainer.run()


if __name__ == "__main__":
    main()
