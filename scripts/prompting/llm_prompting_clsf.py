import traceback
from legm import splitify_namespace, ExperimentManager
from legm.argparse_utils import parse_args_and_metadata

from llm_ml import (
    PromptDataset,
    LMForClassification,
    PromptEvaluator,
    text_preprocessor,
    CONSTANT_ARGS,
    DATASETS,
)
from llm_ml.utils import clean_cuda


# make its own function to avoid memory leaks
def loop(args, metadata):
    exp_manager = ExperimentManager(
        "./logs",
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
        args.text_preprocessor = text_preprocessor[
            DATASETS[args.task].source_domain
        ]()
    else:
        args.text_preprocessor = None

    train_dataset = DATASETS[args.task](
        init__namespace=splitify_namespace(args, "train")
    )
    test_dataset = DATASETS[args.task](
        init__namespace=splitify_namespace(args, "test"),
        annotator_ids=train_dataset.annotators,
    )
    dataset = PromptDataset(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        init__namespace=args,
    )

    model = LMForClassification(
        init__namespace=args,
        labels=dataset.label_set,
        tokenizer=dataset.get_tokenizer(),
    )

    evaluator = PromptEvaluator(
        model=model, test_dataset=dataset, experiment_manager=exp_manager
    )

    evaluator.train()

    clean_cuda(model)


def main():
    grid_args, metadata = parse_args_and_metadata(
        [
            PromptDataset,
            LMForClassification,
            PromptEvaluator,
            ExperimentManager,
        ],
        CONSTANT_ARGS,
        DATASETS,
        "task",
        DATASETS,
    )
    for i, args in enumerate(grid_args):
        try:
            print(f"\nCurrent setting {i + 1}/{len(grid_args)}: {args}\n")
            loop(args, metadata)
        except Exception as e:
            print("\n\n\nError:", traceback.format_exc())
            print("\n\n\nContinuing...\n\n\n")
            clean_cuda()


if __name__ == "__main__":
    main()
