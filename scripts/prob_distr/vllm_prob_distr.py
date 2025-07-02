import traceback
from legm import splitify_namespace, ExperimentManager
from legm.argparse_utils import parse_args_and_metadata
import os

from llm_ml import (
    PromptDataset,
    UnaryBreakdownDataset,
    BinaryBreakdownDataset,
    vLMForClassification,
    vDistributionEstimator,
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
    
    
    print(exp_manager._alternative_experiment_name)

    # this is done after exp_manager.set_namespace_params
    # so as not to log the actual preprocessing function
    if args.text_preprocessor:
        args.text_preprocessor = text_preprocessor[DATASETS[args.task].source_domain]()
    else:
        args.text_preprocessor = None

    print("loading train")

    train_dataset = DATASETS[args.task](
        init__namespace=splitify_namespace(args, "train")
    )

    print("loading test")

    test_dataset = DATASETS[args.task](
        init__namespace=splitify_namespace(args, "test"),
        # annotator_ids=train_dataset.annotators,
    )
    
    # for debugging, setting len(test_dataset) to 10
    # def debug_len(self): return 5
    # test_dataset.__class__.__len__ = debug_len
    
    
    if args.distribution == 'baseline':
        dataset = PromptDataset(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            init__namespace=args,
        )
    elif args.distribution == 'unary':
        dataset = UnaryBreakdownDataset(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            init__namespace=args,
        )
    elif args.distribution == 'binary':
        dataset = BinaryBreakdownDataset(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            init__namespace=args,
        )

    model = vLMForClassification(
        init__namespace=args,
        labels=dataset.label_set,
        tokenizer=dataset.get_tokenizer(),
    )

    evaluator = vDistributionEstimator(
        model=model, test_dataset=dataset, experiment_manager=exp_manager
    )
    
    evaluator.run()

    clean_cuda(model)


def main():
    grid_args, metadata = parse_args_and_metadata(
        [
            PromptDataset,
            vLMForClassification,
            vDistributionEstimator,
            ExperimentManager,
        ],
        CONSTANT_ARGS,
        DATASETS,
        "task",
        DATASETS,
    )
    for i, args in enumerate(grid_args):
        print(args.alternative_experiment_name)
        # import pdb; pdb.set_trace()
        try:
            print(f"\nCurrent setting {i + 1}/{len(grid_args)}: {args}\n")
            loop(args, metadata)
        except Exception as e:
            print("\n\n\nError:", traceback.format_exc())
            print("\n\n\nContinuing...\n\n\n")
            clean_cuda()


if __name__ == "__main__":
    main()