import traceback
from legm import splitify_namespace, ExperimentManager
from legm.argparse_utils import parse_args_and_metadata
import os

from llm_subj import (
    PromptDataset,
    UnaryBreakdownDataset,
    BinaryBreakdownDataset,
    vLMForClassification,
    vDistributionEstimator,
    text_preprocessor,
    CONSTANT_ARGS,
    DATASETS,
)
from llm_subj.utils import clean_cuda


# make its own function to avoid memory leaks
def loop(args, metadata):
    
    # for debugging
    
    # args.alternative_experiment_name = \
        # '/scratch1/mjma/llm_subjectivity/subjective-tasks-llms/logs/GoEmotions/prob_distr/baseline/debugging'
        
    # import shutil
    # experiment_folder = f'logs/MMLUPro/{args.alternative_experiment_name.replace(r"{distribution}", args.distribution)}_0'
    # print(f'EXPERIMENT FOLDER: {experiment_folder}')
    
    # if os.path.exists(experiment_folder):
        # shutil.rmtree(experiment_folder)
    
    # args.alternative_experiment_name = os.path.join(
    #     'prob_distr',
    #     'baseline',
    #     args.alternative_experiment_name
    # )
    
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

"""

score debugging

ember.trainer get_evals_from_dataset (line 1146)
    - model.forward
    - extra variable
    llm_subj.trainers.py get_extra_data_from_model (line 73)
        - return_vals variable
    
1. create Trainer extensions for:
    - Baseline
    - Unary breakdown
    - Binary breakdown

2. Create different prompts
    - Identify where prompts are created
    - Mess around live with prompts themselves
    - Create valid, replicable prompts
        - For binary, flip + average two labels
        
3. Create evaluation metrics
    - Make a script for evaluation (so that Yiorgos can extend)
    - MLE baseline
    - Monte Carlo simulation


dumping metrics into indexed_metrics.yml (will be main yml file)

creation and writing of indexed_metrics.yml: legm.exp_manager.log_metrics (line 1058)
    - just dumps self._best_metric_dict_indexed
        - which is determined in legm.exp_manager.set_best (line 1007)
            - based on the best value in self.metric_dict_indexed
                - which is set in legm.exp_manager.set_metric (line 536, 584)
                    - which is called in legm.exp_manager.set_dict_metrics (line 602, 624)
                        - which is called in ember.trainer.run (line 918, 1140)
                        
Llama tokens:
    'a': 64
    'A': 32
    ' A': 362
    ' a': 264
    
    'b': 65
    'B': 33
    ' B': 426
    ' b': 293

- move seaparate_question_labels to the dataset itself

- unary and binary in-context examples:
    - random every time
    - random, but fixed within a sample
    - fixed across all samples

todo:

- investigate sampling bias (?)
- calculate multi-label distributions for:
    - baseline
    - output
    - unary
    - binary
- multi-label is a series of independent distributions, normalized based on the "none" distribution
    - zero-shot approximation
    - dev set using Platt scaling


"""
