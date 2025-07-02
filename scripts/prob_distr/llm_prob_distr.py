import traceback
from legm import splitify_namespace, ExperimentManager
from legm.argparse_utils import parse_args_and_metadata
import os
import copy

from llm_ml import (
    PromptDataset,
    UnaryBreakdownDataset,
    BinaryBreakdownDataset,
    ICLMultiLabelRatioDataset,
    LMForClassification,
    DistributionEstimator,
    text_preprocessor,
    CONSTANT_ARGS,
    DATASETS,
)
from llm_ml.utils import clean_cuda

model = None


# make its own function to avoid memory leaks
def loop(args, metadata):
    global model
    
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
    
    # if args.task == 'MFRC':
    #     multilabel_ids = []
    #     for i in range(len(train_dataset)):
    #         datum = train_dataset[i]
    #         datum_id = datum['id']
    #         if datum['label'].sum().item() > 1:
    #             multilabel_ids.append(tuple(datum_id.split('__')))
    #     train_dataset.ids = multilabel_ids
        
    #     just_ids = [int(x[0]) for x in train_dataset.ids]
    #     for label in train_dataset.annotator2label_inds['aggregate'].keys():
    #         train_dataset.annotator2label_inds['aggregate'][label] = just_ids
    #     train_dataset.annotator2inds['aggregate'] = just_ids
        
    # import pdb; pdb.set_trace()

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
    elif args.distribution == 'multilabel_ICL':
        dataset = ICLMultiLabelRatioDataset(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            init__namespace=args,
        )
        
    if model is None:
        print('LOADING MODEL')
        model = LMForClassification(
            init__namespace=args,
            labels=dataset.label_set,
            tokenizer=dataset.get_tokenizer(),
        )

    evaluator = DistributionEstimator(
        model=model, test_dataset=dataset, experiment_manager=exp_manager
    )
    
    evaluator.run()

def main():
    global model
    grid_args, metadata = parse_args_and_metadata(
        [
            PromptDataset,
            LMForClassification,
            DistributionEstimator,
            ExperimentManager,
        ],
        CONSTANT_ARGS,
        DATASETS,
        "task",
        DATASETS,
    )
    
    # if seed is -1, then we create four seeds (1, 2, 3, 4)
    if grid_args[0].seed == -1:
        new_args = []
        for i in range(0, 5):
            new_args.append(copy.deepcopy(grid_args[0]))
            new_args[i - 1].seed = i
        grid_args = new_args
        
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
    clean_cuda(model)
        


if __name__ == "__main__":
    main()
