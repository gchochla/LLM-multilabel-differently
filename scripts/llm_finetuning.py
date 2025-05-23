import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import traceback
import gridparse
from legm import splitify_namespace, ExperimentManager
from legm.argparse_utils import add_arguments, add_metadata
from huggingface_hub import login
import wandb

from llm_ml import (
    PromptEvaluator,
    text_preprocessor,
    CONSTANT_ARGS,
)
from llm_ml.utils import clean_cuda
from llm_ml.prompt_dataset import PromptDatasetWithQueryLabels
from llm_ml.datasets import SemEval2018Task1EcDataset, GoEmotionsDataset
from llm_ml.models import LMForFinetuning, LMForClassification
from llm_ml.trainers import FinetuneEvaluator
from transformers import AutoTokenizer
import warnings

warnings.filterwarnings("ignore")


DATASET = dict(
    SemEval=SemEval2018Task1EcDataset
    # GoEmotions=GoEmotionsDataset
)

token = os.getenv('ACCESS_TOKEN')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
login(token)


def parse_args_and_metadata():
    parser = gridparse.GridArgumentParser()
    metadata = {}
    sp = parser.add_subparsers(dest="task")

    for task in DATASET:
        sp_task = sp.add_parser(task)
        metadata[task] = {}

        argparse_args = {}
        for module in [
            DATASET[task],
            # PromptDataset,
            PromptDatasetWithQueryLabels,
            LMForFinetuning,
            LMForClassification,
            PromptEvaluator,
            ExperimentManager,
        ]:
            argparse_args.update(module.argparse_args())

        add_arguments(sp_task, argparse_args, replace_underscores=True)
        add_metadata(metadata[task], argparse_args)

        add_arguments(sp_task, CONSTANT_ARGS, replace_underscores=True)
        add_metadata(metadata[task], CONSTANT_ARGS)

    return parser.parse_args(), metadata


def format_instruction(sample):
    return f"{sample['text']}"


# make its own function to avoid memory leaks
def loop(args, metadata):
    wandb.init(
        project="subjective-tasks-llms-scripts",
        name=args.model_name_or_path + '/' + str(args.num_train_epochs),
    )
    print("\nCurrent setting: ", args, "\n")

    if args.text_preprocessor:
        args.text_preprocessor = text_preprocessor[
            DATASET[args.task].source_domain
        ]()
    else:
        args.text_preprocessor = None

    train_dataset = DATASET[args.task](
        init__namespace=splitify_namespace(args, "train")
    )
    test_dataset = DATASET[args.task](
        init__namespace=splitify_namespace(args, "test")
    )
    train_data = PromptDatasetWithQueryLabels(
        train_dataset=train_dataset,  # train_dataset argument is not considerd as of now.
        test_dataset=train_dataset,
        init__namespace=args,
    )
    val_data = PromptDatasetWithQueryLabels(
        train_dataset=train_dataset,  # train_dataset argument is not considerd as of now.
        test_dataset=test_dataset,
        init__namespace=args,
    )

    train_ds = train_data.get_list()
    val_ds = val_data.get_list()
    print(format_instruction(train_ds[0]))

    model = LMForFinetuning(
        args.model_name_or_path,
        os.path.join(
            args.output_dir, args.model_name_or_path, str(args.num_train_epochs)
        ),
        os.path.join(
            args.log_dir,
            os.path.basename(args.model_name_or_path),
            str(args.num_train_epochs),
        ),
        args.num_train_epochs,
        args.cache_dir,
    )

    model.finetune(train_ds, val_ds, format_instruction)
    clean_cuda(model)

    eval_model_path = os.path.join(
        args.output_dir, args.model_name_or_path, str(args.num_train_epochs)
    )
    if not os.path.exists(eval_model_path):
        print('Finetuned model not present in path')

    tokenizer = AutoTokenizer.from_pretrained(eval_model_path)

    eval_model = LMForClassification(
        model_name_or_path=eval_model_path,
        max_new_tokens=args.max_new_tokens,
        model_dtype=args.model_dtype,
        load_in_8bit=False,
        load_in_4bit=args.load_in_4bit,
        device="auto",
        tokenizer=tokenizer,
        trust_remote_code=False,
        labels=val_data.label_set,
        cache_dir=args.cache_dir,
    )

    evaluator = FinetuneEvaluator(
        model=eval_model,
        test_dataset=val_data,
        labels=val_data.label_set,
        log_dir=os.path.join(
            args.log_dir,
            os.path.basename(args.model_name_or_path),
            str(args.num_train_epochs),
        ),
    )
    metrics = evaluator.evaluate()
    print(metrics)
    # clean_cuda(model)


def main():
    grid_args, metadata = parse_args_and_metadata()

    for args in grid_args:
        try:
            loop(args, metadata)
        except Exception as e:
            print("\n\n\nError:", traceback.format_exc())
            print("\n\n\nContinuing...\n\n\n")
            clean_cuda()


if __name__ == "__main__":
    main()
