from .datasets import (
    SemEval2018Task1EcDataset,
    SemEval2018Task1EcDatasetForTransformers,
    GoEmotionsDataset,
    GoEmotionsDatasetForTransformers,
    MFRCDataset,
    MFRCDatasetForTransformers,
    MMLUProDataset,
    MMLUProDatasetForTransformers,
    BoxesDataset,
    BoxesDatasetForTransformers,
    MSPPodcastDataset,
    MSPPodcastDatasetForTransformers,
    QueerReclaimLexDataset,
    QueerReclaimLexDatasetForTransformers,
    HatexplainDataset,
    HatexplainDatasetForTransformers,
    TRECDataset,
    TRECDatasetForTransformers,
)

DATASETS = dict(
    SemEval=SemEval2018Task1EcDataset,
    GoEmotions=GoEmotionsDataset,
    MFRC=MFRCDataset,
    MMLUPro=MMLUProDataset,
    Boxes=BoxesDataset,
    MSPPodcast=MSPPodcastDataset,
    QueerReclaimLex=QueerReclaimLexDataset,
    Hatexplain=HatexplainDataset,
    TREC=TRECDataset,
)


from .models import LMForClassification, OpenAIClassifier, vLMForClassification
from .prompt_dataset import (
    PromptDataset,
    PromptTextDataset,
    OpenAIPromptTextDataset,
    ReasonablenessPromptDataset,
    OpenAIReasonablenessPromptTextDataset,
)
from .trainers import (
    PromptEvaluator,
    vPromptEvaluator,
    APIPromptEvaluator,
    ReasonablenessEvaluator,
    vReasonablenessEvaluator,
    APIReasonablenessEvaluator,
)
from .utils import twitter_preprocessor, reddit_preprocessor

text_preprocessor = dict(
    Twitter=twitter_preprocessor,
    Reddit=reddit_preprocessor,
    Plain=lambda *a, **k: lambda x: x,
)

CONSTANT_ARGS = dict(
    seed=dict(
        type=int,
        help="random seed",
        metadata=dict(disable_comparison=True),
        searchable=True,
    ),
)
