from typing import Any

from legm import from_namespace

from llm_ml.base_datasets import TokenizationMixin
from llm_ml.benchmarks import (
    SemEval2018Task1Ec,
    GoEmotions,
    MFRC,
    MMLUPro,
    Boxes,
    MSPPodcast,
    QueerReclaimLex,
    Hatexplain,
    TREC,
)


class SemEval2018Task1EcDataset(SemEval2018Task1Ec):
    """Plain text dataset for `SemEval 2018 Task 1: Affect in Tweets`
    (https://competitions.codalab.org/competitions/17751). Class uses
    from_namespace decorator, so it should not be used for inheritance.

    Attributes: Check `SemEval2018Task1Ec`.
    """

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SemEval2018Task1EcDatasetForTransformers(
    TokenizationMixin, SemEval2018Task1Ec
):
    """Dataset with encodings for `transformers`
    for `SemEval 2018 Task 1: Affect in Tweets`"""

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        return (
            SemEval2018Task1Ec.argparse_args()
            | TokenizationMixin.argparse_args()
        )

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dict_tokenize(self.examples, self.preprocessor)


class GoEmotionsDataset(GoEmotions):
    """Plain text dataset for `GoEmotions`. Class uses
    from_namespace decorator, so it should not be used for inheritance.

    Attributes: Check `GoEmotions`.
    """

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class GoEmotionsDatasetForTransformers(TokenizationMixin, GoEmotions):
    """Dataset with encodings for `transformers`
    for `GoEmotions`"""

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        return GoEmotions.argparse_args() | TokenizationMixin.argparse_args()

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dict_tokenize(self.examples, self.preprocessor)


class MFRCDataset(MFRC):
    """Plain text dataset for `MFRC`. Class uses
    from_namespace decorator, so it should not be used for inheritance.

    Attributes: Check `MFRC`.
    """

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class MFRCDatasetForTransformers(TokenizationMixin, MFRC):
    """Dataset with encodings for `transformers`
    for `MFRC`"""

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        return MFRC.argparse_args() | TokenizationMixin.argparse_args()

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dict_tokenize(self.examples, self.preprocessor)


class MMLUProDataset(MMLUPro):
    """Plain text dataset for `MMLUPro`. Class uses
    from_namespace decorator, so it should not be used for inheritance.

    Attributes: Check `MMLUPro`.
    """

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class MMLUProDatasetForTransformers(TokenizationMixin, MMLUPro):
    """Dataset with encodings for `transformers`
    for `MMLUPro`"""

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        return MMLUPro.argparse_args() | TokenizationMixin.argparse_args()

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dict_tokenize(self.examples, self.preprocessor)


class BoxesDataset(Boxes):
    """Plain text dataset for `Boxes`. Class uses
    from_namespace decorator, so it should not be used for inheritance.

    Attributes: Check `Boxes`.
    """

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class BoxesDatasetForTransformers(TokenizationMixin, Boxes):
    """Dataset with encodings for `transformers` for `Boxes`"""

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        return Boxes.argparse_args() | TokenizationMixin.argparse_args()

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dict_tokenize(self.examples, self.preprocessor)


class MSPPodcastDataset(MSPPodcast):
    """Plain text dataset for `MSPPodcast`. Class uses"
    from_namespace decorator, so it should not be used for inheritance.

    Attributes: Check `MSPPodcast`.
    """

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class MSPPodcastDatasetForTransformers(TokenizationMixin, MSPPodcast):
    """Dataset with encodings for `transformers` for `MSPPodcast`"""

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        return MSPPodcast.argparse_args() | TokenizationMixin.argparse_args()

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dict_tokenize(self.examples, self.preprocessor)


class QueerReclaimLexDataset(QueerReclaimLex):
    """Plain text dataset for `QueerReclaimLex`. Class uses
    from_namespace decorator, so it should not be used for inheritance.

    Attributes: Check `QueerReclaimLex`.
    """

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class QueerReclaimLexDatasetForTransformers(TokenizationMixin, QueerReclaimLex):
    """Dataset with encodings for `transformers` for `QueerReclaimLex`"""

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        return (
            QueerReclaimLex.argparse_args() | TokenizationMixin.argparse_args()
        )

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dict_tokenize(self.examples, self.preprocessor)


class HatexplainDataset(Hatexplain):
    """Plain text dataset for `Hatexplain`. Class uses
    from_namespace decorator, so it should not be used for inheritance.

    Attributes: Check `Hatexplain`.
    """

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class HatexplainDatasetForTransformers(TokenizationMixin, Hatexplain):
    """Dataset with encodings for `transformers` for `Hatexplain`"""

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        return Hatexplain.argparse_args() | TokenizationMixin.argparse_args()

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dict_tokenize(self.examples, self.preprocessor)


class TRECDataset(TREC):
    """Plain text dataset for `TREC`. Class uses
    from_namespace decorator, so it should not be used for inheritance.

    Attributes: Check `TREC`.
    """

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class TRECDatasetForTransformers(TokenizationMixin, TREC):
    """Dataset with encodings for `transformers` for `TREC`"""

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        return TREC.argparse_args() | TokenizationMixin.argparse_args()

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dict_tokenize(self.examples, self.preprocessor)
