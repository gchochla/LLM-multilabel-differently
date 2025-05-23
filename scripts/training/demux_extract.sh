#!/bin/bash

python scripts/training/extract_scores.py SemEval \
    --gridparse-config ./configs/SemEval/config.yaml \
    --experiment-name aggregate,None,None,None_1

python scripts/training/extract_scores.py GoEmotions \
    --gridparse-config ./configs/GoEmotions/config.yaml \
    --experiment-name aggregate,None,None,None_1

python scripts/training/extract_scores.py MFRC \
    --gridparse-config ./configs/MFRC/config.yaml \
    --experiment-name aggregate,None,None,None_0

python scripts/training/extract_scores.py Hatexplain \
    --gridparse-config ./configs/Hatexplain/config.yaml \
    --experiment-name aggregate,None,None,None_1

python scripts/training/extract_scores.py MSPPodcast \
    --gridparse-config ./configs/MSPPodcast/config.yaml \
    --experiment-name aggregate,None,None,None_2