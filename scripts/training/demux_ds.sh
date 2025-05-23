#!/bin/bash

python scripts/training/demux.py GoEmotions \
    --gridparse-config ./configs/GoEmotions/config.yaml \
    --train-split train --dev-split dev --test-split dev test \
    --num-train-epoch 20 --max-length 128 --accelerate --warmup 0.1 \
    --model-name bert-base-uncased --text-preprocessor true --seed 0 \
    --train-batch-size 128 --eval-batch-size 256 --early-stopping-patience 3 \
    --intra-loss-coef 0 0.2

python scripts/training/demux.py MFRC \
    --gridparse-config ./configs/MFRC/config.yaml \
    --train-split train --dev-split dev --test-split dev test \
    --num-train-epoch 20 --max-length 128 --accelerate --warmup 0.1 \
    --model-name bert-base-uncased --text-preprocessor true --seed 0 \
    --train-batch-size 128 --eval-batch-size 256 --early-stopping-patience 3 \
    --intra-loss-coef 0 0.2

python scripts/training/demux.py SemEval \
    --gridparse-config ./configs/SemEval/config.yaml \
    --train-split train --dev-split dev --test-split dev test \
    --num-train-epoch 30 --max-length 64 --accelerate --warmup 0.1 \
    --model-name bert-base-uncased --text-preprocessor true --seed 0 \
    --train-batch-size 128 --eval-batch-size 256 --early-stopping-patience 3 \
    --intra-loss-coef 0 0.2

python scripts/training/demux.py Hatexplain \
    --gridparse-config ./configs/Hatexplain/config.yaml \
    --train-split train --dev-split dev --test-split dev test \
    --num-train-epoch 30 --max-length 64 --accelerate --warmup 0.1 \
    --model-name bert-base-uncased --text-preprocessor true --seed 0 \
    --train-batch-size 128 --eval-batch-size 256 --early-stopping-patience 3 --early-stopping-metric f1

python scripts/training/demux.py MSPPodcast \
    --gridparse-config ./configs/MSPPodcast/config.yaml \
    --train-split train --dev-split dev --test-split dev test \
    --num-train-epoch 30 --max-length 64 --accelerate --warmup 0.1 \
    --model-name bert-base-uncased --text-preprocessor true --seed 0 \
    --train-batch-size 128 --eval-batch-size 256 --early-stopping-patience 3 --early-stopping-metric f1
