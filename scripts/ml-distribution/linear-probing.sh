python scripts/prompting/llm_prompting_clsf.py \
    GoEmotions --gridparse-config ./configs/GoEmotions/config.yaml \
    --test-split train --train-split dev --max-new-tokens 10 \
    --model-name-or-path meta-llama/Llama-3.1-8B-Instruct \
    --shot 10 --label-format json --annotation-mode aggregate --load-in-4bit \
    --alternative {model_name_or_path}-lb-{shot}-shot --seed 0 --linear-probing

python scripts/prompting/llm_prompting_clsf.py \
    MFRC --gridparse-config ./configs/MFRC/config.yaml \
    --test-split train --train-split dev --max-new-tokens 10 \
    --model-name-or-path meta-llama/Llama-3.1-8B-Instruct \
    --shot 10 --label-format json --annotation-mode aggregate --load-in-4bit \
    --alternative {model_name_or_path}-lb-{shot}-shot --seed 0 --linear-probing

python scripts/prompting/llm_prompting_clsf.py \
    SemEval --gridparse-config ./configs/SemEval/config.yaml \
    --test-split train --train-split dev --max-new-tokens 10 \
    --model-name-or-path meta-llama/Llama-3.1-8B-Instruct \
    --shot 10 --label-format json --annotation-mode aggregate --load-in-4bit \
    --alternative {model_name_or_path}-lb-{shot}-shot --seed 0 --linear-probing

# get dev and test features

python scripts/prompting/llm_prompting_clsf.py \
    GoEmotions --gridparse-config ./configs/GoEmotions/config.yaml \
    --test-split dev test --train-split train --max-new-tokens 10 \
    --model-name-or-path meta-llama/Llama-3.1-8B-Instruct \
    --shot 10 --label-format json --annotation-mode aggregate --load-in-4bit \
    --alternative {model_name_or_path}-lb-eval-{shot}-shot --seed 0 --linear-probing

python scripts/prompting/llm_prompting_clsf.py \
    MFRC --gridparse-config ./configs/MFRC/config.yaml \
    --test-split dev test --train-split train --max-new-tokens 10 \
    --model-name-or-path meta-llama/Llama-3.1-8B-Instruct \
    --shot 10 --label-format json --annotation-mode aggregate --load-in-4bit \
    --alternative {model_name_or_path}-lb-eval-{shot}-shot --seed 0 --linear-probing

python scripts/prompting/llm_prompting_clsf.py \
    SemEval --gridparse-config ./configs/SemEval/config.yaml \
    --test-split dev test --train-split train --max-new-tokens 10 \
    --model-name-or-path meta-llama/Llama-3.1-8B-Instruct \
    --shot 10 --label-format json --annotation-mode aggregate --load-in-4bit \
    --alternative {model_name_or_path}-lb-eval-{shot}-shot --seed 0 --linear-probing

# 8b instruct
python scripts/ml-distribution/train_probes.py \
    --folder logs/GoEmotions/meta-llama--Llama-3.1-8B-Instruct-lb-10-shot_1 \
    logs/MFRC/meta-llama--Llama-3.1-8B-Instruct-lb-10-shot_0 \
    logs/SemEval/meta-llama--Llama-3.1-8B-Instruct-lb-10-shot_1 \
    --eval-folder logs/GoEmotions/meta-llama--Llama-3.1-8B-Instruct-lb-eval-10-shot_0 \
    logs/MFRC/meta-llama--Llama-3.1-8B-Instruct-lb-eval-10-shot_0 \
    logs/SemEval/meta-llama--Llama-3.1-8B-Instruct-lb-eval-10-shot_0 \
    --output logs/analysis/ml-distr/GoEmotions-lp/linear-probing-3.1-8B-Instruct \
    logs/analysis/ml-distr/MFRC-lp/linear-probing-3.1-8B-Instruct \
    logs/analysis/ml-distr/SemEval-lp/linear-probing-3.1-8B-Instruct

# 70b instruct
python scripts/ml-distribution/train_probes.py \
    --folder logs/GoEmotions/meta-llama--Llama-3.3-70B-Instruct-lb-10-shot_0 \
    logs/MFRC/meta-llama--Llama-3.3-70B-Instruct-lb-10-shot_0 \
    logs/SemEval/meta-llama--Llama-3.3-70B-Instruct-lb-10-shot_0 \
    --eval-folder logs/GoEmotions/meta-llama--Llama-3.3-70B-Instruct-lb-eval-10-shot_0 \
    logs/MFRC/meta-llama--Llama-3.3-70B-Instruct-lb-eval-10-shot_0 \
    logs/SemEval/meta-llama--Llama-3.3-70B-Instruct-lb-eval-10-shot_0 \
    --output logs/analysis/ml-distr/GoEmotions-lp/linear-probing-3.3-70B-Instruct \
    logs/analysis/ml-distr/MFRC-lp/linear-probing-3.3-70B-Instruct \
    logs/analysis/ml-distr/SemEval-lp/linear-probing-3.3-70B-Instruct

# 8b base
python scripts/ml-distribution/train_probes.py \
    --folder logs/GoEmotions/meta-llama--Llama-3.1-8B-lb-10-shot_2 \
    logs/MFRC/meta-llama--Llama-3.1-8B-lb-10-shot_0 \
    logs/SemEval/meta-llama--Llama-3.1-8B-lb-10-shot_1 \
    --output logs/analysis/ml-distr/GoEmotions-lp/linear-probing-3.1-8B \
    logs/analysis/ml-distr/MFRC-lp/linear-probing-3.3-8B \
    logs/analysis/ml-distr/SemEval-lp/linear-probing-3.3-8B

# 70b base
python scripts/ml-distribution/train_probes.py \
    --folder logs/GoEmotions/meta-llama--Llama-3.1-70B-lb-10-shot_0 \
    logs/MFRC/meta-llama--Llama-3.1-70B-lb-10-shot_0 \
    logs/SemEval/meta-llama--Llama-3.1-70B-lb-10-shot_0 \
    --output logs/analysis/ml-distr/GoEmotions-lp/linear-probing-3.1-70B \
    logs/analysis/ml-distr/MFRC-lp/linear-probing-3.1-70B \
    logs/analysis/ml-distr/SemEval-lp/linear-probing-3.1-70B

# boxes
python scripts/ml-distribution/train_probes.py \
    --folder logs/Boxes/meta-llama--Llama-3.1-8B-attn-5-shot_0 \
    logs/Boxes/meta-llama--Llama-3.1-8B-Instruct-attn-5-shot_0 \
    logs/Boxes/meta-llama--Llama-3.1-70B-lb-4-shot_0 \
    logs/Boxes/meta-llama--Llama-3.3-70B-Instruct-lb-4-shot_0 \
    --output logs/analysis/ml-distr/Boxes-lp/linear-probing-3-8B \
    logs/analysis/ml-distr/Boxes-lp/linear-probing-3-8B-Instruct \
    logs/analysis/ml-distr/Boxes-lp/linear-probing-3-70B \
    logs/analysis/ml-distr/Boxes-lp/linear-probing-3-70B-Instruct 
