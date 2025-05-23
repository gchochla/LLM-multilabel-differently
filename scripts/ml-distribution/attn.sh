python scripts/ml-distribution/calc_attn_weights.py \
    --experiment logs/SemEval/meta-llama--Llama-3.1-8B-attn-10-shot_0 \
    logs/SemEval/meta-llama--Llama-3.1-8B-Instruct-attn-10-shot_0 \
    logs/MFRC/meta-llama--Llama-3.1-8B-attn-10-shot_0 \
    logs/MFRC/meta-llama--Llama-3.1-8B-Instruct-attn-10-shot_0 \
    logs/GoEmotions/meta-llama--Llama-3.1-8B-attn-10-shot_0 \
    logs/GoEmotions/meta-llama--Llama-3.1-8B-Instruct-attn-10-shot_0 \
    logs/Boxes/meta-llama--Llama-3.1-8B-attn-5-shot_0 \
    logs/Boxes/meta-llama--Llama-3.1-8B-Instruct-attn-5-shot_0 \
    --output logs/analysis/ml-distr/SemEval-attn/3-8B \
    logs/analysis/ml-distr/SemEval-attn/3-8B-Instruct \
    logs/analysis/ml-distr/MFRC-attn/3-8B \
    logs/analysis/ml-distr/MFRC-attn/3-8B-Instruct \
    logs/analysis/ml-distr/GoEmotions-attn/3-8B \
    logs/analysis/ml-distr/GoEmotions-attn/3-8B-Instruct \
    logs/analysis/ml-distr/Boxes-attn/3-8B \
    logs/analysis/ml-distr/Boxes-attn/3-8B-Instruct
