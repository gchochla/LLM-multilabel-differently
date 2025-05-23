# 8B instruct

python scripts/prompting/llm_prompting_clsf.py \
    SemEval --gridparse-config ./configs/SemEval/config.yaml \
    --test-split train --train-split dev --max-new-tokens 21 \
    --model-name-or-path meta-llama/Llama-3.1-8B \
    --shot 10 --label-format json --annotation-mode aggregate --load-in-4bit \
    --alternative {model_name_or_path}-attn-{shot}-shot --seed 0 --linear-probing

python scripts/prompting/llm_prompting_clsf.py \
    GoEmotions --gridparse-config ./configs/GoEmotions/config.yaml \
    --test-split train --train-split dev --max-new-tokens 21 \
    --model-name-or-path meta-llama/Llama-3.1-8B \
    --shot 10 --label-format json --annotation-mode aggregate --load-in-4bit \
    --alternative {model_name_or_path}-attn-{shot}-shot --seed 0 --linear-probing

python scripts/prompting/llm_prompting_clsf.py \
    MFRC --gridparse-config ./configs/MFRC/config.yaml \
    --test-split train --train-split dev --max-new-tokens 21 \
    --model-name-or-path meta-llama/Llama-3.1-8B \
    --shot 10 --label-format json --annotation-mode aggregate --load-in-4bit \
    --alternative {model_name_or_path}-attn-{shot}-shot --seed 0 --linear-probing

python scripts/prompting/llm_prompting_clsf.py \
    Boxes --gridparse-config ./configs/Boxes/config.yaml \
    --test-split train --train-split dev --max-new-tokens 21 \
    --model-name-or-path meta-llama/Llama-3.1-8B \
    --shot 5 --annotation-mode aggregate --load-in-4bit \
    --alternative {model_name_or_path}-attn-{shot}-shot --seed 0 --linear-probing


# 8B base

python scripts/prompting/llm_prompting_clsf.py \
    SemEval --gridparse-config ./configs/SemEval/config.yaml \
    --test-split train --train-split dev --max-new-tokens 20 \
    --model-name-or-path meta-llama/Llama-3.1-8B \
    --shot 10 --label-format json --annotation-mode aggregate --load-in-4bit \
    --alternative {model_name_or_path}-lb-{shot}-shot --seed 0 --linear-probing

python scripts/prompting/llm_prompting_clsf.py \
    GoEmotions --gridparse-config ./configs/GoEmotions/config.yaml \
    --test-split train --train-split dev --max-new-tokens 20 \
    --model-name-or-path meta-llama/Llama-3.1-8B \
    --shot 10 --label-format json --annotation-mode aggregate --load-in-4bit \
    --alternative {model_name_or_path}-lb-{shot}-shot --seed 0 --linear-probing

python scripts/prompting/llm_prompting_clsf.py \
    MFRC --gridparse-config ./configs/MFRC/config.yaml \
    --test-split train --train-split dev --max-new-tokens 20 \
    --model-name-or-path meta-llama/Llama-3.1-8B \
    --shot 10 --label-format json --annotation-mode aggregate --load-in-4bit \
    --alternative {model_name_or_path}-lb-{shot}-shot --seed 0 --linear-probing

# 70B instruct
python scripts/prompting/llm_prompting_clsf.py \
    SemEval --gridparse-config ./configs/SemEval/config.yaml \
    --test-split train --train-split dev --max-new-tokens 20 \
    --model-name-or-path meta-llama/Llama-3.3-70B-Instruct \
    --shot 10 --label-format json --annotation-mode aggregate --load-in-4bit \
    --alternative {model_name_or_path}-lb-{shot}-shot --seed 0 --linear-probing

python scripts/prompting/llm_prompting_clsf.py \
    GoEmotions --gridparse-config ./configs/GoEmotions/config.yaml \
    --test-split train --train-split dev --max-new-tokens 20 \
    --model-name-or-path meta-llama/Llama-3.3-70B-Instruct \
    --shot 10 --label-format json --annotation-mode aggregate --load-in-4bit \
    --alternative {model_name_or_path}-lb-{shot}-shot --seed 0 --linear-probing

python scripts/prompting/llm_prompting_clsf.py \
    MFRC --gridparse-config ./configs/MFRC/config.yaml \
    --test-split train --train-split dev --max-new-tokens 20 \
    --model-name-or-path meta-llama/Llama-3.3-70B-Instruct \
    --shot 10 --label-format json --annotation-mode aggregate --load-in-4bit \
    --alternative {model_name_or_path}-lb-{shot}-shot --seed 0 --linear-probing

