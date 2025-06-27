while getopts c: flag
do
    case "${flag}" in
        c) cuda=${OPTARG};;
    esac
done

# model=meta-llama/Llama-2-13b-chat-hf
# model=meta-llama/Llama-2-7b-chat-hf
# model=meta-llama/Llama-3.2-1B
# model=meta-llama/Llama-3.1-8B
model=meta-llama/Llama-3.3-70B-Instruct

# override from command line, if provided
model=${4:-$model}

export CUDA_VISIBLE_DEVICES="$3"

id_list=$2

id_file=prob_distr_ids/GoEmotions/$id_list.txt

seed=${5:-0}

echo Using model $model
echo Evaluating distribution type $1
echo Testing on IDs in $id_file
echo Running on GPU $3

if [ "$5" == "vllm" ]; then
    echo Using VLLM

    python scripts/prob_distr/vllm_prob_distr.py \
        GoEmotions \
        --distribution $1 \
        --root-dir /data1/chochlak/goemotions \
        --emotion-clustering-json /data1/chochlak/goemotions/emotion_clustering.json \
        --train-split train \
        --test-split dev \
        --system ' ' \
        --instruction $'Classify the following inputs into none, one, or multiple the following emotions per input: {labels}. Output exactly these emotions and no others.\n' \
        --incontext $'Input: {text}\n{label}\n' \
        --model-name-or-path $model \
        --label-format json \
        --max-new-tokens 18 \
        --accelerate \
        --logging-level debug \
        --annotation-mode aggregate \
        --text-preprocessor false \
        --sampling-strategy multilabel \
        --sentence-model all-mpnet-base-v2 \
        --seed $seed \
        --shot 10 \
        --alternative $id_list/{distribution}/{model_name_or_path} \
        --test-ids-filename $id_file

else
    echo Using HuggingFace

    python scripts/prob_distr/llm_prob_distr.py \
        GoEmotions \
        --distribution $1 \
        --root-dir /data1/chochlak/goemotions \
        --emotion-clustering-json /data1/chochlak/goemotions/emotion_clustering.json \
        --train-split train \
        --test-split dev \
        --system ' ' \
        --instruction $'Classify the following inputs into none, one, or multiple the following emotions per input: {labels}. Output exactly these emotions and no others.\n' \
        --incontext $'Input: {text}\n{label}\n' \
        --model-name-or-path $model \
        --label-format json \
        --max-new-tokens 18 \
        --accelerate \
        --logging-level debug \
        --annotation-mode aggregate \
        --text-preprocessor false \
        --load-in-4bit \
        --sampling-strategy multilabel \
        --sentence-model all-mpnet-base-v2 \
        --seed $seed \
        --shot 10 \
        --alternative $id_list/{distribution}/{model_name_or_path} \
        --test-ids-filename $id_file

fi