# 8b

python scripts/ml-distribution/contrast_distributions_plot.py \
    --exp ./logs/MFRC/meta-llama--Llama-3.1-8B-Instruct-lb-10-shot_0/ \
    --out logs/analysis/ml-distr/MFRC --name "Llama3 8B Instruct on MFRC" --union --max-rank 3

python scripts/ml-distribution/contrast_distributions_plot.py \
    --exp ./logs/SemEval/meta-llama--Llama-3.1-8B-Instruct-lb-10-shot_1/ \
    --out logs/analysis/ml-distr/SemEval --name "Llama3 8B Instruct on SemEval 2018 Task 1" --max-rank 3

python scripts/ml-distribution/contrast_distributions_plot.py \
    --exp ./logs/GoEmotions/meta-llama--Llama-3.1-8B-Instruct-attn-10-shot_0/ \
    --out logs/analysis/ml-distr/GoEmotions --name "Llama3 8B Instruct on GoEmotions" --max-rank 3

python scripts/ml-distribution/contrast_distributions_plot.py \
    --exp ./logs/Boxes/meta-llama--Llama-3.1-8B-Instruct-lb-4-shot_0/ \
    --out logs/analysis/ml-distr/Boxes --name "Llama3 8B Instruct on Boxes" --max-rank 3 #4

# SFT

python scripts/ml-distribution/contrast_distributions_plot.py \
    --exp ./logs/MFRC/SFT-Llama-3.1-8B-Instruct-lb-0-shot_0/ \
    --out logs/analysis/ml-distr/MFRC --name "SFT Llama3 8B Instruct 0-shot on MFRC" --union --max-rank 3

python scripts/ml-distribution/contrast_distributions_plot.py \
    --exp ./logs/SemEval/SFT-Llama-3.1-8B-Instruct-lb-0-shot_0/ \
    --out logs/analysis/ml-distr/SemEval --name "SFT Llama3 8B Instruct 0-shot on SemEval 2018 Task 1" --max-rank 3

python scripts/ml-distribution/contrast_distributions_plot.py \
    --exp ./logs/GoEmotions/SFT-Llama-3.1-8B-Instruct-lb-0-shot_0/ \
    --out logs/analysis/ml-distr/GoEmotions --name "SFT Llama3 8B Instruct 0-shot on GoEmotions" --max-rank 2

python scripts/ml-distribution/contrast_distributions_plot.py \
    --exp ./logs/MFRC/SFT-Llama-3.1-8B-Instruct-lb-10-shot_0/ \
    --out logs/analysis/ml-distr/MFRC --name "SFT Llama3 8B Instruct 10-shot on MFRC" --union --max-rank 3

python scripts/ml-distribution/contrast_distributions_plot.py \
    --exp ./logs/SemEval/SFT-Llama-3.1-8B-Instruct-lb-10-shot_0/ \
    --out logs/analysis/ml-distr/SemEval --name "SFT Llama3 8B Instruct 10-shot on SemEval 2018 Task 1" --max-rank 3

python scripts/ml-distribution/contrast_distributions_plot.py \
    --exp ./logs/GoEmotions/SFT-Llama-3.1-8B-Instruct-lb-10-shot_0/ \
    --out logs/analysis/ml-distr/GoEmotions --name "SFT Llama3 8B Instruct 10-shot on GoEmotions" --max-rank 3

# 8b base

python scripts/ml-distribution/contrast_distributions_plot.py \
    --exp ./logs/MFRC/meta-llama--Llama-3.1-8B-lb-10-shot_0/ \
    --out logs/analysis/ml-distr/MFRC --name "Llama3 8B Base on MFRC" --max-rank 2

python scripts/ml-distribution/contrast_distributions_plot.py \
    --exp ./logs/SemEval/meta-llama--Llama-3.1-8B-lb-10-shot_1/ \
    --out logs/analysis/ml-distr/SemEval --name "Llama3 8B Base on SemEval 2018 Task 1" --max-rank 3 #5

python scripts/ml-distribution/contrast_distributions_plot.py \
    --exp ./logs/GoEmotions/meta-llama--Llama-3.1-8B-attn-10-shot_0/ \
    --out logs/analysis/ml-distr/GoEmotions --name "Llama3 8B Base on GoEmotions" --max-rank 2

# 70b

python scripts/ml-distribution/contrast_distributions_plot.py \
    --exp ./logs/MFRC/meta-llama--Llama-3.3-70B-Instruct-lb-10-shot_0/ \
    --out logs/analysis/ml-distr/MFRC --name "Llama3 70B Instruct on MFRC" --union --max-rank 3

python scripts/ml-distribution/contrast_distributions_plot.py \
    --exp ./logs/SemEval/meta-llama--Llama-3.3-70B-Instruct-lb-10-shot_0/ \
    --out logs/analysis/ml-distr/SemEval --name "Llama3 70B Instruct on SemEval 2018 Task 1" --max-rank 3

python scripts/ml-distribution/contrast_distributions_plot.py \
    --exp ./logs/GoEmotions/meta-llama--Llama-3.3-70B-Instruct-lb-10-shot_0/ \
    --out logs/analysis/ml-distr/GoEmotions --name "Llama3 70B Instruct on GoEmotions" --max-rank 2

python scripts/ml-distribution/contrast_distributions_plot.py \
    --exp ./logs/Boxes/meta-llama--Llama-3.3-70B-Instruct-lb-4-shot_0/ \
    --out logs/analysis/ml-distr/Boxes --name "Llama3 70B Instruct on Boxes" --max-rank 3 #4

# 70b base

python scripts/ml-distribution/contrast_distributions_plot.py \
    --exp ./logs/MFRC/meta-llama--Llama-3.1-70B-lb-10-shot_0/ \
    --out logs/analysis/ml-distr/MFRC --name "Llama3 70B Base on MFRC" --union --max-rank 2

python scripts/ml-distribution/contrast_distributions_plot.py \
    --exp ./logs/SemEval/meta-llama--Llama-3.1-70B-lb-10-shot_0/ \
    --out logs/analysis/ml-distr/SemEval --name "Llama3 70B Base on SemEval 2018 Task 1" --max-rank 3

python scripts/ml-distribution/contrast_distributions_plot.py \
    --exp ./logs/GoEmotions/meta-llama--Llama-3.1-70B-lb-10-shot_0/ \
    --out logs/analysis/ml-distr/GoEmotions --name "Llama3 70B Base on GoEmotions" --max-rank 1


# combined

python scripts/ml-distribution/contrast_distributions_plot_group.py \
    --exp ./logs/SemEval/meta-llama--Llama-3.1-8B-lb-10-shot_1/ \
    ./logs/SemEval/meta-llama--Llama-3.1-8B-Instruct-lb-10-shot_1/ \
    ./logs/SemEval/SFT-Llama-3.1-8B-Instruct-lb-10-shot_0/ \
    ./logs/SemEval/meta-llama--Llama-3.1-70B-lb-10-shot_0/ \
    ./logs/SemEval/meta-llama--Llama-3.3-70B-Instruct-lb-10-shot_0/ \
    ./logs/SemEval/SFT-Llama-3.3-70B-Instruct-10-shot_0/ \
    --out logs/analysis/ml-distr/SemEval-all \
    --name "Llama3 8B Base" "Llama3 8B Instruct" "SFT Llama3 8B Instruct" \
    "Llama3 70B Base" "Llama3 70B Instruct" "SFT Llama3 70B Instruct" \
    --max-rank 2 --dataset "SemEval 2018 Task 1" --no-box --no-x

python scripts/ml-distribution/contrast_distributions_plot_group.py \
    --exp ./logs/GoEmotions/meta-llama--Llama-3.1-8B-attn-10-shot_0/ \
    ./logs/GoEmotions/meta-llama--Llama-3.1-8B-Instruct-attn-10-shot_0/ \
    ./logs/GoEmotions/SFT-Llama-3.1-8B-Instruct-lb-10-shot_0/ \
    ./logs/GoEmotions/meta-llama--Llama-3.1-70B-lb-10-shot_0/ \
    ./logs/GoEmotions/meta-llama--Llama-3.3-70B-Instruct-lb-10-shot_0/ \
    ./logs/GoEmotions/SFT-Llama-3.3-70B-Instruct-10-shot_0/ \
    --out logs/analysis/ml-distr/GoEmotions-all \
    --name "Llama3 8B Base" "Llama3 8B Instruct" "SFT Llama3 8B Instruct" \
    "Llama3 70B Base" "Llama3 70B Instruct" "SFT Llama3 70B Instruct" \
    --max-rank 2 2 2 1 2 2 --dataset "GoEmotions" --no-legend --no-box --no-x

python scripts/ml-distribution/contrast_distributions_plot_group.py \
    --exp ./logs/MFRC/meta-llama--Llama-3.1-8B-lb-10-shot_0/ \
    ./logs/MFRC/meta-llama--Llama-3.1-8B-Instruct-lb-10-shot_0/ \
    ./logs/MFRC/SFT-Llama-3.1-8B-Instruct-lb-10-shot_0/ \
    ./logs/MFRC/meta-llama--Llama-3.1-70B-lb-10-shot_0/ \
    ./logs/MFRC/meta-llama--Llama-3.3-70B-Instruct-lb-10-shot_0/ \
    ./logs/MFRC/SFT-Llama-3.3-70B-Instruct-10-shot_0/ \
    --out logs/analysis/ml-distr/MFRC-all \
    --name "Llama3 8B Base" "Llama3 8B Instruct" "SFT Llama3 8B Instruct" \
    "Llama3 70B Base" "Llama3 70B Instruct" "SFT Llama3 70B Instruct" \
    --max-rank 2 --dataset "MFRC" --no-legend --no-box --no-x

python scripts/ml-distribution/contrast_distributions_plot_group.py \
    --exp ./logs/Boxes/meta-llama--Llama-3.1-8B-attn-5-shot_0/ \
    ./logs/Boxes/meta-llama--Llama-3.1-8B-Instruct-attn-5-shot_0/ \
    ./logs/Boxes/meta-llama--Llama-3.1-70B-lb-4-shot_0/ \
    ./logs/Boxes/meta-llama--Llama-3.3-70B-Instruct-lb-4-shot_0/ \
    --out logs/analysis/ml-distr/Boxes-all \
    --name "Llama3 8B Base" "Llama3 8B Instruct" \
    "Llama3 70B Base" "Llama3 70B Instruct" \
    --max-rank 2 --dataset "Boxes" --no-legend --no-box

# only box plot

python scripts/ml-distribution/contrast_distributions_plot_group.py \
    --exp ./logs/SemEval/meta-llama--Llama-3.1-8B-lb-10-shot_1/ \
    ./logs/SemEval/meta-llama--Llama-3.1-8B-Instruct-lb-10-shot_1/ \
    ./logs/SemEval/SFT-Llama-3.1-8B-Instruct-lb-10-shot_0/ \
    ./logs/SemEval/meta-llama--Llama-3.1-70B-lb-10-shot_0/ \
    ./logs/SemEval/meta-llama--Llama-3.3-70B-Instruct-lb-10-shot_0/ \
    ./logs/SemEval/SFT-Llama-3.3-70B-Instruct-10-shot_0/ \
    --out logs/analysis/ml-distr/SemEval-all-no-violin \
    --name "Llama3 8B Base" "Llama3 8B Instruct" "SFT Llama3 8B Instruct" \
    "Llama3 70B Base" "Llama3 70B Instruct" "SFT Llama3 70B Instruct" \
    --max-rank 2 --dataset "SemEval 2018 Task 1" --no-violin --no-x

python scripts/ml-distribution/contrast_distributions_plot_group.py \
    --exp ./logs/GoEmotions/meta-llama--Llama-3.1-8B-attn-10-shot_0/ \
    ./logs/GoEmotions/meta-llama--Llama-3.1-8B-Instruct-attn-10-shot_0/ \
    ./logs/GoEmotions/SFT-Llama-3.1-8B-Instruct-lb-10-shot_0/ \
    ./logs/GoEmotions/meta-llama--Llama-3.1-70B-lb-10-shot_0/ \
    ./logs/GoEmotions/meta-llama--Llama-3.3-70B-Instruct-lb-10-shot_0/ \
    ./logs/GoEmotions/SFT-Llama-3.3-70B-Instruct-10-shot_0/ \
    --out logs/analysis/ml-distr/GoEmotions-all-no-violin \
    --name "Llama3 8B Base" "Llama3 8B Instruct" "SFT Llama3 8B Instruct" \
    "Llama3 70B Base" "Llama3 70B Instruct" "SFT Llama3 70B Instruct" \
    --max-rank 2 2 2 1 2 2 --dataset "GoEmotions" --no-legend --no-violin --no-x

python scripts/ml-distribution/contrast_distributions_plot_group.py \
    --exp ./logs/MFRC/meta-llama--Llama-3.1-8B-lb-10-shot_0/ \
    ./logs/MFRC/meta-llama--Llama-3.1-8B-Instruct-lb-10-shot_0/ \
    ./logs/MFRC/SFT-Llama-3.1-8B-Instruct-lb-10-shot_0/ \
    ./logs/MFRC/meta-llama--Llama-3.1-70B-lb-10-shot_0/ \
    ./logs/MFRC/meta-llama--Llama-3.3-70B-Instruct-lb-10-shot_0/ \
    ./logs/MFRC/SFT-Llama-3.3-70B-Instruct-10-shot_0/ \
    --out logs/analysis/ml-distr/MFRC-all-no-violin \
    --name "Llama3 8B Base" "Llama3 8B Instruct" "SFT Llama3 8B Instruct" \
    "Llama3 70B Base" "Llama3 70B Instruct" "SFT Llama3 70B Instruct" \
    --max-rank 2 --dataset "MFRC" --no-legend --no-violin --no-x

python scripts/ml-distribution/contrast_distributions_plot_group.py \
    --exp ./logs/Boxes/meta-llama--Llama-3.1-8B-attn-5-shot_0/ \
    ./logs/Boxes/meta-llama--Llama-3.1-8B-Instruct-attn-5-shot_0/ \
    ./logs/Boxes/meta-llama--Llama-3.1-70B-lb-4-shot_0/ \
    ./logs/Boxes/meta-llama--Llama-3.3-70B-Instruct-lb-4-shot_0/ \
    --out logs/analysis/ml-distr/Boxes-all-no-violin \
    --name "Llama3 8B Base" "Llama3 8B Instruct" \
    "Llama3 70B Base" "Llama3 70B Instruct" \
    --max-rank 2 --dataset "Boxes" --no-legend --no-violin

# python scripts/ml-distribution/contrast_distributions_plot_group.py \
#     --exp ./logs/Boxes/meta-llama--Llama-3.1-8B-Instruct-attn-5-shot_0/ \
#     ./logs/Boxes/meta-llama--Llama-3.1-8B-attn-5-shot_0/ \
#     ./logs/Boxes/meta-llama--Llama-3.3-70B-Instruct-lb-4-shot_0/ \
#     ./logs/Boxes/meta-llama--Llama-3.1-70B-lb-4-shot_0/ \
#     --out logs/analysis/ml-distr/Boxes-all \
#     --name "Llama3 8B Instruct" "Llama3 8B Base" \
#     "SFT Llama3 8B Instruct" "Llama3 70B Instruct" "Llama3 70B Base" \
#     --max-rank 2 --dataset "Boxes"

# only top and only bottom

python scripts/ml-distribution/contrast_distributions_plot_group_singles.py \
    --exp ./logs/SemEval/meta-llama--Llama-3.1-8B-lb-10-shot_1/ \
    ./logs/SemEval/meta-llama--Llama-3.1-8B-Instruct-lb-10-shot_1/ \
    ./logs/SemEval/SFT-Llama-3.1-8B-Instruct-lb-10-shot_0/ \
    ./logs/SemEval/meta-llama--Llama-3.1-70B-lb-10-shot_0/ \
    ./logs/SemEval/meta-llama--Llama-3.3-70B-Instruct-lb-10-shot_0/ \
    ./logs/SemEval/SFT-Llama-3.3-70B-Instruct-10-shot_0/ \
    --out logs/analysis/ml-distr/SemEval-all \
    --name "Llama3 8B Base" "Llama3 8B Instruct" "SFT Llama3 8B Instruct" \
    "Llama3 70B Base" "Llama3 70B Instruct" "SFT Llama3 70B Instruct" \
    --max-rank 2 --dataset "SemEval 2018 Task 1" --no-box --no-x

python scripts/ml-distribution/contrast_distributions_plot_group_singles.py \
    --exp ./logs/GoEmotions/meta-llama--Llama-3.1-8B-attn-10-shot_0/ \
    ./logs/GoEmotions/meta-llama--Llama-3.1-8B-Instruct-attn-10-shot_0/ \
    ./logs/GoEmotions/SFT-Llama-3.1-8B-Instruct-lb-10-shot_0/ \
    ./logs/GoEmotions/meta-llama--Llama-3.1-70B-lb-10-shot_0/ \
    ./logs/GoEmotions/meta-llama--Llama-3.3-70B-Instruct-lb-10-shot_0/ \
    ./logs/GoEmotions/SFT-Llama-3.3-70B-Instruct-10-shot_0/ \
    --out logs/analysis/ml-distr/GoEmotions-all \
    --name "Llama3 8B Base" "Llama3 8B Instruct" "SFT Llama3 8B Instruct" \
    "Llama3 70B Base" "Llama3 70B Instruct" "SFT Llama3 70B Instruct" \
    --max-rank 2 2 2 1 2 2 --dataset "GoEmotions" --no-legend --no-box --no-x

python scripts/ml-distribution/contrast_distributions_plot_group_singles.py \
    --exp ./logs/MFRC/meta-llama--Llama-3.1-8B-lb-10-shot_0/ \
    ./logs/MFRC/meta-llama--Llama-3.1-8B-Instruct-lb-10-shot_0/ \
    ./logs/MFRC/SFT-Llama-3.1-8B-Instruct-lb-10-shot_0/ \
    ./logs/MFRC/meta-llama--Llama-3.1-70B-lb-10-shot_0/ \
    ./logs/MFRC/meta-llama--Llama-3.3-70B-Instruct-lb-10-shot_0/ \
    ./logs/MFRC/SFT-Llama-3.3-70B-Instruct-10-shot_0/ \
    --out logs/analysis/ml-distr/MFRC-all \
    --name "Llama3 8B Base" "Llama3 8B Instruct" "SFT Llama3 8B Instruct" \
    "Llama3 70B Base" "Llama3 70B Instruct" "SFT Llama3 70B Instruct" \
    --max-rank 2 --dataset "MFRC" --no-legend --no-box --no-x

python scripts/ml-distribution/contrast_distributions_plot_group_singles.py \
    --exp ./logs/Boxes/meta-llama--Llama-3.1-8B-attn-5-shot_0/ \
    ./logs/Boxes/meta-llama--Llama-3.1-8B-Instruct-attn-5-shot_0/ \
    ./logs/Boxes/meta-llama--Llama-3.1-70B-lb-4-shot_0/ \
    ./logs/Boxes/meta-llama--Llama-3.3-70B-Instruct-lb-4-shot_0/ \
    --out logs/analysis/ml-distr/Boxes-all \
    --name "Llama3 8B Base" "Llama3 8B Instruct" \
    "Llama3 70B Base" "Llama3 70B Instruct" \
    --max-rank 2 --dataset "Boxes" --no-legend --no-box