import yaml
import os
import numpy as np

import sys
sys.path.append('scripts/prob_distr')

import matplotlib.pyplot as plt

from distribution_estimation import load_data_from_yaml

def get_graph_probs(data):
    
    probs = None
    for example_id, example in data.items():
        if len(example['test_scores']) == 0:
            continue
        # for SemEval, don't have baseline, so just use 'ratio_0.4' from the ICL experiments (basically the same as baseline, which has 0.5 ratio multilabel)
        if 'ratio_0.0' in example_id or 'ratio_0.2' in example_id or 'ratio_0.6' in example_id or 'ratio_0.8' in example_id or 'ratio_1.0' in example_id:
            continue
        
        distr = example['test_scores']
        if 'none' in distr:
            del distr['none']
        
        if probs is None:
            probs = []
            for _ in range(len(distr)):
                probs.append([])
        sorted_probs = sorted(list(distr.values()), reverse=True)
        if sorted_probs[0] < 0.3 or len(example['test_preds']) < 1: # ignore 'none' examples and noisy examples (less than 0.3 for top label)
            continue
        for i, prob in enumerate(sorted_probs):
            probs[i].append(prob)
        
    return [x for x in probs if len(x) > 0]               
                
def plot_multilabel_icl(yaml_files, save_path):
    
    colors = ['blue', 'red', 'green']
    labels = ['GoEmotions', 'MFRC', 'SemEval']

    plt.figure(figsize=(12, 7))  # Move outside the loop so all datasets are in one figure

    # Offset each dataset on the x-axis for clearer separation
    x_offset = 0.25

    for i, (yaml_file, color, label) in enumerate(zip(yaml_files, colors, labels)):
        data = load_data_from_yaml(yaml_file)
        points = get_graph_probs(data)
        
        for x, probs in enumerate(points):
            jitter = x_offset * (i - 1)  # -0.25, 0.0, 0.25 for three datasets
            jittered_x = x + jitter + 0.12 * (np.random.random(len(probs)) - 0.5)
            plt.scatter(jittered_x, probs, color=color, alpha=0.08)
            if x == 0:
                plt.scatter(-1, -1, color=color, alpha=1, label=label)
            # Only label once per dataset to avoid repeated legend entries

    plt.xlabel("Sorted Label Index", fontsize=22)
    plt.ylabel("Label Probability", fontsize=22)
    plt.xticks([0, 1, 2, 3, 4, 5, 6], fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlim(-0.5, 6.5)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend(loc='upper right', fontsize=20)
    plt.tight_layout()
    
    plt.savefig(save_path)
    plt.close()
    
        
if __name__ == '__main__':
    
    # yaml_files = [
    #  'logs/GoEmotions/main_test_set/baseline/meta-llama--Llama-3.3-70B-Instruct_0/indexed_metrics.yml',
    #  'logs/MFRC/main_test_set/baseline/meta-llama--Llama-3.3-70B-Instruct_0/indexed_metrics.yml',
    #  'logs/SemEval/main_test_set/multilabel_ICL/meta-llama--Llama-3.3-70B-Instruct_0/indexed_metrics.yml',
    # ]
    
    # plot_multilabel_icl(yaml_files, save_path='scripts/prob_distr/figures/spikiness.png')
    
    yaml_files = [
    'logs/GoEmotions/main_test_set/baseline/Qwen--Qwen2.5-7B-Instruct_0/indexed_metrics.yml',
     'logs/MFRC/main_test_set/baseline/Qwen--Qwen2.5-7B-Instruct_0/indexed_metrics.yml',
     'logs/SemEval/main_test_set/multilabel_ICL/Qwen--Qwen2.5-7B-Instruct_0/indexed_metrics.yml',
    ]
    plot_multilabel_icl(yaml_files, save_path='scripts/prob_distr/figures_qwen/spikiness.png')