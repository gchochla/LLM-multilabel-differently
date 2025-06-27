import yaml
import numpy as np

import sys
sys.path.append('scripts/prob_distr')

import matplotlib.pyplot as plt

from distribution_estimation import load_data_from_yaml

def get_label_order_accuracy_from_example(example):
    """
    Returns a list of 1's and 0's with length equal to number of labels predicted;
    1 if the label is in the ground truth, 0 otherwise.
    """
    
    label_indices = []
    for label in example['test_preds']:
        # Find the index where this label appears in the output text
        label_idx = example['test_outs'].find(label)
        if label_idx == -1:  # If label not found in text
            label_idx = float('inf')  # Put at end of list
        label_indices.append((label, label_idx))
    
    # Sort labels by their index in the text
    sorted_labels = [label for label, _ in sorted(label_indices, key=lambda x: x[1])]
    
    label_order_accuracies = [1 if label in example['test_gt'] else 0 for label in sorted_labels]
    
    return label_order_accuracies[:3]
    
def get_graph_probs(data):
    label_order_accuracies = [0] * 999
    label_order_counts = [0] * 999
    for example_id, example in data.items():
        if len(example['test_preds']) < 2 or len(example['test_scores']) == 0: # need at least two labels
            continue
        
        example_label_order_accuracies = get_label_order_accuracy_from_example(example)
        for i, acc in enumerate(example_label_order_accuracies):
            label_order_accuracies[i] += acc
            label_order_counts[i] += 1
            
    final_accs = []
    for acc, count in zip(label_order_accuracies, label_order_counts):
        if count > 10:
            final_accs.append(acc/count)
    
    return final_accs

def plot_label_order_accuracies(yaml_files, save_path, color='blue'):
    
    potential_datasets = ['GoEmotions', 'MFRC', 'SemEval']
    
    # results[dataset][model] = order_accs
    results = {d: {} for d in potential_datasets}
    
    for yml_file in yaml_files:
        upper_yml_file = yml_file
        # yml_file = yml_file.lower()
        if 'semeval' in yml_file.lower():
            dataset = 'SemEval'
        elif 'mfrc' in yml_file.lower():
            dataset = 'MFRC'
        elif 'goemotions' in yml_file.lower():
            dataset = 'GoEmotions'
        else:
            raise ValueError(f'Unknown dataset: {yml_file}')
        if '3.1' in yml_file:
            model = '8B Instruct'
        elif '3.2' in yml_file:
            model = '1B Instruct'
        elif '3.3' in yml_file:
            model = '70B Instruct'
        elif 'qwen' in yml_file.lower():
            model = 'Qwen 7B Instruct'
        else:
            raise ValueError(f'Unknown model: {yml_file}')
        
        results[dataset][model] = []
        yaml_data = yaml.safe_load(open(upper_yml_file, 'r'))
        # for seed in [1, 2, 3, 4]:
        for seed in [0]:
            seed_data = load_data_from_yaml(upper_yml_file, seed=seed, existing_data=yaml_data)
            results[dataset][model].append(get_graph_probs(seed_data)[:2])
        print(dataset, model, results[dataset][model])
        
    plt.figure(figsize=(10, 7))
    
    # Set up colors and line styles
    colors = {'GoEmotions': 'blue', 'MFRC': 'red'}
    line_styles = {'1B Instruct': ':',
                   '8B Instruct': '--',
                   '70B Instruct': '-.',
                   'Qwen 7B Instruct': '-'}
    
    # Plot each dataset and model combination
    for dataset in results:
        for model in results[dataset]:
            accuracies = results[dataset][model]
            
            # Pad arrays with nan to make them same length
            max_len = max(len(acc) for acc in accuracies)
            padded_accuracies = np.array([np.pad(acc, (0, max_len - len(acc)), constant_values=np.nan) for acc in accuracies])
            x = list(range(1, max_len + 1))  # Offset by 1 to start at 1
            
            # Calculate mean and std across seeds, ignoring nan values
            mean_accuracies = np.nanmean(padded_accuracies, axis=0)
            std_accuracies = np.nanstd(padded_accuracies, axis=0)
            
            # for i in reversed(range(len(mean_accuracies))):
            #     nan_count = np.isnan(padded_accuracies[i]).sum()
            #     if nan_count >= len(mean_accuracies) / 2:
            #         mean_accuracies = mean_accuracies[:i]
            #         std_accuracies = std_accuracies[:i]
            #         x = x[:i]
                
                
            plt.plot(x, mean_accuracies, 
                    color=colors[dataset], 
                    linestyle=line_styles[model],
                    marker='o',
                    alpha=0.6)
            
            # Add shaded standard deviation
            plt.fill_between(x, 
                           mean_accuracies - std_accuracies / 2,
                           mean_accuracies + std_accuracies / 2,
                           color=colors[dataset],
                           alpha=0.1)
    
    plt.ylim(0.2, 1)
    plt.xlim(0.95, 2.05)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1], fontsize=26)
    plt.xticks([1, 2], fontsize=26)
    
    plt.xlabel('Label Order', fontsize=26)
    plt.ylabel('Accuracy', fontsize=26)
    
    # Create separate legends for datasets and models
    dataset_handles = [plt.Line2D([0], [0], color=color, label=f'{dataset}', linestyle='-') 
                      for dataset, color in colors.items()]
    model_handles = [plt.Line2D([0], [0], color='black', label=f'{model}', linestyle=style) 
                    for model, style in line_styles.items()]
    
    plt.legend(handles=dataset_handles + model_handles, 
               fontsize=17,
              loc='upper right')
    
    plt.tight_layout()
    
    plt.savefig(save_path)
    plt.close()
    
        
if __name__ == '__main__':
    
    datasets = [
        'MFRC',
        'GoEmotions',
    ]
    
    models = [
        'meta-llama--Llama-3.2-1B-Instruct_0',
        'meta-llama--Llama-3.1-8B-Instruct_0',
        'meta-llama--Llama-3.3-70B-Instruct_0',
        'Qwen--Qwen2.5-7B-Instruct_0',
    ]
    
    yaml_files = []
    
    for dataset in datasets:
        for model in models:
            yaml_file = f'logs/{dataset}/big_multilabel/baseline/{model}/indexed_metrics.yml'
            yaml_files.append(yaml_file)
            
    
    plot_label_order_accuracies(yaml_files, save_path='scripts/prob_distr/figures/label_order_accuracies.pdf')