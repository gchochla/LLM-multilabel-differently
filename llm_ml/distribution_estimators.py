from abc import ABC, abstractmethod
from scipy.special import softmax
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from llm_ml.load_dataset_annotations import get_dataset_label_order

class DistributionEstimator(ABC):
    ACCEPTED_TYPES = ['baseline', 'output', 'unary', 'binary', 'binary_outcome', 'max', 'SFT_output', 'SFT_max']
    def __init__(self, dataset, estimation_type):
        if estimation_type not in DistributionEstimator.ACCEPTED_TYPES:
            raise Exception(f'{estimation_type} must be one of: {DistributionEstimator.ACCEPTED_TYPES}')
        self.estimation_type = estimation_type
        
        """
        Labels are saved alphabetically into the yaml file but
        annotations are based on a custom label order.
        """
        self.label_order = get_dataset_label_order(dataset)
        
    def estimate(self, data):
        if self.estimation_type == 'baseline':
            return self.baseline_estimate(data)
        elif self.estimation_type == 'output' or self.estimation_type == 'SFT_output':
            return self.output_estimate(data)
        elif self.estimation_type == 'unary':
            return self.unary_estimate(data)
        elif self.estimation_type == 'binary':
            return self.binary_estimate(data, use_outcome=False)
        elif self.estimation_type == 'binary_outcome':
            return self.binary_estimate(data, use_outcome=True)
        elif self.estimation_type == 'max' or self.estimation_type == 'SFT_max':
            return self.max_estimate(data)
    
    def calibrate(self, validation_data, annotations):
        if self.estimation_type == 'baseline':
            self.baseline_calibrate(validation_data, annotations)
        if self.estimation_type == 'unary':
            self.unary_calibrate(validation_data, annotations)
        if self.estimation_type == 'binary':
            self.binary_calibrate(validation_data, annotations)
            
    def bradley_terry_model(matchups, num_labels, num_iters=5000, lr=1e-2, l2_lambda=0.01):
        probs = torch.nn.Parameter(torch.randn(num_labels))
        optimizer = torch.optim.Adam([probs], lr=lr)
        
        # Early stopping parameters
        best_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for iter in range(num_iters):
            loss = 0
            for (i, j, p) in matchups:
                est_i = probs[i]
                est_j = probs[j]
                # never zero
                diff = torch.sigmoid(est_i - est_j) + 1e-7
                loss += -(p * torch.log(diff) + (1 - p) * torch.log(1 - diff))
            
            # Add L2 regularization
            l2_reg = l2_lambda * torch.sum(probs ** 2)
            loss += l2_reg
            
            # Early stopping check
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        probs = probs.detach().cpu().tolist()
        return probs
            
    @abstractmethod
    def baseline_estimate(self, data): pass
    
    @abstractmethod
    def output_estimate(self, data): pass
    
    @abstractmethod
    def unary_estimate(self, data): pass
    
    @abstractmethod
    def binary_estimate(self, data): pass
    
    def baseline_calibrate(self, validation_data, annotations): pass
    
    def unary_calibrate(self, validation_data, annotations): pass
    
    def binary_calibrate(self, validation_data, annotations): pass
    
class SingleLabelEstimator(DistributionEstimator):
    def baseline_estimate(self, data):
        distributions = {}
        for example, datum in data.items():
            distributions[example] = datum['test_scores']
        return distributions
    
    def output_estimate(self, data, epsilon=1e-8):
        distributions = {}
        for example, datum in data.items():
            distributions[example] = {k: 1 - epsilon if k == datum['test_preds'] else epsilon for k in datum['test_scores'].keys()}
            if 'none' in distributions:
                del distributions['none']
        return distributions

    def unary_estimate(self, data):
        unary_distributions = {}
        
        labels = set()
        
        for unary_example, datum in data.items():
            # in the form {example}_unary_{unary_label}, i.e. 123_unary_happy
            segments = unary_example.split('_')
            unary_label = segments[-1]
            example = '_'.join(segments[:-2])
            
            labels.add(unary_label)
            
            if example not in unary_distributions:
                unary_distributions[example] = {}
            unary_distributions[example][unary_label] = datum['test_scores']['reasonable']
            
        # now, need to normalize
        labels = self.label_order
        
        for example, unnormalized_probabilities in unary_distributions.items():
            
            unnormalized_list = [unnormalized_probabilities[label] for label in labels]
            normalized_list = [float(p) for p in softmax(unnormalized_list)]
            assert len(normalized_list) == len(labels)
            
            unary_distributions[example] = {label: p for label, p in zip(labels, normalized_list)}
        
        return unary_distributions

    def binary_estimate(self, data, use_outcome=False):
        all_matchups = {}
        
        label_to_index = {l: i for i, l in enumerate(self.label_order)}
        index_to_label = {i: l for l, i in label_to_index.items()}
        
        for binary_example, datum in data.items():
            # in the form {example}_binary_{first}_{second}, i.e. 123_binary_4_7
            segments = binary_example.split('_')
            second_label = segments[-1]
            first_label = segments[-2]
            example = '_'.join(segments[:-3])
            
            second_index = label_to_index[second_label]
            first_index = label_to_index[first_label]
            
            if example not in all_matchups:
                all_matchups[example] = [] # will be tuple of (first player, second player, score)
            
            # default to first player probabilities
            if not use_outcome:
                all_matchups[example].append((first_index, second_index, datum['test_scores']['a']))
            else:
                all_matchups[example].append((first_index, second_index, 1 if datum['test_scores']['a'] > datum['test_scores']['b'] else 0))
            
        from multiprocessing import Pool
        from functools import partial

        # Create a pool of workers
        with Pool() as pool:
            # Create a partial function with fixed num_labels parameter
            process_func = partial(DistributionEstimator.bradley_terry_model, num_labels=len(self.label_order))
            
            # Process examples in parallel
            example_to_raw_probs = dict(zip(
                all_matchups.keys(),
                tqdm(
                    pool.imap(process_func, all_matchups.values()),
                    total=len(all_matchups),
                    desc='Running Bradley-Terry model'
                )
            ))

        probabilities = {}
        
        for example, raw_probs in example_to_raw_probs.items():
            example_probabilities = F.softmax(torch.tensor(raw_probs), dim=0)
            probabilities[example] = {index_to_label[i]: p for i, p in enumerate(example_probabilities.tolist())}
        
        return probabilities

class MultiLabelEstimator(DistributionEstimator):
    
    def __init__(self, dataset, estimation_type):
        super().__init__(dataset, estimation_type)
        
        self.label_order.append('none')
        
        self.regressor = PlattScaler()
        
    
    def baseline_calibrate(self, validation_data, annotations):
        
        individual_logit_diffs, individual_labels = [], []
        
        for example_id, example_data in validation_data.items():
            raw_logits = example_data['test_logits']
            none_logit_diffs = []
            for ordered_label in self.label_order:
                none_logit_diffs.append(raw_logits[ordered_label] - raw_logits['none'])
            
            annotation_labels = annotations[example_id]['labels']
            
            # averaged annotation_labels, i.e. if 3 out of 4 annotators assign a label, the value is 0.75
            labels = [sum(x) / len(x) for x in zip(*annotation_labels)]
            
            assert len(labels) == len(none_logit_diffs)
            for none_diff, label in zip(none_logit_diffs, labels):
                individual_logit_diffs.append(none_diff)
                individual_labels.append(label)
        
        self.regressor.fit(individual_logit_diffs, individual_labels)
    
    def baseline_estimate(self, data):
        
        distributions = {}
        
        for example_id, example_data in data.items():
            raw_logits = example_data['test_logits']
            distributions[example_id] = {}
            
            # NEED TO FIX: 'none' should always appear
            if 'none' not in raw_logits:
                raw_logits['none'] = 0
            
            for label, logit in raw_logits.items():
                if label != 'none':
                    distributions[example_id][label] = self.regressor(logit - raw_logits['none']).item()
        
        return distributions
    
    def max_estimate(self, data, epsilon=1e-8):
        distributions = {}
        for example, datum in data.items():
            distributions[example] = {label: epsilon for label in self.label_order}
            
            if 'test_all_scores' in datum:
                all_scores = datum['test_all_scores']
            else:
                all_scores = datum['test_scores']
                print('aaa')
            
            for scores in all_scores:
                for label, score in scores.items():
                    if label != 'none':
                        distributions[example][label] = max(distributions[example][label], score)
            if sum(distributions[example].values()) < epsilon * 10 and not isinstance(datum['test_scores'], list):
                distributions[example] = {label: datum['test_scores'][label] for label in self.label_order if label != 'none'}
            
        return distributions
            
    
    def output_estimate(self, data, epsilon=1e-8):
        distributions = {}
        for example, datum in data.items():
            distributions[example] = {k: 1 - epsilon if k in datum['test_preds'] else epsilon for k in self.label_order if k != 'none'}
        return distributions
    
    def unary_estimate(self, data):
        unary_distributions = {}
        
        for unary_example, datum in data.items():
            # in the form {example}_unary_{unary_label}, i.e. 123_unary_happy
            segments = unary_example.split('_')
            unary_label = segments[-1]
            example = '_'.join(segments[:-2])
            
            if example not in unary_distributions:
                unary_distributions[example] = {}
            unary_distributions[example][unary_label] = datum['test_scores']['reasonable']
        
        return unary_distributions
        
    
    def binary_estimate(self, data, use_outcome=False):
        all_matchups = {}
        
        label_to_index = {l: i for i, l in enumerate(self.label_order)}
        index_to_label = {i: l for l, i in label_to_index.items()}
        
        for binary_example, datum in data.items():
            # in the form {example}_binary_{first}_{second}, i.e. 123_binary_4_7
            segments = binary_example.split('_')
            second_label = segments[-1]
            first_label = segments[-2]
            example = '_'.join(segments[:-3])
            
            second_index = label_to_index[second_label]
            first_index = label_to_index[first_label]
            
            if example not in all_matchups:
                all_matchups[example] = [] # will be tuple of (first player, second player, score)
            
            # default to first player probabilities
            if not use_outcome:
                all_matchups[example].append((first_index, second_index, datum['test_scores']['a']))
            else:
                all_matchups[example].append((first_index, second_index, 1 if datum['test_scores']['a'] > datum['test_scores']['b'] else 0))
          
          
        from multiprocessing import Pool
        from functools import partial

        # Create a pool of workers
        with Pool() as pool:
            # Create a partial function with fixed num_labels parameter
            process_func = partial(DistributionEstimator.bradley_terry_model, num_labels=len(self.label_order))
            
            # Process examples in parallel
            example_to_raw_probs = dict(zip(
                all_matchups.keys(),
                tqdm(
                    pool.imap(process_func, all_matchups.values()),
                    total=len(all_matchups),
                    desc='Running Bradley-Terry model'
                )
            ))

        probabilities = {}
        
        for example, raw_probs in example_to_raw_probs.items():
            none_prob = raw_probs[label_to_index['none']]
            probabilities[example] = {
                'zzz_raw_logits': {}
            }
            
            for label in self.label_order:
                probabilities[example][label] = self.regressor(raw_probs[label_to_index[label]] - none_prob).item()
                probabilities[example]['zzz_raw_logits'][label] = raw_probs[label_to_index[label]]
        
        return probabilities

    
class PlattScaler(nn.Module):
    """
    Trained weighted logistic regression, also known as Platt scaling.
    """
    def __init__(self):
        super().__init__()
        
        # set default values
        self.w = nn.Parameter(torch.tensor([1.0]))
        self.b = nn.Parameter(torch.tensor([0.0]))
        
    def forward(self, logit):
        weighted_logit = self.w * logit + self.b
        prob = torch.sigmoid(weighted_logit)
        return prob
    
    def fit(self, logit_diffs, true_probs, epochs=1000, lr=0.01):
        
        X = torch.tensor(logit_diffs, dtype=torch.float32)
        y = torch.tensor(true_probs, dtype=torch.float32)
        
        optimizer = optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.BCELoss()
        
        for _ in range(epochs):
            optimizer.zero_grad()
            predicted = self.forward(X)
            loss = loss_fn(predicted, y)
            loss.backward()
            optimizer.step()
        
        print(f'Learned parameters for w ({self.w.data.item()}) and b ({self.b.data.item()})')