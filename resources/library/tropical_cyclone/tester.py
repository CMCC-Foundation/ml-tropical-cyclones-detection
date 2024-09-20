import numpy as np
import torch
import torch_geometric
from tqdm import tqdm

import sys
sys.path.append('../library')
import tropical_cyclone as tc

class GraphTester:
    def __init__(
        self,
        device: torch.device,
        loader: torch_geometric.loader.DataLoader,
        model: tc.models.BaseLightningModuleGNN,
        n_cyclones: int,
        nodes_per_graph: int
    ):
        self.device = device
        self.loader = loader
        self.model = model
        self.n_cyclones = n_cyclones
        self.nodes_per_graph = nodes_per_graph
    
    @torch.no_grad()
    def get_metrics(self,
                    threshold: float = 0.5):
    
        # confusion matrix
        true_pos = 0
        false_pos = 0
        true_neg = 0
        false_neg = 0

        # average distance between predicted and actual cyclones
        dist_sum = 0

        self.model.eval()

        # patch_id to keep track(the first dataset_test.n_cy patches are true cyclones)
        patch_id = 0

        for batch in tqdm(self.loader, desc="Testing the model", unit="batch"):
            batch = batch.to(self.device)
            pred = self.model(batch)

            start = 0
            end = self.nodes_per_graph
            
            # iterate over every patch in the batch
            while start != batch.y.shape[0]:
                max_pred, id_pred = torch.max(pred[start:end], 0)
                _, id_target = torch.max(batch.y[start:end], 0)

                # cyclone patch found
                if max_pred > threshold:
                    # retrieve cyclone 40x40 coordinates
                    row_pred = id_pred % 40
                    col_pred = id_pred // 40
                    row_target = id_target % 40
                    col_target = id_target // 40

                    # calculate Euclidean distance and sum up
                    dist_sum += np.sqrt((row_pred.cpu()-row_target.cpu())**2 + (col_pred.cpu()-col_target.cpu())**2).item()

                    if patch_id < self.n_cyclones:
                        true_pos += 1
                    else:
                        false_pos += 1
                else:
                    if patch_id < self.n_cyclones:
                        false_neg += 1
                    else:
                        true_neg += 1

                patch_id += 1
                start = end
                end += self.nodes_per_graph
        
        print(f'Threshold: {threshold}')
        print(f'\tTrue  positives: {true_pos}')
        print(f'\tFalse positives: {false_pos}')
        print(f'\tTrue  negatives: {true_neg}')
        print(f'\tFalse negatives: {false_neg}')
        
        # in case of division by zero I'm returning an error string with all the metrics that were involved
        zero_division = ''
        
        try:
            dist_sum = dist_sum / (true_pos+false_pos)
        except ZeroDivisionError:
            print("Error: Division by zero in average distance calculation, leaving distance untouched")
            zero_division += 'distance, '
        
        try:
            accuracy = (true_pos+true_neg) / (true_pos+false_pos+true_neg+false_neg)
        except ZeroDivisionError:
            print("Error: Division by zero in accuracy calculation, keeping just the numerator")
            accuracy = (true_pos+true_neg)
            zero_division += 'accuracy, '
        
        try:
            precision = true_pos/(true_pos+false_pos)
        except ZeroDivisionError:
            print("Error: Division by zero in precision calculation, keeping just the numerator")
            precision = true_pos
            zero_division += 'precision, '
        
        try:
            recall = true_pos/(true_pos+false_neg)
        except ZeroDivisionError:
            print("Error: Division by zero in recall calculation, keeping just the numerator")
            recall = true_pos
            zero_division += 'recall, '
        
        try:
            F1_score = 2*(precision*recall)/(precision+recall)
        except ZeroDivisionError:
            print("Error: Division by zero in F1_score calculation, keeping just the numerator")
            F1_score = 2*(precision*recall)
            zero_division += 'F1_score, '
        
        print(f'\tAverage distance between preds and targets: {dist_sum:.3f}')
        print(f'\tAccuracy:  {accuracy:.6f}')
        print(f'\tPrecision: {precision:.6f}')
        print(f'\tRecall:    {recall:.6f}')
        print(f'\tF1 score:  {F1_score:.6f}')
    
        metrics = {
            'avg_dist': dist_sum,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'F1 score': F1_score,
            'zero_division': zero_division
        }
        return metrics