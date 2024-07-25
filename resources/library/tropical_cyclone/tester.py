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
        
        # printing these before the metrics calculation, so in case of divisions by zero it would still be possible to have a look at them
        print(f'Threshold: {threshold}')
        print(f'\ttrue_pos, true_neg, false_pos, false_neg: {true_pos, true_neg, false_pos, false_neg}')
        
        dist_sum = dist_sum / (true_pos+false_pos)
        accuracy = (true_pos+true_neg) / (true_pos+false_pos+true_neg+false_neg)
        precision = true_pos/(true_pos+false_pos)
        recall = true_pos/(true_pos+false_neg)
        F1_score = 2*(precision*recall)/(precision+recall)
        
        print(f'\tAverage distance between preds and targets: {dist_sum:.3f}')
        print(f'\tAccuracy:\t{accuracy:.6f}')
        print(f'\tPrecision:\t{precision:.6f}')
        print(f'\tRecall:\t\t{recall:.6f}')
        print(f'\tF1 score:\t{F1_score:.6f}')
    
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'F1 score': F1_score
        }
        return metrics