import lightning.pytorch as pl
import lightning as L
from typing import Any
import torch.nn.functional as F
import torch.nn as nn
import torch
import torch_geometric
from torch_geometric.utils import dropout_edge

# Provenance logger (needed only when working with GNNs)
try:
    import sys
    sys.path.append('../../yProvML')
    import prov4ml
except ImportError:
    print('Library prov4ml not found, keep executing...')

class BaseLightningModule(L.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.callback_metrics = {}

    def training_step(self, batch, batch_idx):
        # get data from the batch
        x, y = batch
        # forward pass
        y_pred = self(x)
        # compute loss
        loss = self.loss(y_pred, y)
        # define log dictionary
        log_dict = {'train_loss': loss}
        # compute metrics
        for metric in self.metrics:
            metric_value = metric(y_pred, y)
            log_dict.update({f'train_{metric.name}' : metric_value})
            self.log(f'train_{metric.name}', metric_value, prog_bar=True)
        # log the outputs
        self.callback_metrics = {**self.callback_metrics, **log_dict}
        self.log('train_loss', loss, prog_bar=True)
        # return the loss
        return {'loss':loss}

    def validation_step(self, batch, batch_idx):
        # get data from the batch
        x, y = batch
        # forward pass
        y_pred = self(x)
        # compute loss
        loss = self.loss(y_pred, y)
        # define log dictionary
        log_dict = {'val_loss': loss}
        # compute metrics
        for metric in self.metrics:
            metric_value = metric(y_pred, y)
            log_dict.update({f'val_{metric.name}' : metric_value})
            self.log(f'val_{metric.name}', metric_value, prog_bar=True)
        # log the outputs
        self.callback_metrics = {**self.callback_metrics, **log_dict}
        self.log('val_loss', loss, prog_bar=True)
        # return the loss
        return {'loss':loss}

    def configure_optimizers(self):
        return {'optimizer': self.optimizer, 'lr_scheduler': self.lr_scheduler}



class VGG_V1(BaseLightningModule):
    def __init__(self, 
            in_channels: int, 
            out_channels: int, 
            activation = 'nn.Identity', 
            kernel_size: int = 3, 
            dtype = torch.float32) -> None:
        super().__init__()
        activation = eval(activation)
        self.vgg = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=kernel_size, padding='same', dtype=dtype), activation(), 
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding="same", dtype=dtype), activation(), 
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding="same", dtype=dtype), activation(), 
            nn.MaxPool2d(kernel_size=2, stride=2), 

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding="same", dtype=dtype), activation(), 
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding="same", dtype=dtype), activation(), 
            nn.MaxPool2d(kernel_size=2, stride=2), 

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, padding="same", dtype=dtype), activation(), 
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2, padding="same", dtype=dtype), activation(), 
            nn.MaxPool2d(kernel_size=2, stride=2), 

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=2, padding="valid", dtype=dtype), activation(), 
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2, padding="valid", dtype=dtype), activation(), 
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2, padding="valid", dtype=dtype), activation(), 
            nn.MaxPool2d(kernel_size=2, stride=2), 

            nn.Flatten(), 

            nn.Linear(in_features=512, out_features=512, dtype=dtype), activation(), 
            nn.Linear(in_features=512, out_features=256, dtype=dtype), activation(), 
            nn.Linear(in_features=256, out_features=128, dtype=dtype), activation(), 
            nn.Linear(in_features=128, out_features=64, dtype=dtype), activation(), 
            nn.Linear(in_features=64, out_features=out_channels, dtype=dtype), 
        )
        with torch.no_grad():
            for module in self.vgg.modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0, std=0.05)
                    nn.init.normal_(module.bias, mean=0, std=0.05)

    def forward(self, x: torch.Tensor) -> Any:
        x = self.vgg(x)
        return x



class VGG_V2(BaseLightningModule):
    def __init__(self, 
            in_channels: int, 
            out_channels: int, 
            activation = 'nn.Identity', 
            kernel_size: int = 3) -> None:
        super().__init__()
        activation = eval(activation)
        self.vgg = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=kernel_size, padding='same'), activation(), 
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=kernel_size, padding="same"), activation(), 
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=kernel_size, padding="same"), activation(), 
            nn.MaxPool2d(kernel_size=2, stride=2), 

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding="same"), activation(), 
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding="same"), activation(), 
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding="same"), activation(), 
            nn.MaxPool2d(kernel_size=2, stride=2), 

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding="same"), activation(), 
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding="same"), activation(), 
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding="same"), activation(), 
            nn.MaxPool2d(kernel_size=2, stride=2), 

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, padding="same"), activation(), 
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2, padding="same"), activation(), 
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2, padding="same"), activation(), 

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=2, padding="valid"), activation(), 
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2, padding="valid"), activation(), 
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2, padding="valid"), activation(), 
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2, padding="valid"), activation(), 

            nn.Flatten(), 

            nn.Linear(in_features=512, out_features=1024), activation(), 
            nn.Linear(in_features=1024, out_features=512), activation(), 
            nn.Linear(in_features=512, out_features=256), activation(), 
            nn.Linear(in_features=256, out_features=128), activation(), 

            nn.Linear(in_features=128, out_features=out_channels), 
        )
        with torch.no_grad():
            for module in self.vgg.modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0, std=0.05)
                    nn.init.normal_(module.bias, mean=0, std=0.05)

    def forward(self, x: torch.Tensor) -> Any:
        x = self.vgg(x)
        return x



class VGG_V3(BaseLightningModule):
    def __init__(self, 
            in_channels: int, 
            out_channels: int, 
            activation: str = 'nn.Identity', 
            kernel_size: int = 3, 
            init_std: float = 0.05, 
        ) -> None:
        super().__init__()
        activation = eval(activation)
        self.vgg = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=kernel_size, padding='same'), 
            activation(), 
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=kernel_size, padding="same"), 
            activation(), 
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=kernel_size, padding="same"), 
            activation(), 
            nn.MaxPool2d(kernel_size=2, stride=2), 

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding="same"), 
            activation(), 
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding="same"), 
            activation(), 
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding="same"), 
            activation(), 
            nn.MaxPool2d(kernel_size=2, stride=2), 

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding="same"), 
            activation(), 
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding="same"), 
            activation(), 
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding="same"), 
            activation(), 
            nn.MaxPool2d(kernel_size=2, stride=2), 

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, padding="same"), 
            activation(), 
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2, padding="same"), 
            activation(), 
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2, padding="same"), 
            activation(), 

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=2, padding="valid"), 
            activation(), 
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2, padding="valid"), 
            activation(), 

            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=2, padding="valid"), 
            activation(), 
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=2, padding="valid"), 
            activation(), 

            nn.Flatten(), 

            nn.Linear(in_features=1024, out_features=1024), 
            activation(), 
            nn.Linear(in_features=1024, out_features=512), 
            activation(), 
            nn.Linear(in_features=512, out_features=512), 
            activation(), 
            nn.Linear(in_features=512, out_features=256), 
            activation(), 
            nn.Linear(in_features=256, out_features=out_channels)
        )
        self._init_normal(init_std)

    def forward(self, x: torch.Tensor):
        x = self.vgg(x)
        return x

    def _init_normal(self, std: float = 0.05):
        with torch.no_grad():
            for module in self.vgg.modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0, std=std)
                    nn.init.normal_(module.bias, mean=0, std=std)

        

class BaseLightningModuleGNN(pl.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.callback_metrics = {}
    
    def training_step(self, batch, batch_idx):
        # forward pass + loss computation
        y_pred = self(batch)
        loss = self.loss(y_pred, batch.y)
        
        # TODO can mix the loss with something else, like the recall and the distance from the cyclone, if there is one to calculate
        
        # log metric in provenance logger
        prov4ml.log_metric("BCE_train", float(loss), prov4ml.Context.TRAINING, step=self.current_epoch)
        
        # define log dictionary
        log_dict = {'train_loss': loss}
        # compute metrics
        for metric in self.metrics:
            log_dict.update({f'train_{metric.name}' : metric(y_pred, batch.y)})
        # log the outputs
        self.callback_metrics = {**self.callback_metrics, **log_dict}
        
        # log the train_loss
        self.log('train_loss', loss)
        
        return {'loss':loss}

    def validation_step(self, batch, batch_idx):
        # forward pass + loss computation
        y_pred = self(batch)
        loss = self.loss(y_pred, batch.y)
        
        # log metric in provenance logger
        prov4ml.log_metric("BCE_eval", float(loss), prov4ml.Context.VALIDATION, step=self.current_epoch)
        
        # define log dictionary
        log_dict = {'val_loss': loss}
        # compute metrics
        for metric in self.metrics:
            log_dict.update({f'val_{metric.name}' : metric(y_pred, batch.y)})
        # log the outputs
        self.callback_metrics = {**self.callback_metrics, **log_dict}
        
        # log the val_loss
        self.log('val_loss', loss)
        
        return {'loss':loss}
    
    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": self.lr_scheduler,
        }

    def on_validation_model_eval(self) -> None:
        self.eval()
    
    def on_validation_model_train(self) -> None:
        self.train()
        prov4ml.log_system_metrics(prov4ml.Context.TRAINING, step=self.current_epoch)
        prov4ml.log_carbon_metrics(prov4ml.Context.TRAINING, step=self.current_epoch)
        prov4ml.log_current_execution_time("train_step", prov4ml.Context.TRAINING, self.current_epoch)
        
    def on_test_model_train(self) -> None:
        self.train()
    def on_test_model_eval(self) -> None:
        self.eval()
    def on_predict_model_eval(self) -> None:
        self.eval()


        
class GraphUNet(BaseLightningModuleGNN):
    def __init__(self,
            in_channels: int,
            hid_channels: int,
            out_channels: int,
            K_pool: int,
            nodes_per_graph: int,
            edge_dropout_rate: float,
            node_dropout_rate: float,
            activation: str,
            *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        
        self.edge_dropout_rate = edge_dropout_rate
        self.node_dropout_rate = node_dropout_rate
        self.activation = eval(activation)
        
        # Top-K pooling setup
        if K_pool == -1:
            K_pool = nodes_per_graph / 2
        pool_ratios = [K_pool / nodes_per_graph, 0.5]
        self.unet = torch_geometric.nn.GraphUNet(in_channels=in_channels,
                                                 hidden_channels=hid_channels,
                                                 out_channels=out_channels,
                                                 depth=3,
                                                 pool_ratios=pool_ratios)
        
    def forward(self, data):
        edge_index, _ = dropout_edge(data.edge_index, p=self.edge_dropout_rate, force_undirected=True, training=self.training)
        x = F.dropout(data.x, p=self.node_dropout_rate, training=self.training)
        x = self.unet(x, edge_index)
        return self.activation(x)
