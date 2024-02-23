import lightning.pytorch as pl
from typing import Any
import torch.nn as nn
import torch



class BaseLightningModule(pl.LightningModule):
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
            log_dict.update({f'train_{metric.name}' : metric(y_pred, y)})
        # log the outputs
        self.callback_metrics = {**self.callback_metrics, **log_dict}
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
            log_dict.update({f'val_{metric.name}' : metric(y_pred, y)})
        # log the outputs
        self.callback_metrics = {**self.callback_metrics, **log_dict}
        # return the loss
        return {'loss':loss}

    def on_validation_model_eval(self) -> None:
        self.eval()
    def on_validation_model_train(self) -> None:
        self.train()
    def on_test_model_train(self) -> None:
        self.train()
    def on_test_model_eval(self) -> None:
        self.eval()
    def on_predict_model_eval(self) -> None:
        self.eval()



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
