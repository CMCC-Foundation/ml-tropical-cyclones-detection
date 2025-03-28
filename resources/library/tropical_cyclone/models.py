from tropical_cyclone.layers import PatchEmbed, BasicLayer, Downsample, Mlp

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)
from timm.layers import to_2tuple, trunc_normal_
import lightning.pytorch as pl
import lightning as L
from typing import Any
import torch.nn.functional as F
import torch.nn as nn
import torch
import torch_geometric
from torch_geometric.utils import dropout_edge


# parent class to pass down the tracking capabilities of itwinai
class LightningTrackingParent(L.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.callback_metrics = {}
        self._trn_loss = {'sum': 0, 'steps': 0}
        self._vld_loss = {'sum': 0, 'steps': 0}
        self._training_metrics = {'steps' : 0, 'metrics' : {}}
        self._validation_metrics = {'steps' : 0, 'metrics' : {}}
    
    def on_train_epoch_end(self) -> None:
        if self.logger.experiment is not None:
            context='training'
            self.logger.experiment.log(item=self.current_epoch, identifier="epoch", kind='metric', step=self.current_epoch, context=context)
            self.logger.experiment.log(item=self, identifier=f"model_version_{self.current_epoch}", kind='model_version', step=self.current_epoch, context=context)
            self.logger.experiment.log(item=None, identifier=None, kind='system', step=self.current_epoch, context=context)
            self.logger.experiment.log(item=None, identifier=None, kind='carbon', step=self.current_epoch, context=context)
            self.logger.experiment.log(item=None, identifier="train_epoch_time", kind='execution_time', step=self.current_epoch, context=context)
            self.logger.experiment.log(item=self._trn_loss['sum']/self._trn_loss['steps'], identifier="training_loss", kind='metric', step=self.current_epoch, context=context)
        self._training_metrics = {'steps' : 0, 'metrics' : {}}
        self._trn_loss = {'sum': 0, 'steps': 0}
    
    def on_validation_epoch_end(self) -> None:
        if self.logger.experiment is not None:
            context='validation'
            self.logger.experiment.log(item=self.current_epoch, identifier="epoch", kind='metric', step=self.current_epoch, context=context)
            self.logger.experiment.log(item=self, identifier=f"model_version_{self.current_epoch}", kind='model_version', step=self.current_epoch, context=context)
            self.logger.experiment.log(item=None, identifier=None, kind='system', step=self.current_epoch, context=context)
            self.logger.experiment.log(item=None, identifier=None, kind='system', step=self.current_epoch, context=context)
            self.logger.experiment.log(item=None, identifier=None, kind='carbon', step=self.current_epoch, context=context)
            self.logger.experiment.log(item=None, identifier="validation_epoch_time", kind='execution_time', step=self.current_epoch, context=context)
            self.logger.experiment.log(item=self._vld_loss['sum']/self._vld_loss['steps'], identifier="validation_loss", kind='metric', step=self.current_epoch, context=context)
        self._validation_metrics = {'steps' : 0, 'metrics' : {}}
        self._vld_loss = {'sum': 0, 'steps': 0}
        
    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": self.lr_scheduler,
        }


class BaseLightningModule(LightningTrackingParent):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        # get data from the batch
        x, y = batch
        # forward pass
        y_pred = self(x)
        # compute loss
        loss = self.loss(y_pred, y)
        # define log dictionary
        log_dict = {"train_loss": loss}
        # compute metrics
        for metric in self.metrics:
            metric_value = metric(y_pred, y)
            log_dict.update({f"train_{metric.name}": metric_value})
            self.log(f"train_{metric.name}", metric_value, prog_bar=True)
        # log the outputs
        self.callback_metrics = {**self.callback_metrics, **log_dict}
        self.log("train_loss", loss, prog_bar=True)
        
        # mlflow logs
        self._training_loss = loss
        self._trn_loss['sum'] += loss
        self._trn_loss['steps'] += 1
        
        # return the loss
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        # get data from the batch
        x, y = batch
        # forward pass
        y_pred = self(x)
        # compute loss
        loss = self.loss(y_pred, y)
        # define log dictionary
        log_dict = {"val_loss": loss}
        # compute metrics
        for metric in self.metrics:
            metric_value = metric(y_pred, y)
            log_dict.update({f"val_{metric.name}": metric_value})
            self.log(f"val_{metric.name}", metric_value, prog_bar=True)
        # log the outputs
        self.callback_metrics = {**self.callback_metrics, **log_dict}
        self.log("val_loss", loss, prog_bar=True)
        
        # mlflow logs
        self._validation_loss = loss
        self._vld_loss['sum'] += loss
        self._vld_loss['steps'] += 1
        
        # return the loss
        return {"loss": loss}


class VGG_V1(BaseLightningModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation="nn.Identity",
        kernel_size: int = 3,
        dtype=torch.float32,
    ) -> None:
        super().__init__()
        activation = eval(activation)
        self.vgg = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=kernel_size,
                padding="same",
                dtype=dtype,
            ),
            activation(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=kernel_size,
                padding="same",
                dtype=dtype,
            ),
            activation(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=kernel_size,
                padding="same",
                dtype=dtype,
            ),
            activation(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding="same",
                dtype=dtype,
            ),
            activation(),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                padding="same",
                dtype=dtype,
            ),
            activation(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=2,
                padding="same",
                dtype=dtype,
            ),
            activation(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=2,
                padding="same",
                dtype=dtype,
            ),
            activation(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=2,
                padding="valid",
                dtype=dtype,
            ),
            activation(),
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=2,
                padding="valid",
                dtype=dtype,
            ),
            activation(),
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=2,
                padding="valid",
                dtype=dtype,
            ),
            activation(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=512, dtype=dtype),
            activation(),
            nn.Linear(in_features=512, out_features=256, dtype=dtype),
            activation(),
            nn.Linear(in_features=256, out_features=128, dtype=dtype),
            activation(),
            nn.Linear(in_features=128, out_features=64, dtype=dtype),
            activation(),
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
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation="nn.Identity",
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        activation = eval(activation)
        self.vgg = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=kernel_size,
                padding="same",
            ),
            activation(),
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=kernel_size, padding="same"
            ),
            activation(),
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=kernel_size, padding="same"
            ),
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
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=2, padding="valid"
            ),
            activation(),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=2, padding="valid"
            ),
            activation(),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=2, padding="valid"
            ),
            activation(),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=2, padding="valid"
            ),
            activation(),
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=1024),
            activation(),
            nn.Linear(in_features=1024, out_features=512),
            activation(),
            nn.Linear(in_features=512, out_features=256),
            activation(),
            nn.Linear(in_features=256, out_features=128),
            activation(),
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
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = "nn.Identity",
        kernel_size: int = 3,
        init_std: float = 0.05,
    ) -> None:
        super().__init__()
        activation = eval(activation)
        self.vgg = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=kernel_size,
                padding="same",
            ),
            activation(),
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=kernel_size, padding="same"
            ),
            activation(),
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=kernel_size, padding="same"
            ),
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
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=2, padding="valid"
            ),
            activation(),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=2, padding="valid"
            ),
            activation(),
            nn.Conv2d(
                in_channels=512, out_channels=1024, kernel_size=2, padding="valid"
            ),
            activation(),
            nn.Conv2d(
                in_channels=1024, out_channels=1024, kernel_size=2, padding="valid"
            ),
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
            nn.Linear(in_features=256, out_features=out_channels),
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


class SwinTropicalCyclone(BaseLightningModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        img_size=40,
        patch_size=1,
        win_size=4,
        embed_dim=96,
        depths=[2, 4],
        num_heads=[4, 8],
        pretrained_window_sizes=(0, 0),
        use_checkpoint=True,
        dim_factor=1,
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        swin_block_version=2,
        norm_layer=nn.LayerNorm,
        embed_patch_norm=True,
        ape=True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = len(depths)
        self.ape = ape
        patch_size = to_2tuple(patch_size)
        win_size = to_2tuple(win_size)

        # patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            in_channels=in_channels,
            patch_size=patch_size,
            embed_dim=embed_dim,
            norm_layer=norm_layer if embed_patch_norm else None,
            use_bias=True,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        # positional dropout
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # create resolutions list
        input_resolutions = [
            (
                -(patches_resolution[0] // -(2**i)),
                -(patches_resolution[1] // -(2**i)),
            )
            for i in range(self.num_layers)
        ]

        # build model encoder layers
        self.transformer = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * dim_factor**i_layer),
                dim_factor=dim_factor,
                input_resolution=input_resolutions[i_layer],
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                win_size=win_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                operation=Downsample,
                norm_layer=norm_layer,
                pretrained_window_size=pretrained_window_sizes[i_layer],
                sealand_attn_mask=None,
                use_checkpoint=use_checkpoint,
                version=swin_block_version,
            )
            self.transformer.append(layer)

        dim = int(embed_dim * dim_factor**self.num_layers)
        self.norm = norm_layer(dim) if norm_layer is not None else nn.Identity()
        self.flatten = nn.Flatten()
        self.head = Mlp(in_features=dim, hidden_features=128, out_features=out_channels)

        # apply weight initialization
        self.apply(self._init_weights)
        for bly in self.transformer:
            bly._init_respostnorm()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"cpb_mlp", "logit_scale", "relative_position_bias_table"}

    def forward_features(self, x: torch.Tensor):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.transformer:
            x = layer(x)
        x = self.norm(x)
        x = self.flatten(x)
        x = self.head(x)
        return x

    def forward(self, x: torch.Tensor):
        x = self.forward_features(x)
        return x

    def configure_activation_checkpointing(self):
        """Configure activation checkpointing.

        This is required in order to compute gradients without running out of memory.
        """
        apply_activation_checkpointing(
            self, check_fn=lambda x: (isinstance(x, (BasicLayer, PatchEmbed)))
        )


class BaseLightningModuleGNN(LightningTrackingParent):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        # forward pass + loss computation
        y_pred = self(batch)
        loss = self.loss(y_pred, batch.y)

        # define log dictionary
        log_dict = {"train_loss": loss}
        # compute metrics
        for metric in self.metrics:
            log_dict.update({f"train_{metric.name}": metric(y_pred, batch.y)})
        # log the outputs
        self.callback_metrics = {**self.callback_metrics, **log_dict}

        # log the train_loss
        self.log("train_loss", loss)

        # mlflow logs
        self._training_loss = loss
        self._trn_loss['sum'] += loss
        self._trn_loss['steps'] += 1
        
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        # forward pass + loss computation
        y_pred = self(batch)
        loss = self.loss(y_pred, batch.y)
        
        # define log dictionary
        log_dict = {"val_loss": loss}
        # compute metrics
        for metric in self.metrics:
            log_dict.update({f"val_{metric.name}": metric(y_pred, batch.y)})
        # log the outputs
        self.callback_metrics = {**self.callback_metrics, **log_dict}

        # log the val_loss
        self.log("val_loss", loss)
        
        # mlflow logs
        self._validation_loss = loss
        self._vld_loss['sum'] += loss
        self._vld_loss['steps'] += 1
        
        return {"loss": loss}

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


class GraphUNet(BaseLightningModuleGNN):
    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        out_channels: int,
        K_pool: int,
        nodes_per_graph: int,
        edge_dropout_rate: float,
        node_dropout_rate: float,
        activation: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.edge_dropout_rate = edge_dropout_rate
        self.node_dropout_rate = node_dropout_rate
        self.activation = eval(activation)

        # Top-K pooling setup
        if K_pool == -1:
            K_pool = nodes_per_graph / 2
        pool_ratios = [K_pool / nodes_per_graph, 0.5]
        self.unet = torch_geometric.nn.GraphUNet(
            in_channels=in_channels,
            hidden_channels=hid_channels,
            out_channels=out_channels,
            depth=3,
            pool_ratios=pool_ratios,
        )

    def forward(self, data):
        edge_index, _ = dropout_edge(
            data.edge_index,
            p=self.edge_dropout_rate,
            force_undirected=True,
            training=self.training,
        )
        x = F.dropout(data.x, p=self.node_dropout_rate, training=self.training)
        x = self.unet(x, edge_index)
        return self.activation(x)
