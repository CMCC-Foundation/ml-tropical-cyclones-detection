from typing import Any, Iterable, List, Literal, Optional, Tuple, Mapping, Union, cast
from lightning.pytorch.utilities.model_summary import ModelSummary
from lightning.fabric.accelerators import Accelerator
from lightning_utilities import apply_to_collection
from lightning.fabric.strategies import Strategy
from lightning.fabric.loggers import Logger
from functools import partial
import lightning as L
from tqdm import tqdm
import logging
import torch
import os

from tropical_cyclone.utils import seed_everything



class FabricTrainer:
    def __init__(
        self,
        accelerator: Union[str, Accelerator] = "auto",
        strategy: Union[str, Strategy] = "auto",
        devices: Union[List[int], str, int] = "auto",
        num_nodes: int = 1, 
        precision: Union[str, int] = "32-true",
        plugins: Optional[Union[str, Any]] = None,
        callbacks: Optional[Union[List[Any], Any]] = None,
        loggers: Optional[Union[Logger, List[Logger]]] = None,
        max_epochs: Optional[int] = 1000,
        max_steps: Optional[int] = None,
        grad_accum_steps: int = 1,
        limit_train_batches: Union[int, float] = float("inf"),
        limit_val_batches: Union[int, float] = float("inf"),
        validation_frequency: int = 1,
        use_distributed_sampler: bool = True,
        checkpoint_dir: str = None, 
        checkpoint_frequency: int = 1, 
        seed: int = 42, 
    ) -> None:
        """Exemplary Trainer with Fabric. This is a very simple trainer focused on readablity but with reduced
        featureset. As a trainer with more included features, we recommend using the
        :class:`lightning.pytorch.Trainer`.

        Args:
            accelerator: The hardware to run on. Possible choices are:
                ``"cpu"``, ``"cuda"``, ``"mps"``, ``"gpu"``, ``"tpu"``, ``"auto"``.
            strategy: Strategy for how to run across multiple devices. Possible choices are:
                ``"dp"``, ``"ddp"``, ``"ddp_spawn"``, ``"deepspeed"``, ``"fsdp"``.
            devices: Number of devices to train on (``int``),
                which GPUs to train on (``list`` or ``str``), or ``"auto"``.
                The value applies per node.
            precision: Double precision (``"64"``), full precision (``"32"``), half precision AMP (``"16-mixed"``),
                or bfloat16 precision AMP (``"bf16-mixed"``).
            plugins: One or several custom plugins
            callbacks: A single callback or a list of callbacks. The following hooks are supported:
                - on_train_epoch_start
                - on train_epoch_end
                - on_train_batch_start
                - on_train_batch_end
                - on_before_backward
                - on_after_backward
                - on_before_zero_grad
                - on_before_optimizer_step
                - on_validation_model_eval
                - on_validation_model_train
                - on_validation_epoch_start
                - on_validation_epoch_end
                - on_validation_batch_start
                - on_validation_batch_end

            loggers: A single logger or a list of loggers. See :meth:`~lightning.fabric.fabric.Fabric.log` for more
                information.

            max_epochs: The maximum number of epochs to train
            max_steps: The maximum number of (optimizer) steps to train
            grad_accum_steps: How many batches to process before each optimizer step
            limit_train_batches: Limits the number of train batches per epoch
                If greater than number of batches in the dataloader, this has no effect.
            limit_val_batches: Limits the number of validation batches per epoch.
                If greater than number of batches in the dataloader, this has no effect.
            validation_frequency: How many epochs to run before each validation epoch.
            use_distributed_sampler: Wraps the sampler of each dataloader with a respective distributed-aware sampler
                in case of distributed training.
            checkpoint_dir: Directory to store checkpoints to.
            checkpoint_frequency: How many epochs to run before each checkpoint is written.

        Warning:
            callbacks written for the lightning trainer (especially making assumptions on the trainer), won't work!

        """
        # init fabric accelerator
        self.fabric = L.Fabric(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            num_nodes=num_nodes,
            precision=precision,
            plugins=plugins,
            callbacks=callbacks,
            loggers=loggers,
        )

        # launch fabric
        self.fabric.launch()

        # get info about the nodes
        self.world_size = self.fabric.world_size
        self.node_rank = self.fabric.node_rank
        self.global_rank = self.fabric.global_rank
        self.local_rank = self.fabric.local_rank

        # store random seed
        self.seed = seed

        # seed the run (add the global rank to differentiate the parallel executions)
        seed_everything(seed=self.seed+self.global_rank)

        self.global_step = 0
        self.grad_accum_steps: int = grad_accum_steps
        self.current_epoch = 0

        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.should_stop = False

        # ensures limit_X_batches is either int or inf
        if not isinstance(limit_train_batches, int):
            assert limit_train_batches == float("inf")

        if not isinstance(limit_val_batches, int):
            assert limit_val_batches == float("inf")

        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.validation_frequency = validation_frequency
        self.use_distributed_sampler = use_distributed_sampler
        self._current_train_return: Union[torch.Tensor, Mapping[str, Any]] = {}
        self._current_val_return: Optional[Union[torch.Tensor, Mapping[str, Any]]] = {}

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_frequency = checkpoint_frequency

    def setup(
        self, 
        model: L.LightningModule, 
        optimizer_cls, 
        optimizer_args, 
        scheduler_cls = None, 
        scheduler_args = None, 
        checkpoint: str = None, 
    ):
        """The fabric model and optimizer setup for the training.
        """
        # log
        logging.info(f'Fabric Trainer Setup')

        # load previous model if provided
        if checkpoint:
            state_dict = self.fabric.load(checkpoint)
            model.load_state_dict(state_dict['model'])
            # log
            logging.info(f'Model checkpoint provided. Restored weights from checkpoint at {checkpoint}')

        # print model summary
        if self.fabric.global_rank == 0: print(ModelSummary(model=model, max_depth=2))

        # setup the model in sync
        self.model = self.fabric.setup_module(model)

        # log
        logging.info(f'Model distributed in sync')

        # init the optimizer
        self.optimizer = optimizer_cls(self.model.parameters(), **optimizer_args)

        # log
        logging.info(f'Optimizer initialized {optimizer_cls}')

        # init the scheduler if provided
        if scheduler_cls and scheduler_args:
            self.scheduler_cfg = {
                'scheduler': scheduler_cls(optimizer = self.optimizer, **scheduler_args), 
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1,
                }
            # log
            logging.info(f'Scheduler initialized {scheduler_cls}')
        else:
            self.scheduler_cfg = None

        # load previous optimizer if provided
        if checkpoint and 'optimizer' in state_dict.keys():
            self.optimizer.load_state_dict(state_dict['optimizer'])
            # log
            logging.info(f'Restored optimizer from checkpoint at {checkpoint}')

        # setup optimizer among replicas in sync
        self.optimizer = self.fabric.setup_optimizers(self.optimizer)

        # log
        logging.info(f'Optimizer distributed in sync')

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        # ckpt_path: Optional[str] = None,
    ):
        """The main entrypoint of the trainer, triggering the actual training.

        Args:
            model: the LightningModule to train.
                Can have the same hooks as :attr:`callbacks` (see :meth:`MyCustomTrainer.__init__`).
            train_loader: the training dataloader. Has to be an iterable returning batches.
            val_loader: the validation dataloader. Has to be an iterable returning batches.
                If not specified, no validation will run.
            ckpt_path: Path to previous checkpoints to resume training from.
                If specified, will always look for the latest checkpoint within the given directory.

        """

        # setup dataloaders
        train_loader = self.fabric.setup_dataloaders(train_loader, use_distributed_sampler=self.use_distributed_sampler)
        if val_loader is not None:
            val_loader = self.fabric.setup_dataloaders(val_loader, use_distributed_sampler=self.use_distributed_sampler)

        # log
        logging.info(f'Dataloaders distributed in sync')

        self.call("on_fit_start", trainer=self, pl_module=self.model)

        # log
        logging.info(f'Training started')

        while not self.should_stop:
            # log
            logging.info(f'Epoch [{self.current_epoch+1}/{self.max_epochs}]')

            # execute training loop
            self.train_loop(self.model, self.optimizer, train_loader, limit_batches=self.limit_train_batches, scheduler_cfg=self.scheduler_cfg)

            # execute validation loop
            if self.should_validate:
                self.val_loop(self.model, val_loader, limit_batches=self.limit_val_batches)

            # step with the LR scheduler
            self.step_scheduler(self.model, self.scheduler_cfg, level="epoch", current_value=self.current_epoch)

            # add 1 to the current epoch
            self.current_epoch += 1

            # stopping condition on epoch level
            if self.max_epochs is not None and self.current_epoch >= self.max_epochs:
                self.should_stop = True

            # assemble state (current epoch and global step will be added in save)
            state = {"model": self.model, "optimizer": self.optimizer, 'scheduler': self.scheduler_cfg}

            # save model state
            self.save(state)

        # reset for next fit call
        self.should_stop = False

    def train_loop(
        self,
        model: L.LightningModule,
        optimizer: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        limit_batches: Union[int, float] = float("inf"),
        scheduler_cfg: Optional[Mapping[str, Union[L.fabric.utilities.types.LRScheduler, bool, str, int]]] = None,
    ):
        """The training loop running a single training epoch.

        Args:
            model: the LightningModule to train
            optimizer: the optimizer, optimizing the LightningModule.
            train_loader: The dataloader yielding the training batches.
            limit_batches: Limits the batches during this training epoch.
                If greater than the number of batches in the ``train_loader``, this has no effect.
            scheduler_cfg: The learning rate scheduler configuration.
                Have a look at :meth:`~lightning.pytorch.core.LightningModule.configure_optimizers`
                for supported values.

        """
        self.call("on_train_epoch_start", trainer=self, pl_module=self.model)

        # print the epoch number
        if self.fabric.global_rank == 0: print(f'Epoch [{self.current_epoch+1}/{self.max_epochs}]')

        # set the tqdm progress bar
        iterable = self.progbar_wrapper(train_loader, total=min(len(train_loader), limit_batches), desc=f"Train")

        for batch_idx, batch in enumerate(iterable):
            # end epoch if stopping training completely or max batches for this epoch reached
            if self.should_stop or batch_idx >= limit_batches:
                break

            self.call("on_train_batch_start", trainer=self, pl_module=self.model, batch=batch, batch_idx=batch_idx)

            # check if optimizer should step in gradient accumulation
            should_optim_step = batch_idx % self.grad_accum_steps == 0

            if should_optim_step:
                # currently only supports a single optimizer
                self.call("on_before_optimizer_step", trainer=self, pl_module=self.model, optimizer=optimizer)

                # optimizer step runs train step internally through closure
                optimizer.step(partial(self.training_step, model=model, batch=batch, batch_idx=batch_idx))

                self.call("on_before_zero_grad", trainer=self, pl_module=self.model, optimizer=optimizer)

                # zero grad the optimizer
                optimizer.zero_grad()
            else:
                # gradient accumulation -> no optimizer step
                self.training_step(model=model, batch=batch, batch_idx=batch_idx)

            self.call("on_train_batch_end", trainer=self, pl_module=self.model, outputs=self._current_train_return, batch=batch, batch_idx=batch_idx)

            # this guard ensures, we only step the scheduler once per global step
            if should_optim_step:
                self.step_scheduler(model, scheduler_cfg, level="step", current_value=self.global_step)

            # add output values to progress bar
            self._format_iterable(iterable, self.model.callback_metrics, "", "train")

            # only increase global step if optimizer stepped
            self.global_step += int(should_optim_step)

            # stopping criterion on step level
            if self.max_steps is not None and self.global_step >= self.max_steps:
                self.should_stop = True
                break

        self.call("on_train_epoch_end", trainer=self, pl_module=self.model)

    def val_loop(
        self,
        model: L.LightningModule,
        val_loader: Optional[torch.utils.data.DataLoader],
        limit_batches: Union[int, float] = float("inf"),
    ):
        """The validation loop ruunning a single validation epoch.

        Args:
            model: the LightningModule to evaluate
            val_loader: The dataloader yielding the validation batches.
            limit_batches: Limits the batches during this validation epoch.
                If greater than the number of batches in the ``val_loader``, this has no effect.

        """
        # no validation if val_loader wasn't passed
        if val_loader is None:
            return

        self.fabric.call("on_validation_model_eval")  # calls `model.eval()`

        # disable gradient computation
        torch.set_grad_enabled(False)

        self.call("on_validation_epoch_start", trainer=self, pl_module=self.model)

        # create validation tqdm iterable
        iterable = self.progbar_wrapper(val_loader, total=min(len(val_loader), limit_batches), desc="Valid")

        # iterate over validation batches
        for batch_idx, batch in enumerate(iterable):
            # end epoch if stopping training completely or max batches for this epoch reached
            if self.should_stop or batch_idx >= limit_batches:
                break

            self.call("on_validation_batch_start", trainer=self, pl_module=self.model, batch=batch, batch_idx=batch_idx)

            # apply model validation step and return the step outputs
            outputs = model.validation_step(batch, batch_idx)

            # avoid gradients in stored/accumulated values -> prevents potential OOM
            outputs = apply_to_collection(outputs, torch.Tensor, lambda x: x.detach())

            self.call("on_validation_batch_end", trainer=self, pl_module=self.model, outputs=outputs, batch=batch, batch_idx=batch_idx)

            # store detached outputs
            self._current_val_return = outputs

            # update tqdm iterable with the model metrics
            self._format_iterable(iterable, self.model.callback_metrics, "", "val")

        # get model callbacks metrics for before calling the callbacks
        self.callback_metrics = self.model.callback_metrics

        self.call("on_validation_epoch_end", trainer=self, pl_module=self.model)

        self.fabric.call("on_validation_model_train")
        torch.set_grad_enabled(True)

    def training_step(self, model: L.LightningModule, batch: Any, batch_idx: int) -> torch.Tensor:
        """A single training step, running forward and backward. The optimizer step is called separately, as this is
        given as a closure to the optimizer step.

        Args:
            model: the lightning module to train
            batch: the batch to run the forward on
            batch_idx: index of the current batch w.r.t the current epoch

        """
        # apply model training step and return the step outputs
        outputs: Union[torch.Tensor, Mapping[str, Any]] = model.training_step(batch, batch_idx=batch_idx)

        # get the loss value from the outputs
        loss = outputs if isinstance(outputs, torch.Tensor) else outputs["loss"]

        self.call("on_before_backward", trainer=self, pl_module=self.model, loss=loss)

        # backward the loss
        self.fabric.backward(loss)

        self.call("on_after_backward", trainer=self, pl_module=self.model)

        # avoid gradients in stored/accumulated values -> prevents potential OOM
        self._current_train_return = apply_to_collection(outputs, dtype=torch.Tensor, function=lambda x: x.detach())

        return loss

    def step_scheduler(
        self,
        model: L.LightningModule,
        scheduler_cfg: Optional[Mapping[str, Union[L.fabric.utilities.types.LRScheduler, bool, str, int]]],
        level: Literal["step", "epoch"],
        current_value: int,
    ) -> None:
        """Steps the learning rate scheduler if necessary.

        Args:
            model: The LightningModule to train
            scheduler_cfg: The learning rate scheduler configuration.
                Have a look at :meth:`lightning.pytorch.LightningModule.configure_optimizers` for supported values.
            level: whether we are trying to step on epoch- or step-level
            current_value: Holds the current_epoch if ``level==epoch``, else holds the ``global_step``

        """
        # no scheduler
        if scheduler_cfg is None:
            return

        # wrong interval (step vs. epoch)
        if scheduler_cfg["interval"] != level:
            return

        # right interval, but wrong step wrt frequency
        if current_value % cast(int, scheduler_cfg["frequency"]) != 0:
            return

        # assemble potential monitored values
        possible_monitor_vals = {None: None}
        if isinstance(self._current_train_return, torch.Tensor):
            possible_monitor_vals.update({"train_loss": self._current_train_return.item()})
        elif isinstance(self._current_train_return, Mapping):
            possible_monitor_vals.update({"train_" + k: v for k, v in self._current_train_return.items()})

        if isinstance(self._current_val_return, torch.Tensor):
            possible_monitor_vals.update({"val_loss": self._current_val_return})
        elif isinstance(self._current_val_return, Mapping):
            possible_monitor_vals.update({"val_" + k: v for k, v in self._current_val_return.items()})

        try:
            monitor = possible_monitor_vals[cast(Optional[str], scheduler_cfg["monitor"])]
        except KeyError as ex:
            possible_keys = list(possible_monitor_vals.keys())
            raise KeyError(
                f"monitor {scheduler_cfg['monitor']} is invalid. Possible values are {possible_keys}."
            ) from ex

        # rely on model hook for actual step
        model.lr_scheduler_step(scheduler_cfg["scheduler"], monitor)

    @property
    def should_validate(self) -> bool:
        """Whether to currently run validation."""
        return self.current_epoch % self.validation_frequency == 0

    def progbar_wrapper(self, iterable: Iterable, total: int, **kwargs: Any):
        """Wraps the iterable with tqdm for global rank zero.

        Args:
            iterable: the iterable to wrap with tqdm
            total: the total length of the iterable, necessary in case the number of batches was limited.

        """
        if self.fabric.is_global_zero:
            return tqdm(iterable, total=total, **kwargs)
        return iterable

    def load(self, state: Optional[Mapping], path: str) -> None:
        """Loads a checkpoint from a given file into state.

        Args:
            state: a mapping contaning model, optimizer and lr scheduler
            path: the path to load the checkpoint from

        """
        if state is None:
            state = {}

        remainder = self.fabric.load(path, state)
        self.global_step = remainder.pop("global_step")
        self.current_epoch = remainder.pop("current_epoch")

        if remainder:
            raise RuntimeError(f"Unused Checkpoint Values: {remainder}")

    def save(self, state: Optional[Mapping]) -> None:
        """Saves a checkpoint to the ``checkpoint_dir``

        Args:
            state: A mapping containing model, optimizer and lr scheduler.

        """
        if state is None:
            state = {}

        state.update(global_step=self.global_step, current_epoch=self.current_epoch)

        if self.checkpoint_dir:
            self.fabric.save(os.path.join(self.checkpoint_dir, f"epoch-{self.current_epoch:04d}.ckpt"), state)

    @staticmethod
    def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
        """Returns the latest checkpoint from the ``checkpoint_dir``

        Args:
            checkpoint_dir: the directory to search for checkpoints

        """
        if not os.path.isdir(checkpoint_dir):
            return None

        items = sorted(os.listdir(checkpoint_dir))

        if not items:
            return None

        return os.path.join(checkpoint_dir, items[-1])

    def _parse_optimizers_schedulers(
        self, configure_optim_output
    ) -> Tuple[
        Optional[L.fabric.utilities.types.Optimizable],
        Optional[Mapping[str, Union[L.fabric.utilities.types.LRScheduler, bool, str, int]]],
    ]:
        """Recursively parses the output of :meth:`lightning.pytorch.LightningModule.configure_optimizers`.

        Args:
            configure_optim_output: The output of ``configure_optimizers``.
                For supported values, please refer to :meth:`lightning.pytorch.LightningModule.configure_optimizers`.

        """
        _lr_sched_defaults = {"interval": "epoch", "frequency": 1, "monitor": "val_loss"}

        # single optimizer
        if isinstance(configure_optim_output, L.fabric.utilities.types.Optimizable):
            return configure_optim_output, None

        # single lr scheduler
        if isinstance(configure_optim_output, L.fabric.utilities.types.LRScheduler):
            return None, _lr_sched_defaults.update(scheduler=configure_optim_output)

        # single lr scheduler config
        if isinstance(configure_optim_output, Mapping):
            _lr_sched_defaults.update(configure_optim_output)
            return None, _lr_sched_defaults

        # list or tuple
        if isinstance(configure_optim_output, (list, tuple)):
            if all(isinstance(_opt_cand, L.fabric.utilities.types.Optimizable) for _opt_cand in configure_optim_output):
                # single optimizer in list
                if len(configure_optim_output) == 1:
                    return configure_optim_output[0][0], None

                raise NotImplementedError("BYOT only supports a single optimizer")

            if all(
                isinstance(_lr_cand, (L.fabric.utilities.types.LRScheduler, Mapping))
                for _lr_cand in configure_optim_output
            ):
                # single scheduler in list
                if len(configure_optim_output) == 1:
                    return None, self._parse_optimizers_schedulers(configure_optim_output[0])[1]

            # optimizer and lr scheduler
            elif len(configure_optim_output) == 2:
                opt_cands, lr_cands = (
                    self._parse_optimizers_schedulers(configure_optim_output[0])[0],
                    self._parse_optimizers_schedulers(configure_optim_output[1])[1],
                )
                return opt_cands, lr_cands

        return None, None

    @staticmethod
    def _format_iterable(
        prog_bar, 
        candidates: Optional[Union[torch.Tensor, Mapping[str, Union[torch.Tensor, float, int]]]], 
        prefix: str, 
        status: str = 'train'
    ):
        """Adds values as postfix string to progressbar.

        Args:
            prog_bar: a progressbar (on global rank zero) or an iterable (every other rank).
            candidates: the values to add as postfix strings to the progressbar.
            prefix: the prefix to add to each of these values.

        """
        if isinstance(prog_bar, tqdm) and candidates is not None:
            postfix_str = ""
            float_candidates = apply_to_collection(candidates, torch.Tensor, lambda x: x.item())
            if isinstance(candidates, torch.Tensor):
                key = f" {prefix}_loss" if prefix!="" else f" loss"
                postfix_str += f"{key}: {float_candidates:.3f}"
            elif isinstance(candidates, Mapping):
                for k, v in float_candidates.items():
                    if status in k:
                        # k = k.split(f'{status}_')[-1]
                        pre =  f" {prefix}_{k}" if prefix!="" else f" {k}"
                        postfix_str += f"{pre}: {v:.3f}"
            if postfix_str:
                prog_bar.set_postfix_str(postfix_str)

    def call(self, hook_name, **kwargs):
        call_args = kwargs.copy()
        try:
            self.fabric.call(hook_name=hook_name, **call_args)
        except Exception as e:
            del call_args['trainer']; del call_args['pl_module']
            try:
                self.fabric.call(hook_name=hook_name, **call_args)
            except Exception as k:
                pass
