from lightning import Trainer, LightningModule
import pandas as pd
import os



class FabricBenchmark:
    def __init__(self, filename) -> None:
        self.filename = filename

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.global_rank == 0:
            # get the metrics from the trainer
            metrics = trainer.callback_metrics
            # create csv file if not exists
            if not os.path.exists(self.filename):
                # define csv columns
                columns = [key for key in metrics.keys()]
                # create empty DataFrame
                self.df = pd.DataFrame(columns=columns)
                # save the DataFrame to disk
                self.df.to_csv(self.filename)
            else:
                # get the DataFrame from disk
                self.df = pd.read_csv(self.filename, index_col=0)
                # get the columns
                columns = self.df.columns
            # create the row to be added to DataFrame
            row = [metrics[col].item() for col in columns]
            # add to the DataFrame the row
            self.df.loc[len(self.df.index)] = row
            # store the data to the csv file
            self.df.to_csv(self.filename)
        return



class FabricCheckpoint:
    def __init__(self, dst, monitor: str = 'val_loss', verbose: bool = False) -> None:
        self.dst = dst
        self.monitor = monitor
        self.verbose = verbose
        self.global_min_loss = 9999.9
        self.df = None

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # only process 0
        if trainer.global_rank == 0:
            # collect training and validation metrics
            self.__collect_metrics(trainer.callback_metrics)
            # save checkpoint if necessary
            self.__checkpoint(trainer)

    def __collect_metrics(self, metrics):
        # define csv columns
        columns = [key for key in metrics.keys()]
        data = {}
        for col in columns: data.update({col: [metrics[col].item()]})
        df = pd.DataFrame(data=data)
        if self.df is None:
            self.df = df
        else:
            self.df = pd.concat([self.df, df]).reset_index(drop=True)

    def __checkpoint(self, trainer):
        # get the loss list that we are monitoring
        losses = self.df[self.monitor].to_numpy()
        # get the current loss
        cur_loss = losses[-1]
        # if we reached a new minimum
        if cur_loss < self.global_min_loss:
            # get the checkpoint output filename
            path = os.path.join(self.dst, f"epoch-{trainer.current_epoch+1:04d}-val_loss-{cur_loss:.2f}.ckpt")
            # update the global minimum with the new one
            self.global_min_loss = cur_loss
            # eventually print the update
            if self.verbose: print(f'Epoch [{trainer.current_epoch+1}/{trainer.max_epochs}]: {self.monitor} improved from {self.global_min_loss} to {cur_loss}, saving checkpoint to {path}')
            # save the model to disk
            trainer.fabric.save(path, {'model':trainer.model, 'optimizer':trainer.optimizer})
