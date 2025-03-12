from lightning.pytorch.callbacks import Callback
from lightning import Trainer, LightningModule
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests
import json
import os


class BenchmarkCSV(Callback):
    def __init__(self, filename) -> None:
        self.filename = filename

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if trainer.global_rank == 0:
            # get the metrics from the trainer
            metrics = pl_module.callback_metrics
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


class DiscordLog(Callback):
    def __init__(
        self,
        model_name: str = "default",
        webhook_url: str = None,
        benchmark_csv: str = None,
        msg_every_n_epochs: int = 1,
        plot_every_n_epochs: int = 5,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.webhook_url = webhook_url
        self.benchmark_csv = benchmark_csv
        self.msg_every_n_epochs = msg_every_n_epochs
        self.plot_every_n_epochs = plot_every_n_epochs

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        # only process 0
        if trainer.global_rank == 0:
            # send a message only if we have a `webhook_url` and `msg_every_n_epochs` epochs have passed
            if (
                self.webhook_url
                and (trainer.current_epoch + 1) % self.msg_every_n_epochs == 0
            ):
                try:
                    self.__log_message(trainer=trainer)
                except Exception as e:
                    print(f"Error encountered on discord callback. {e}")
                # send a message only if we have a `webhook_url` and a `benchmark_csv` and `plot_every_n_epochs` epochs have passed
            if (
                self.webhook_url
                and self.benchmark_csv
                and (trainer.current_epoch + 1) % self.plot_every_n_epochs == 0
                and trainer.current_epoch > 1
            ):
                try:
                    self.__log_plot(trainer=trainer)
                    self.__log_grad_plot(trainer=trainer)
                except Exception as e:
                    print(f"Error encountered on discord callback. {e}")

    def __log_message(self, trainer: Trainer):
        # get the metrics from the trainer
        metrics = trainer.callback_metrics
        # create message header
        message = f"[{self.model_name}] Epoch [{trainer.current_epoch+1}/{trainer.max_epochs}]\n"
        # put metrics information for each message row
        for key, value in metrics.items():
            message += f"   {key}: {value.item():.4f}\n"
        # create data message
        data = {"content": message}
        # post to the message to the webhook
        requests.post(
            self.webhook_url,
            data=json.dumps(data),
            headers={"Content-Type": "application/json"},
        )

    def __log_plot(self, trainer: Trainer):
        """In order to generate the plots, `trainer` must contain metrics in the format:
        Train : `train_{key}` for each passed metrics
        Valid : `val_{key}` for each passed metrics

        """
        df = pd.read_csv(self.benchmark_csv, index_col=0)
        # get the metrics from the trainer
        metrics = trainer.callback_metrics
        # get metrics keys
        metrics_keys = [m.split("train_")[-1] for m in metrics if m.startswith("train")]
        # for each key
        for key in metrics_keys:
            plt.figure(figsize=(6, 3))
            plt.plot(
                np.arange(len(df)),
                df[f"train_{key}"],
                label=f"Train {key.capitalize()}",
            )
            plt.plot(
                np.arange(len(df)), df[f"val_{key}"], label=f"Valid {key.capitalize()}"
            )
            plt.title(f"{self.model_name} benchmark")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            outfile = os.path.join(
                os.path.dirname(self.benchmark_csv),
                f"plot_{key}_{trainer.current_epoch}.png",
            )
            plt.savefig(outfile, dpi=200)
            # prepare a payload to send the image
            message = f"Metrics {key.capitalize()} Plot"
            files = {
                "payload_json": (
                    None,
                    '{"content": "' + message + '"}',
                ),  # None in this tuple sets no filename and is needed to send the text
                f"{outfile}": open(outfile, "rb"),
            }
            # post to the message to the webhook
            requests.post(self.webhook_url, files=files)
            os.remove(outfile)

    def __log_grad_plot(self, trainer: Trainer):
        df = pd.read_csv(self.benchmark_csv, index_col=0)
        # get the metrics from the trainer
        metrics = trainer.callback_metrics
        if not "grad" in metrics.keys():
            return
        plt.figure(figsize=(6, 3))
        plt.plot(np.arange(len(df)), df[f"grad"], label=f"Gradient Norm")
        plt.title(f"{self.model_name} benchmark")
        plt.xlabel("Epochs")
        plt.ylabel("Norm")
        plt.legend()
        outfile = os.path.join(
            os.path.dirname(self.benchmark_csv),
            f"plot_grad_{trainer.current_epoch}.png",
        )
        plt.savefig(outfile, dpi=200)
        # prepare a payload to send the image
        message = f"Gradient Norm Plot"
        files = {
            "payload_json": (
                None,
                '{"content": "' + message + '"}',
            ),  # None in this tuple sets no filename and is needed to send the text
            f"{outfile}": open(outfile, "rb"),
        }
        # post to the message to the webhook
        requests.post(self.webhook_url, files=files)
        os.remove(outfile)
