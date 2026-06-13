import os
import json
import logging
import torch

logger = logging.getLogger(__name__)


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            logger.info(
                f"EarlyStopping counter: {self.counter}/{self.patience}"
            )
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info("Early stopping triggered")
        else:
            self.best_loss = val_loss
            self.counter = 0

    def state_dict(self):
        return {
            "best_loss": self.best_loss,
            "counter": self.counter,
            "should_stop": self.should_stop,
        }

    def load_state_dict(self, state):
        self.best_loss = state["best_loss"]
        self.counter = state["counter"]
        self.should_stop = state["should_stop"]


class ModelCheckpoint:
    def __init__(self, save_dir="./checkpoints", save_best=True, monitor="val_loss"):
        self.save_dir = save_dir
        self.save_best = save_best
        self.monitor = monitor
        self.best_metric = None
        os.makedirs(save_dir, exist_ok=True)

    def __call__(self, model, metric_value, epoch):
        if self.monitor == "val_loss":
            is_better = (
                self.best_metric is None or metric_value < self.best_metric
            )
        else:
            is_better = (
                self.best_metric is None or metric_value > self.best_metric
            )

        last_path = os.path.join(self.save_dir, "last_model.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                self.monitor: metric_value,
            },
            last_path,
        )

        if is_better and self.save_best:
            self.best_metric = metric_value
            best_path = os.path.join(self.save_dir, "best_model.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    self.monitor: metric_value,
                },
                best_path,
            )
            logger.info(
                f"Best model saved with {self.monitor}={metric_value:.6f}"
            )


class TrainingLogger:
    def __init__(self, log_dir="./logs", log_interval=10):
        self.log_dir = log_dir
        self.log_interval = log_interval
        self.history = []
        os.makedirs(log_dir, exist_ok=True)

    def __call__(self, epoch, phase, loss, metrics=None, global_step=None):
        entry = {
            "epoch": epoch,
            "phase": phase,
            "loss": loss,
        }
        if metrics:
            entry.update(metrics)
        if global_step is not None:
            entry["global_step"] = global_step
        self.history.append(entry)

        if phase == "train" and global_step and global_step % self.log_interval == 0:
            logger.info(
                f"Epoch [{epoch}] Step [{global_step}] Loss: {loss:.6f}"
            )

    def save_history(self, filename="training_log.json"):
        path = os.path.join(self.log_dir, filename)
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Training log saved to {path}")


class CallbackManager:
    def __init__(self, config):
        cb_config = config.get("callbacks", {})

        self.early_stopping = None
        self.checkpoint = None
        self.logger_cb = None

        es_cfg = cb_config.get("early_stopping", {})
        if es_cfg.get("enabled", True):
            self.early_stopping = EarlyStopping(
                patience=es_cfg.get("patience", 10),
                min_delta=es_cfg.get("min_delta", 0.001),
            )

        ckpt_cfg = cb_config.get("checkpoint", {})
        if ckpt_cfg.get("enabled", True):
            self.checkpoint = ModelCheckpoint(
                save_dir=ckpt_cfg.get("save_dir", "./checkpoints"),
                save_best=ckpt_cfg.get("save_best", True),
                monitor=ckpt_cfg.get("monitor", "val_loss"),
            )

        log_cfg = cb_config.get("logging", {})
        self.logger_cb = TrainingLogger(
            log_dir=log_cfg.get("log_dir", "./logs"),
            log_interval=log_cfg.get("log_interval", 10),
        )

    def on_epoch_end(self, model, epoch, val_loss, metrics=None):
        if self.checkpoint:
            self.checkpoint(model, val_loss, epoch)

        if self.logger_cb:
            self.logger_cb(epoch, "val", val_loss, metrics)

        should_stop = False
        if self.early_stopping:
            self.early_stopping(val_loss)
            should_stop = self.early_stopping.should_stop

        return should_stop

    def on_train_step(self, epoch, loss, global_step):
        if self.logger_cb:
            self.logger_cb(epoch, "train", loss, global_step=global_step)

    def on_train_end(self):
        if self.logger_cb:
            self.logger_cb.save_history()
