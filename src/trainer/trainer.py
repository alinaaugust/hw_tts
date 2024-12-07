from pathlib import Path

import pandas as pd

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]

        audio = batch["audio"]
        mel_spec = batch["mel_spec"]
        gen_audio = self.model(mel_spec)
        batch["generated_audio"] = gen_audio

        self.optimizer_disc.zero_grad()

        output_p, output_pred_p, _, _ = self.mpd(audio, gen_audio.detach())
        output_s, output_pred_s, _, _ = self.msd(audio, gen_audio.detach())

        mpd_loss = self.criterion.discriminator_gan_loss(output_pred_p, output_p)
        msd_loss = self.criterion.discriminator_gan_loss(output_pred_s, output_s)
        loss = mpd_loss + msd_loss

        loss.backward()
        self._clip_grad_norm(self.model.mpd)
        self._clip_grad_norm(self.model.msd)
        self.optimizer_disc.step()
        if self.lr_scheduler_disc is not None:
            self.lr_scheduler_disc.step()

        batch.update({"mpd_loss": mpd_loss, "msd_loss": msd_loss, "disc_loss": loss})

        self.optimizer_gen.zero_grad()

        mel_spec_loss = self.criterion.mel_loss(self.mel_spec(gen_audio), mel_spec)
        output_p, output_pred_p, feature_maps_p, feature_maps_pred_p = self.mpd(
            audio, gen_audio
        )
        output_s, output_pred_s, feature_maps_s, feature_maps_pred_s = self.msd(
            audio, gen_audio
        )
        gan_loss = self.criterion.generator_gan_loss(
            output_pred_p, output_p
        ) + self.criterion.generator_gan_loss(output_pred_s, output_s)
        fm_loss = self.criterion.fm_loss(
            feature_maps_pred_p, feature_maps_p
        ) + self.criterion.fm_loss(feature_maps_pred_s, feature_maps_s)
        loss = fm_loss + gan_loss + mel_spec_loss
        loss.backward()
        self._clip_grad_norm(self.model.generator)
        self.optimizer_gen.step()
        if self.lr_scheduler_gen is not None:
            self.lr_scheduler_gen.step()

        batch.update(
            {
                "mel_spec_loss": mel_spec_loss,
                "feature_matching_loss": fm_loss,
                "generator_adv_loss": gan_loss,
                "generator_loss": loss,
            }
        )

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            self.log_spectrogram(**batch)
            self.log_predictions(**batch)
        else:
            # Log Stuff
            self.log_spectrogram(**batch)
            self.log_predictions(**batch)

    def log_spectrogram(self, batch):
        spectrogram_for_plot = batch["mel_spec"][0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image("Initial melspectrogram", image)
        spectrogram_for_plot = self.mel_spec(batch["generated_audio"])[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image("Generated melspectrogram", image)

    def log_predictions(self, batch, examples_to_log=10):
        result = {}
        examples_to_log = min(examples_to_log, batch["audio"].shape[0])

        tuples = list(zip(batch["audio"], batch["generated_audio"]))

        for idx, (audio, gen_audio) in enumerate(tuples[:examples_to_log]):
            result[idx] = {
                "audio": self.writer.wandb.Audio(
                    audio.squeeze(0).detach().cpu().numpy(), sample_rate=22050
                ),
                "generated_audio": self.writer.wandb.Audio(
                    gen_audio.squeeze(0).detach().cpu().numpy(), sample_rate=22050
                ),
            }
        self.writer.add_table(
            "predictions", pd.DataFrame.from_dict(result, orient="index")
        )
