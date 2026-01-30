from tqdm import tqdm
import time
import math

import torch
import wandb
import numpy as np

from .base_trainer import BaseTrainer

class ClassificationTrainer(BaseTrainer):
    
    def __init__(self, optimizer, loss, train_metrics, val_metrics):
        super().__init__(optimizer, loss, train_metrics, val_metrics)
    
    def train_one_epoch(self, model, epoch, train_dataloader, nb_classes, device, wandb_run):
        running_loss = 0.
        
        model.train()
        
        for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), unit="Batch", desc=f"Train epoch {epoch}"):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            self.optimizer.zero_grad()
            output_logits = self.model_prediction(model, images)
            loss = self.loss.compute_loss(output_logits, labels)
            loss.backward()
            if wandb_run is not None:
                wandb_run.log({"train_loss": loss})
            self.optimizer.step()
            
            running_loss += loss.item()

        return running_loss / len(train_dataloader)
    
    def log_image_table(self, images, predicted, labels, nb_classes, wandb_run, probs, table_name):
        # Create a wandb Table to log images, labels and predictions to
        table = wandb.Table(
            columns=["image", "pred", "target"] + [f"score_{i}" for i in range(nb_classes)]
        )

        # Handle both torch tensors and numpy arrays
        if isinstance(images, torch.Tensor):
            images = images.to("cpu")
            predicted = predicted.to("cpu") 
            labels = labels.to("cpu")
            probs = probs.to("cpu")
            for img, pred, targ, prob in zip(images, predicted, labels, probs):
                table.add_data(wandb.Image(img[0].numpy() * 255), pred, targ, *prob.numpy())
        else:
            # Handle numpy arrays
            images = np.transpose(images, (0, 3, 1, 2))
            for img, pred, targ, prob in zip(images, predicted, labels, probs):
                table.add_data(wandb.Image(img), pred, targ, *prob)

        wandb_run.log({table_name: table}, commit=False)

    def validate_model(self, model, test_dl, noisy_test_dl, nb_classes, device, wandb_run, save_path="", log_images=True, batch_idx=2):
        if save_path != "":
            weights = torch.load(save_path, map_location=device)
            model.load_state_dict(weights)
        model.eval()
        
        for idx, dataloader in enumerate([test_dl, noisy_test_dl]):
            # Collect all logits and labels for plotting
            all_logits = []
            all_labels = []
            
            with torch.inference_mode():
                for idx, (images, processed_images, labels) in enumerate(dataloader):
                    processed_images, labels = processed_images.to(device), labels.to(device)

                    output_logits = self.model_prediction(model, processed_images)
                    _, predicted = torch.max(output_logits, 1)
                    
                    # Store logits and labels for plotting
                    all_logits.append(output_logits)
                    all_labels.append(labels)

                    if log_images and wandb_run is not None and idx == 0:
                        name = "predictions_table" if idx == 0 else "noisy_predictions_table"
                        self.log_image_table(images, predicted, labels, nb_classes, wandb_run, output_logits.softmax(dim=1), name)
                    
            # Concatenate all logits and labels
            all_logits = torch.cat(all_logits, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            
            # Save plots
            if save_path != "":
                suffix = "_noisy" if idx == 1 else ""
                plot_path = save_path.replace('.pth', f'_metrics_plot{suffix}.png')
                self.val_metrics.plot_metrics(all_logits, all_labels, plot_path)
                if wandb_run is not None:
                    wandb_run.log({f"metrics_plot_{suffix}": wandb.Image(plot_path)})

        metrics_results = self.val_metrics.compute_metrics(self, model, test_dl, device, samples_set="test")
        noisy_metrics_results = self.val_metrics.compute_metrics(self, model, noisy_test_dl, device, samples_set="test_noisy")
        FPS = self._model_FPS(model, processed_images, device)
        FLOPS = self._model_FLOPS(model, processed_images, len(test_dl), device)
        metrics_results = metrics_results | noisy_metrics_results | {"FPS": FPS, "FLOPS": FLOPS, "train_time": self.train_time}

        if wandb_run is not None:
            for k, v in metrics_results.items():
                wandb_run.summary[f"{k}"] = v

        return metrics_results
    
    def evaluate(self, model, epoch, val_dataloader, nb_classes, device, wandb_run):
        running_loss = 0.
        
        model.eval()
        
        with torch.no_grad():
            for i, vdata in tqdm(enumerate(val_dataloader), total=len(val_dataloader), unit="Batch", desc=f"Val epoch {epoch}"):
                images, processed_images, labels = vdata
                processed_images, labels = processed_images.to(device), labels.to(device)

                output_logits = self.model_prediction(model, processed_images)
                loss = self.loss.compute_loss(output_logits, labels)

                running_loss += loss.item()

        return running_loss / len(val_dataloader)
