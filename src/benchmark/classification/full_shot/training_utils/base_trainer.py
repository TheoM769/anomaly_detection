import typing
import time
from abc import ABC, abstractmethod
from tqdm import tqdm
import gc

from torch.utils.data import DataLoader
import torch
import wandb

from .loss import Loss
from .metrics import Metric
from fvcore.nn import FlopCountAnalysis

class BaseTrainer(ABC):
    
    def __init__(self, optimizer, loss: Loss, train_metrics: Metric, val_metrics: Metric):
        
        self.optimizer = optimizer
        self.loss = loss
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.train_time = 0
    
    
    def model_prediction(self, model, images):
        return model(images)
    
    @abstractmethod
    def train_one_epoch(self, model, epoch, train_dataloader, nb_classes, device, wandb_run):
        pass

    @abstractmethod
    def log_image_table(self, images, predicted, labels, nb_classes, wandb_run, probs):
        pass

    @abstractmethod
    def validate_model(self, model, test_dl, nb_classes, device, wandb_run, log_images=True, batch_idx=0):
        pass

    @abstractmethod
    def evaluate(self, model, epoch, val_dataloader, nb_classes, device, wandb_run):
        pass
    
    def train(self, model, epochs, train_dataloader, validation_dataloader, nb_classes, device, metric_steps=5, save_path="", wandb_run=None, model_artifact=None, save_model=False, patience=10, min_delta=0.001):
        best_AP = 0.
        model.to(device)
        
        # Early stopping variables
        patience_counter = 0
        
        start_time = time.time()
        for epoch in range(epochs):
            model.train()
            avg_loss = self.train_one_epoch(model, epoch, train_dataloader, nb_classes, device, wandb_run)
            model.eval()
            print(f"train loss: {avg_loss}")

            if epoch % metric_steps == 0:
                metrics_results = self.train_metrics.compute_metrics(self, model, train_dataloader, device)
                vmetrics_results = self.val_metrics.compute_metrics(self, model, validation_dataloader, device, samples_set="val")
                print(vmetrics_results)
                avg_vloss = self.evaluate(model, epoch, validation_dataloader, nb_classes, device, wandb_run)
                if wandb_run is not None:
                    wandb_log = metrics_results | vmetrics_results | {"avg_val_loss": avg_vloss}
                    wandb_run.log(wandb_log)
                else:
                    vmetrics_results = self.train_metrics.compute_metrics(self, model, validation_dataloader, device, samples_set="val")
                    # Early stopping check
                current_AP = vmetrics_results["val_AP"]

                if current_AP > best_AP + min_delta:
                    best_AP = current_AP
                    patience_counter = 0
                    if len(save_path) > 1:
                        torch.save(model.state_dict(), save_path)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping triggered after {epoch + 1} epochs")
                        break
                
                if wandb_run is not None:
                    wandb_log = {"avg_train_loss": avg_loss, "epoch": epoch}
                    wandb_run.log(wandb_log)
            
        # if model_artifact is not None:
        #     model_artifact.add_file(save_path)
        #     wandb.save(save_path)
        #     if save_model:
        #         wandb_run.log_artifact(model_artifact)

        end_time = time.time()
        self.train_time = (end_time - start_time) / 60 # in minutes

    def _model_FPS(self, model, image_batch, device):
        image_batch = image_batch.to(device)
        # Warmup
        with torch.no_grad():
            for _ in tqdm(range(10), desc="Warmup"):
                _ = model(image_batch)
                if device == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                elif device == "mps":
                    torch.mps.synchronize()
            gc.collect()
        
        # Measure FPS
        times = []
        with torch.no_grad():
            for _ in tqdm(range(50), desc="FPS measurement"):  # Run multiple iterations for more stable measurement
                start = time.time()
                _ = model(image_batch)
                if device == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                elif device == "mps":
                    torch.mps.synchronize()
                end = time.time()
                times.append(end - start)
            gc.collect()
        
        # Calculate average FPS
        avg_time = sum(times) / len(times)
        fps = len(image_batch) / avg_time  # FPS = batch_size / time_per_batch
        return round(fps, 3)
    
    def _model_FLOPS(self, model, image_batch, nb_batches, device):
        with torch.inference_mode():
            # Take a single image from the batch and move to device
            image = image_batch[0].unsqueeze(0).to(device)

            # Compute FLOPs for a forward pass
            flops = FlopCountAnalysis(model, image)
            forward_flops = self._compute_forward_flops(flops)
            backward_flops = self._compute_backward_flops(flops, model)

        return forward_flops + 2 * backward_flops
    
    def _compute_forward_flops(self, flops: FlopCountAnalysis):
        total_forward_flops = flops.total()
        return total_forward_flops
    
    def _compute_backward_flops(self, flops: FlopCountAnalysis, model):
        analysis = flops.by_module()
        del analysis[""]
        del analysis["model"]
        params = []
        sum_flops = 0
        last_item = "start"
        for k, v in analysis.items():
            for model_params_name, model_params in dict(model.named_parameters()).items():
                if k in model_params_name and model_params.requires_grad and last_item not in k:
                    sum_flops += v
                    params.append(k)
                    last_item = k
                    break

        return sum_flops
