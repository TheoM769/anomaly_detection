import typing
from tqdm import tqdm
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns

from torcheval.metrics import MulticlassRecall, MulticlassPrecision, MulticlassF1Score, MulticlassAUPRC
from sklearn.metrics import precision_recall_curve, average_precision_score
import torch
import numpy as np

class Metric(ABC):
    
    def __init__(self, selected_metrics: list[str]):
        self.selected_metrics = selected_metrics
    
    @abstractmethod
    def compute_metrics(self, pred, labels):
        pass

        
class ClassificationMetrics(Metric):
    
    def __init__(self, nb_classes: int, selected_metrics: list[str], threshold_step: int = 20, train_ratio: float = 1.0, cost_weights: dict = None):
        super().__init__(selected_metrics)
        self.nb_classes = nb_classes
        self.threshold_step = threshold_step
        self.train_ratio = train_ratio
        self.epsilon = 1e-10
        self.cost_weights = cost_weights
        self.cls_mapping = {"f1": MulticlassF1Score(average="macro", num_classes=self.nb_classes),
                                 "precision": MulticlassPrecision(average="macro", num_classes=self.nb_classes),
                                 "recall": MulticlassRecall(average="macro", num_classes=self.nb_classes),
                                 "AP": MulticlassAUPRC(average="macro", num_classes=self.nb_classes),
                                 "f1_max": self._compute_f1_max,
                                 "precision@0.95": self._compute_precision_at_recall,
                                 "precision@0.90": self._compute_precision_at_recall2,
                                 "optimal_cost": self.compute_optimal_threshold_for_cost
                                }
    
    def _compute_f1_max(self, logits, labels):
        probs = torch.softmax(logits, dim=1)
        probs, labels = probs.detach().cpu(), labels.detach().cpu()
        precisions, recalls, thresholds = precision_recall_curve(labels, probs[:, 1])
        f1_scores = (2 * precisions * recalls) / (precisions + recalls + self.epsilon)
        max_f1_idx = np.nanargmax(f1_scores)
        f1_clf = f1_scores[max_f1_idx]
        precision_at_max_f1 = precisions[max_f1_idx]
        recall_at_max_f1 = recalls[max_f1_idx]
        return f1_clf, precision_at_max_f1, recall_at_max_f1
    
    def _compute_precision_at_recall(self, logits, labels):
        probs = torch.softmax(logits, dim=1)
        probs, labels = probs.detach().cpu(), labels.detach().cpu()
        precisions, recalls, thresholds = precision_recall_curve(labels, probs[:, 1])
        recall_target = 0.95
        idx = np.where(recalls >= recall_target)[0]
        if len(idx) > 0:
            precision_at_target_recall = max(precisions[idx])
        else:
            precision_at_target_recall = 0
        return precision_at_target_recall
    
    def _compute_precision_at_recall2(self, logits, labels):
        probs = torch.softmax(logits, dim=1)
        probs, labels = probs.detach().cpu(), labels.detach().cpu()
        precisions, recalls, thresholds = precision_recall_curve(labels, probs[:, 1])
        recall_target = 0.90
        idx = np.where(recalls >= recall_target)[0]
        if len(idx) > 0:
            precision_at_target_recall = max(precisions[idx])
        else:
            precision_at_target_recall = 0
        return precision_at_target_recall
    
    def compute_confusion_matrix_components(self, logits, labels, class_idx=None, threshold=0.5):
        """
        Compute true positives, true negatives, false positives, and false negatives.
        
        Args:
            logits: Model output logits (tensor)
            labels: Ground truth labels (tensor)
            class_idx: For multiclass, specify which class to compute metrics for.
                      If None, assumes binary classification with class 1 as positive.
            threshold: Decision threshold for binary classification (default: 0.5)
        
        Returns:
            dict: Dictionary containing 'tp', 'tn', 'fp', 'fn'
        """
        logits = logits.detach().cpu()
        labels = labels.detach().cpu()
        
        # Get predictions from logits
        if logits.dim() > 1 and logits.size(1) > 1:
            # Multiclass case - convert to probabilities and apply threshold
            if class_idx is None:
                class_idx = 1  # For binary classification, assume class 1 is positive
            probs = torch.softmax(logits, dim=1)
            predictions = (probs[:, class_idx] >= threshold).long()
        else:
            # Binary case with single output
            probs = torch.sigmoid(logits.squeeze())
            predictions = (probs >= threshold).long()
            class_idx = 1
        
        # Convert to binary problem for the specified class
        if logits.dim() > 1 and logits.size(1) > 2:  # Multiclass
            binary_labels = (labels == class_idx).long()
        else:  # Binary
            binary_labels = labels
        
        # Compute confusion matrix components
        tp = torch.sum((predictions == 1) & (binary_labels == 1)).item()
        tn = torch.sum((predictions == 0) & (binary_labels == 0)).item()
        fp = torch.sum((predictions == 1) & (binary_labels == 0)).item()
        fn = torch.sum((predictions == 0) & (binary_labels == 1)).item()
        
        return {
            'tp': tp,
            'tn': tn, 
            'fp': fp,
            'fn': fn
        }
    
    def compute_optimal_threshold_for_cost(self, logits, labels, cost_weights, thresholds=None):
        """
        Find the optimal threshold that minimizes the expected cost per sample.
        
        Args:
            logits: Model output logits (tensor)
            labels: Ground truth labels (tensor)
            cost_weights: Dictionary with weights for each confusion matrix component
                         e.g., {'tp': 10, 'tn': 1, 'fp': -5, 'fn': -20}
                         Positive weights represent benefits (we want more of these)
                         Negative weights represent penalties (we want fewer of these)
            thresholds: List of thresholds to evaluate. If None, uses 100 evenly spaced values [0, 1]
        
        Returns:
            dict: Dictionary containing 'optimal_threshold', 'min_expected_cost', 'threshold_costs'
        """
        # Default thresholds if not provided
        if thresholds is None:
            thresholds = np.linspace(0, 1, 100)
        
        expected_costs = []
        threshold_details = []
        
        for threshold in thresholds:
            # Use the existing confusion matrix function
            confusion_matrix = self.compute_confusion_matrix_components(logits, labels, threshold=threshold)
            
            # Calculate total samples
            total_samples = (confusion_matrix['tp'] + confusion_matrix['tn'] + 
                           confusion_matrix['fp'] + confusion_matrix['fn'])
            
            total_gain = (cost_weights.get('tp', 0) * confusion_matrix['tp'] + 
                          cost_weights.get('tn', 0) * confusion_matrix['tn'] + 
                          cost_weights.get('fp', 0) * confusion_matrix['fp'] + 
                          cost_weights.get('fn', 0) * confusion_matrix['fn'])
            
            # Normalize by number of samples to get expected cost per sample
            expected_gain = total_gain / total_samples if total_samples > 0 else 0
            
            expected_costs.append(expected_gain)
            threshold_details.append({
                'threshold': threshold,
                'expected_gain': expected_gain,
                'total_gain': total_gain,
                'total_samples': total_samples,
                **confusion_matrix  # Unpack tp, tn, fp, fn
            })
        
        # Find optimal threshold
        max_gain_idx = np.argmax(expected_costs)
        optimal_threshold = thresholds[max_gain_idx]
        max_expected_gain = expected_costs[max_gain_idx]
        
        return {
            'optimal_threshold': optimal_threshold,
            'max_expected_gain': max_expected_gain,
            'threshold_costs': threshold_details,
            'all_expected_costs': expected_costs,
            'all_thresholds': thresholds
        }
    
    def compute_metrics(self, trainer, model, dataloader, device, samples_set="train"):
        computed_metrics = {}
        
        # Single pass through dataloader to collect all logits and labels
        all_logits = []
        all_labels = []
        total_samples = len(dataloader.dataset)
        samples_to_use = int(total_samples * self.train_ratio) if samples_set == "train" else total_samples
        samples_processed = 0
        
        with torch.no_grad():
            print(f"Collecting predictions for {samples_set} set...")
            for batch in tqdm(dataloader, desc=f"Computing predictions for {samples_set}"):
                if samples_processed >= samples_to_use:
                    break
                    
                if len(batch) == 3:
                    images, processed_images, labels = batch
                else:
                    processed_images, labels = batch
                processed_images, labels = processed_images.to(device), labels.to(device)
                output_logits = trainer.model_prediction(model, processed_images)
                all_logits.append(output_logits)
                all_labels.append(labels)
                samples_processed += len(labels)
            
            # Concatenate all predictions
            all_logits = torch.cat(all_logits, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            
            # Now compute all metrics using the pre-computed logits
            for m in tqdm(self.selected_metrics, total=len(self.selected_metrics), unit="Metric", desc=f"Computing {samples_set} metrics"):
                if m in ["f1_max", "precision@0.95", "precision@0.90", "optimal_cost"]:
                    # These metrics need all logits at once
                    if m == "optimal_cost":
                        if self.cost_weights is None:
                            raise ValueError("cost_weights must be provided in constructor to use optimal_cost metric")
                        result = self.cls_mapping[m](all_logits, all_labels, self.cost_weights)
                    else:
                        result = self.cls_mapping[m](all_logits, all_labels)
                else:
                    # Standard metrics that can work with all data at once
                    metric = self.cls_mapping[m]
                    
                    # Convert logits to probabilities for AP computation
                    if m == "AP":
                        output_for_metric = torch.softmax(all_logits, dim=1)
                    else:
                        output_for_metric = all_logits
                    
                    # Update metric with all data at once
                    metric.update(output_for_metric, all_labels)
                    result = metric.compute()
                    metric.reset()
                    
                # Store results
                if isinstance(result, tuple):
                    for r, name in zip(result, ["f1", "precision", "recall"]):
                        computed_metrics[samples_set + "_" + m + "_" + name] = r
                elif isinstance(result, dict) and m == "optimal_cost":
                    # Handle the cost optimization result
                    computed_metrics[samples_set + "_optimal_threshold"] = result['optimal_threshold']
                    computed_metrics[samples_set + "_expected_gain"] = result['max_expected_gain']
                else:
                    computed_metrics[samples_set + "_" + m] = result
        
        return computed_metrics

    def plot_metrics(self, logits, labels, save_path):
        """
        Create and save plots for expected gain curve and AP curve with optimal threshold.
        
        Args:
            logits: Model output logits
            labels: Ground truth labels
            save_path: Path to save the plots
        """
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot 1: Expected Gain Curve
        if self.cost_weights is not None:
            result = self.compute_optimal_threshold_for_cost(logits, labels, self.cost_weights)
            thresholds = result['all_thresholds']
            expected_costs = result['all_expected_costs']
            optimal_threshold = result['optimal_threshold']
            
            ax1.plot(thresholds, expected_costs, 'b-', label='Expected Gain')
            ax1.axvline(x=optimal_threshold, color='r', linestyle='--', 
                       label=f'Optimal Threshold: {optimal_threshold:.3f}')
            ax1.set_xlabel('Threshold')
            ax1.set_ylabel('Expected Gain')
            ax1.set_title('Expected Gain vs Threshold')
            ax1.legend()
            ax1.grid(True)
        
        # Plot 2: AP Curve
        probs = torch.softmax(logits, dim=1).detach().cpu()
        labels = labels.detach().cpu()
        
        # For each class
        for i in range(self.nb_classes):
            precision, recall, thresholds = precision_recall_curve(
                (labels == i).float(), probs[:, i])
            ap = average_precision_score((labels == i).float(), probs[:, i])
            
            ax2.plot(recall, precision, label=f'Class {i} (AP={ap:.3f})')
            
            # If we have cost weights, mark the optimal threshold
            if self.cost_weights is not None:
                optimal_prob = result['optimal_threshold']
                # Find the recall at the optimal threshold
                optimal_idx = np.argmin(np.abs(thresholds - optimal_prob))
                if optimal_idx < len(recall):
                    ax2.plot(recall[optimal_idx], precision[optimal_idx], 'ro',
                            label=f'Optimal Threshold: {optimal_prob:.3f}')
        
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curves')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()