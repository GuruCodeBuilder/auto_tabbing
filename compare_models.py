import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import time
import os
from tqdm import tqdm
import pickle

# Import from local modules
from data import cqt_data, split_training_valid, labels, SAMPLE_RATE_REF
from train import GuitarDataset, GuitarCNN
from train_old_rl import GuitarEnv

# Try to import stable_baselines3 for DQN model
try:
    from stable_baselines3 import DQN

    STABLE_BASELINES_AVAILABLE = True
except ImportError:
    print(
        "Warning: stable_baselines3 not available. DQN model comparison will be skipped."
    )
    STABLE_BASELINES_AVAILABLE = False

# Constants
IS_USING_FULL_CQT = False
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelComparator:
    def __init__(self):
        self.training_data, self.validation_data = split_training_valid(
            cqt_data, training_size=0.8
        )
        self.labels = labels
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.labels)

        # Results storage
        self.results = {"CNN": {}, "DQN": {}}

        # Create results directory
        os.makedirs("./comparison_results", exist_ok=True)
        os.makedirs("./comparison_results/graphs", exist_ok=True)

    def load_cnn_model(self, model_path="guitar_cnn_model.pth"):
        """Load the CNN model"""
        try:
            # Create dataset to get input shape
            val_dataset = GuitarDataset(
                self.validation_data, self.labels, IS_USING_FULL_CQT
            )
            sample_input, _ = val_dataset[0]
            input_shape = sample_input.shape
            num_classes = len(self.labels)

            # Initialize and load model
            model = GuitarCNN(input_shape, num_classes).to(DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.eval()

            print(f"✅ CNN model loaded successfully from {model_path}")
            return model, val_dataset
        except Exception as e:
            print(f"❌ Error loading CNN model: {e}")
            return None, None

    def load_dqn_model(self, model_path="dqn_guitar.zip"):
        """Load the DQN model"""
        if not STABLE_BASELINES_AVAILABLE:
            print("❌ stable_baselines3 not available for DQN model")
            return None, None

        try:
            env = GuitarEnv()
            model = DQN.load(model_path, env=env)
            print(f"✅ DQN model loaded successfully from {model_path}")
            return model, env
        except Exception as e:
            print(f"❌ Error loading DQN model: {e}")
            return None, None

    def evaluate_cnn_model(self, model, dataset):
        """Evaluate CNN model performance"""
        print("🔍 Evaluating CNN model...")

        val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

        predictions = []
        true_labels = []
        inference_times = []

        model.eval()
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="CNN Evaluation"):
                data, target = data.to(DEVICE), target.to(DEVICE)

                # Measure inference time
                start_time = time.time()
                output = model(data)
                end_time = time.time()

                # Calculate predictions
                pred = output.argmax(dim=1)

                predictions.extend(pred.cpu().numpy())
                true_labels.extend(target.cpu().numpy())
                inference_times.extend(
                    [(end_time - start_time) / len(data)] * len(data)
                )

        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        avg_inference_time = np.mean(inference_times)

        # Store results
        self.results["CNN"] = {
            "accuracy": accuracy,
            "predictions": predictions,
            "true_labels": true_labels,
            "inference_times": inference_times,
            "avg_inference_time": avg_inference_time,
            "classification_report": classification_report(
                true_labels, predictions, target_names=self.labels, output_dict=True
            ),
        }

        print(f"✅ CNN Accuracy: {accuracy:.4f}")
        print(f"✅ CNN Average Inference Time: {avg_inference_time:.6f}s per sample")

        return self.results["CNN"]

    def evaluate_dqn_model(self, model, env):
        """Evaluate DQN model performance"""
        print("🔍 Evaluating DQN model...")

        predictions = []
        true_labels = []
        inference_times = []

        # Reset environment
        obs, _ = env.reset()

        for i in tqdm(range(len(self.validation_data)), desc="DQN Evaluation"):
            # Get true label
            true_label_str = self.validation_data.iloc[i]["LABEL"]
            # Find the index of the label in self.labels list
            true_label = list(self.labels).index(true_label_str)
            true_labels.append(true_label)

            # Get observation for current step
            if i < len(self.validation_data) - 1:  # Not the last sample
                obs = self.validation_data.iloc[i][
                    "CQT_DATA_FULL" if IS_USING_FULL_CQT else "CQT_DATA_MEAN_TRIMMED"
                ]  # DQN expects 2D input as trained

            # Measure inference time
            start_time = time.time()
            action, _ = model.predict(obs, deterministic=True)
            end_time = time.time()

            predictions.append(action)
            inference_times.append(end_time - start_time)

        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        avg_inference_time = np.mean(inference_times)

        # Store results
        self.results["DQN"] = {
            "accuracy": accuracy,
            "predictions": predictions,
            "true_labels": true_labels,
            "inference_times": inference_times,
            "avg_inference_time": avg_inference_time,
            "classification_report": classification_report(
                true_labels, predictions, target_names=self.labels, output_dict=True
            ),
        }

        print(f"✅ DQN Accuracy: {accuracy:.4f}")
        print(f"✅ DQN Average Inference Time: {avg_inference_time:.6f}s per sample")

        return self.results["DQN"]

    def create_accuracy_comparison_plot(self):
        """Create accuracy comparison bar plot"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Overall accuracy comparison
        models = []
        accuracies = []

        if "CNN" in self.results and self.results["CNN"]:
            models.append("CNN")
            accuracies.append(self.results["CNN"]["accuracy"])

        if "DQN" in self.results and self.results["DQN"]:
            models.append("DQN")
            accuracies.append(self.results["DQN"]["accuracy"])

        bars = ax1.bar(models, accuracies, color=["#2E86AB", "#A23B72"], alpha=0.7)
        ax1.set_ylabel("Accuracy")
        ax1.set_title("Overall Model Accuracy Comparison")
        ax1.set_ylim(0, 1)

        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.01,
                f"{acc:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # Inference time comparison
        inference_times = []
        if "CNN" in self.results and self.results["CNN"]:
            inference_times.append(self.results["CNN"]["avg_inference_time"])

        if "DQN" in self.results and self.results["DQN"]:
            inference_times.append(self.results["DQN"]["avg_inference_time"])

        if len(inference_times) == len(models):
            bars2 = ax2.bar(
                models, inference_times, color=["#2E86AB", "#A23B72"], alpha=0.7
            )
            ax2.set_ylabel("Average Inference Time (seconds)")
            ax2.set_title("Model Inference Speed Comparison")

            # Add value labels on bars
            for bar, time_val in zip(bars2, inference_times):
                ax2.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height(),
                    f"{time_val:.6f}s",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

        plt.tight_layout()
        plt.savefig(
            "./comparison_results/graphs/accuracy_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

    def create_confusion_matrices(self):
        """Create confusion matrices for both models"""
        n_models = sum(
            1
            for model in ["CNN", "DQN"]
            if model in self.results and self.results[model]
        )

        if n_models == 0:
            return

        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
        if n_models == 1:
            axes = [axes]

        idx = 0
        for model_name in ["CNN", "DQN"]:
            if model_name in self.results and self.results[model_name]:
                cm = confusion_matrix(
                    self.results[model_name]["true_labels"],
                    self.results[model_name]["predictions"],
                )

                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=self.labels,
                    yticklabels=self.labels,
                    ax=axes[idx],
                )
                axes[idx].set_title(f"{model_name} Confusion Matrix")
                axes[idx].set_xlabel("Predicted Label")
                axes[idx].set_ylabel("True Label")
                idx += 1

        plt.tight_layout()
        plt.savefig(
            "./comparison_results/graphs/confusion_matrices.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

    def create_per_class_accuracy_plot(self):
        """Create per-class accuracy comparison"""
        if not any(
            self.results[model] for model in ["CNN", "DQN"] if model in self.results
        ):
            return

        fig, ax = plt.subplots(figsize=(15, 8))

        x = np.arange(len(self.labels))
        width = 0.35

        cnn_scores = []
        dqn_scores = []

        # Extract per-class accuracies
        if "CNN" in self.results and self.results["CNN"]:
            cnn_report = self.results["CNN"]["classification_report"]
            cnn_scores = [
                cnn_report.get(label, {}).get("f1-score", 0) for label in self.labels
            ]

        if "DQN" in self.results and self.results["DQN"]:
            dqn_report = self.results["DQN"]["classification_report"]
            dqn_scores = [
                dqn_report.get(label, {}).get("f1-score", 0) for label in self.labels
            ]

        # Plot bars
        if cnn_scores and dqn_scores:
            ax.bar(
                x - width / 2,
                cnn_scores,
                width,
                label="CNN",
                color="#2E86AB",
                alpha=0.7,
            )
            ax.bar(
                x + width / 2,
                dqn_scores,
                width,
                label="DQN",
                color="#A23B72",
                alpha=0.7,
            )
        elif cnn_scores:
            ax.bar(x, cnn_scores, width, label="CNN", color="#2E86AB", alpha=0.7)
        elif dqn_scores:
            ax.bar(x, dqn_scores, width, label="DQN", color="#A23B72", alpha=0.7)

        ax.set_xlabel("Guitar Note Classes")
        ax.set_ylabel("F1-Score")
        ax.set_title("Per-Class Performance Comparison (F1-Score)")
        ax.set_xticks(x)
        ax.set_xticklabels(self.labels, rotation=45, ha="right")
        ax.legend()
        ax.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(
            "./comparison_results/graphs/per_class_accuracy.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

    def create_inference_time_distribution(self):
        """Create inference time distribution plots"""
        models_with_times = [
            model
            for model in ["CNN", "DQN"]
            if model in self.results
            and self.results[model]
            and "inference_times" in self.results[model]
        ]

        if not models_with_times:
            return

        fig, axes = plt.subplots(
            1, len(models_with_times), figsize=(6 * len(models_with_times), 5)
        )
        if len(models_with_times) == 1:
            axes = [axes]

        colors = ["#2E86AB", "#A23B72"]

        for idx, model_name in enumerate(models_with_times):
            times = self.results[model_name]["inference_times"]
            axes[idx].hist(
                times,
                bins=50,
                alpha=0.7,
                color=colors[idx % len(colors)],
                edgecolor="black",
                linewidth=0.5,
            )
            axes[idx].set_xlabel("Inference Time (seconds)")
            axes[idx].set_ylabel("Frequency")
            axes[idx].set_title(f"{model_name} Inference Time Distribution")
            axes[idx].axvline(
                np.mean(times),
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {np.mean(times):.6f}s",
            )
            axes[idx].legend()

        plt.tight_layout()
        plt.savefig(
            "./comparison_results/graphs/inference_time_distribution.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

    def save_detailed_results(self):
        """Save detailed comparison results to files"""
        # Save summary results
        summary = {"model_comparison_summary": {}}

        for model_name in ["CNN", "DQN"]:
            if model_name in self.results and self.results[model_name]:
                summary["model_comparison_summary"][model_name] = {
                    "accuracy": self.results[model_name]["accuracy"],
                    "avg_inference_time": self.results[model_name][
                        "avg_inference_time"
                    ],
                    "total_samples_tested": len(
                        self.results[model_name]["predictions"]
                    ),
                }

        # Save to JSON-like format
        with open("./comparison_results/comparison_summary.txt", "w") as f:
            f.write("=" * 50 + "\n")
            f.write("GUITAR NOTE CLASSIFICATION MODEL COMPARISON\n")
            f.write("=" * 50 + "\n\n")

            for model_name in ["CNN", "DQN"]:
                if model_name in summary["model_comparison_summary"]:
                    results = summary["model_comparison_summary"][model_name]
                    f.write(f"{model_name} MODEL RESULTS:\n")
                    f.write(
                        f"  Accuracy: {results['accuracy']:.4f} ({results['accuracy'] * 100:.2f}%)\n"
                    )
                    f.write(
                        f"  Avg Inference Time: {results['avg_inference_time']:.6f} seconds\n"
                    )
                    f.write(f"  Samples Tested: {results['total_samples_tested']}\n\n")

            # Add detailed classification reports
            for model_name in ["CNN", "DQN"]:
                if model_name in self.results and self.results[model_name]:
                    f.write(f"\n{model_name} DETAILED CLASSIFICATION REPORT:\n")
                    f.write("-" * 40 + "\n")
                    report = self.results[model_name]["classification_report"]

                    for label in self.labels:
                        if label in report:
                            metrics = report[label]
                            f.write(
                                f"{label:>10}: Precision={metrics['precision']:.3f}, "
                                f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}\n"
                            )

                    if "macro avg" in report:
                        macro = report["macro avg"]
                        f.write(
                            f"\nMacro Avg: Precision={macro['precision']:.3f}, "
                            f"Recall={macro['recall']:.3f}, F1={macro['f1-score']:.3f}\n"
                        )

        # Save raw results as pickle for further analysis
        with open("./comparison_results/raw_results.pkl", "wb") as f:
            pickle.dump(self.results, f)

        print("✅ Detailed results saved to ./comparison_results/")

    def run_comparison(
        self, cnn_model_path="guitar_cnn_model.pth", dqn_model_path="dqn_guitar.zip"
    ):
        """Run complete model comparison"""
        print("🚀 Starting Model Comparison...")
        print("=" * 60)

        # Load and evaluate CNN model
        cnn_model, cnn_dataset = self.load_cnn_model(cnn_model_path)
        if cnn_model is not None:
            self.evaluate_cnn_model(cnn_model, cnn_dataset)

        print("-" * 40)

        # Load and evaluate DQN model
        dqn_model, dqn_env = self.load_dqn_model(dqn_model_path)
        if dqn_model is not None:
            self.evaluate_dqn_model(dqn_model, dqn_env)

        print("=" * 60)
        print("📊 Generating Comparison Visualizations...")

        # Generate all visualizations
        self.create_accuracy_comparison_plot()
        self.create_confusion_matrices()
        self.create_per_class_accuracy_plot()
        self.create_inference_time_distribution()

        # Save detailed results
        self.save_detailed_results()

        print(
            "✅ Model comparison completed! Check ./comparison_results/ for detailed outputs."
        )


if __name__ == "__main__":
    # Set up matplotlib for better plots
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")

    # Run the comparison
    comparator = ModelComparator()
    comparator.run_comparison()
