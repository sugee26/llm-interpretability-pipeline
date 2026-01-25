"""Main interpretable NLP classification pipeline."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from .data.dataset import TextClassificationDataset
from .interpretability import (
    AttentionVisualizer,
    IntegratedGradientsExplainer,
    LIMEExplainer,
    SHAPExplainer,
)
from .utils.metrics import compute_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InterpretableNLPPipeline:
    """
    End-to-end interpretable NLP classification pipeline.

    This pipeline provides transparent text classification with built-in
    interpretability methods including SHAP, LIME, attention visualization,
    and integrated gradients.

    Args:
        model_name: HuggingFace model identifier (e.g., 'distilbert-base-uncased')
        num_labels: Number of classification labels
        interpretability_methods: List of methods to enable ['shap', 'lime', 'attention', 'ig']
        device: Device to run on ('cuda', 'cpu', or 'auto')
        max_length: Maximum sequence length for tokenization

    Example:
        >>> pipeline = InterpretableNLPPipeline(
        ...     model_name="distilbert-base-uncased",
        ...     num_labels=2,
        ...     interpretability_methods=["shap", "attention"]
        ... )
        >>> pipeline.fit(train_texts, train_labels)
        >>> predictions, explanations = pipeline.predict_with_explanations(test_texts)
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 2,
        interpretability_methods: Optional[List[str]] = None,
        device: str = "auto",
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.interpretability_methods = interpretability_methods or ["shap", "attention"]

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            output_attentions=True,
            output_hidden_states=True,
        )
        self.model.to(self.device)

        # Initialize interpretability components
        self._init_explainers()

        # Training state
        self.is_trained = False
        self.training_history = []
        self.label_names = None

    def _init_explainers(self):
        """Initialize interpretability explainers based on configured methods."""
        self.explainers = {}

        if "shap" in self.interpretability_methods:
            self.explainers["shap"] = SHAPExplainer(
                self.model, self.tokenizer, self.device
            )

        if "lime" in self.interpretability_methods:
            self.explainers["lime"] = LIMEExplainer(
                self.model, self.tokenizer, self.device
            )

        if "attention" in self.interpretability_methods:
            self.explainers["attention"] = AttentionVisualizer(
                self.model, self.tokenizer, self.device
            )

        if "ig" in self.interpretability_methods:
            self.explainers["ig"] = IntegratedGradientsExplainer(
                self.model, self.tokenizer, self.device
            )

    def fit(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        warmup_ratio: float = 0.1,
        label_names: Optional[List[str]] = None,
    ) -> Dict:
        """
        Train the classification model.

        Args:
            train_texts: List of training text samples
            train_labels: List of training labels (integers)
            val_texts: Optional validation texts
            val_labels: Optional validation labels
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for AdamW optimizer
            warmup_ratio: Ratio of warmup steps
            label_names: Optional list of label names for interpretability

        Returns:
            Dictionary containing training history and metrics
        """
        self.label_names = label_names or [f"Label_{i}" for i in range(self.num_labels)]

        # Create datasets
        train_dataset = TextClassificationDataset(
            train_texts, train_labels, self.tokenizer, self.max_length
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_loader = None
        if val_texts and val_labels:
            val_dataset = TextClassificationDataset(
                val_texts, val_labels, self.tokenizer, self.max_length
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, warmup_steps, total_steps
        )

        # Training loop
        self.model.train()
        self.training_history = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

            for batch in progress_bar:
                optimizer.zero_grad()

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                loss = outputs.loss
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item()})

            avg_loss = epoch_loss / len(train_loader)
            epoch_metrics = {"epoch": epoch + 1, "train_loss": avg_loss}

            # Validation
            if val_loader:
                val_metrics = self._evaluate(val_loader)
                epoch_metrics.update(val_metrics)
                logger.info(
                    f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, "
                    f"Val Accuracy={val_metrics['accuracy']:.4f}"
                )
            else:
                logger.info(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}")

            self.training_history.append(epoch_metrics)

        self.is_trained = True
        return {"history": self.training_history}

    def _evaluate(self, dataloader: DataLoader) -> Dict:
        """Evaluate model on a dataloader."""
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        self.model.train()
        return compute_metrics(all_labels, all_preds)

    def predict(
        self,
        texts: Union[str, List[str]],
        return_probs: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions on input texts.

        Args:
            texts: Single text or list of texts
            return_probs: Whether to return probability distributions

        Returns:
            Predicted labels (and probabilities if requested)
        """
        if isinstance(texts, str):
            texts = [texts]

        self.model.eval()
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                    padding=True,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                pred = torch.argmax(probs, dim=-1)

                all_preds.append(pred.cpu().numpy()[0])
                all_probs.append(probs.cpu().numpy()[0])

        preds = np.array(all_preds)
        probs = np.array(all_probs)

        if return_probs:
            return preds, probs
        return preds

    def predict_with_explanations(
        self,
        texts: Union[str, List[str]],
        methods: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Make predictions with interpretability explanations.

        Args:
            texts: Single text or list of texts
            methods: Interpretability methods to use (defaults to all configured)

        Returns:
            Tuple of (predictions, explanations_dict)
        """
        if isinstance(texts, str):
            texts = [texts]

        methods = methods or list(self.explainers.keys())

        predictions, probs = self.predict(texts, return_probs=True)

        explanations = {
            "predictions": predictions,
            "probabilities": probs,
            "label_names": self.label_names,
            "explanations": {},
        }

        for method in methods:
            if method in self.explainers:
                logger.info(f"Generating {method.upper()} explanations...")
                method_explanations = []
                for text in tqdm(texts, desc=f"{method.upper()}"):
                    exp = self.explainers[method].explain(text)
                    method_explanations.append(exp)
                explanations["explanations"][method] = method_explanations

        return predictions, explanations

    def explain_shap(
        self,
        text: str,
        num_samples: int = 100,
        visualize: bool = True,
    ):
        """
        Generate SHAP explanation for a single text.

        Args:
            text: Input text to explain
            num_samples: Number of samples for SHAP
            visualize: Whether to display visualization

        Returns:
            SHAP values and visualization
        """
        if "shap" not in self.explainers:
            raise ValueError("SHAP explainer not initialized")

        return self.explainers["shap"].explain(
            text, num_samples=num_samples, visualize=visualize
        )

    def explain_lime(
        self,
        text: str,
        num_features: int = 10,
        visualize: bool = True,
    ):
        """
        Generate LIME explanation for a single text.

        Args:
            text: Input text to explain
            num_features: Number of top features to show
            visualize: Whether to display visualization

        Returns:
            LIME explanation object
        """
        if "lime" not in self.explainers:
            raise ValueError("LIME explainer not initialized")

        return self.explainers["lime"].explain(
            text, num_features=num_features, visualize=visualize
        )

    def visualize_attention(
        self,
        text: str,
        layer: Optional[int] = None,
        head: Optional[int] = None,
        aggregate: bool = True,
    ):
        """
        Visualize attention patterns for a text.

        Args:
            text: Input text
            layer: Specific layer to visualize (None for all)
            head: Specific attention head (None for all)
            aggregate: Whether to aggregate across heads

        Returns:
            Attention visualization
        """
        if "attention" not in self.explainers:
            raise ValueError("Attention visualizer not initialized")

        return self.explainers["attention"].visualize(
            text, layer=layer, head=head, aggregate=aggregate
        )

    def explain_integrated_gradients(
        self,
        text: str,
        n_steps: int = 50,
        visualize: bool = True,
    ):
        """
        Generate Integrated Gradients explanation.

        Args:
            text: Input text
            n_steps: Number of interpolation steps
            visualize: Whether to display visualization

        Returns:
            Attribution scores for each token
        """
        if "ig" not in self.explainers:
            raise ValueError("Integrated Gradients explainer not initialized")

        return self.explainers["ig"].explain(
            text, n_steps=n_steps, visualize=visualize
        )

    def save(self, path: str):
        """Save the pipeline to disk."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

        # Save additional config
        import json
        config = {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "max_length": self.max_length,
            "interpretability_methods": self.interpretability_methods,
            "label_names": self.label_names,
        }
        with open(f"{path}/pipeline_config.json", "w") as f:
            json.dump(config, f)

        logger.info(f"Pipeline saved to {path}")

    @classmethod
    def load(cls, path: str, device: str = "auto") -> "InterpretableNLPPipeline":
        """Load a pipeline from disk."""
        import json

        with open(f"{path}/pipeline_config.json", "r") as f:
            config = json.load(f)

        pipeline = cls(
            model_name=path,
            num_labels=config["num_labels"],
            interpretability_methods=config["interpretability_methods"],
            device=device,
            max_length=config["max_length"],
        )
        pipeline.label_names = config["label_names"]
        pipeline.is_trained = True

        logger.info(f"Pipeline loaded from {path}")
        return pipeline
