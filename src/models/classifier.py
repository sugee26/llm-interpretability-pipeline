"""Classifier models with interpretability support."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class TransformerClassifier(nn.Module):
    """
    Transformer-based text classifier with interpretability hooks.

    This classifier wraps HuggingFace transformers and adds hooks for
    extracting intermediate representations useful for interpretability.

    Args:
        model_name: HuggingFace model identifier
        num_labels: Number of classification labels
        dropout: Dropout rate for classifier head
        freeze_encoder: Whether to freeze transformer weights

    Example:
        >>> classifier = TransformerClassifier("bert-base-uncased", num_labels=2)
        >>> logits, hidden_states = classifier(input_ids, attention_mask)
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
    ):
        super().__init__()

        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(
            model_name,
            config=self.config,
            output_hidden_states=True,
            output_attentions=True,
        )

        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Storage for interpretability
        self._hidden_states = None
        self._attentions = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            labels: Optional labels for loss computation
            return_hidden_states: Whether to return hidden states

        Returns:
            Dictionary with logits, loss (if labels provided), and optionally hidden states
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Store for interpretability
        self._hidden_states = outputs.hidden_states
        self._attentions = outputs.attentions

        # Pool using CLS token
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)

        logits = self.classifier(pooled)

        result = {"logits": logits}

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            result["loss"] = loss_fn(logits, labels)

        if return_hidden_states:
            result["hidden_states"] = outputs.hidden_states
            result["attentions"] = outputs.attentions

        return result

    def get_hidden_states(self) -> Optional[Tuple[torch.Tensor, ...]]:
        """Get hidden states from last forward pass."""
        return self._hidden_states

    def get_attentions(self) -> Optional[Tuple[torch.Tensor, ...]]:
        """Get attention weights from last forward pass."""
        return self._attentions

    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get input embeddings for interpretability methods."""
        return self.encoder.embeddings.word_embeddings(input_ids)


class EnsembleClassifier:
    """
    Ensemble of multiple classifiers for robust predictions.

    Combines predictions from multiple models and provides
    uncertainty estimates useful for interpretability.

    Args:
        models: List of classifier models
        weights: Optional weights for each model

    Example:
        >>> ensemble = EnsembleClassifier([model1, model2, model3])
        >>> predictions, uncertainty = ensemble.predict(input_ids, attention_mask)
    """

    def __init__(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None,
    ):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)

        if len(self.weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")

        # Normalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_all_predictions: bool = False,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Get ensemble predictions.

        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            return_all_predictions: Whether to return individual model predictions

        Returns:
            Tuple of (predictions, metadata with uncertainty)
        """
        all_probs = []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                outputs = model(input_ids, attention_mask)
                probs = torch.softmax(outputs["logits"], dim=-1)
                all_probs.append(probs.cpu().numpy())

        all_probs = np.array(all_probs)  # (n_models, batch, n_classes)

        # Weighted average
        weighted_probs = np.zeros_like(all_probs[0])
        for i, (probs, weight) in enumerate(zip(all_probs, self.weights)):
            weighted_probs += probs * weight

        predictions = np.argmax(weighted_probs, axis=-1)

        # Compute uncertainty metrics
        uncertainty = {
            "entropy": self._compute_entropy(weighted_probs),
            "variance": all_probs.var(axis=0).mean(axis=-1),
            "agreement": self._compute_agreement(all_probs),
        }

        if return_all_predictions:
            uncertainty["individual_predictions"] = [
                np.argmax(p, axis=-1) for p in all_probs
            ]

        return predictions, uncertainty

    def _compute_entropy(self, probs: np.ndarray) -> np.ndarray:
        """Compute prediction entropy."""
        # Avoid log(0)
        probs = np.clip(probs, 1e-10, 1.0)
        return -np.sum(probs * np.log(probs), axis=-1)

    def _compute_agreement(self, all_probs: np.ndarray) -> np.ndarray:
        """Compute agreement ratio among models."""
        predictions = np.argmax(all_probs, axis=-1)  # (n_models, batch)
        mode_predictions = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(),
            axis=0,
            arr=predictions,
        )
        agreement = (predictions == mode_predictions).mean(axis=0)
        return agreement


class AttentionPoolingClassifier(nn.Module):
    """
    Classifier using attention-based pooling instead of CLS token.

    This can provide more interpretable pooling weights showing
    which tokens contributed most to the classification.

    Args:
        model_name: HuggingFace model identifier
        num_labels: Number of classification labels
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
    ):
        super().__init__()

        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)

        self.attention = nn.Linear(self.config.hidden_size, 1)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

        self._pooling_weights = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with attention pooling."""
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        hidden_states = outputs.last_hidden_state  # (batch, seq, hidden)

        # Compute attention weights
        attention_scores = self.attention(hidden_states).squeeze(-1)  # (batch, seq)

        # Mask padding tokens
        attention_scores = attention_scores.masked_fill(
            attention_mask == 0, float("-inf")
        )

        attention_weights = torch.softmax(attention_scores, dim=-1)
        self._pooling_weights = attention_weights

        # Weighted sum
        pooled = torch.bmm(
            attention_weights.unsqueeze(1), hidden_states
        ).squeeze(1)

        logits = self.classifier(pooled)

        result = {"logits": logits, "pooling_weights": attention_weights}

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            result["loss"] = loss_fn(logits, labels)

        return result

    def get_pooling_weights(self) -> Optional[torch.Tensor]:
        """Get attention pooling weights from last forward pass."""
        return self._pooling_weights
