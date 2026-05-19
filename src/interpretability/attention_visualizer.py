"""Attention visualization for transformer models."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


class AttentionVisualizer:
    """
    Visualize attention patterns in transformer models.

    This class extracts and visualizes attention weights from transformer
    layers, helping understand what the model focuses on when making predictions.

    Args:
        model: HuggingFace transformer model (must have output_attentions=True)
        tokenizer: HuggingFace tokenizer
        device: Device to run on

    Example:
        >>> visualizer = AttentionVisualizer(model, tokenizer, device)
        >>> visualizer.visualize("The movie was fantastic!", layer=11)
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def _get_attention(self, text: str) -> Tuple[torch.Tensor, List[str]]:
        """
        Extract attention weights for a text.

        Args:
            text: Input text

        Returns:
            Tuple of (attention_weights, tokens)
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Newer transformers default to SDPA, which silently drops output_attentions.
        # Force eager attention for this forward pass only.
        prev_impl = getattr(self.model.config, "_attn_implementation", None)
        try:
            self.model.config._attn_implementation = "eager"
            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True)
        finally:
            if prev_impl is not None:
                self.model.config._attn_implementation = prev_impl

        if not outputs.attentions:
            raise RuntimeError(
                "Model did not return attentions. Load the model with "
                "AutoModelForSequenceClassification.from_pretrained(..., "
                "attn_implementation='eager')."
            )

        # Shape: (num_layers, batch, num_heads, seq_len, seq_len)
        attentions = torch.stack(outputs.attentions)

        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        return attentions, tokens

    def explain(
        self,
        text: str,
        layer: Optional[int] = None,
        head: Optional[int] = None,
        aggregate_method: str = "mean",
    ) -> Dict:
        """
        Extract attention-based explanation.

        Args:
            text: Input text
            layer: Specific layer (None for all)
            head: Specific head (None for all)
            aggregate_method: How to aggregate ('mean', 'max', 'cls')

        Returns:
            Dictionary with attention data and token importances
        """
        attentions, tokens = self._get_attention(text)
        attentions = attentions.cpu().numpy()

        # Select layer(s)
        if layer is not None:
            attentions = attentions[layer:layer+1]

        # Select head(s)
        if head is not None:
            attentions = attentions[:, :, head:head+1]

        # Aggregate attention
        if aggregate_method == "mean":
            # Average across layers, batch, and heads
            aggregated = attentions.mean(axis=(0, 1, 2))
        elif aggregate_method == "max":
            aggregated = attentions.max(axis=(0, 1, 2))
        elif aggregate_method == "cls":
            # Attention from CLS token to other tokens
            aggregated = attentions[:, :, :, 0, :].mean(axis=(0, 1, 2))
        else:
            raise ValueError(f"Unknown aggregate_method: {aggregate_method}")

        # Token importance (sum of attention received)
        token_importance = aggregated.sum(axis=0)
        token_importance = token_importance / token_importance.sum()

        # Get prediction
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=-1).item()

        return {
            "text": text,
            "tokens": tokens,
            "attention_matrix": aggregated,
            "token_importance": token_importance.tolist(),
            "token_attributions": [
                {"token": tok, "importance": float(imp), "position": i}
                for i, (tok, imp) in enumerate(zip(tokens, token_importance))
            ],
            "predicted_class": predicted_class,
            "prediction_proba": probs.cpu().numpy()[0].tolist(),
            "num_layers": attentions.shape[0] if layer is None else 1,
            "num_heads": attentions.shape[2] if head is None else 1,
        }

    def visualize(
        self,
        text: str,
        layer: Optional[int] = None,
        head: Optional[int] = None,
        aggregate: bool = True,
        figsize: Tuple[int, int] = (12, 8),
    ):
        """
        Visualize attention patterns.

        Args:
            text: Input text
            layer: Specific layer to visualize
            head: Specific attention head
            aggregate: Whether to aggregate across heads
            figsize: Figure size
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        attentions, tokens = self._get_attention(text)
        attentions = attentions.cpu().numpy()

        # Truncate tokens for visualization
        max_tokens = 50
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            attentions = attentions[:, :, :, :max_tokens, :max_tokens]

        if aggregate:
            # Aggregate and show heatmap
            if layer is not None:
                attn_matrix = attentions[layer, 0].mean(axis=0)
                title = f"Attention (Layer {layer}, Aggregated)"
            else:
                attn_matrix = attentions.mean(axis=(0, 1, 2))
                title = "Attention (All Layers, Aggregated)"

            fig, ax = plt.subplots(figsize=figsize)
            sns.heatmap(
                attn_matrix,
                xticklabels=tokens,
                yticklabels=tokens,
                cmap="Blues",
                ax=ax,
            )
            ax.set_title(title)
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.show()

        else:
            # Show individual heads
            if layer is None:
                layer = -1  # Last layer

            num_heads = attentions.shape[2]
            cols = 4
            rows = (num_heads + cols - 1) // cols

            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
            axes = axes.flatten()

            for h in range(num_heads):
                attn_matrix = attentions[layer, 0, h]
                ax = axes[h]
                sns.heatmap(
                    attn_matrix,
                    xticklabels=tokens if h >= num_heads - cols else False,
                    yticklabels=tokens if h % cols == 0 else False,
                    cmap="Blues",
                    ax=ax,
                    cbar=False,
                )
                ax.set_title(f"Head {h}")

            # Hide empty subplots
            for idx in range(num_heads, len(axes)):
                axes[idx].axis("off")

            plt.suptitle(f"Attention Heads (Layer {layer})")
            plt.tight_layout()
            plt.show()

    def plot_token_importance(
        self,
        text: str,
        layer: Optional[int] = None,
        top_k: int = 15,
        figsize: Tuple[int, int] = (10, 6),
    ):
        """
        Plot token importance based on attention.

        Args:
            text: Input text
            layer: Specific layer (None for aggregated)
            top_k: Number of top tokens to show
            figsize: Figure size
        """
        import matplotlib.pyplot as plt

        explanation = self.explain(text, layer=layer)
        attributions = explanation["token_attributions"]

        # Sort by importance and get top-k
        sorted_attr = sorted(
            attributions,
            key=lambda x: x["importance"],
            reverse=True
        )[:top_k]

        tokens = [a["token"] for a in sorted_attr]
        importance = [a["importance"] for a in sorted_attr]

        fig, ax = plt.subplots(figsize=figsize)
        y_pos = np.arange(len(tokens))
        ax.barh(y_pos, importance, color="steelblue", alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(tokens)
        ax.invert_yaxis()
        ax.set_xlabel("Attention-based Importance")
        ax.set_title("Token Importance from Attention")
        plt.tight_layout()
        plt.show()

    def get_attention_rollout(
        self,
        text: str,
        discard_ratio: float = 0.9,
    ) -> Dict:
        """
        Compute attention rollout for better interpretability.

        Attention rollout recursively multiplies attention weights across
        layers to capture information flow.

        Args:
            text: Input text
            discard_ratio: Ratio of lowest attention weights to discard

        Returns:
            Dictionary with rollout attention and token attributions
        """
        attentions, tokens = self._get_attention(text)
        attentions = attentions.cpu().numpy()

        # Average across heads
        attention_heads_mean = attentions.mean(axis=2)  # (layers, batch, seq, seq)

        # Add residual connections
        num_layers, batch_size, seq_len, _ = attention_heads_mean.shape
        eye = np.eye(seq_len)

        # Apply attention rollout
        rollout = eye.copy()
        for layer_attn in attention_heads_mean[:, 0]:  # Iterate through layers
            # Add residual
            layer_attn = 0.5 * layer_attn + 0.5 * eye

            # Discard lowest attention weights
            if discard_ratio > 0:
                flat = layer_attn.flatten()
                threshold = np.percentile(flat, discard_ratio * 100)
                layer_attn = np.where(layer_attn < threshold, 0, layer_attn)

            # Renormalize
            layer_attn = layer_attn / layer_attn.sum(axis=-1, keepdims=True)

            # Multiply with accumulated rollout
            rollout = np.matmul(layer_attn, rollout)

        # Get CLS attention to other tokens
        cls_attention = rollout[0]
        cls_attention = cls_attention / cls_attention.sum()

        return {
            "text": text,
            "tokens": tokens,
            "rollout_attention": rollout,
            "cls_attention": cls_attention.tolist(),
            "token_attributions": [
                {"token": tok, "importance": float(imp), "position": i}
                for i, (tok, imp) in enumerate(zip(tokens, cls_attention))
            ],
        }
