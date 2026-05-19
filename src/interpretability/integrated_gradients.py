"""Integrated Gradients for transformer interpretability."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

try:
    from captum.attr import (
        IntegratedGradients,
        LayerIntegratedGradients,
        TokenReferenceBase,
    )
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False


class IntegratedGradientsExplainer:
    """
    Integrated Gradients explainer for transformer models.

    Integrated Gradients is an axiomatic attribution method that attributes
    the prediction to input features by accumulating gradients along the
    path from a baseline to the input.

    Args:
        model: HuggingFace transformer model
        tokenizer: HuggingFace tokenizer
        device: Device to run on

    Example:
        >>> explainer = IntegratedGradientsExplainer(model, tokenizer, device)
        >>> explanation = explainer.explain("This is great!")
        >>> explainer.visualize(explanation)
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
    ):
        if not CAPTUM_AVAILABLE:
            raise ImportError(
                "captum package required. Install with: pip install captum"
            )

        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

        # Get the embedding layer
        self.embedding_layer = self._get_embedding_layer()

        # Reference token (PAD token)
        self.ref_token_id = tokenizer.pad_token_id
        if self.ref_token_id is None:
            self.ref_token_id = tokenizer.unk_token_id
        if self.ref_token_id is None:
            self.ref_token_id = 0

    def _get_embedding_layer(self):
        """Get the embedding layer from the model."""
        # Try common embedding layer names
        if hasattr(self.model, "distilbert"):
            return self.model.distilbert.embeddings.word_embeddings
        elif hasattr(self.model, "bert"):
            return self.model.bert.embeddings.word_embeddings
        elif hasattr(self.model, "roberta"):
            return self.model.roberta.embeddings.word_embeddings
        elif hasattr(self.model, "transformer"):
            if hasattr(self.model.transformer, "wte"):
                return self.model.transformer.wte
            elif hasattr(self.model.transformer, "word_embedding"):
                return self.model.transformer.word_embedding
        else:
            # Generic attempt
            for name, module in self.model.named_modules():
                if "embedding" in name.lower() and "word" in name.lower():
                    return module
        raise ValueError("Could not find embedding layer")

    def _forward_func(
        self,
        input_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        target_class: int,
    ) -> torch.Tensor:
        """Forward function for Integrated Gradients."""
        outputs = self.model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
        )
        logits = outputs.logits
        return logits[:, target_class]

    def _construct_baseline(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Construct a baseline (reference) input."""
        return torch.full_like(input_ids, self.ref_token_id)

    def explain(
        self,
        text: str,
        n_steps: int = 50,
        target_class: Optional[int] = None,
        visualize: bool = False,
        internal_batch_size: int = 5,
    ) -> Dict:
        """
        Generate Integrated Gradients explanation.

        Args:
            text: Input text to explain
            n_steps: Number of interpolation steps
            target_class: Class to explain (None for predicted class)
            visualize: Whether to display visualization
            internal_batch_size: Batch size for gradient computation

        Returns:
            Dictionary containing attributions and metadata
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Get prediction
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=-1).item()

        if target_class is None:
            target_class = predicted_class

        # Get embeddings
        input_embeds = self.embedding_layer(input_ids)
        baseline_ids = self._construct_baseline(input_ids)
        baseline_embeds = self.embedding_layer(baseline_ids)

        # Create IG instance
        ig = IntegratedGradients(
            lambda embeds: self._forward_func(embeds, attention_mask, target_class)
        )

        # Compute attributions
        attributions = ig.attribute(
            inputs=input_embeds,
            baselines=baseline_embeds,
            n_steps=n_steps,
            internal_batch_size=internal_batch_size,
        )

        # Sum across embedding dimension to get per-token attribution
        token_attributions = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()

        # Normalize
        token_attributions = token_attributions / np.abs(token_attributions).max()

        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        result = {
            "text": text,
            "tokens": tokens,
            "attributions": token_attributions.tolist(),
            "token_attributions": [
                {"token": tok, "attribution": float(attr), "position": i}
                for i, (tok, attr) in enumerate(zip(tokens, token_attributions))
            ],
            "predicted_class": predicted_class,
            "target_class": target_class,
            "prediction_proba": probs.cpu().numpy()[0].tolist(),
        }

        if visualize:
            self.visualize(result)

        return result

    def visualize(
        self,
        explanation: Dict,
        figsize: Tuple[int, int] = (12, 4),
    ):
        """
        Visualize Integrated Gradients attributions.

        Args:
            explanation: Explanation dict from explain()
            figsize: Figure size for the plot
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        tokens = explanation["tokens"]
        attributions = explanation["attributions"]

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Create colormap (red for negative, green for positive)
        cmap = plt.cm.RdYlGn
        norm = mcolors.Normalize(vmin=-1, vmax=1)

        # Plot tokens with colored backgrounds
        y_position = 0.5
        x_position = 0.02

        for i, (token, attr) in enumerate(zip(tokens, attributions)):
            # Skip special tokens
            if token in ["[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"]:
                continue

            color = cmap(norm(attr))
            bbox = dict(
                boxstyle="round,pad=0.3",
                facecolor=color,
                edgecolor="none",
                alpha=0.7,
            )
            ax.text(
                x_position,
                y_position,
                token.replace("##", ""),
                fontsize=12,
                bbox=bbox,
                transform=ax.transAxes,
            )
            x_position += len(token.replace("##", "")) * 0.015 + 0.02

            if x_position > 0.95:
                y_position -= 0.15
                x_position = 0.02

        ax.axis("off")
        ax.set_title(
            f"Integrated Gradients Attribution "
            f"(Class {explanation['target_class']})\n"
            f"Green = Positive, Red = Negative",
            fontsize=12,
        )

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation="horizontal", pad=0.1, shrink=0.6)
        cbar.set_label("Attribution Score")

        plt.tight_layout()
        plt.show()

        # Also show bar chart of top attributions
        self._plot_attribution_bars(explanation)

    def _plot_attribution_bars(
        self,
        explanation: Dict,
        top_k: int = 15,
        figsize: Tuple[int, int] = (10, 6),
    ):
        """Plot bar chart of top attributions."""
        import matplotlib.pyplot as plt

        # Filter out special tokens and get top-k
        attrs = [
            a for a in explanation["token_attributions"]
            if a["token"] not in ["[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"]
        ]

        # Sort by absolute attribution
        attrs = sorted(attrs, key=lambda x: abs(x["attribution"]), reverse=True)[:top_k]

        tokens = [a["token"].replace("##", "") for a in attrs]
        values = [a["attribution"] for a in attrs]
        colors = ["green" if v > 0 else "red" for v in values]

        fig, ax = plt.subplots(figsize=figsize)
        y_pos = np.arange(len(tokens))
        ax.barh(y_pos, values, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(tokens)
        ax.invert_yaxis()
        ax.set_xlabel("Attribution Score")
        ax.set_title(f"Top Token Attributions (Class {explanation['target_class']})")
        ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
        plt.tight_layout()
        plt.show()

    def compare_classes(
        self,
        text: str,
        n_steps: int = 50,
    ) -> Dict:
        """
        Compare attributions across all classes.

        Args:
            text: Input text
            n_steps: Number of interpolation steps

        Returns:
            Dictionary with attributions for each class
        """
        num_classes = self.model.config.num_labels
        results = {
            "text": text,
            "class_explanations": {},
        }

        for class_idx in range(num_classes):
            exp = self.explain(text, n_steps=n_steps, target_class=class_idx)
            results["class_explanations"][class_idx] = {
                "attributions": exp["attributions"],
                "token_attributions": exp["token_attributions"],
            }

        results["tokens"] = exp["tokens"]
        results["predicted_class"] = exp["predicted_class"]
        results["prediction_proba"] = exp["prediction_proba"]

        return results
