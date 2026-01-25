"""SHAP (SHapley Additive exPlanations) for text classification."""

from typing import Dict, List, Optional, Union

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class SHAPExplainer:
    """
    SHAP explainer for transformer-based text classification.

    Uses SHAP values to explain model predictions by computing the contribution
    of each token to the final prediction. This provides a game-theoretic
    approach to feature importance.

    Args:
        model: HuggingFace transformer model
        tokenizer: HuggingFace tokenizer
        device: Device to run on

    Example:
        >>> explainer = SHAPExplainer(model, tokenizer, device)
        >>> explanation = explainer.explain("This movie was great!")
        >>> explainer.visualize(explanation)
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
    ):
        if not SHAP_AVAILABLE:
            raise ImportError("shap package required. Install with: pip install shap")

        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def _predict_proba(self, texts: List[str]) -> np.ndarray:
        """Get prediction probabilities for a list of texts."""
        self.model.eval()
        all_probs = []

        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                all_probs.append(probs.cpu().numpy()[0])

        return np.array(all_probs)

    def explain(
        self,
        text: str,
        num_samples: int = 100,
        visualize: bool = False,
        target_class: Optional[int] = None,
    ) -> Dict:
        """
        Generate SHAP explanation for a text.

        Args:
            text: Input text to explain
            num_samples: Number of perturbation samples
            visualize: Whether to display visualization
            target_class: Specific class to explain (None for predicted class)

        Returns:
            Dictionary containing SHAP values and metadata
        """
        # Create masker for text
        masker = shap.maskers.Text(self.tokenizer)

        # Create explainer
        explainer = shap.Explainer(
            self._predict_proba,
            masker,
            output_names=[f"class_{i}" for i in range(self.model.config.num_labels)],
        )

        # Compute SHAP values
        shap_values = explainer([text], fixed_context=1, batch_size=5)

        # Get tokens
        tokens = self.tokenizer.tokenize(text)

        # Get prediction
        probs = self._predict_proba([text])[0]
        predicted_class = np.argmax(probs)

        if target_class is None:
            target_class = predicted_class

        result = {
            "text": text,
            "tokens": tokens,
            "shap_values": shap_values,
            "predicted_class": int(predicted_class),
            "prediction_proba": probs.tolist(),
            "target_class": target_class,
            "token_attributions": self._extract_token_attributions(
                shap_values, tokens, target_class
            ),
        }

        if visualize:
            self.visualize(result)

        return result

    def _extract_token_attributions(
        self,
        shap_values,
        tokens: List[str],
        target_class: int,
    ) -> List[Dict]:
        """Extract per-token attributions from SHAP values."""
        values = shap_values.values[0][:, target_class]
        data = shap_values.data[0]

        attributions = []
        for i, (token, value) in enumerate(zip(data, values)):
            attributions.append({
                "token": str(token),
                "attribution": float(value),
                "position": i,
            })

        # Sort by absolute attribution
        attributions.sort(key=lambda x: abs(x["attribution"]), reverse=True)
        return attributions

    def visualize(
        self,
        explanation: Dict,
        max_display: int = 20,
    ):
        """
        Visualize SHAP explanation.

        Args:
            explanation: Explanation dict from explain()
            max_display: Maximum number of tokens to display
        """
        import matplotlib.pyplot as plt

        shap_values = explanation["shap_values"]
        target_class = explanation["target_class"]

        # SHAP text plot
        shap.plots.text(shap_values[:, :, target_class])

        # Bar plot of top attributions
        attributions = explanation["token_attributions"][:max_display]
        tokens = [a["token"] for a in attributions]
        values = [a["attribution"] for a in attributions]
        colors = ["green" if v > 0 else "red" for v in values]

        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = np.arange(len(tokens))
        ax.barh(y_pos, values, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(tokens)
        ax.invert_yaxis()
        ax.set_xlabel("SHAP Value")
        ax.set_title(f"Token Contributions (Class {target_class})")
        ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
        plt.tight_layout()
        plt.show()

    def explain_batch(
        self,
        texts: List[str],
        num_samples: int = 100,
    ) -> List[Dict]:
        """
        Generate SHAP explanations for multiple texts.

        Args:
            texts: List of input texts
            num_samples: Number of perturbation samples

        Returns:
            List of explanation dictionaries
        """
        return [self.explain(text, num_samples=num_samples) for text in texts]
