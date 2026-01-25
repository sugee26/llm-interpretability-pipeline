"""LIME (Local Interpretable Model-agnostic Explanations) for text classification."""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

try:
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False


class LIMEExplainer:
    """
    LIME explainer for text classification models.

    LIME creates locally faithful explanations by approximating the model's
    behavior around a specific prediction using a simpler, interpretable model.

    Args:
        model: HuggingFace transformer model
        tokenizer: HuggingFace tokenizer
        device: Device to run on
        class_names: Optional list of class names

    Example:
        >>> explainer = LIMEExplainer(model, tokenizer, device)
        >>> explanation = explainer.explain("This product is amazing!")
        >>> explainer.visualize(explanation)
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
        class_names: Optional[List[str]] = None,
    ):
        if not LIME_AVAILABLE:
            raise ImportError("lime package required. Install with: pip install lime")

        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

        num_labels = model.config.num_labels
        self.class_names = class_names or [f"Class_{i}" for i in range(num_labels)]

        self.lime_explainer = LimeTextExplainer(
            class_names=self.class_names,
            split_expression=r"\W+",
            bow=False,
        )

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
        num_features: int = 10,
        num_samples: int = 500,
        visualize: bool = False,
        target_class: Optional[int] = None,
    ) -> Dict:
        """
        Generate LIME explanation for a text.

        Args:
            text: Input text to explain
            num_features: Number of top features to include
            num_samples: Number of perturbed samples
            visualize: Whether to display visualization
            target_class: Specific class to explain

        Returns:
            Dictionary containing LIME explanation and metadata
        """
        # Get prediction
        probs = self._predict_proba([text])[0]
        predicted_class = np.argmax(probs)

        if target_class is None:
            target_class = predicted_class

        # Generate LIME explanation
        lime_exp = self.lime_explainer.explain_instance(
            text,
            self._predict_proba,
            num_features=num_features,
            num_samples=num_samples,
            labels=[target_class],
        )

        # Extract feature weights
        feature_weights = lime_exp.as_list(label=target_class)

        result = {
            "text": text,
            "predicted_class": int(predicted_class),
            "prediction_proba": probs.tolist(),
            "target_class": target_class,
            "lime_explanation": lime_exp,
            "feature_weights": [
                {"word": word, "weight": weight}
                for word, weight in feature_weights
            ],
            "intercept": lime_exp.intercept[target_class],
            "local_prediction": lime_exp.local_pred[0] if lime_exp.local_pred is not None else None,
        }

        if visualize:
            self.visualize(result)

        return result

    def visualize(
        self,
        explanation: Dict,
        figsize: Tuple[int, int] = (10, 6),
    ):
        """
        Visualize LIME explanation.

        Args:
            explanation: Explanation dict from explain()
            figsize: Figure size for the plot
        """
        import matplotlib.pyplot as plt

        lime_exp = explanation["lime_explanation"]
        target_class = explanation["target_class"]

        # Use LIME's built-in visualization
        fig = lime_exp.as_pyplot_figure(label=target_class)
        fig.set_size_inches(figsize)
        plt.title(f"LIME Explanation (Class: {self.class_names[target_class]})")
        plt.tight_layout()
        plt.show()

        # Also show inline HTML if in notebook
        try:
            from IPython.display import display, HTML
            display(HTML(lime_exp.as_html()))
        except ImportError:
            pass

    def explain_with_highlighted_text(
        self,
        text: str,
        num_features: int = 10,
        num_samples: int = 500,
    ) -> str:
        """
        Generate HTML with highlighted words based on LIME weights.

        Args:
            text: Input text
            num_features: Number of features to highlight
            num_samples: Number of perturbed samples

        Returns:
            HTML string with highlighted text
        """
        explanation = self.explain(
            text,
            num_features=num_features,
            num_samples=num_samples,
        )

        # Build word -> weight mapping
        weight_dict = {
            fw["word"].lower(): fw["weight"]
            for fw in explanation["feature_weights"]
        }

        # Tokenize and highlight
        words = text.split()
        html_parts = []

        for word in words:
            clean_word = "".join(c for c in word.lower() if c.isalnum())
            if clean_word in weight_dict:
                weight = weight_dict[clean_word]
                if weight > 0:
                    color = f"rgba(0, 255, 0, {min(abs(weight) * 2, 0.8)})"
                else:
                    color = f"rgba(255, 0, 0, {min(abs(weight) * 2, 0.8)})"
                html_parts.append(
                    f'<span style="background-color: {color}; padding: 2px;">{word}</span>'
                )
            else:
                html_parts.append(word)

        return " ".join(html_parts)

    def compare_explanations(
        self,
        texts: List[str],
        num_features: int = 10,
    ) -> Dict:
        """
        Compare LIME explanations across multiple texts.

        Args:
            texts: List of texts to compare
            num_features: Number of features per explanation

        Returns:
            Dictionary with comparative analysis
        """
        explanations = [
            self.explain(text, num_features=num_features)
            for text in texts
        ]

        # Find common important features
        all_features = set()
        for exp in explanations:
            for fw in exp["feature_weights"]:
                all_features.add(fw["word"].lower())

        # Build comparison matrix
        comparison = {
            "texts": texts,
            "predictions": [exp["predicted_class"] for exp in explanations],
            "feature_matrix": {},
        }

        for feature in all_features:
            comparison["feature_matrix"][feature] = []
            for exp in explanations:
                weight = 0.0
                for fw in exp["feature_weights"]:
                    if fw["word"].lower() == feature:
                        weight = fw["weight"]
                        break
                comparison["feature_matrix"][feature].append(weight)

        return comparison
