"""Text preprocessing utilities."""

import re
from typing import List, Optional

import numpy as np


class TextPreprocessor:
    """
    Text preprocessing pipeline with configurable steps.

    Args:
        lowercase: Convert text to lowercase
        remove_urls: Remove URLs from text
        remove_mentions: Remove @mentions
        remove_hashtags: Remove #hashtags
        remove_special_chars: Remove special characters
        remove_numbers: Remove numeric characters
        min_length: Minimum text length (shorter texts are filtered)

    Example:
        >>> preprocessor = TextPreprocessor(lowercase=True, remove_urls=True)
        >>> clean_texts = preprocessor.transform(["Check out https://example.com!"])
    """

    def __init__(
        self,
        lowercase: bool = True,
        remove_urls: bool = True,
        remove_mentions: bool = False,
        remove_hashtags: bool = False,
        remove_special_chars: bool = False,
        remove_numbers: bool = False,
        min_length: Optional[int] = None,
    ):
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
        self.remove_special_chars = remove_special_chars
        self.remove_numbers = remove_numbers
        self.min_length = min_length

        # Compile regex patterns
        self.url_pattern = re.compile(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        )
        self.mention_pattern = re.compile(r"@\w+")
        self.hashtag_pattern = re.compile(r"#\w+")
        self.special_chars_pattern = re.compile(r"[^a-zA-Z0-9\s]")
        self.numbers_pattern = re.compile(r"\d+")
        self.whitespace_pattern = re.compile(r"\s+")

    def _preprocess_single(self, text: str) -> str:
        """Preprocess a single text."""
        if self.remove_urls:
            text = self.url_pattern.sub(" ", text)

        if self.remove_mentions:
            text = self.mention_pattern.sub(" ", text)

        if self.remove_hashtags:
            text = self.hashtag_pattern.sub(" ", text)

        if self.remove_special_chars:
            text = self.special_chars_pattern.sub(" ", text)

        if self.remove_numbers:
            text = self.numbers_pattern.sub(" ", text)

        if self.lowercase:
            text = text.lower()

        # Normalize whitespace
        text = self.whitespace_pattern.sub(" ", text).strip()

        return text

    def transform(
        self,
        texts: List[str],
        return_mask: bool = False,
    ) -> List[str]:
        """
        Transform a list of texts.

        Args:
            texts: List of texts to preprocess
            return_mask: If True, also return boolean mask of kept texts

        Returns:
            Preprocessed texts (and optionally mask if return_mask=True)
        """
        processed = [self._preprocess_single(t) for t in texts]

        if self.min_length is not None:
            mask = [len(t) >= self.min_length for t in processed]
            if return_mask:
                return [t for t, m in zip(processed, mask) if m], mask
            return [t for t, m in zip(processed, mask) if m]

        if return_mask:
            return processed, [True] * len(processed)
        return processed

    def fit_transform(self, texts: List[str], **kwargs) -> List[str]:
        """Fit and transform (for API compatibility with sklearn)."""
        return self.transform(texts, **kwargs)


class LabelEncoder:
    """
    Encode string labels to integers and back.

    Example:
        >>> encoder = LabelEncoder()
        >>> encoded = encoder.fit_transform(["positive", "negative", "positive"])
        >>> decoded = encoder.inverse_transform([0, 1, 0])
    """

    def __init__(self):
        self.label_to_id = {}
        self.id_to_label = {}
        self.is_fitted = False

    def fit(self, labels: List[str]) -> "LabelEncoder":
        """Fit the encoder to labels."""
        unique_labels = sorted(set(labels))
        self.label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}
        self.is_fitted = True
        return self

    def transform(self, labels: List[str]) -> np.ndarray:
        """Transform labels to integers."""
        if not self.is_fitted:
            raise ValueError("LabelEncoder must be fitted before transform")
        return np.array([self.label_to_id[label] for label in labels])

    def fit_transform(self, labels: List[str]) -> np.ndarray:
        """Fit and transform labels."""
        self.fit(labels)
        return self.transform(labels)

    def inverse_transform(self, ids: List[int]) -> List[str]:
        """Transform integers back to labels."""
        if not self.is_fitted:
            raise ValueError("LabelEncoder must be fitted before inverse_transform")
        return [self.id_to_label[idx] for idx in ids]

    @property
    def classes_(self) -> List[str]:
        """Return list of label names."""
        return list(self.label_to_id.keys())

    @property
    def num_classes(self) -> int:
        """Return number of classes."""
        return len(self.label_to_id)
