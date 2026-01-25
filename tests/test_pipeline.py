"""Tests for the InterpretableNLPPipeline."""

import pytest
import numpy as np
import sys
sys.path.insert(0, "..")


class TestTextPreprocessor:
    """Tests for text preprocessing utilities."""

    def test_lowercase(self):
        from src.data.preprocessing import TextPreprocessor

        preprocessor = TextPreprocessor(lowercase=True)
        result = preprocessor.transform(["HELLO World"])
        assert result[0] == "hello world"

    def test_remove_urls(self):
        from src.data.preprocessing import TextPreprocessor

        preprocessor = TextPreprocessor(remove_urls=True)
        result = preprocessor.transform(["Check out https://example.com for more"])
        assert "https" not in result[0]
        assert "example.com" not in result[0]

    def test_min_length_filter(self):
        from src.data.preprocessing import TextPreprocessor

        preprocessor = TextPreprocessor(min_length=10)
        result = preprocessor.transform(["short", "this is a longer text"])
        assert len(result) == 1
        assert "longer" in result[0]


class TestLabelEncoder:
    """Tests for label encoding."""

    def test_fit_transform(self):
        from src.data.preprocessing import LabelEncoder

        encoder = LabelEncoder()
        labels = ["positive", "negative", "positive", "neutral"]
        encoded = encoder.fit_transform(labels)

        assert len(encoded) == 4
        assert encoder.num_classes == 3

    def test_inverse_transform(self):
        from src.data.preprocessing import LabelEncoder

        encoder = LabelEncoder()
        labels = ["cat", "dog", "cat"]
        encoded = encoder.fit_transform(labels)
        decoded = encoder.inverse_transform(encoded.tolist())

        assert decoded == labels


class TestMetrics:
    """Tests for metrics computation."""

    def test_compute_metrics(self):
        from src.utils.metrics import compute_metrics

        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1]

        metrics = compute_metrics(y_true, y_pred)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_classification_report(self):
        from src.utils.metrics import classification_report

        y_true = [0, 1, 1, 0]
        y_pred = [0, 1, 0, 0]

        report = classification_report(y_true, y_pred, output_dict=True)

        assert "0" in report or "accuracy" in report
        assert "weighted avg" in report


class TestDataset:
    """Tests for dataset classes."""

    def test_text_classification_dataset(self):
        from src.data.dataset import TextClassificationDataset
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        texts = ["Hello world", "Test sentence"]
        labels = [0, 1]

        dataset = TextClassificationDataset(texts, labels, tokenizer, max_length=32)

        assert len(dataset) == 2

        sample = dataset[0]
        assert "input_ids" in sample
        assert "attention_mask" in sample
        assert "labels" in sample


# Integration tests (require model download, skip if no network)
@pytest.mark.slow
class TestPipelineIntegration:
    """Integration tests for the full pipeline."""

    @pytest.fixture
    def pipeline(self):
        from src.pipeline import InterpretableNLPPipeline

        return InterpretableNLPPipeline(
            model_name="distilbert-base-uncased",
            num_labels=2,
            interpretability_methods=["attention"],
            max_length=64,
        )

    def test_pipeline_init(self, pipeline):
        assert pipeline.model is not None
        assert pipeline.tokenizer is not None
        assert "attention" in pipeline.explainers

    def test_predict(self, pipeline):
        texts = ["This is great!", "This is bad."]
        predictions = pipeline.predict(texts)

        assert len(predictions) == 2
        assert all(p in [0, 1] for p in predictions)

    def test_predict_with_probs(self, pipeline):
        texts = ["Test text"]
        predictions, probs = pipeline.predict(texts, return_probs=True)

        assert len(predictions) == 1
        assert probs.shape == (1, 2)
        assert np.isclose(probs.sum(), 1.0, atol=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
