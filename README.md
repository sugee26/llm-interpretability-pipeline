---
title: LLM Interpretability Pipeline
emoji: 🔍
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: SHAP + Integrated Gradients + Attention on DistilBERT-SST2
---

# LLM Interpretability Pipeline

A comprehensive, interpretable machine learning pipeline for NLP classification tasks. This pipeline emphasizes transparency and explainability at every stage, from data preprocessing to model predictions.

## Live demo

Type any sentence in the box at the top of this Space — you'll see three independent attribution methods (Integrated Gradients, last-layer Attention, SHAP) run side-by-side on DistilBERT fine-tuned for SST-2 sentiment. Watching the three methods agree (or disagree) is a quick read on how robust the model's reasoning is for that input.

Source code: `examples/serve_demo.py` · API: `POST /explain {"text": "...", "top_k": 8}`

## Features

- **Multiple Model Support**: BERT, DistilBERT, RoBERTa, and traditional ML models
- **Built-in Interpretability**:
  - SHAP (SHapley Additive exPlanations)
  - LIME (Local Interpretable Model-agnostic Explanations)
  - Attention Visualization
  - Integrated Gradients
  - Token Attribution
- **End-to-End Pipeline**: Data loading → Preprocessing → Training → Evaluation → Interpretation
- **Experiment Tracking**: Built-in logging and metrics tracking
- **Modular Design**: Easy to extend and customize

## Installation

```bash
# Clone the repository
git clone https://github.com/sugeerth/llm-interpretability-pipeline.git
cd llm-interpretability-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from src.pipeline import InterpretableNLPPipeline

# Initialize pipeline
pipeline = InterpretableNLPPipeline(
    model_name="distilbert-base-uncased",
    num_labels=2,
    interpretability_methods=["shap", "attention", "lime"]
)

# Train on your data
pipeline.fit(train_texts, train_labels)

# Get predictions with explanations
predictions, explanations = pipeline.predict_with_explanations(test_texts)

# Visualize attention patterns
pipeline.visualize_attention(text="This movie was absolutely fantastic!")

# Generate SHAP explanations
pipeline.explain_shap(text="The product quality is terrible.")
```

## Project Structure

```
llm-interpretability-pipeline/
├── src/
│   ├── models/           # Model architectures
│   ├── interpretability/ # SHAP, LIME, Attention viz
│   ├── data/             # Data loading and preprocessing
│   └── utils/            # Helper functions
├── notebooks/            # Jupyter notebooks with examples
├── configs/              # Configuration files
├── tests/                # Unit tests
└── examples/             # Example scripts
```

## Interpretability Methods

### 1. SHAP Values
Compute feature importance using Shapley values from game theory.

```python
from src.interpretability import SHAPExplainer

explainer = SHAPExplainer(model, tokenizer)
shap_values = explainer.explain(text)
explainer.visualize(shap_values)
```

### 2. LIME Explanations
Generate local interpretable explanations.

```python
from src.interpretability import LIMEExplainer

explainer = LIMEExplainer(model, tokenizer)
explanation = explainer.explain(text, num_features=10)
```

### 3. Attention Visualization
Visualize transformer attention patterns.

```python
from src.interpretability import AttentionVisualizer

visualizer = AttentionVisualizer(model, tokenizer)
visualizer.plot_attention_heads(text, layer=11)
```

### 4. Integrated Gradients
Attribute predictions to input features using gradients.

```python
from src.interpretability import IntegratedGradientsExplainer

explainer = IntegratedGradientsExplainer(model, tokenizer)
attributions = explainer.explain(text)
```

## Configuration

Configure the pipeline using YAML files:

```yaml
# configs/default.yaml
model:
  name: "distilbert-base-uncased"
  num_labels: 2
  max_length: 512

training:
  batch_size: 16
  learning_rate: 2e-5
  epochs: 3

interpretability:
  methods: ["shap", "lime", "attention"]
  shap_samples: 100
  lime_num_features: 10
```

## Examples

See the `notebooks/` directory for detailed examples:
- `01_basic_classification.ipynb` - Basic text classification
- `02_interpretability_demo.ipynb` - All interpretability methods
- `03_custom_models.ipynb` - Using custom model architectures

## License

MIT License

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{llm_interpretability_pipeline,
  title = {LLM Interpretability Pipeline},
  author = {Sugeerth Murugesan},
  year = {2024},
  url = {https://github.com/sugeerth/llm-interpretability-pipeline}
}
```
