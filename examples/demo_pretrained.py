"""
Demo of llm-interpretability-pipeline on a pretrained DistilBERT SST-2 model.
Runs: Integrated Gradients, last-layer attention, SHAP token attributions.
No training needed — just downloads the fine-tuned model from HF.

Usage:
    source /Users/fullfocus/interpretabile/.venv/bin/activate
    python3 examples/demo_pretrained.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.interpretability.shap_explainer import SHAPExplainer
from src.interpretability.attention_visualizer import AttentionVisualizer
from src.interpretability.integrated_gradients import IntegratedGradientsExplainer

MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
device = torch.device("cpu")

print(f"Loading: {MODEL}")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, attn_implementation="eager"
).to(device)
model.eval()

text = "The movie was visually stunning but the plot was completely incoherent."
print(f"\nINPUT: {text!r}")

with torch.no_grad():
    inputs = tokenizer(text, return_tensors="pt").to(device)
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    pred_idx = int(probs.argmax())
    label = ["NEGATIVE", "POSITIVE"][pred_idx]
    print(f"PREDICTION: {label}  (neg={probs[0]:.3f}, pos={probs[1]:.3f})")

print("\n--- Integrated Gradients (Captum) ---")
try:
    ig = IntegratedGradientsExplainer(model, tokenizer, device)
    ig_result = ig.explain(text, target_class=pred_idx, n_steps=30)
    for item in sorted(
        ig_result["token_attributions"],
        key=lambda x: -abs(x["attribution"]),
    )[:8]:
        print(f"  {item['token']:15s}  attr={item['attribution']:+.4f}")
except Exception as e:
    print(f"  (failed: {type(e).__name__}: {e})")

print("\n--- Attention (last layer, CLS attending to each token) ---")
try:
    av = AttentionVisualizer(model, tokenizer, device)
    attn_t, toks = av._get_attention(text)
    last = attn_t[-1, 0]
    cls_to_all = last[:, 0, :].mean(0)
    pairs = sorted(zip(toks, cls_to_all.tolist()), key=lambda x: -x[1])[:8]
    for tok, w in pairs:
        print(f"  CLS -> {tok:15s}  attn={w:.3f}")
except Exception as e:
    print(f"  (failed: {type(e).__name__}: {e})")

print("\n--- SHAP token attributions (using max_evals alias) ---")
try:
    shap_exp = SHAPExplainer(model, tokenizer, device)
    sv = shap_exp.explain(text, max_evals=32, target_class=pred_idx)
    for item in sv.get("token_attributions", [])[:8]:
        print(f"  {item['token']:15s}  contrib={item['attribution']:+.4f}")
except Exception as e:
    print(f"  (failed: {type(e).__name__}: {e})")

print("\nDone.")
