"""
FastAPI web UI for llm-interpretability-pipeline.
Wraps SHAP + Integrated Gradients + Attention on a pretrained DistilBERT SST-2.

Run:
    source /Users/fullfocus/interpretabile/.venv/bin/activate
    python3 examples/serve_demo.py
Then open http://localhost:8766/
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.interpretability.shap_explainer import SHAPExplainer
from src.interpretability.attention_visualizer import AttentionVisualizer
from src.interpretability.integrated_gradients import IntegratedGradientsExplainer

MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
device = torch.device("cpu")

print(f"[serve_demo] loading {MODEL} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, attn_implementation="eager"
).to(device).eval()

ig = IntegratedGradientsExplainer(model, tokenizer, device)
av = AttentionVisualizer(model, tokenizer, device)
shap_exp = SHAPExplainer(model, tokenizer, device)
print("[serve_demo] explainers ready.")

app = FastAPI(title="llm-interpretability-pipeline demo")


class ExplainReq(BaseModel):
    text: str
    top_k: int = 8


@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!doctype html><html><head><meta charset=utf-8>
<title>llm-interpretability-pipeline</title>
<style>
  body { font-family: -apple-system, system-ui, sans-serif; max-width: 900px;
         margin: 2rem auto; padding: 0 1rem; color: #222; }
  h1 { font-size: 1.3rem; margin-bottom: 0.2rem; }
  .sub { color: #777; margin-bottom: 1rem; font-size: 0.9rem; }
  textarea { width: 100%; padding: 0.6rem; font-size: 1rem; min-height: 4rem;
             border: 1px solid #ccc; border-radius: 6px; }
  button { padding: 0.5rem 1rem; font-size: 1rem; background: #2563eb;
           color: white; border: 0; border-radius: 6px; cursor: pointer; }
  .col { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; margin-top: 1rem; }
  .card { border: 1px solid #e5e7eb; border-radius: 8px; padding: 0.8rem;
          background: #fafafa; font-size: 0.85rem; }
  .card h3 { margin: 0 0 0.4rem; font-size: 0.95rem; }
  .row { display: flex; justify-content: space-between; padding: 2px 0;
         font-variant-numeric: tabular-nums; font-family: ui-monospace, monospace; }
  .pos { color: #b91c1c; } .neg { color: #166534; }
  .pred { padding: 0.5rem 0.8rem; background: #eef; border-radius: 6px;
          display: inline-block; margin-top: 0.6rem; }
</style></head><body>
<h1>llm-interpretability-pipeline</h1>
<div class=sub>DistilBERT SST-2 · Integrated Gradients + Attention + SHAP</div>
<textarea id=t>The movie was visually stunning but the plot was completely incoherent.</textarea>
<div style="margin-top:0.5rem"><button onclick=go()>Explain</button>
  <span id=status style="margin-left:1rem;color:#777"></span></div>
<div id=pred class=pred style="display:none"></div>
<div class=col id=out></div>
<script>
async function go() {
  const text = document.getElementById('t').value;
  const status = document.getElementById('status');
  const out = document.getElementById('out');
  const pred = document.getElementById('pred');
  status.textContent = 'running 3 explainers...';
  out.innerHTML = ''; pred.style.display = 'none';
  const r = await fetch('/explain', {method:'POST', headers:{'content-type':'application/json'},
    body: JSON.stringify({text, top_k: 8})});
  const j = await r.json();
  status.textContent = '';
  pred.style.display = 'inline-block';
  pred.textContent = `Prediction: ${j.label}  (neg=${j.probs[0].toFixed(3)}, pos=${j.probs[1].toFixed(3)})`;
  const card = (title, items, key) => {
    const lines = items.map(it => {
      const v = it[key];
      const cls = v >= 0 ? 'pos' : 'neg';
      return `<div class=row><span>${it.token}</span><span class=${cls}>${v>=0?'+':''}${v.toFixed(3)}</span></div>`;
    }).join('');
    return `<div class=card><h3>${title}</h3>${lines}</div>`;
  };
  out.innerHTML = card('Integrated Gradients', j.ig, 'attribution')
    + card('Attention (last layer, CLS→tok)', j.attn, 'attribution')
    + card('SHAP', j.shap, 'attribution');
}
</script></body></html>
"""


@app.post("/explain")
def explain(req: ExplainReq):
    text = req.text
    k = req.top_k
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt").to(device)
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].tolist()
    pred_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
    label = ["NEGATIVE", "POSITIVE"][pred_idx]

    # IG
    try:
        ig_r = ig.explain(text, target_class=pred_idx, n_steps=30)
        ig_items = sorted(ig_r["token_attributions"],
                          key=lambda x: -abs(x["attribution"]))[:k]
        ig_items = [{"token": x["token"], "attribution": x["attribution"]} for x in ig_items]
    except Exception as e:
        ig_items = [{"token": f"err: {type(e).__name__}", "attribution": 0.0}]

    # Attention
    try:
        attn_t, toks = av._get_attention(text)
        cls_to_all = attn_t[-1, 0][:, 0, :].mean(0).tolist()
        attn_items = sorted(zip(toks, cls_to_all), key=lambda x: -x[1])[:k]
        attn_items = [{"token": t, "attribution": float(w)} for t, w in attn_items]
    except Exception as e:
        attn_items = [{"token": f"err: {type(e).__name__}", "attribution": 0.0}]

    # SHAP
    try:
        sv = shap_exp.explain(text, max_evals=32, target_class=pred_idx)
        shap_items = sv.get("token_attributions", [])[:k]
        shap_items = [{"token": x["token"], "attribution": x["attribution"]} for x in shap_items]
    except Exception as e:
        shap_items = [{"token": f"err: {type(e).__name__}", "attribution": 0.0}]

    return JSONResponse({
        "label": label, "probs": probs,
        "ig": ig_items, "attn": attn_items, "shap": shap_items,
    })


if __name__ == "__main__":
    import uvicorn
    # PORT defaults to 8766 locally, 7860 on HF Spaces (their convention).
    port = int(os.environ.get("PORT", "8766"))
    # Bind to 0.0.0.0 inside containers (HF Spaces); fall back to localhost
    # when running on a workstation.
    host = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"
    uvicorn.run(app, host=host, port=port)
