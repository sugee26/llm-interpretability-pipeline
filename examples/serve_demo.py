"""
Enhanced interpretability dashboard for DistilBERT-SST2.

Surfaces:
  - 3-method token attribution (Integrated Gradients, Attention, SHAP)
  - Logit lens: per-layer P(class) trajectory
  - Attention head entropy heatmap (6 layers x 12 heads)
  - Top classifier neurons (CLS dim * classifier weight per predicted class)
  - Per-token input gradient saliency
  - Method agreement (Jaccard over top-K tokens) + entropy stats

Run:
    source /Users/fullfocus/interpretabile/.venv/bin/activate
    python3 examples/serve_demo.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
import torch.nn.functional as F
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

ig_exp = IntegratedGradientsExplainer(model, tokenizer, device)
av_exp = AttentionVisualizer(model, tokenizer, device)
shap_exp = SHAPExplainer(model, tokenizer, device)
print("[serve_demo] explainers ready.")

NUM_LAYERS = model.config.num_hidden_layers       # 6 for DistilBERT
NUM_HEADS = model.config.num_attention_heads      # 12
HIDDEN = model.config.dim                         # 768
LABELS = ["NEGATIVE", "POSITIVE"]


def _entropy(p, eps=1e-12):
    p = p.clamp_min(eps)
    return float(-(p * p.log()).sum())


def _saliency(text: str, target_class: int):
    """Per-token input-gradient saliency (L2 norm of grad w.r.t. embedding)."""
    enc = tokenizer(text, return_tensors="pt").to(device)
    ids = enc["input_ids"]
    mask = enc["attention_mask"]
    toks = tokenizer.convert_ids_to_tokens(ids[0])
    with torch.enable_grad():
        model.zero_grad(set_to_none=True)
        embeds = model.distilbert.embeddings(ids).detach().clone()
        embeds.requires_grad_(True)
        embeds.retain_grad()
        out = model.distilbert(inputs_embeds=embeds, attention_mask=mask)
        cls = out.last_hidden_state[:, 0]
        pre = F.relu(model.pre_classifier(cls))
        logits = model.classifier(pre)
        score = logits[0, target_class]
        score.backward()
    grad = embeds.grad
    if grad is None:
        return [{"token": t, "saliency": 0.0} for t in toks]
    sal = grad[0].norm(dim=-1).detach().tolist()
    sal_max = max(sal) or 1.0
    sal_norm = [v / sal_max for v in sal]
    return [{"token": t, "saliency": s} for t, s in zip(toks, sal_norm)]


def _logit_lens(text: str):
    """Apply pre_classifier + classifier to every layer's CLS hidden state."""
    enc = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.distilbert(**enc, output_hidden_states=True)
        per_layer = []
        for li, h in enumerate(out.hidden_states):    # length = num_layers + 1
            cls = h[:, 0]
            pre = F.relu(model.pre_classifier(cls))
            logits = model.classifier(pre)
            p = F.softmax(logits, dim=-1)[0].tolist()
            per_layer.append({"layer": li, "p_neg": p[0], "p_pos": p[1]})
        # final (real) prediction
        final = model.classifier(F.relu(model.pre_classifier(out.last_hidden_state[:, 0])))
        p_final = F.softmax(final, dim=-1)[0].tolist()
    return {"per_layer": per_layer, "p_final": p_final}


def _attention_entropy(text: str):
    """Per-head entropy of CLS-row attention; identifies focused vs diffuse heads."""
    enc = tokenizer(text, return_tensors="pt").to(device)
    prev = getattr(model.config, "_attn_implementation", None)
    model.config._attn_implementation = "eager"
    try:
        with torch.no_grad():
            out = model.distilbert(**enc, output_attentions=True)
    finally:
        if prev is not None:
            model.config._attn_implementation = prev
    attns = out.attentions             # tuple of (1, n_heads, seq, seq)
    rows = []
    for li, a in enumerate(attns):
        cls_row = a[0, :, 0, :]        # (n_heads, seq)
        ents = [_entropy(cls_row[h]) for h in range(cls_row.shape[0])]
        max_ent = math.log(cls_row.shape[1]) or 1.0
        rows.append({"layer": li, "entropy": ents,
                     "focus": [1.0 - (e / max_ent) for e in ents]})
    return rows


def _top_neurons(text: str, pred_idx: int, top_k=12):
    """Decompose classifier logit into per-neuron contributions.

    contrib[d] = pre_activation[d] * classifier.weight[pred_idx, d]
    The bias is constant per class and doesn't help rank dims, so we drop it.
    """
    enc = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.distilbert(**enc)
        cls = out.last_hidden_state[:, 0]
        pre = F.relu(model.pre_classifier(cls))[0]            # (hidden,)
    w_pred = model.classifier.weight[pred_idx]                # (hidden,)
    w_other = model.classifier.weight[1 - pred_idx]
    contrib_pred = (pre * w_pred).detach().tolist()
    contrib_other = (pre * w_other).detach().tolist()
    idx = sorted(range(HIDDEN), key=lambda i: -abs(contrib_pred[i]))[:top_k]
    return [
        {
            "neuron": int(i),
            "activation": float(pre[i].item()),
            "contrib_predicted": float(contrib_pred[i]),
            "contrib_other": float(contrib_other[i]),
        }
        for i in idx
    ]


def _agreement(ig_items, attn_items, shap_items, k=5):
    """Jaccard agreement between top-k token sets across the 3 methods."""
    def cleanset(items):
        return {str(x["token"]).strip().lower().lstrip("##") for x in items[:k]}
    a, b, c = cleanset(ig_items), cleanset(attn_items), cleanset(shap_items)
    def jac(x, y):
        u = x | y
        return (len(x & y) / len(u)) if u else 0.0
    return {
        "ig_vs_attn": jac(a, b),
        "ig_vs_shap": jac(a, c),
        "attn_vs_shap": jac(b, c),
        "mean": (jac(a, b) + jac(a, c) + jac(b, c)) / 3,
    }


app = FastAPI(title="llm-interpretability-pipeline")


class ExplainReq(BaseModel):
    text: str
    top_k: int = 10


HTML = r"""
<!doctype html><html><head><meta charset=utf-8>
<title>LLM Interpretability — DistilBERT SST-2</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  :root {
    --bg:#0e1117; --panel:#161b22; --border:#30363d; --text:#e6edf3;
    --muted:#8b949e; --accent:#58a6ff; --pos:#3fb950; --neg:#f85149;
  }
  html,body{background:var(--bg);color:var(--text);
    font-family:-apple-system,system-ui,"Segoe UI",sans-serif;
    margin:0; padding:0;}
  .wrap{max-width:1280px;margin:1.5rem auto;padding:0 1.2rem;}
  h1{font-size:1.3rem;margin:0 0 .2rem;}
  .sub{color:var(--muted);font-size:.85rem;margin-bottom:1rem;}
  textarea{width:100%;padding:.7rem;font-size:.95rem;min-height:3.6rem;
    background:#0d1117;color:var(--text);border:1px solid var(--border);
    border-radius:6px;box-sizing:border-box;font-family:inherit;}
  button{padding:.55rem 1.2rem;font-size:.95rem;background:var(--accent);
    color:#0d1117;border:0;border-radius:6px;cursor:pointer;font-weight:600;}
  button:disabled{opacity:.4;cursor:not-allowed;}
  .metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:.6rem;margin:.9rem 0;}
  .metric{background:var(--panel);border:1px solid var(--border);
    border-radius:8px;padding:.6rem .8rem;}
  .metric .label{color:var(--muted);font-size:.7rem;letter-spacing:.06em;
    text-transform:uppercase;}
  .metric .value{font-size:1.3rem;font-weight:600;margin-top:.15rem;
    font-variant-numeric:tabular-nums;}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:.8rem;}
  .grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:.6rem;}
  .card{background:var(--panel);border:1px solid var(--border);
    border-radius:8px;padding:.8rem;margin-top:.6rem;}
  .card h3{margin:0 0 .4rem;font-size:.92rem;font-weight:600;}
  .card p.help{color:var(--muted);font-size:.78rem;margin:0 0 .55rem;
    line-height:1.4;}
  .row{display:flex;justify-content:space-between;padding:1px 0;
    font-variant-numeric:tabular-nums;font-family:ui-monospace,monospace;
    font-size:.84rem;}
  .pos{color:var(--neg);} .neg{color:var(--pos);}
  /* names match attribution direction: positive contribution to NEGATIVE
     class shows in red, positive contribution to POSITIVE class in green */
  .pred{display:inline-block;padding:.45rem .9rem;border-radius:6px;
    margin-top:.4rem;font-size:.88rem;}
  .pred.NEGATIVE{background:#3a1f1f;color:var(--neg);}
  .pred.POSITIVE{background:#1d2f1f;color:var(--pos);}
  .status{color:var(--muted);font-size:.8rem;margin-left:.7rem;}
  .legend{color:var(--muted);font-size:.72rem;margin-top:.3rem;}
  @media (max-width:900px){.grid,.grid3,.metrics{grid-template-columns:1fr;}}
</style></head><body><div class=wrap>
<h1>LLM Interpretability — DistilBERT SST-2</h1>
<div class=sub>Token attribution · logit lens · attention entropy · classifier neurons · saliency · cross-method agreement</div>

<textarea id=t>The movie was visually stunning but the plot was completely incoherent.</textarea>
<div style="margin-top:.5rem;display:flex;align-items:center">
  <button id=btn onclick=go()>Explain</button>
  <span id=status class=status></span>
  <span id=pred class=pred style="display:none;margin-left:auto"></span>
</div>

<div class=metrics id=metrics style="display:none">
  <div class=metric><div class=label>Prediction confidence</div><div class=value id=m_conf>-</div></div>
  <div class=metric><div class=label>Decision layer (lens flip)</div><div class=value id=m_layer>-</div></div>
  <div class=metric><div class=label>Mean attn entropy (bits)</div><div class=value id=m_ent>-</div></div>
  <div class=metric><div class=label>Method agreement (top-5)</div><div class=value id=m_agree>-</div></div>
</div>

<div id=out style="display:none">
  <div class=grid3>
    <div class=card>
      <h3>Integrated Gradients</h3>
      <p class=help>How much each token's embedding pushed toward the prediction, via path-integrated gradients (Captum).</p>
      <div id=col_ig></div>
    </div>
    <div class=card>
      <h3>Attention (last layer, CLS→token)</h3>
      <p class=help>Average over 12 heads of attention from CLS to each input token in the final layer.</p>
      <div id=col_attn></div>
    </div>
    <div class=card>
      <h3>SHAP</h3>
      <p class=help>Shapley-value token contributions to the predicted class probability.</p>
      <div id=col_shap></div>
    </div>
  </div>

  <div class=grid>
    <div class=card>
      <h3>Logit Lens — confidence by layer</h3>
      <p class=help>Apply the classifier head to the CLS hidden state at every layer. Watch the model's "running guess" crystallize. Early flip = easy input; late flip = hard / hallucination-prone.</p>
      <div id=plot_lens style="height:280px"></div>
    </div>
    <div class=card>
      <h3>Attention head focus (6 layers × 12 heads)</h3>
      <p class=help>Brighter = more focused (low entropy) attention from CLS. Dimmer = diffuse / less informative head.</p>
      <div id=plot_attn_heatmap style="height:280px"></div>
    </div>
  </div>

  <div class=grid>
    <div class=card>
      <h3>Top classifier neurons</h3>
      <p class=help>Decomposes the final logit. <i>contrib = pre-activation × classifier weight</i>, per hidden-dim index. Same neuron's pull on the other class shown for contrast.</p>
      <div id=plot_neurons style="height:300px"></div>
    </div>
    <div class=card>
      <h3>Input gradient saliency (per token)</h3>
      <p class=help>L2 norm of ∂logit/∂embedding per token, normalized. Tokens the model is most sensitive to — bigger = bigger flip if removed.</p>
      <div id=plot_saliency style="height:300px"></div>
    </div>
  </div>

  <div class=card>
    <h3>Cross-method agreement</h3>
    <p class=help>Jaccard overlap between the top-5 tokens chosen by each pair of methods. High agreement = robust attribution; low agreement = methods disagree on what mattered.</p>
    <div id=plot_agree style="height:200px"></div>
  </div>
</div>

<script>
const NEG_COLOR='#f85149', POS_COLOR='#3fb950', NEUTRAL='#8b949e';
const LAYOUT_BASE = {paper_bgcolor:'#161b22', plot_bgcolor:'#161b22',
  font:{color:'#e6edf3', family:'-apple-system, system-ui, sans-serif', size:11},
  margin:{l:50, r:10, t:25, b:40}, hovermode:'closest'};

function row(tok, val, key) {
  const cls = val >= 0 ? 'pos' : 'neg';
  const sign = val >= 0 ? '+' : '';
  return `<div class=row><span>${tok}</span><span class=${cls}>${sign}${val.toFixed(3)}</span></div>`;
}

async function go() {
  const btn = document.getElementById('btn');
  const text = document.getElementById('t').value;
  const status = document.getElementById('status');
  const out = document.getElementById('out');
  const pred = document.getElementById('pred');
  const metrics = document.getElementById('metrics');
  btn.disabled = true;
  status.textContent = 'running 5 explainers (~10–25 s)...';
  out.style.display = 'none'; pred.style.display = 'none'; metrics.style.display = 'none';
  const t0 = performance.now();
  try {
    const r = await fetch('/explain', {method:'POST',
      headers:{'content-type':'application/json'},
      body: JSON.stringify({text, top_k: 10})});
    if (!r.ok) throw new Error('http ' + r.status);
    const j = await r.json();
    const dt = ((performance.now() - t0)/1000).toFixed(1);
    status.textContent = `done in ${dt}s`;
    render(j);
  } catch (e) {
    status.textContent = 'error: ' + e.message;
  } finally {
    btn.disabled = false;
  }
}

function render(j) {
  // header
  const pred = document.getElementById('pred');
  pred.textContent = `${j.label}  ·  P=${j.probs[j.pred_idx].toFixed(3)}`;
  pred.className = 'pred ' + j.label;
  pred.style.display = 'inline-block';

  // metrics
  document.getElementById('m_conf').textContent = (j.probs[j.pred_idx]*100).toFixed(1) + '%';
  document.getElementById('m_layer').textContent =
    j.lens.decision_layer !== null ? `L${j.lens.decision_layer} / L${j.lens.per_layer.length-1}` : '—';
  document.getElementById('m_ent').textContent = j.entropy_mean.toFixed(2);
  document.getElementById('m_agree').textContent = (j.agreement.mean*100).toFixed(0) + '%';
  document.getElementById('metrics').style.display = 'grid';

  // attribution columns
  const col = (id, items, key) => {
    document.getElementById(id).innerHTML =
      items.slice(0, 10).map(x => row(x.token, x[key], key)).join('');
  };
  col('col_ig', j.ig, 'attribution');
  col('col_attn', j.attn, 'attribution');
  col('col_shap', j.shap, 'attribution');

  // logit lens
  const lensX = j.lens.per_layer.map(d => d.layer);
  Plotly.newPlot('plot_lens', [
    {x: lensX, y: j.lens.per_layer.map(d => d.p_pos), type:'scatter', mode:'lines+markers',
      name:'P(positive)', line:{color: POS_COLOR, width:3}, marker:{size:8}},
    {x: lensX, y: j.lens.per_layer.map(d => d.p_neg), type:'scatter', mode:'lines+markers',
      name:'P(negative)', line:{color: NEG_COLOR, width:3}, marker:{size:8}},
    {x: lensX, y: lensX.map(() => 0.5), type:'scatter', mode:'lines',
      line:{color:NEUTRAL, dash:'dot', width:1}, showlegend:false, hoverinfo:'skip'},
  ], {...LAYOUT_BASE, xaxis:{title:'Layer', dtick:1, gridcolor:'#30363d'},
       yaxis:{title:'P(class)', range:[0,1], gridcolor:'#30363d'},
       legend:{orientation:'h', y:1.15}}, {responsive:true, displayModeBar:false});

  // attention heatmap
  const z = j.attn_heatmap.map(r => r.focus);  // 1 - entropy/log(seq)
  Plotly.newPlot('plot_attn_heatmap', [{
    z: z, type:'heatmap', colorscale:'Viridis',
    x: Array.from({length: z[0].length}, (_,i) => 'H'+i),
    y: z.map((_,i) => 'L'+i),
    hovertemplate: 'Layer %{y}, Head %{x}<br>focus %{z:.3f}<extra></extra>',
    colorbar:{title:'focus', thickness:10},
  }], {...LAYOUT_BASE, xaxis:{title:'Head', side:'top'}, yaxis:{autorange:'reversed', title:'Layer'}},
     {responsive:true, displayModeBar:false});

  // top neurons
  const ns = j.neurons;
  const labels = ns.map(n => `n${n.neuron}`);
  Plotly.newPlot('plot_neurons', [
    {x: labels, y: ns.map(n => n.contrib_predicted),
      type:'bar', name:`→ ${j.label}`,
      marker:{color: ns.map(n => n.contrib_predicted >= 0
        ? (j.label === 'POSITIVE' ? POS_COLOR : NEG_COLOR) : NEUTRAL)}},
    {x: labels, y: ns.map(n => n.contrib_other),
      type:'bar', name:`→ ${j.label === 'POSITIVE' ? 'NEGATIVE' : 'POSITIVE'}`,
      marker:{color: NEUTRAL}, opacity:0.55},
  ], {...LAYOUT_BASE, barmode:'group', xaxis:{title:'CLS-head hidden dim'},
       yaxis:{title:'contribution to logit', gridcolor:'#30363d'},
       legend:{orientation:'h', y:1.15}}, {responsive:true, displayModeBar:false});

  // saliency
  Plotly.newPlot('plot_saliency', [{
    x: j.saliency.map(s => s.token),
    y: j.saliency.map(s => s.saliency),
    type:'bar', marker:{color: '#58a6ff'},
    hovertemplate: '%{x}: %{y:.3f}<extra></extra>',
  }], {...LAYOUT_BASE, xaxis:{title:'token', tickangle:-30},
        yaxis:{title:'‖∂logit/∂embed‖ (normalized)', range:[0,1], gridcolor:'#30363d'}},
     {responsive:true, displayModeBar:false});

  // agreement gauge
  const pairs = [
    ['IG vs Attention', j.agreement.ig_vs_attn],
    ['IG vs SHAP', j.agreement.ig_vs_shap],
    ['Attention vs SHAP', j.agreement.attn_vs_shap],
    ['mean', j.agreement.mean],
  ];
  Plotly.newPlot('plot_agree', [{
    x: pairs.map(p => p[1]*100),
    y: pairs.map(p => p[0]),
    type:'bar', orientation:'h',
    marker:{color: pairs.map(p => p[1] >= 0.5 ? POS_COLOR : (p[1] >= 0.25 ? '#d29922' : NEG_COLOR))},
    text: pairs.map(p => (p[1]*100).toFixed(0)+'%'),
    textposition:'auto', textfont:{color:'#0d1117'},
  }], {...LAYOUT_BASE, xaxis:{range:[0,100], title:'% top-5 token overlap', gridcolor:'#30363d'},
        yaxis:{autorange:'reversed'}}, {responsive:true, displayModeBar:false});

  document.getElementById('out').style.display = 'block';
}
</script>
</div></body></html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


@app.post("/explain")
def explain(req: ExplainReq):
    text = req.text
    k = req.top_k

    with torch.no_grad():
        enc = tokenizer(text, return_tensors="pt").to(device)
        logits = model(**enc).logits
        probs = F.softmax(logits, dim=-1)[0].tolist()
    pred_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
    label = LABELS[pred_idx]

    # ---- token-level attribution: IG, Attention, SHAP ----
    try:
        ig = ig_exp.explain(text, target_class=pred_idx, n_steps=30)
        ig_items = sorted(ig["token_attributions"],
                          key=lambda x: -abs(x["attribution"]))[:k]
        ig_items = [{"token": x["token"], "attribution": x["attribution"]} for x in ig_items]
    except Exception as e:
        ig_items = [{"token": f"err: {type(e).__name__}", "attribution": 0.0}]

    try:
        attn_t, toks = av_exp._get_attention(text)
        cls_to_all = attn_t[-1, 0][:, 0, :].mean(0).tolist()
        attn_items = sorted(zip(toks, cls_to_all), key=lambda x: -x[1])[:k]
        attn_items = [{"token": t, "attribution": float(w)} for t, w in attn_items]
    except Exception as e:
        attn_items = [{"token": f"err: {type(e).__name__}", "attribution": 0.0}]

    try:
        sv = shap_exp.explain(text, max_evals=32, target_class=pred_idx)
        shap_items = sv.get("token_attributions", [])[:k]
        shap_items = [{"token": x["token"], "attribution": x["attribution"]}
                      for x in shap_items]
    except Exception as e:
        shap_items = [{"token": f"err: {type(e).__name__}", "attribution": 0.0}]

    # ---- model-internal views ----
    lens = _logit_lens(text)
    # decision layer: first layer where pred class wins (>0.5)
    decision_layer = None
    target_key = "p_pos" if pred_idx == 1 else "p_neg"
    for d in lens["per_layer"]:
        if d[target_key] > 0.5:
            decision_layer = d["layer"]
            break
    lens["decision_layer"] = decision_layer

    attn_heatmap = _attention_entropy(text)
    # mean entropy in bits
    all_ent = [e for r in attn_heatmap for e in r["entropy"]]
    entropy_mean = (sum(all_ent) / len(all_ent)) / math.log(2)

    neurons = _top_neurons(text, pred_idx, top_k=12)
    saliency = _saliency(text, pred_idx)
    agreement = _agreement(ig_items, attn_items, shap_items)

    return JSONResponse({
        "label": label,
        "pred_idx": pred_idx,
        "probs": probs,
        "ig": ig_items,
        "attn": attn_items,
        "shap": shap_items,
        "lens": lens,
        "attn_heatmap": attn_heatmap,
        "entropy_mean": entropy_mean,
        "neurons": neurons,
        "saliency": saliency,
        "agreement": agreement,
        "model_meta": {
            "name": MODEL,
            "num_layers": NUM_LAYERS,
            "num_heads": NUM_HEADS,
            "hidden": HIDDEN,
        },
    })


@app.get("/healthz")
def healthz():
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8766"))
    host = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"
    uvicorn.run(app, host=host, port=port)
