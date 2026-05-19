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
    --bg:#ffffff; --panel:#ffffff; --border:#e5e7eb; --text:#111827;
    --muted:#6b7280; --soft:#f9fafb; --accent:#2563eb;
    --pos:#16a34a; --neg:#dc2626; --neutral:#9ca3af;
  }
  *{box-sizing:border-box;}
  html,body{background:var(--bg);color:var(--text);
    font-family:-apple-system,system-ui,"Segoe UI",sans-serif;
    margin:0;padding:0;line-height:1.45;}
  .wrap{max-width:1180px;margin:2rem auto;padding:0 1.5rem;}
  header{margin-bottom:1.4rem;}
  h1{font-size:1.55rem;font-weight:700;margin:0 0 .3rem;letter-spacing:-.01em;}
  .sub{color:var(--muted);font-size:.92rem;}
  textarea{width:100%;padding:.85rem 1rem;font-size:1rem;min-height:3.5rem;
    background:var(--soft);color:var(--text);border:1px solid var(--border);
    border-radius:10px;font-family:inherit;resize:vertical;}
  textarea:focus{outline:none;border-color:var(--accent);background:#fff;}
  .controls{margin-top:.7rem;display:flex;align-items:center;gap:.8rem;}
  button{padding:.6rem 1.4rem;font-size:.95rem;background:var(--accent);
    color:#fff;border:0;border-radius:8px;cursor:pointer;font-weight:600;
    transition:opacity .15s;}
  button:hover:not(:disabled){opacity:.88;}
  button:disabled{opacity:.45;cursor:not-allowed;}
  .status{color:var(--muted);font-size:.88rem;}
  .pred{margin-left:auto;display:inline-flex;align-items:center;gap:.6rem;
    padding:.5rem 1rem;border-radius:8px;font-size:.95rem;font-weight:600;}
  .pred.NEGATIVE{background:#fef2f2;color:var(--neg);}
  .pred.POSITIVE{background:#f0fdf4;color:var(--pos);}
  .pred .conf{font-variant-numeric:tabular-nums;font-weight:500;
    color:var(--muted);font-size:.85rem;}

  .metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:.7rem;
    margin:1.5rem 0 1rem;}
  .metric{background:var(--soft);border:1px solid var(--border);
    border-radius:10px;padding:.85rem 1rem;}
  .metric .label{color:var(--muted);font-size:.72rem;letter-spacing:.05em;
    text-transform:uppercase;font-weight:600;}
  .metric .value{font-size:1.55rem;font-weight:700;margin-top:.2rem;
    font-variant-numeric:tabular-nums;color:var(--text);letter-spacing:-.02em;}
  .metric .hint{color:var(--muted);font-size:.75rem;margin-top:.1rem;}

  .panel{background:var(--panel);border:1px solid var(--border);
    border-radius:12px;padding:1.2rem;margin-top:1rem;}
  .panel h2{margin:0 0 .35rem;font-size:1.02rem;font-weight:600;
    letter-spacing:-.005em;}
  .panel .why{color:var(--muted);font-size:.84rem;margin:0 0 1rem;
    max-width:780px;line-height:1.5;}
  .panel .plot{height:280px;}
  .panel .plot.tall{height:340px;}
  .row2{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-top:1rem;}
  @media (max-width:880px){
    .metrics{grid-template-columns:repeat(2,1fr);}
    .row2{grid-template-columns:1fr;}
    .pred{margin-left:0;}
    .controls{flex-wrap:wrap;}
  }
</style></head><body><div class=wrap>

<header>
  <h1>LLM Interpretability — DistilBERT SST-2</h1>
  <div class=sub>Five views into how the model decided. Type a sentence below.</div>
</header>

<textarea id=t>The movie was visually stunning but the plot was completely incoherent.</textarea>
<div class=controls>
  <button id=btn onclick=go()>Explain</button>
  <span id=status class=status></span>
  <span id=pred class=pred style="display:none"></span>
</div>

<div class=metrics id=metrics style="display:none">
  <div class=metric>
    <div class=label>Confidence</div>
    <div class=value id=m_conf>—</div>
    <div class=hint>P(predicted class)</div>
  </div>
  <div class=metric>
    <div class=label>Decision Layer</div>
    <div class=value id=m_layer>—</div>
    <div class=hint>where the model commits</div>
  </div>
  <div class=metric>
    <div class=label>Attention Focus</div>
    <div class=value id=m_ent>—</div>
    <div class=hint>lower entropy = sharper</div>
  </div>
  <div class=metric>
    <div class=label>Method Agreement</div>
    <div class=value id=m_agree>—</div>
    <div class=hint>across IG · Attn · SHAP</div>
  </div>
</div>

<div id=out style="display:none">

  <div class=panel>
    <h2>1 · Which tokens drove the prediction</h2>
    <p class=why>Three independent attribution methods scored each token. Bars show the average score; agreement across methods is the trust signal.</p>
    <div id=p1 class="plot tall"></div>
  </div>

  <div class=row2>
    <div class=panel>
      <h2>2 · When did the model decide</h2>
      <p class=why>Apply the classifier head to the CLS state at every layer. The crossing point is when the prediction crystallizes.</p>
      <div id=p2 class=plot></div>
    </div>
    <div class=panel>
      <h2>3 · Where the attention looks</h2>
      <p class=why>6 layers × 12 heads. Bright = a focused head (low entropy); dim = diffuse / less informative.</p>
      <div id=p3 class=plot></div>
    </div>
  </div>

  <div class=row2>
    <div class=panel>
      <h2>4 · Which hidden units fired</h2>
      <p class=why>Top classifier neurons by contribution to the predicted logit. The grey bar is the same neuron's pull on the opposite class.</p>
      <div id=p4 class=plot></div>
    </div>
    <div class=panel>
      <h2>5 · Do the methods agree</h2>
      <p class=why>Jaccard overlap between the top-5 tokens picked by each pair of methods. High = robust attribution; low = methods disagree on what mattered.</p>
      <div id=p5 class=plot></div>
    </div>
  </div>

</div>

<script>
const POS='#16a34a', NEG='#dc2626', BLUE='#2563eb', GREY='#9ca3af', AMBER='#d97706';
const LAYOUT_BASE = {
  paper_bgcolor:'#ffffff', plot_bgcolor:'#ffffff',
  font:{color:'#111827', family:'-apple-system, system-ui, sans-serif', size:11.5},
  margin:{l:60, r:18, t:18, b:50}, hovermode:'closest',
  xaxis:{linecolor:'#e5e7eb', gridcolor:'#f3f4f6', zerolinecolor:'#e5e7eb'},
  yaxis:{linecolor:'#e5e7eb', gridcolor:'#f3f4f6', zerolinecolor:'#e5e7eb'},
};

async function go() {
  const btn = document.getElementById('btn');
  const text = document.getElementById('t').value;
  const status = document.getElementById('status');
  const out = document.getElementById('out');
  const pred = document.getElementById('pred');
  const metrics = document.getElementById('metrics');
  btn.disabled = true;
  status.textContent = 'analyzing (~10–25s) …';
  out.style.display = 'none'; pred.style.display = 'none'; metrics.style.display = 'none';
  const t0 = performance.now();
  try {
    const r = await fetch('/explain', {method:'POST',
      headers:{'content-type':'application/json'},
      body: JSON.stringify({text, top_k: 12})});
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

// merge IG / Attention / SHAP into one token-importance view.
// Each method picks top tokens with absolute scores normalized to [-1, +1] for
// the pull-toward-prediction direction; "+" means pushes toward predicted class.
function buildTokenImportance(j) {
  const isPos = j.pred_idx === 1;
  const normalize = (items, ref) => {
    const max = Math.max(...items.map(x => Math.abs(x.attribution))) || 1;
    return Object.fromEntries(items.map(x => [x.token.toString().trim(), x.attribution / max]));
  };
  const igMap = normalize(j.ig);
  const attnMap = Object.fromEntries(j.attn.map(x => {
    const max = Math.max(...j.attn.map(y => Math.abs(y.attribution))) || 1;
    return [x.token.toString().trim(), x.attribution / max];
  }));
  const shapMap = normalize(j.shap);
  // union of tokens, sorted by mean absolute score
  const allToks = new Set([...Object.keys(igMap), ...Object.keys(attnMap), ...Object.keys(shapMap)]);
  const rows = [...allToks]
    .filter(t => t && t !== '[CLS]' && t !== '[SEP]' && t !== '[PAD]')
    .map(t => ({
      tok: t,
      ig: igMap[t] || 0,
      attn: attnMap[t] || 0,
      shap: shapMap[t] || 0,
    }))
    .map(r => ({...r, mean_abs: (Math.abs(r.ig) + Math.abs(r.attn) + Math.abs(r.shap)) / 3}))
    .sort((a, b) => b.mean_abs - a.mean_abs)
    .slice(0, 12);
  return rows.reverse(); // bottom-to-top in horizontal chart
}

function render(j) {
  // header pill
  const pred = document.getElementById('pred');
  pred.innerHTML = `${j.label} <span class=conf>P = ${j.probs[j.pred_idx].toFixed(3)}</span>`;
  pred.className = 'pred ' + j.label;
  pred.style.display = 'inline-flex';

  // top metrics
  document.getElementById('m_conf').textContent = (j.probs[j.pred_idx]*100).toFixed(1) + '%';
  document.getElementById('m_layer').textContent =
    j.lens.decision_layer !== null ? `L${j.lens.decision_layer} / L${j.lens.per_layer.length-1}` : '—';
  document.getElementById('m_ent').textContent = j.entropy_mean.toFixed(2) + ' bits';
  document.getElementById('m_agree').textContent = (j.agreement.mean*100).toFixed(0) + '%';
  document.getElementById('metrics').style.display = 'grid';

  // PANEL 1: Token importance (3 methods combined)
  const tok = buildTokenImportance(j);
  Plotly.newPlot('p1', [
    {y: tok.map(r => r.tok), x: tok.map(r => r.ig),
      type:'bar', orientation:'h', name:'Integrated Gradients',
      marker:{color: BLUE}},
    {y: tok.map(r => r.tok), x: tok.map(r => r.attn),
      type:'bar', orientation:'h', name:'Attention',
      marker:{color: AMBER}},
    {y: tok.map(r => r.tok), x: tok.map(r => r.shap),
      type:'bar', orientation:'h', name:'SHAP',
      marker:{color: POS}},
  ], {...LAYOUT_BASE, barmode:'group', height:340,
       margin:{l:90, r:18, t:18, b:42},
       xaxis:{...LAYOUT_BASE.xaxis, title:'token contribution (normalized)', range:[-1.05, 1.05]},
       yaxis:{...LAYOUT_BASE.yaxis, automargin:true},
       legend:{orientation:'h', y:-0.18, x:0.5, xanchor:'center'}},
     {responsive:true, displayModeBar:false});

  // PANEL 2: Logit lens
  const lx = j.lens.per_layer.map(d => d.layer);
  const predTrace = j.pred_idx === 1
    ? j.lens.per_layer.map(d => d.p_pos)
    : j.lens.per_layer.map(d => d.p_neg);
  Plotly.newPlot('p2', [
    {x: lx, y: predTrace, type:'scatter', mode:'lines+markers',
      name:`P(${j.label.toLowerCase()})`,
      line:{color: j.pred_idx === 1 ? POS : NEG, width:3, shape:'spline'},
      marker:{size:9}},
    {x: lx, y: lx.map(() => 0.5), type:'scatter', mode:'lines',
      line:{color:GREY, dash:'dot', width:1.5}, name:'50% threshold',
      hoverinfo:'skip'},
  ], {...LAYOUT_BASE,
       xaxis:{...LAYOUT_BASE.xaxis, title:'Layer', dtick:1},
       yaxis:{...LAYOUT_BASE.yaxis, title:'Confidence', range:[0,1], tickformat:'.0%'},
       showlegend:false},
     {responsive:true, displayModeBar:false});

  // PANEL 3: Attention focus heatmap
  const z = j.attn_heatmap.map(r => r.focus);
  Plotly.newPlot('p3', [{
    z: z, type:'heatmap',
    colorscale:[[0,'#f3f4f6'],[0.5,'#93c5fd'],[1,'#1d4ed8']],
    x: Array.from({length: z[0].length}, (_,i) => i),
    y: z.map((_,i) => i),
    hovertemplate: 'Layer %{y}, Head %{x}<br>focus %{z:.2f}<extra></extra>',
    colorbar:{title:'', thickness:10, len:0.85, tickfont:{size:10}},
  }], {...LAYOUT_BASE,
       xaxis:{...LAYOUT_BASE.xaxis, title:'Head', dtick:1, side:'bottom'},
       yaxis:{...LAYOUT_BASE.yaxis, title:'Layer', dtick:1, autorange:'reversed'}},
     {responsive:true, displayModeBar:false});

  // PANEL 4: Top neurons
  const ns = j.neurons.slice(0, 10);
  const predColor = j.pred_idx === 1 ? POS : NEG;
  Plotly.newPlot('p4', [
    {x: ns.map(n => `n${n.neuron}`),
      y: ns.map(n => n.contrib_predicted),
      type:'bar', name:`→ ${j.label}`,
      marker:{color: predColor},
      hovertemplate:'%{x}<br>contrib %{y:+.3f}<extra></extra>'},
    {x: ns.map(n => `n${n.neuron}`),
      y: ns.map(n => n.contrib_other),
      type:'bar', name:`→ opposite class`,
      marker:{color: GREY},
      hovertemplate:'%{x}<br>contrib %{y:+.3f}<extra></extra>'},
  ], {...LAYOUT_BASE, barmode:'group',
       xaxis:{...LAYOUT_BASE.xaxis, title:'classifier hidden-dim index'},
       yaxis:{...LAYOUT_BASE.yaxis, title:'contribution to logit', zeroline:true},
       legend:{orientation:'h', y:-0.22, x:0.5, xanchor:'center'}},
     {responsive:true, displayModeBar:false});

  // PANEL 5: Method agreement
  const pairs = [
    ['IG vs Attention', j.agreement.ig_vs_attn],
    ['IG vs SHAP', j.agreement.ig_vs_shap],
    ['Attention vs SHAP', j.agreement.attn_vs_shap],
  ];
  Plotly.newPlot('p5', [{
    y: pairs.map(p => p[0]),
    x: pairs.map(p => p[1]*100),
    type:'bar', orientation:'h',
    marker:{color: pairs.map(p =>
      p[1] >= 0.5 ? POS : (p[1] >= 0.25 ? AMBER : NEG))},
    text: pairs.map(p => (p[1]*100).toFixed(0)+'%'),
    textposition:'outside', textfont:{size:12, color:'#111827'},
    hovertemplate:'%{y}: %{x:.0f}%<extra></extra>',
    cliponaxis: false,
  }], {...LAYOUT_BASE,
        xaxis:{...LAYOUT_BASE.xaxis, range:[0,115], title:'% top-5 token overlap',
               ticksuffix:'%', dtick:25},
        yaxis:{...LAYOUT_BASE.yaxis, autorange:'reversed', automargin:true},
        margin:{l:130, r:30, t:18, b:42}},
     {responsive:true, displayModeBar:false});

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
