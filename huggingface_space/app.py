"""
🎓 AI Essay Evaluator — Capstone Demo (LT6582)
Deploy: Hugging Face Spaces / Gradio
Real RAG + Agentic Workflow + AES Scoring + Radar Chart
"""
import os, re, json, glob
import numpy as np, faiss, gradio as gr
from fastembed import TextEmbedding
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
import plotly.graph_objects as go

# ---- API key ----
KEY = os.environ.get("DEEPSEEK_API_KEY")
if not KEY and os.path.exists(".env"):
    for line in open(".env"):
        if line.startswith("DEEPSEEK_API_KEY"):
            KEY = line.split("=", 1)[1].strip()
client = OpenAI(api_key=KEY, base_url="https://api.deepseek.com")
MODEL = "deepseek-chat"

# ---- KB + FAISS ----
chunks = []
for fp in sorted(glob.glob("knowledge_base/*.md")):
    for seg in re.split(r'(?m)^### ', open(fp, encoding="utf-8").read()):
        seg = re.sub(r'<!--.*?-->', '', seg, flags=re.S).strip()
        if seg: chunks.append("### " + seg)
embedder = TextEmbedding()
def embed(texts):
    v = np.array(list(embedder.embed(texts)), dtype="float32"); faiss.normalize_L2(v); return v
index = faiss.IndexFlatIP(384); index.add(embed(chunks)) if chunks else None
def retrieve(q, k=4):
    if not chunks: return []
    _, idx = index.search(embed([q]), k)
    return [chunks[i] for i in idx[0]]

def call(sys, user, max_tokens=1500, temp=0.1, json_mode=False):
    kw = {"response_format": {"type": "json_object"}} if json_mode else {}
    r = client.chat.completions.create(model=MODEL, temperature=temp, max_tokens=max_tokens,
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}], **kw)
    return r.choices[0].message.content.strip()

# ---- AES Scoring ----
SCORING_PROMPT = """You are an expert English proficiency assessor. Evaluate this ELL essay holistically on a 1.0-5.0 scale using 0.5 increments. Consider vocabulary, grammar, sentence structure, organization, and coherence together.

1.0 Minimal — Extremely limited vocabulary, pervasive errors, no organization.
2.0 Emerging — Basic vocabulary, frequent errors, limited organization.
3.0 Developing — Adequate vocabulary, noticeable errors, basic structure.
4.0 Proficient — Good range, few errors, clear organization.
5.0 Advanced — Rich vocabulary, rare errors, sophisticated structure.

Respond ONLY as JSON: {"overall":3.0,"grammar":3.0,"vocabulary":2.5,"organization":3.0,"content":3.0}"""

def score_essay(essay):
    raw = call(SCORING_PROMPT, f"Essay:\n\n{essay}", 200, temp=0.1, json_mode=True)
    try:
        scores = json.loads(raw)
        return {k: float(v) for k, v in scores.items()}
    except:
        return {"overall": 3.0, "grammar": 3.0, "vocabulary": 3.0, "organization": 3.0, "content": 3.0}

# ---- Radar Chart ----
def radar_chart(scores):
    dims = ["Grammar", "Vocabulary", "Organization", "Content"]
    vals = [scores.get("grammar", 0), scores.get("vocabulary", 0),
            scores.get("organization", 0), scores.get("content", 0)]
    vals.append(vals[0]); dims.append(dims[0])
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=vals, theta=dims, fill="toself",
        line_color="#6366f1", fillcolor="rgba(99,102,241,0.25)",
        name="Essay Scores"))
    fig.add_trace(go.Scatterpolar(r=[scores.get("overall",0)]*5,
        theta=["Grammar","Vocabulary","Organization","Content","Grammar"],
        line_color="#f43f5e", line_dash="dash", name="Holistic"))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0,5.5], tickvals=[1,2,3,4,5])),
        showlegend=False, margin=dict(l=30,r=30,t=20,b=20), height=320,
        paper_bgcolor="rgba(0,0,0,0)", font=dict(size=12))
    return fig

# ---- Agentic Workflow ----
EXPERTS = {"Grammar": "grammar subject-verb agreement tense articles run-on",
           "Vocabulary": "vocabulary word choice Chinglish collocation repetition",
           "Organisation": "organisation coherence topic sentence transitions",
           "Content": "content ideas argument development support"}
def expert(dim, essay):
    kb = "\n\n".join(retrieve(EXPERTS[dim] + " " + essay[:600], 4))
    sys = (f"You assess ONLY the '{dim}' dimension. Use the reference; do not invent issues.\n"
           f"=== REFERENCE ===\n{kb}\n=== END ===\nReport at most 5 most significant problems. "
           'Respond ONLY as JSON: {"findings":[{"quote":"...","issue":"...","correction":"..."}]}.')
    raw = call(sys, f"Essay:\n\n{essay}", 2000, json_mode=True)
    try: fs = json.loads(raw).get("findings", [])
    except: fs = [json.loads(m) for m in re.findall(r'\{[^{}]*\}', raw) if '"issue"' in m or '"quote"' in m]
    return [dict(f, dimension=dim) for f in fs][:5]

def agentic_feedback(essay, lvl):
    with ThreadPoolExecutor(max_workers=4) as ex:
        findings = sum(ex.map(lambda d: expert(d, essay), EXPERTS.keys()), [])
    ne = re.sub(r'\s+', ' ', essay.lower())
    survivors, dropped = [], []
    for f in findings:
        q = re.sub(r'\s+', ' ', f.get("quote", "").lower()).strip()
        (dropped if (q and len(q)>8 and q not in ne and q[:30] not in ne) else survivors).append(f)
    if survivors:
        sys = ("Fact-check each error. FALSE only if: quote not in essay, pure style, or not actually wrong. "
               'Respond ONLY as JSON: {"verdicts":[true,false,...]}.')
        payload = f"Essay:\n\n{essay}\n\nErrors:\n{json.dumps([{'quote':f.get('quote',''),'issue':f.get('issue','')} for f in survivors])}"
        try: verd = json.loads(call(sys, payload, 800, json_mode=True)).get("verdicts",[])
        except: verd = [True]*len(survivors)
        verd += [True]*(len(survivors)-len(verd))
        kept = [f for f,v in zip(survivors,verd) if v]
        dropped += [f for f,v in zip(survivors,verd) if not v]
    else: kept = []
    BASE = ("You are a supportive English writing teacher (student: {lvl} proficiency). "
            "Using ONLY verified findings (no new errors), write encouraging feedback:\n"
            "🌟 1 thing done well, then 4 sections: Grammar, Vocabulary, Organisation, Content. "
            "Each section: quote error → explain → improved version. End with 2-3 tips.")
    fb = call(BASE.format(lvl=lvl),
              f"Essay:\n\n{essay}\n\nVerified findings:\n{json.dumps(kept, ensure_ascii=False)}", 1500)
    log = (f"### 🔬 Pipeline\n**{len(findings)}** initial → self-correction kept **{len(kept)}**, "
           f"dropped **{len(dropped)}**\n")
    if dropped:
        log += "\n> ⚠️ Dropped (hallucinated/style):\n" + "\n".join(
            f"> - [{d['dimension']}] {d.get('issue','')[:80]}" for d in dropped[:5])
    return fb, log

def single_pass(essay, lvl, use_rag):
    BASE = ("You are a supportive university writing teacher (student: {lvl} proficiency). "
            "Begin with one thing done well, then 4 sections: (1) Grammar (2) Vocabulary (3) Organisation (4) Content. "
            "Each: quote example → explain issue → improved version. End with 2-3 tips. Only comment on real issues.")
    sys = BASE.format(lvl=lvl)
    if use_rag:
        kb = "\n\n".join(retrieve(essay, 6))
        sys = f"Ground feedback in this reference; do not invent.\n=== REFERENCE ===\n{kb}\n=== END ===\n\n" + sys
    return call(sys, f"Essay:\n\n{essay}", 1500), f"Mode: {'RAG ON' if use_rag else 'RAG OFF'} | Single-pass LLM call"

# ---- Main generate ----
def generate(essay, level, mode):
    if not KEY: return "⚠️ API key not set. Add DEEPSEEK_API_KEY as Space secret.", None, ""
    if not essay.strip(): return "Please paste an essay.", None, ""
    essay_text = essay.strip()
    scores = score_essay(essay_text)
    chart = radar_chart(scores)
    score_md = (f"### 📊 Scores\n"
                f"| Holistic | Grammar | Vocabulary | Organisation | Content |\n"
                f"|---|---|---|---|---|\n"
                f"| **{scores['overall']}** | {scores['grammar']} | {scores['vocabulary']} | "
                f"{scores['organization']} | {scores['content']} |")
    if mode == "Agentic (4 experts + self-correction)":
        fb, log = agentic_feedback(essay_text, level)
        return fb, chart, score_md + "\n\n" + log
    elif mode == "Single-pass + RAG":
        fb, log = single_pass(essay_text, level, use_rag=True)
        return fb, chart, score_md + "\n\n" + log
    else:
        fb, log = single_pass(essay_text, level, use_rag=False)
        return fb, chart, score_md + "\n\n" + log

# ---- UI ----
CSS = """
.gradio-container { max-width: 1100px !important; }
.header { text-align: center; padding: 20px; background: linear-gradient(135deg, #6366f1, #8b5cf6); color: white; border-radius: 12px; margin-bottom: 16px; }
.header h1 { font-size: 28px; margin: 0; }
.header p { opacity: 0.9; margin: 4px 0 0; }
.tag { display: inline-block; background: rgba(255,255,255,0.2); padding: 2px 10px; border-radius: 10px; font-size: 12px; margin: 2px; }
"""

with gr.Blocks(title="🎓 AI Essay Evaluator — LT6582 Capstone", css=CSS, theme=gr.themes.Soft()) as demo:
    gr.HTML("""
    <div class="header">
      <h1>🎓 AI Essay Evaluator</h1>
      <p>Agentic Workflow + RAG + Self-Correction — LT6582 Capstone Project</p>
      <p>
        <span class="tag">4 Expert Agents</span>
        <span class="tag">RAG Retrieval</span>
        <span class="tag">Self-Correction</span>
        <span class="tag">AES Scoring</span>
        <span class="tag">DeepSeek V3</span>
      </p>
    </div>
    """)
    with gr.Row():
        with gr.Column(scale=2):
            essay = gr.Textbox(label="📝 Student Essay", lines=12, placeholder="Paste an English essay here (100-600 words)...")
            with gr.Row():
                level = gr.Dropdown(["low", "mid", "high"], value="mid", label="Proficiency")
                mode = gr.Radio(["Agentic (4 experts + self-correction)", "Single-pass + RAG", "Single-pass (no RAG)"],
                                value="Agentic (4 experts + self-correction)", label="Mode")
            btn = gr.Button("🚀 Generate Feedback", variant="primary", size="lg")
        with gr.Column(scale=1):
            chart = gr.Plot(label="Radar Chart")

    out = gr.Markdown(label="📋 Feedback")
    detail = gr.Markdown(label="🔬 Pipeline Log", visible=True)

    gr.Markdown("---
*Built for LT6582 Capstone — Meng Ni • Agentic Workflow + RAG + Self-Correction*")

    btn.click(generate, [essay, level, mode], [out, chart, detail])

if __name__ == "__main__":
    demo.launch()
