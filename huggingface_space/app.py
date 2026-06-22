"""
Gradio demo: AI Essay Feedback (LT6582 capstone)
Real RAG (fastembed BGE-small + FAISS) + Agentic Workflow (4 experts -> self-correction -> compose).
Set DEEPSEEK_API_KEY as a Space secret (or env var). No GPU needed.
"""
import os, re, json, glob
import numpy as np, faiss, gradio as gr
from fastembed import TextEmbedding
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

# ---- key (HF Space secret / env / local .env) ----
KEY = os.environ.get("DEEPSEEK_API_KEY")
if not KEY and os.path.exists(".env"):
    for line in open(".env"):
        if line.startswith("DEEPSEEK_API_KEY"):
            KEY = line.split("=", 1)[1].strip()
client = OpenAI(api_key=KEY, base_url="https://api.deepseek.com")
MODEL = "deepseek-chat"

# ---- KB + FAISS (built once at startup) ----
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

BASE_SYS = ("You are a supportive university English writing teacher giving feedback to an English Language "
            "Learner (proficiency: {lvl}). Begin with one thing done well, then four headings (1) Grammar and "
            "Sentence Structure, (2) Vocabulary Use, (3) Organisation and Coherence, (4) Content and Ideas — each "
            "quoting an example, explaining the issue, and giving an improved version — then 2-3 actionable tips. "
            "Only comment on issues actually present.")

def single_pass(essay, lvl, use_rag):
    sys = BASE_SYS.format(lvl=lvl)
    if use_rag:
        kb = "\n\n".join(retrieve(essay, 6))
        sys = (f"Ground your feedback in this reference knowledge; do not invent issues.\n=== REFERENCE ===\n{kb}\n=== END ===\n\n") + sys
    return call(sys, f"Essay:\n\n{essay}", 1500), ""

EXPERTS = {"Grammar": "grammar subject-verb agreement tense articles run-on",
           "Vocabulary": "vocabulary word choice Chinglish collocation repetition",
           "Organisation": "organisation coherence topic sentence transitions",
           "Content": "content ideas argument development support"}
def expert(dim, essay):
    kb = "\n\n".join(retrieve(EXPERTS[dim] + " " + essay[:600], 4))
    sys = (f"You assess ONLY the '{dim}' dimension of an ELL essay. Use the reference; do not invent issues.\n"
           f"=== REFERENCE ===\n{kb}\n=== END ===\nReport at most 5 most significant problems. "
           'Respond ONLY as JSON: {"findings":[{"quote":"...","issue":"...","correction":"..."}]}.')
    raw = call(sys, f"Essay:\n\n{essay}", 2000, json_mode=True)
    try: fs = json.loads(raw).get("findings", [])
    except Exception:
        fs = [json.loads(m) for m in re.findall(r'\{[^{}]*\}', raw) if '"issue"' in m or '"quote"' in m]
    return [dict(f, dimension=dim) for f in fs][:5]

def agentic(essay, lvl):
    with ThreadPoolExecutor(max_workers=4) as ex:
        findings = sum(ex.map(lambda d: expert(d, essay), EXPERTS.keys()), [])
    ne = re.sub(r'\s+', ' ', essay.lower())
    survivors, dropped = [], []
    for f in findings:
        q = re.sub(r'\s+', ' ', f.get("quote", "").lower()).strip()
        (dropped if (q and len(q) > 8 and q not in ne and q[:30] not in ne) else survivors).append(f)
    if survivors:
        sys = ("Fact-check each flagged error. Mark FALSE only if: the quote is not in the essay, OR it is purely "
               "stylistic, OR the essay is not actually wrong there. Otherwise TRUE. "
               'Respond ONLY as JSON: {"verdicts":[true,false,...]} in order.')
        payload = "Essay:\n\n" + essay + "\n\nErrors:\n" + json.dumps(
            [{"quote": f.get("quote",""), "issue": f.get("issue","")} for f in survivors], ensure_ascii=False)
        try: verd = json.loads(call(sys, payload, 800, json_mode=True)).get("verdicts", [])
        except Exception: verd = [True]*len(survivors)
        verd += [True]*(len(survivors)-len(verd))
        kept = [f for f, v in zip(survivors, verd) if v]
        dropped += [f for f, v in zip(survivors, verd) if not v]
    else:
        kept = []
    sys = (f"You are a supportive English writing teacher (student proficiency: {lvl}). Using ONLY the verified "
           "findings (do not add new errors), write encouraging feedback: one thing done well, then four headings "
           "(Grammar, Vocabulary, Organisation, Content) each quoting the example + improved version, then 2-3 tips.")
    fb = call(sys, f"Essay:\n\n{essay}\n\nVerified findings:\n{json.dumps(kept, ensure_ascii=False)}", 1500)
    log = (f"**Agentic pipeline:** {len(findings)} initial findings → "
           f"self-correction kept {len(kept)}, dropped {len(dropped)} as unsupported/stylistic.\n\n")
    if dropped:
        log += "**Dropped:**\n" + "\n".join(f"- [{d['dimension']}] {d.get('issue','')[:90]}" for d in dropped)
    return fb, log

def generate(essay, level, mode):
    if not KEY:
        return "⚠️ DEEPSEEK_API_KEY not set (add it as a Space secret).", ""
    if not essay.strip():
        return "请粘贴一篇作文。", ""
    if mode == "Agentic (4 experts + self-correction)":
        return agentic(essay, level)
    return single_pass(essay, level, use_rag=(mode == "Single-pass + RAG"))

with gr.Blocks(title="AI Essay Feedback — LT6582") as demo:
    gr.Markdown("# 📝 AI Essay Feedback (LT6582)\nReal RAG + Agentic Workflow over DeepSeek. Paste an essay, pick a mode.")
    with gr.Row():
        with gr.Column():
            essay = gr.Textbox(label="Student essay", lines=14, placeholder="Paste the essay here...")
            level = gr.Dropdown(["low", "mid", "high"], value="mid", label="Proficiency level")
            mode = gr.Radio(["Agentic (4 experts + self-correction)", "Single-pass + RAG", "Single-pass (no RAG)"],
                            value="Agentic (4 experts + self-correction)", label="Mode")
            btn = gr.Button("Generate feedback", variant="primary")
        with gr.Column():
            out = gr.Markdown(label="Feedback")
            with gr.Accordion("Pipeline details (agentic)", open=False):
                log = gr.Markdown()
    btn.click(generate, [essay, level, mode], [out, log])

if __name__ == "__main__":
    demo.launch()
