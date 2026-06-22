"""
RAG ablation pipeline (Part 2 AWE, RQ2).
Real retrieval: fastembed (BGE-small) + FAISS over the knowledge base.
For each of the 5 essays, generate feedback with RAG OFF and RAG ON, at temps 0.1 and 0.7.
Same base prompt both arms; the ONLY difference is the injected retrieved knowledge => clean ablation.
KB is the bootstrap placeholder; swap in 黄小倩's docs later and re-run (same script).
"""
import os, re, json, glob
import numpy as np
import faiss
from fastembed import TextEmbedding
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

BASE = "results"
KB_DIR = os.path.join(BASE, "knowledge_base_bootstrap")
OUT = os.path.join(BASE, "rag_ablation")
os.makedirs(OUT, exist_ok=True)

# --- key ---
for line in open(".env"):
    if line.startswith("DEEPSEEK_API_KEY"):
        os.environ["DEEPSEEK_API_KEY"] = line.split("=", 1)[1].strip()
client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
MODEL = "deepseek-chat"

# --- 1. load + chunk KB (one chunk per '### ' entry) ---
chunks = []
for fp in sorted(glob.glob(os.path.join(KB_DIR, "*.md"))):
    src = os.path.basename(fp)
    raw = open(fp, encoding="utf-8").read()
    for m in re.split(r'(?m)^### ', raw):
        m = m.strip()
        if not m or m.startswith("<!--"):
            continue
        m = re.sub(r'<!--.*?-->', '', m, flags=re.S).strip()
        if m:
            chunks.append({"source": src, "text": "### " + m})
print(f"KB chunks: {len(chunks)} from {KB_DIR}")

# --- 2. embed + FAISS index ---
embedder = TextEmbedding()  # BAAI/bge-small-en-v1.5
def embed(texts):
    vecs = np.array(list(embedder.embed(texts)), dtype="float32")
    faiss.normalize_L2(vecs)
    return vecs
kb_vecs = embed([c["text"] for c in chunks])
index = faiss.IndexFlatIP(kb_vecs.shape[1])
index.add(kb_vecs)
print(f"FAISS index built: dim={kb_vecs.shape[1]}, n={index.ntotal}")

def retrieve(query, k=6):
    qv = embed([query[:2000]])
    scores, idx = index.search(qv, k)
    return [(chunks[i], float(scores[0][j])) for j, i in enumerate(idx[0])]

# --- 3. prompts (identical base; RAG only prepends a reference block) ---
BASE_SYS = """You are a supportive university English writing teacher giving feedback to an English Language Learner. Adapt your language to the student's proficiency level. Structure your feedback as follows:
- Begin with one short paragraph noting something the student did well.
- Then give feedback under four headings: (1) Grammar and Sentence Structure, (2) Vocabulary Use, (3) Organisation and Coherence, (4) Content and Ideas. For each, quote a specific example from the essay, explain the issue clearly, and give a concrete improved version.
- End with 2-3 specific, actionable tips.
Be encouraging and constructive. Only comment on issues that are actually present in the essay."""

RAG_PREFIX = """Use the following reference knowledge (scoring rubric, common error patterns for Chinese learners of English, and academic writing norms) to ground your feedback. When you point out an error, base it on these references where relevant, and do not invent issues that are not supported by the essay.

=== REFERENCE KNOWLEDGE (retrieved) ===
{kb}
=== END REFERENCE KNOWLEDGE ===

"""

def feedback(essay, level, temp, use_rag):
    sys = BASE_SYS
    retrieved = []
    if use_rag:
        hits = retrieve(essay, k=6)
        retrieved = [{"source": h[0]["source"], "tag": h[0]["text"][:60], "score": round(h[1], 3)} for h in hits]
        kb_txt = "\n\n".join(h[0]["text"] for h in hits)
        sys = RAG_PREFIX.format(kb=kb_txt) + BASE_SYS
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": sys},
                  {"role": "user", "content": f"Here is the student essay (proficiency: {level}):\n\n{essay}"}],
        temperature=temp, max_tokens=1200,
    )
    return resp.choices[0].message.content.strip(), retrieved

# --- 4. run all combos ---
essays = json.load(open(os.path.join(BASE, "part2_essays.json")))
jobs = []
for e in essays:
    for temp in (0.1, 0.7):
        for use_rag in (False, True):
            jobs.append((e, temp, use_rag))

def run(job):
    e, temp, use_rag = job
    arm = "RAGon" if use_rag else "RAGoff"
    fb, retr = feedback(e["text"], e["proficiency"], temp, use_rag)
    return {"essay_id": e["essay_id"], "proficiency": e["proficiency"], "temp": temp,
            "arm": arm, "n_words": len(fb.split()), "retrieved": retr, "feedback": fb}

results = []
with ThreadPoolExecutor(max_workers=6) as ex:
    for r in ex.map(run, jobs):
        results.append(r)
        print(f"  {r['essay_id']} t={r['temp']} {r['arm']:7s} -> {r['n_words']} words"
              + (f", retrieved {len(r['retrieved'])} chunks" if r['arm']=='RAGon' else ""))

# --- 5. save: full json + readable side-by-side text ---
json.dump(results, open(os.path.join(OUT, "rag_ablation_results.json"), "w"), ensure_ascii=False, indent=2)
with open(os.path.join(OUT, "rag_ablation_readable.md"), "w", encoding="utf-8") as f:
    for e in essays:
        for temp in (0.1, 0.7):
            off = next(r for r in results if r["essay_id"]==e["essay_id"] and r["temp"]==temp and r["arm"]=="RAGoff")
            on  = next(r for r in results if r["essay_id"]==e["essay_id"] and r["temp"]==temp and r["arm"]=="RAGon")
            f.write(f"\n\n{'='*90}\n# {e['essay_id']} ({e['proficiency']}) — temperature {temp}\n{'='*90}\n")
            f.write(f"\n## Retrieved KB chunks (RAG on):\n")
            for h in on["retrieved"]:
                f.write(f"  - [{h['score']}] {h['tag']}  ({h['source']})\n")
            f.write(f"\n## --- RAG OFF ({off['n_words']} words) ---\n{off['feedback']}\n")
            f.write(f"\n## --- RAG ON ({on['n_words']} words) ---\n{on['feedback']}\n")

# quick descriptive summary
print("\n=== descriptive summary (avg feedback length, words) ===")
for arm in ("RAGoff", "RAGon"):
    ws = [r["n_words"] for r in results if r["arm"]==arm]
    print(f"  {arm}: mean={np.mean(ws):.0f} words over {len(ws)} outputs")
print(f"\nSaved: {OUT}/rag_ablation_results.json and rag_ablation_readable.md")
