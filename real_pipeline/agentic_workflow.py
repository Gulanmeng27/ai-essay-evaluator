"""
Real Agentic Workflow for essay feedback (Part 2 / RQ2 — agentic vs single-pass).
Pipeline per essay:
  A. Decompose -> 4 dimension EXPERTS (Grammar/Vocabulary/Organisation/Content),
     each retrieves its own KB evidence (RAG) and returns structured findings.
  B. SELF-CORRECTION: verify every finding against the essay (quote must exist +
     LLM judges if it is a real error); drop unsupported/hallucinated findings.
  C. COMPOSE: write the final friendly 4-dimension feedback from verified findings only.
Outputs the feedback + a log of how many findings the self-correction step removed.
"""
import os, re, json, glob
import numpy as np, faiss
from fastembed import TextEmbedding
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

BASE = "results"
KB_DIR = os.path.join(BASE, "knowledge_base_bootstrap")
OUT = os.path.join(BASE, "agentic"); os.makedirs(OUT, exist_ok=True)
for line in open(".env"):
    if line.startswith("DEEPSEEK_API_KEY"):
        os.environ["DEEPSEEK_API_KEY"] = line.split("=", 1)[1].strip()
client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
MODEL = "deepseek-chat"
TEMP = 0.1

# --- KB + FAISS (reuse bootstrap KB) ---
chunks = []
for fp in sorted(glob.glob(os.path.join(KB_DIR, "*.md"))):
    raw = open(fp, encoding="utf-8").read()
    for seg in re.split(r'(?m)^### ', raw):
        seg = re.sub(r'<!--.*?-->', '', seg, flags=re.S).strip()
        if seg: chunks.append("### " + seg)
embedder = TextEmbedding()
def embed(texts):
    v = np.array(list(embedder.embed(texts)), dtype="float32"); faiss.normalize_L2(v); return v
kb_vecs = embed(chunks)
index = faiss.IndexFlatIP(kb_vecs.shape[1]); index.add(kb_vecs)
def retrieve(query, k=4):
    s, idx = index.search(embed([query]), k)
    return [chunks[i] for i in idx[0]]

def call(sys, user, max_tokens=1500, json_mode=False):
    kw = {"response_format": {"type": "json_object"}} if json_mode else {}
    r = client.chat.completions.create(model=MODEL, temperature=TEMP, max_tokens=max_tokens,
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}], **kw)
    return r.choices[0].message.content.strip()

def salvage_findings(raw):
    """Robust parse: full JSON first; if truncated, recover complete {...} objects."""
    try:
        return json.loads(raw).get("findings", [])
    except Exception:
        out = []
        for m in re.findall(r'\{[^{}]*\}', raw):
            try:
                o = json.loads(m)
                if "issue" in o or "quote" in o: out.append(o)
            except Exception:
                pass
        return out

# --- A. dimension experts ---
EXPERTS = {
    "Grammar":      "grammar errors subject-verb agreement tense articles run-on sentences",
    "Vocabulary":   "vocabulary word choice precision Chinglish collocation repetition",
    "Organisation": "organisation coherence topic sentence transitions paragraph structure",
    "Content":      "content ideas argument development relevance support",
}
def expert(dim, essay):
    kb = "\n\n".join(retrieve(EXPERTS[dim] + " " + essay[:600], k=4))
    sys = (f"You are an expert assessing ONLY the '{dim}' dimension of an English Language Learner's essay. "
           f"Use the reference knowledge below to ground your judgement; do not invent issues.\n\n"
           f"=== REFERENCE ({dim}) ===\n{kb}\n=== END ===\n\n"
           "Report ONLY the most significant problems in THIS dimension (AT MOST 5; pick the ones that most "
           "affect the writing). For each: the exact problematic span quoted verbatim from the essay, a "
           "one-line issue description, and a corrected version. "
           'Respond ONLY as JSON: {"findings":[{"quote":"...","issue":"...","correction":"..."}]}. '
           "If there are no real problems, return an empty list.")
    raw = call(sys, f"Essay:\n\n{essay}", 2000, json_mode=True)
    return [dict(f, dimension=dim) for f in salvage_findings(raw)][:5]

# --- B. self-correction (verify each finding) ---
def verify(essay, findings):
    kept, dropped = [], []
    norm = lambda s: re.sub(r'\s+', ' ', s.lower()).strip()
    ne = norm(essay)
    survivors = []
    for f in findings:
        q = norm(f.get("quote", ""))
        if q and len(q) > 8 and q not in ne and q[:30] not in ne and q[-30:] not in ne:
            f["drop_reason"] = "quote not found in essay"; dropped.append(f)
        else:
            survivors.append(f)
    if survivors:
        sys = ("You are a fact-checker auditing flagged errors in a student essay. "
               "Judge each flagged item ON ITS OWN MERITS (ignore which category it was filed under). "
               "Mark it FALSE only if one of these holds: (a) the quoted span does NOT appear in the essay; "
               "(b) it is a purely stylistic preference or acceptable variation, not an actual mistake; or "
               "(c) the essay is not actually wrong there (the issue is mistaken). "
               "Otherwise, if it identifies a genuine grammar/word/spelling/structure error that is really "
               "present, mark it TRUE. Keep real errors even if similar to another item. "
               'Respond ONLY as JSON: {"verdicts":[true,false,...]} in the same order as the input.')
        payload = "Essay:\n\n" + essay + "\n\nFlagged errors:\n" + json.dumps(
            [{"quote": f.get("quote",""), "issue": f.get("issue","")} for f in survivors], ensure_ascii=False)
        try:
            verd = json.loads(call(sys, payload, 800, json_mode=True)).get("verdicts", [])
        except Exception:
            verd = [True] * len(survivors)
        verd = verd + [True] * (len(survivors) - len(verd))
        for f, v in zip(survivors, verd):
            if v: kept.append(f)
            else: dropped.append(dict(f, drop_reason="LLM: not a real error"))
    return kept, dropped

# --- C. compose ---
def compose(essay, level, kept):
    sys = ("You are a supportive university English writing teacher giving feedback to an English Language Learner "
           f"(proficiency: {level}). Using ONLY the verified findings provided (do not add new errors), write "
           "encouraging feedback: start with one thing done well, then four headings (1) Grammar and Sentence "
           "Structure, (2) Vocabulary Use, (3) Organisation and Coherence, (4) Content and Ideas — each quoting the "
           "example, explaining the issue, and giving the improved version — then end with 2-3 actionable tips.")
    user = f"Essay:\n\n{essay}\n\nVerified findings:\n{json.dumps(kept, ensure_ascii=False, indent=2)}"
    return call(sys, user, 1500)

def run_essay(e):
    essay, level = e["text"], e["proficiency"]
    with ThreadPoolExecutor(max_workers=4) as ex:
        all_findings = sum(ex.map(lambda d: expert(d, essay), EXPERTS.keys()), [])
    # dedupe by (dimension, normalized quote)
    seen, uniq = set(), []
    for f in all_findings:
        key = (f.get("dimension"), re.sub(r'\s+', ' ', f.get("quote", "").lower()).strip()[:60])
        if key not in seen:
            seen.add(key); uniq.append(f)
    kept, dropped = verify(essay, uniq)
    fb = compose(essay, level, kept)
    return {"essay_id": e["essay_id"], "proficiency": level, "arm": "agentic",
            "n_findings_initial": len(uniq), "n_kept": len(kept), "n_dropped": len(dropped),
            "kept": kept, "dropped": dropped, "feedback": fb, "n_words": len(fb.split())}

essays = json.load(open(os.path.join(BASE, "part2_essays.json")))
results = [run_essay(e) for e in essays]
for r in results:
    print(f"  {r['essay_id']}: initial {r['n_findings_initial']} -> kept {r['n_kept']}, "
          f"self-correction dropped {r['n_dropped']}  ({r['n_words']} words)")

json.dump(results, open(os.path.join(OUT, "agentic_results.json"), "w"), ensure_ascii=False, indent=2)
tot_init = sum(r["n_findings_initial"] for r in results); tot_drop = sum(r["n_dropped"] for r in results)
print(f"\nSelf-correction removed {tot_drop}/{tot_init} initially-flagged findings "
      f"({tot_drop/max(tot_init,1):.1%}) as unsupported/hallucinated.")
with open(os.path.join(OUT, "agentic_readable.md"), "w", encoding="utf-8") as f:
    for r in results:
        f.write(f"\n\n{'='*80}\n# {r['essay_id']} ({r['proficiency']}) — AGENTIC\n")
        f.write(f"initial findings {r['n_findings_initial']} | kept {r['n_kept']} | dropped {r['n_dropped']}\n{'='*80}\n")
        if r["dropped"]:
            f.write("\n## Dropped by self-correction:\n")
            for d in r["dropped"]:
                f.write(f"  - [{d['dimension']}] {d.get('issue','')[:80]} ({d.get('drop_reason','')})\n")
        f.write(f"\n## Final feedback:\n{r['feedback']}\n")
print("Saved:", OUT)
