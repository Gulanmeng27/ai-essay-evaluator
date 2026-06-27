"""
HIGH-RECALL variant of the agentic workflow (recall optimisation, reduces 漏报/FN).
Changes vs agentic_workflow.py (everything else identical, so the comparison is clean):
  1. Per-expert cap raised 5 -> 10, and prompt asks for ALL genuine problems (not "most significant only").
  2. NEW 5th expert: a MECHANICS surface-error sweep that exhaustively lists EVERY
     spelling / article / agreement / tense / preposition / punctuation slip.
  3. Cross-dimension dedupe by quoted span (avoids inflating recall with duplicates).
  4. Self-correction verifier KEPT unchanged (so precision/FP stays controlled).
Outputs -> results/agentic_highrecall/.  Compare n_kept vs the baseline results/agentic/.
"""
import os, re, json, glob
import numpy as np, faiss
from fastembed import TextEmbedding
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "results")
KB_DIR = os.environ.get("KB_DIR", os.path.join(BASE, "knowledge_base_bootstrap"))
OUT = os.environ.get("OUT_DIR", os.path.join(BASE, "agentic_highrecall")); os.makedirs(OUT, exist_ok=True)
for line in open(os.path.join(ROOT, ".env")):
    if line.startswith("DEEPSEEK_API_KEY"):
        os.environ["DEEPSEEK_API_KEY"] = line.split("=", 1)[1].strip()
client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
MODEL = "deepseek-chat"
TEMP = 0.1
CAP = 10  # was 5

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

# --- A. dimension experts (recall-oriented prompt + higher cap) ---
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
           f"Be EXHAUSTIVE: report ALL genuine problems in THIS dimension that are really present in the essay, "
           f"including less-obvious ones — do NOT limit yourself to only the most significant (up to {CAP}). "
           "For each: the exact problematic span quoted verbatim from the essay, a one-line issue description, "
           "and a corrected version. "
           'Respond ONLY as JSON: {"findings":[{"quote":"...","issue":"...","correction":"..."}]}. '
           "Only report problems that are actually present; if none, return an empty list.")
    raw = call(sys, f"Essay:\n\n{essay}", 3000, json_mode=True)
    return [dict(f, dimension=dim) for f in salvage_findings(raw)][:CAP]

# --- A2. NEW mechanics / surface-error exhaustive sweep (recall booster) ---
def surface_sweep(essay):
    sys = ("You are a meticulous proofreader. Scan the English Language Learner essay below and list EVERY "
           "surface-level mechanical error you can find — spelling, missing/incorrect articles (a/an/the), "
           "subject-verb agreement, verb tense, plural/singular, prepositions, capitalisation, and punctuation. "
           "Be exhaustive and do NOT skip minor ones or summarise; one entry per distinct error instance. "
           "For each: the exact span quoted verbatim, a one-line issue, and the correction. "
           'Respond ONLY as JSON: {"findings":[{"quote":"...","issue":"...","correction":"..."}]}. '
           "Only include errors actually present.")
    raw = call(sys, f"Essay:\n\n{essay}", 3000, json_mode=True)
    return [dict(f, dimension="Mechanics") for f in salvage_findings(raw)][:20]

# --- B. self-correction verifier (UNCHANGED from baseline) ---
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
            verd = json.loads(call(sys, payload, 1500, json_mode=True)).get("verdicts", [])
        except Exception:
            verd = [True] * len(survivors)
        verd = verd + [True] * (len(survivors) - len(verd))
        for f, v in zip(survivors, verd):
            if v: kept.append(f)
            else: dropped.append(dict(f, drop_reason="LLM: not a real error"))
    return kept, dropped

def compose(essay, level, kept):
    sys = ("You are a supportive university English writing teacher giving feedback to an English Language Learner "
           f"(proficiency: {level}). Using ONLY the verified findings provided (do not add new errors), write "
           "encouraging feedback: start with one thing done well, then four headings (1) Grammar and Sentence "
           "Structure, (2) Vocabulary Use, (3) Organisation and Coherence, (4) Content and Ideas — each quoting the "
           "examples, explaining the issues, and giving the improved versions — then end with 2-3 actionable tips.")
    user = f"Essay:\n\n{essay}\n\nVerified findings:\n{json.dumps(kept, ensure_ascii=False, indent=2)}"
    return call(sys, user, 2500)

def run_essay(e):
    essay, level = e["text"], e["proficiency"]
    with ThreadPoolExecutor(max_workers=5) as ex:
        dim_jobs = list(EXPERTS.keys())
        dim_findings = sum(ex.map(lambda d: expert(d, essay), dim_jobs), [])
    mech = surface_sweep(essay)
    all_findings = dim_findings + mech
    # cross-dimension dedupe by normalized quoted span (keep first)
    seen, uniq = set(), []
    for f in all_findings:
        key = re.sub(r'\s+', ' ', f.get("quote", "").lower()).strip()[:60]
        if key and key not in seen:
            seen.add(key); uniq.append(f)
    kept, dropped = verify(essay, uniq)
    return {"essay_id": e["essay_id"], "proficiency": level, "arm": "agentic_highrecall",
            "n_findings_initial": len(uniq), "n_kept": len(kept), "n_dropped": len(dropped),
            "kept": kept, "dropped": dropped, "feedback": compose(essay, level, kept),
            "n_words": 0}

essays = json.load(open(os.path.join(BASE, "part2_essays.json")))
results = []
for e in essays:
    r = run_essay(e); r["n_words"] = len(r["feedback"].split()); results.append(r)
    print(f"  {r['essay_id']}: initial {r['n_findings_initial']} -> kept {r['n_kept']}, "
          f"dropped {r['n_dropped']}  ({r['n_words']} words)")

json.dump(results, open(os.path.join(OUT, "agentic_highrecall_results.json"), "w"), ensure_ascii=False, indent=2)
tot_init = sum(r["n_findings_initial"] for r in results)
tot_kept = sum(r["n_kept"] for r in results)
tot_drop = sum(r["n_dropped"] for r in results)
print(f"\nTOTAL initial {tot_init} | kept {tot_kept} | dropped {tot_drop} ({tot_drop/max(tot_init,1):.1%})")
print(f"avg kept/essay = {tot_kept/len(results):.1f}")
