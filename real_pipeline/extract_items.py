"""
Step 1 of audit (③): decompose free-form feedback into discrete flagged-error ITEMS.
Scope: temp=0.1, 5 essays x {RAG off, RAG on} = 10 feedback texts.
Each item = one concrete error the AI flagged: {dimension, quote, issue, correction}.
Output feeds the HTML audit tool for 卢晓琳.
"""
import os, json
from openai import OpenAI

BASE = "results"
for line in open(".env"):
    if line.startswith("DEEPSEEK_API_KEY"):
        os.environ["DEEPSEEK_API_KEY"] = line.split("=", 1)[1].strip()
client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")

EXTRACT_SYS = """You are a precise information extractor. You are given AI-generated feedback on a student's English essay. Extract EVERY concrete error or problem the feedback explicitly flags. For each flagged error output an object with:
- "dimension": one of "Grammar", "Vocabulary", "Organisation", "Content", "Other" (which heading it falls under)
- "quote": the exact span of the STUDENT'S essay the feedback quotes as problematic (verbatim; empty string if the feedback gives no specific quote)
- "issue": a one-sentence description of what the feedback says is wrong
- "correction": the corrected/improved version the feedback suggests (empty string if none)

Only include concrete, specific flagged errors. Do NOT include praise, general encouragement, or the closing tips. Respond with ONLY a JSON array of these objects, nothing else."""

results = json.load(open(os.path.join(BASE, "rag_ablation", "rag_ablation_results.json")))
essays = {e["essay_id"]: e for e in json.load(open(os.path.join(BASE, "part2_essays.json")))}

targets = [r for r in results if r["temp"] == 0.1]
out_items = []
item_id = 0
for r in targets:
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "system", "content": EXTRACT_SYS},
                  {"role": "user", "content": r["feedback"]}],
        temperature=0.0, max_tokens=2000,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content.strip()
    try:
        parsed = json.loads(raw)
        items = parsed if isinstance(parsed, list) else parsed.get("items") or parsed.get("errors") or next((v for v in parsed.values() if isinstance(v, list)), [])
    except Exception as e:
        items = []
        print(f"  parse fail {r['essay_id']} {r['arm']}: {e}")
    for it in items:
        item_id += 1
        out_items.append({
            "item_id": item_id,
            "essay_id": r["essay_id"],
            "proficiency": r["proficiency"],
            "arm": r["arm"],            # RAGoff / RAGon
            "dimension": it.get("dimension", "Other"),
            "quote": it.get("quote", ""),
            "issue": it.get("issue", ""),
            "correction": it.get("correction", ""),
        })
    print(f"  {r['essay_id']:12s} {r['arm']:7s}: {len(items)} items")

bundle = {
    "essays": [{"essay_id": e["essay_id"], "proficiency": e["proficiency"], "text": e["text"]} for e in essays.values()],
    "items": out_items,
}
outp = os.path.join(BASE, "audit", "audit_items.json")
os.makedirs(os.path.dirname(outp), exist_ok=True)
json.dump(bundle, open(outp, "w"), ensure_ascii=False, indent=2)
print(f"\nTotal items: {len(out_items)}  ({sum(1 for i in out_items if i['arm']=='RAGoff')} off / {sum(1 for i in out_items if i['arm']=='RAGon')} on)")
print("Saved:", outp)
