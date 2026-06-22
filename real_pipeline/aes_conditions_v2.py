"""
AES Part 1 — 4-condition LLM scoring, matching the PROPOSAL design (Section 3.2).
Prompt A = Direct scoring (rubric embedded). Prompt B = Score by dimension then aggregate.
Temperatures 0.1 and 0.7 (DeepSeek min ~0.01; 0.1 approximates deterministic).
Conditions: A-0(0.1), A-7(0.7), B-0(0.1), B-7(0.7).
Same 100-essay dev split as the notebook: train.sample(n=300, seed=42).iloc[200:].
"""
import os, re, csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd, numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from openai import OpenAI

HERE = os.path.dirname(os.path.abspath(__file__))
for line in open(".env"):
    if line.strip().startswith("DEEPSEEK_API_KEY"):
        os.environ["DEEPSEEK_API_KEY"] = line.split("=", 1)[1].strip()
client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
MODEL = "deepseek-chat"

# ---------- Prompt A: direct holistic scoring with rubric embedded ----------
PROMPT_A = """You are an expert English language proficiency assessor. Your task is to evaluate an English Language Learner (ELL) essay holistically on a scale from 1.0 to 5.0, using 0.5 increments.

A holistic score reflects the overall effectiveness of the writing, considering vocabulary, grammar, sentence structure, organization, and coherence together, not as a simple average of separate traits.

Use the following detailed rubric to determine the score:

1.0 - Minimal
- Vocabulary: Extremely limited; may rely on memorized phrases or single words.
- Grammar: Pervasive, basic errors that severely distort meaning.
- Sentences: Very short, fragmented, or repetitive; no sentence variety.
- Organization/Cohesion: No clear organization; ideas are disconnected.
- Overall: The text is extremely difficult to understand.

2.0 - Emerging
- Vocabulary: Basic, high-frequency words; little variety or precision.
- Grammar: Frequent errors in simple structures (e.g., subject-verb agreement, tense); meaning is often unclear.
- Sentences: Mostly simple sentences; attempts at compound sentences usually contain errors.
- Organization/Cohesion: Limited logical order; minimal use of basic linking words (and, but, so) often misused.
- Overall: The text conveys only a basic idea with significant effort from the reader.

3.0 - Developing
- Vocabulary: Adequate for everyday topics; some attempts at less common words may be inaccurate.
- Grammar: Errors are frequent but generally do not obscure meaning; some control of basic structures.
- Sentences: A mix of simple and compound sentences; occasional attempts at complex sentences, though they may contain errors.
- Organization/Cohesion: A basic structure is present; uses some common cohesive devices (e.g., first, also, because) but may be mechanical or inconsistent.
- Overall: The text is generally understandable despite noticeable errors and limited sophistication.

4.0 - Proficient
- Vocabulary: A good range of vocabulary, including some precise or idiomatic expressions.
- Grammar: Good control; errors are few and do not impede communication. Some minor slips may occur.
- Sentences: Uses a variety of sentence structures, including effective complex sentences.
- Organization/Cohesion: Clear, logical progression of ideas; uses a range of cohesive devices appropriately.
- Overall: The text is clear, well-organized, and reads smoothly with only occasional errors.

5.0 - Advanced
- Vocabulary: Rich, precise, and natural vocabulary; effective use of idiomatic and nuanced language.
- Grammar: High degree of grammatical accuracy; errors are very rare and insignificant.
- Sentences: A wide range of sophisticated sentence structures used flexibly and correctly.
- Organization/Cohesion: Excellent overall structure; cohesion is managed skillfully and unobtrusively.
- Overall: The text is fully successful in conveying its message, demonstrating advanced writing competence.

Scoring notes:
- Use 0.5 increments (e.g., 3.5) when an essay meets most criteria of the higher level but is still notably closer to the lower level overall.
- Ignore the essay's content or opinions; judge only the quality of the written English.

Respond with ONLY a single number (e.g., 3.0 or 3.5). Do not include any other text, explanation, or commentary."""

# ---------- Prompt B: score by dimension, then aggregate to holistic ----------
PROMPT_B = """You are an expert English language proficiency assessor evaluating an English Language Learner (ELL) essay. Assess the essay on each of the following six analytic dimensions, using a 1.0-5.0 scale in 0.5 increments (1.0 = minimal proficiency, 5.0 = advanced proficiency):

- Cohesion: organization, logical flow, and use of cohesive devices.
- Syntax: grammatical sentence structure and sentence variety.
- Vocabulary: range, precision, and appropriateness of word choice.
- Phraseology: control of collocations, idioms, and multi-word expressions.
- Grammar: morphological and grammatical accuracy.
- Conventions: spelling, punctuation, and mechanics.

First assign a score to each of the six dimensions. Then determine a single holistic Overall score (1.0-5.0, 0.5 increments) that reflects the overall effectiveness of the writing.

Respond in EXACTLY this format, numbers only, no commentary or explanation:
Cohesion: <score>
Syntax: <score>
Vocabulary: <score>
Phraseology: <score>
Grammar: <score>
Conventions: <score>
Overall: <score>"""

CONDITIONS = [   # name, prompt, temp, max_tokens, mode
    ("A-0", PROMPT_A, 0.1, 10,  "single"),
    ("A-7", PROMPT_A, 0.7, 10,  "single"),
    ("B-0", PROMPT_B, 0.1, 80,  "dim"),
    ("B-7", PROMPT_B, 0.7, 80,  "dim"),
]

def parse_single(reply):
    nums = re.findall(r"\d+\.?\d*", reply)
    return max(1.0, min(5.0, float(nums[0]))) if nums else None

def parse_dim(reply):
    m = re.search(r"overall\s*[:=]\s*(\d+\.?\d*)", reply, re.IGNORECASE)
    if m:
        return max(1.0, min(5.0, float(m.group(1))))
    # fallback: mean of any dimension numbers found
    nums = [float(x) for x in re.findall(r"\d+\.?\d*", reply)]
    nums = [n for n in nums if 0 < n <= 5]
    return max(1.0, min(5.0, sum(nums)/len(nums))) if nums else None

def score(essay, prompt, temp, max_tokens, mode):
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": prompt},
                      {"role": "user", "content": f"Score this essay:\n\n{essay}"}],
            temperature=temp, max_tokens=max_tokens,
        )
        reply = resp.choices[0].message.content.strip()
        return (parse_single(reply) if mode == "single" else parse_dim(reply)), reply
    except Exception as e:
        return None, f"ERROR: {e}"

df = pd.read_csv(os.path.join(HERE, "ELLIPSE_train.csv"))
dev = df.sample(n=300, random_state=42).reset_index(drop=True).iloc[200:].reset_index(drop=True)
essays = dev["full_text"].astype(str).tolist()
human = np.array(dev["Overall"].tolist(), dtype=float)
hm, hs = human.mean(), human.std()
print(f"dev: {len(essays)} essays, human mean={hm:.3f} std={hs:.3f}\n")

out = [{"Overall": human[i]} for i in range(len(essays))]
summary = []
for name, prompt, temp, mt, mode in CONDITIONS:
    preds = [None]*len(essays)
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(score, essays[i], prompt, temp, mt, mode): i for i in range(len(essays))}
        for f in as_completed(futs):
            i = futs[f]; preds[i], _ = f.result()
    valid = [(human[i], preds[i]) for i in range(len(essays)) if preds[i] is not None]
    yh = np.array([v[0] for v in valid]); yp = np.array([v[1] for v in valid])
    r, _ = pearsonr(yh, yp); mse = mean_squared_error(yh, yp)
    p = np.array([x for x in preds if x is not None])
    mse_bias = mean_squared_error(yh, yp + (yh.mean() - yp.mean()))
    fails = preds.count(None)
    summary.append((name, prompt is PROMPT_A and "A:direct" or "B:by-dim", temp, r, mse, mse_bias, p.mean(), fails))
    for i in range(len(essays)): out[i][f"{name}_pred"] = preds[i]
    print(f"{name} (t={temp}): n={len(valid)} fail={fails}  r={r:.4f}  MSE={mse:.4f}  MSE(bias-corr)={mse_bias:.4f}  pred_mean={p.mean():.2f}")

with open(os.path.join(HERE, "aes_proposal_predictions.csv"), "w", newline="") as f:
    cols = ["Overall"] + [f"{c[0]}_pred" for c in CONDITIONS]
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for row in out: w.writerow({k: row.get(k) for k in cols})

print("\n=== SUMMARY (proposal design: 2 prompts x 2 temps) ===")
print(f"{'Cond':<6}{'Prompt':<12}{'Temp':<7}{'Pearson r':<12}{'MSE':<9}{'MSE(calib)':<12}{'pred_mean':<10}")
for name, ptype, temp, r, mse, mseb, pm, fails in summary:
    print(f"{name:<6}{ptype:<12}{temp:<7}{r:<12.4f}{mse:<9.4f}{mseb:<12.4f}{pm:<10.2f}")
print(f"\nLR baseline (from Part1): r=0.4438  MSE=0.3494   |   human mean={hm:.2f}")
print("Saved: aes_proposal_predictions.csv")
