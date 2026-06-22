"""
③ Audit metrics. Run AFTER 卢晓琳 (and ideally a 2nd coder) export their audit JSON.
Usage:  python3 compute_audit_metrics.py audit_coder1.json [audit_coder2.json]
Outputs: FP rate, FN count, dimension-accuracy, mean pedagogical utility, tone distribution
         — broken down by RAG arm (off vs on) => fills Ch5.3 table + RQ2 evidence.
         With 2 coders: Cohen's kappa on the TP/FP verdict (inter-rater reliability).
"""
import sys, json
from collections import defaultdict

def load(path):
    return json.load(open(path, encoding="utf-8"))

def metrics_for(coder):
    by_arm = defaultdict(lambda: {"TP":0,"FP":0,"unsure":0,"dim_ok":0,"dim_tot":0,"util":[]})
    for it in coder["items"]:
        a = it["arm"]; v = it.get("verdict","")
        if v in ("TP","FP","unsure"): by_arm[a][v]+=1
        if it.get("dim_ok") in ("yes","no"):
            by_arm[a]["dim_tot"]+=1
            if it["dim_ok"]=="yes": by_arm[a]["dim_ok"]+=1
        if it.get("utility"): by_arm[a]["util"].append(int(it["utility"]))
    fn = defaultdict(int); tone = defaultdict(lambda: defaultdict(int))
    for f in coder.get("feedbacks",[]):
        if f.get("missed"): fn[f["arm"]] += len([x for x in f["missed"].split("\n") if x.strip()])
        if f.get("tone"): tone[f["arm"]][f["tone"]]+=1
    return by_arm, fn, tone

def report(coder):
    by_arm, fn, tone = metrics_for(coder)
    print(f"\n=== Coder: {coder.get('coder','?')} ===")
    print(f"{'Arm':<8}{'flagged':<9}{'TP':<5}{'FP':<5}{'FP rate':<9}{'dim acc':<9}{'mean util':<11}{'FN(missed)':<11}{'tone'}")
    for arm in ("RAGoff","RAGon"):
        d=by_arm[arm]; flagged=d["TP"]+d["FP"]
        fpr = d["FP"]/flagged if flagged else 0
        dacc = d["dim_ok"]/d["dim_tot"] if d["dim_tot"] else 0
        mu = sum(d["util"])/len(d["util"]) if d["util"] else 0
        tdist=dict(tone[arm])
        print(f"{arm:<8}{flagged:<9}{d['TP']:<5}{d['FP']:<5}{fpr:<9.1%}{dacc:<9.1%}{mu:<11.2f}{fn[arm]:<11}{tdist}")

def kappa(c1, c2):
    # Cohen's kappa on TP/FP verdict over items both coders judged
    m1={i["item_id"]:i.get("verdict") for i in c1["items"]}
    m2={i["item_id"]:i.get("verdict") for i in c2["items"]}
    ids=[k for k in m1 if m1[k] in("TP","FP") and m2.get(k) in("TP","FP")]
    if not ids: print("\n[kappa] no overlapping TP/FP judgements"); return
    agree=sum(1 for k in ids if m1[k]==m2[k])/len(ids)
    # expected
    from collections import Counter
    n=len(ids); c1c=Counter(m1[k] for k in ids); c2c=Counter(m2[k] for k in ids)
    pe=sum((c1c[l]/n)*(c2c[l]/n) for l in("TP","FP"))
    k=(agree-pe)/(1-pe) if pe!=1 else 1.0
    print(f"\n[Inter-rater] n={n} overlapping items | observed agreement={agree:.1%} | Cohen's kappa={k:.3f}")

files=sys.argv[1:]
if not files:
    print("usage: python3 compute_audit_metrics.py coder1.json [coder2.json]"); sys.exit()
coders=[load(f) for f in files]
for c in coders: report(c)
if len(coders)>=2: kappa(coders[0],coders[1])
print("\nNote: RAG-on flagged more items; the question is whether the EXTRA flags are TP (more thorough) or FP (more hallucination). Compare FP rate across arms.")
