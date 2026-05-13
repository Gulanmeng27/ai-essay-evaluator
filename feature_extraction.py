import re
import pandas as pd
import numpy as np
from lexical_diversity import lex_div as ld


def extract_features(essay_text):
    words = essay_text.split()
    num_words = len(words)

    sentences = re.split(r'[.!?]+', essay_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    num_sent = len(sentences)

    paragraphs = [p.strip() for p in essay_text.split('\n') if p.strip()]
    num_para = len(paragraphs) if len(paragraphs) > 0 else 1
    num_word_div_para = num_words / num_para if num_para > 0 else num_words

    types = set(w.lower() for w in words)
    ttr = len(types) / num_words * 100 if num_words > 0 else 0

    try:
        mtld = ld.mtld(essay_text)
    except Exception:
        mtld = 40.0

    return {
        "num_words": num_words,
        "num_sent": num_sent,
        "TTR": ttr,
        "MTLD": mtld,
        "num_word_div_para": num_word_div_para,
    }


def extract_features_batch(essays):
    return pd.DataFrame([extract_features(e) for e in essays])
