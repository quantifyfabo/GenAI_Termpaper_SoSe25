# -*- coding: utf-8 -*-
"""
Robust BERTopic pipeline for ~1000 short English texts (NYT).
- No UMAP (more stable)
- Normalized embeddings (euclidean ~= cosine)
- HDBSCAN 'leaf' + small min_cluster_size (enough raw clusters)
- CountVectorizer with English stopwords + custom newsy terms + bigrams
- Outlier reassignment to reduce -1
- Plotly visualizations open in default browser (Spyder-safe)
- Extra: TF-IDF fallback to compute clean topwords if BERTopic representation misbehaves
"""

# ========= Environment & Imports =========
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # avoid HF tokenizers warning

import numpy as np
import pandas as pd

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sentence_transformers import SentenceTransformer
from hdbscan import HDBSCAN
from bertopic import BERTopic

# ========= 1) Load data =========
DATA_PATH = "/Users/fabiangi/Documents/Goethe Uni/Uni_Projects/CSS_Gen_AI/Paper_GenAI_SoSe25/Data/NYT_Dataset_Base.csv"
NYT_Data_Base = pd.read_csv(DATA_PATH)
texts = NYT_Data_Base["plain_text_hard"].fillna("").astype(str).tolist()

# If you want to strictly keep only your 4 categories, uncomment:
# allowed = {"Climate Change","Migration Policy","Economic Inequality","Digital Surveillance"}
# mask = NYT_Data_Base["search_term"].isin(allowed)
# texts = NYT_Data_Base.loc[mask, "plain_text_hard"].fillna("").astype(str).tolist()

# ========= 2) Stopwords & Vectorizer =========
# Custom NYT-ish stopwords seen in your outputs; merge with sklearn English stopwords
custom_stop = {
    "ms","mr","mrs","percent","state","states","city","cities",
    "official","officials","york","new","times","said","say","says"
}
# IMPORTANT: use a LIST, not a set
stopwords = list(text.ENGLISH_STOP_WORDS.union(custom_stop))

vectorizer_model = CountVectorizer(
    stop_words=stopwords,                   # ensure common stopwords are removed
    ngram_range=(1, 2),                     # bigrams like "climate change", "data privacy"
    min_df=2,                               # robust on ~1000 docs
    max_df=0.90,
    max_features=25000,
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b",  # alphabetic tokens, length>=3, hyphens ok
    lowercase=True
)

# ========= 3) Embeddings (normalized) =========
embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
embeddings = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=True)
# normalize_embeddings=True -> euclidean distance approximates cosine

# ========= 4) HDBSCAN (no UMAP) =========
hdbscan_model = HDBSCAN(
    min_cluster_size=6,              # small enough to avoid collapsing to 3 topics
    min_samples=1,                   # tolerant, yields more raw clusters
    metric="euclidean",              # ok since embeddings are normalized
    cluster_selection_method="leaf", # more fine-grained than 'eom'
    prediction_data=True
)

# ========= 5) (Optional) Raw model for diagnostics (no reduction) =========
topic_model_raw = BERTopic(
    embedding_model=embedder,
    vectorizer_model=vectorizer_model,
    umap_model=None,
    hdbscan_model=hdbscan_model,
    calculate_probabilities=False,
    verbose=False
)
_raw_topics, _ = topic_model_raw.fit_transform(texts, embeddings=embeddings)
raw_topic_ids = [t for t in topic_model_raw.get_topics().keys() if t != -1]
print(f"[Diagnostics] Raw (pre-reduction) topic count (excl. -1): {len(raw_topic_ids)}")

# ========= 6) Final model with reduction =========
# Empirically on your corpus: nr_topics=5 yields 4 real topics (plus optional -1)
topic_model = BERTopic(
    embedding_model=embedder,        # needed for reduce_outliers(strategy="embeddings")
    vectorizer_model=vectorizer_model,
    umap_model=None,
    hdbscan_model=hdbscan_model,
    nr_topics=5,                     # <-- important for your data
    calculate_probabilities=True,
    verbose=False
)

topics, probs = topic_model.fit_transform(texts, embeddings=embeddings)

# ========= 7) Reassign outliers (-1) =========
topics = topic_model.reduce_outliers(texts, topics, strategy="embeddings")
topic_model.update_topics(texts, topics=topics)   # DO NOT pass embeddings here (v0.17.3)

# ========= 8) Attach results =========
NYT_Data_Base["bert_topic_id"]   = topics
NYT_Data_Base["bert_topic_conf"] = (np.max(probs, axis=1) if probs is not None else np.nan)

# ========= 9) Topwords (two ways) =========
# 9a) From BERTopic (expected to be clean if vectorizer_model is used)
def show_topwords_from_bertopic(model, topn=15, skip_outlier=False):
    topic_ids = sorted(model.get_topics().keys())
    for tid in topic_ids:
        if skip_outlier and tid == -1:
            continue
        words = [w for w, _ in model.get_topic(tid)[:topn]]
        print(f"Topic {tid}: {', '.join(words)}")

print("\n=== Top words per topic (BERTopic) ===")
show_topwords_from_bertopic(topic_model, topn=15, skip_outlier=False)

# 9b) Fallback: compute clean topwords via TF-IDF per topic (guaranteed no 'the/of/and…')
def compute_topwords_tfidf(texts_list, topic_ids, topn=15):
    """Compute top words per topic via TF-IDF on aggregated per-topic texts."""
    # Build per-topic corpora
    from collections import defaultdict
    agg = defaultdict(list)
    for txt, tid in zip(texts_list, topic_ids):
        agg[tid].append(txt)
    topic_docs = {tid: " ".join(docs) for tid, docs in agg.items()}

    # TF-IDF with SAME stopwording/bigrams/token rules as the vectorizer_model
    tfidf = TfidfVectorizer(
        stop_words=stopwords,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.90,
        max_features=25000,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b",
        lowercase=True
    )
    # Fit on per-topic documents
    tids_sorted = sorted(topic_docs.keys())
    docs_sorted = [topic_docs[tid] for tid in tids_sorted]
    X = tfidf.fit_transform(docs_sorted)
    feat = np.array(tfidf.get_feature_names_out())

    topwords = {}
    for row_idx, tid in enumerate(tids_sorted):
        row = X[row_idx].toarray().ravel()
        if row.sum() == 0:
            topwords[tid] = []
            continue
        idx = np.argsort(row)[::-1][:topn]
        topwords[tid] = feat[idx].tolist()
    return topwords

# If BERTopic still shows stopwords (shouldn't), print the TF-IDF fallback:
print("\n=== Top words per topic (TF-IDF fallback) ===")
tfidf_top = compute_topwords_tfidf(texts, topics, topn=15)
for tid in sorted(tfidf_top.keys()):
    print(f"Topic {tid}: {', '.join(tfidf_top[tid])}")

# ========= 10) Counts =========
print("\n=== Assignment counts (incl. -1) ===")
print(pd.Series(topics).value_counts().sort_index())

# map topics to dataset
topic_map = {
    0: "Digital Surveillance",
    2: "Climate Change",
    1: "Migration Policy",
    3: "Economic Inequality",
    4: "Other",        # if exists after reduction
    -1: "Unidentified"
}
NYT_Data_Base["bert_topic_name"] = NYT_Data_Base["bert_topic_id"].map(topic_map).fillna("Unidentified")

# visualize

try:
    fig = topic_model.visualize_barchart(top_n_topics=5); fig.show("browser")
    fig2 = topic_model.visualize_topics();               fig2.show("browser")
except Exception as e:
    try:
        fig.write_html("bert_topics_barchart.html")
        fig2.write_html("bert_topics_scatter.html")
        print("Saved HTML: bert_topics_barchart.html, bert_topics_scatter.html")
    except Exception as ee:
        print("Visualization error:", ee)

#%% safe as csv

NYT_Data_Base.to_csv("NYT_Dataset_w_Topics_v2.csv", index=False, encoding="utf-8")
print("Saved:", "NYT_Dataset_w_Topics_v2.csv")


# Part 2 - Just for Comparison (these are not the final results of BERTopic)
#%% BERTopic Alternative with Plain Text instead of plain_text_hard // now contains search terms within the text corpora // helps for additional comparison

import re
import numpy as np
import spacy
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN

nlp = spacy.load("en_core_web_sm", disable=["parser","ner"])
def to_lemmas_nouns_adjs(text: str) -> str:
    if not isinstance(text, str):
        return ""
    doc = nlp(text)
    toks = []
    for t in doc:
        if t.is_stop or t.is_punct or t.is_space:
            continue
        if t.pos_ in {"NOUN","ADJ"}:
            lemma = t.lemma_.lower()
            if re.match(r"^[a-z][a-z\-]{1,}$", lemma):
                toks.append(lemma)
    return " ".join(toks)

NYT_Data_Base["plain_text_lemma"] = (
    NYT_Data_Base["plain_text"].fillna("").apply(to_lemmas_nouns_adjs)
)

docs = NYT_Data_Base["plain_text_lemma"].tolist()

vectorizer = CountVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=3,
    max_df=1.0
)

embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(docs, batch_size=64, show_progress_bar=True)

umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.05,
                  metric="cosine", random_state=42)
hdbscan_model = HDBSCAN(min_cluster_size=20, min_samples=5,
                        metric="euclidean", cluster_selection_method="eom",
                        prediction_data=True)

topic_model = BERTopic(
    vectorizer_model=vectorizer,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    calculate_probabilities=True,
    verbose=False
)

topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)
res = topic_model.reduce_topics(docs, nr_topics=4)
topic_model = res[0] if isinstance(res, tuple) else res

new_topics, new_probs = topic_model.transform(docs, embeddings=embeddings)

best_topic = np.array(new_topics)
best_conf  = new_probs.max(axis=1)

threshold = 0.35
assigned = np.where(best_conf >= threshold, best_topic, -1)

def get_top_words(model, topn=15):
    out = {}
    for tid in model.get_topic_info().Topic.unique():
        if tid == -1:
            continue
        out[int(tid)] = [w for w, _ in model.get_topic(int(tid))[:topn]]
    return out

topic_topwords = get_top_words(topic_model, topn=15)

NYT_Data_Base["bert_topic_id"]    = assigned
NYT_Data_Base["bert_topic_conf"]  = best_conf
NYT_Data_Base["bert_topic_words"] = [
    ", ".join(topic_topwords.get(int(t), [])) if t != -1 else ""
    for t in assigned
]

print("Assignment counts (incl. -1 by threshold):")
print(NYT_Data_Base["bert_topic_id"].value_counts(dropna=False).sort_index())
print("\nTop words per topic:")
for k, ws in topic_topwords.items():
    print(f"Topic {k}: {', '.join(ws)}")
