#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 12:00:14 2025

@author: fabiangi
"""

# Packages
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

#%% Load Data
NYT_Data_Base = pd.read_csv('/Users/fabiangi/Documents/Goethe Uni/Uni_Projects/CSS_Gen_AI/Paper_GenAI_SoSe25/Data/NYT_Dataset_Base.csv')



#%% LDA with SK-Learn // Run LDA with n=4 topics // Assign most likely topic to each case // Label Topic based on Top Words to related Search Term

# --- 1) Vectorize (bag-of-words, English stopwords) ---
texts = NYT_Data_Base["plain_text_hard"].fillna("")
vectorizer = CountVectorizer(stop_words="english", min_df=5, max_df=0.8)
X = vectorizer.fit_transform(texts)

# --- 2) Fit LDA (5 topics: 4 known + 1 unidentified) ---
lda = LatentDirichletAllocation(
    n_components=4,
    max_iter=20,
    learning_method="batch",
    random_state=42,
)
lda.fit(X)

# --- 3) Inspect top words per topic ---
def top_words_per_topic(model, feature_names, topn=15):
    tops = {}
    for k, comp in enumerate(model.components_):
        idx = np.argsort(comp)[::-1][:topn]
        tops[k] = [feature_names[i] for i in idx]
    return tops

feature_names = vectorizer.get_feature_names_out()
topic_topwords = top_words_per_topic(lda, feature_names, topn=15)

print("Top words per topic:")
for k, words in topic_topwords.items():
    print(f"Topic {k}:", ", ".join(words))

# --- 4) Document-topic assignments (with optional 'unidentified' threshold) ---
doc_topic = lda.transform(X)                       # shape: (n_docs, 5)
best_topic = doc_topic.argmax(axis=1)              # argmax topic id
best_conf  = doc_topic.max(axis=1)                 # max probability

threshold = 0.35                                   # tune as needed
unidentified_label = -1                         # Set treshhold so that only cases with coef >35 are used otherwise unidentified
assigned_topic = np.where(best_conf >= threshold, best_topic, unidentified_label)

# --- 5) Attach results to DataFrame ---
NYT_Data_Base["lda_topic_id"]   = assigned_topic
NYT_Data_Base["lda_topic_conf"] = best_conf
NYT_Data_Base["lda_topic_words"] = [
    ", ".join(topic_topwords[t]) if t != unidentified_label else ""
    for t in assigned_topic
]

# --- 6) Quick summary ---
print("\nAssignment counts (including unidentified = -1):")
print(NYT_Data_Base["lda_topic_id"].value_counts(dropna=False).sort_index())

# Maps Names to Topics 
# Map LDA topic IDs to human-readable labels
# Adjust the mapping if your topic IDs change across runs.
topic_map = {
    0: "Digital Surveillance",
    1: "Migration Policy",
    2: "Economic Inequality",
    3: "Climate Change",
    -1: "Unidentified",   # keep for thresholded cases
}

# Create a new column with names
NYT_Data_Base["lda_topic_name"] = NYT_Data_Base["lda_topic_id"].map(topic_map).fillna("Unidentified")

# Optional: make it a categorical column with a fixed order
order = ["Climate Change", "Digital Surveillance", "Economic Inequality", "Migration Policy", "Unidentified"]
NYT_Data_Base["lda_topic_name"] = pd.Categorical(NYT_Data_Base["lda_topic_name"], categories=order, ordered=False)

# Quick check
print(NYT_Data_Base["lda_topic_name"].value_counts(dropna=False))


#%% BERT Topic

# BERTopic on plain_text_hard (exactly 4 topics; consistent stopwords with LDA)
# Requirements (run once): pip install bertopic sentence-transformers umap-learn hdbscan

import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN

# --- 1) Input documents (cleaned; no search-term leakage) ---
docs = NYT_Data_Base["plain_text_hard"].fillna("").tolist()

# --- 2) Vectorizer (same stopwords as LDA) ---
vectorizer = CountVectorizer(
    stop_words="english",   # same as your LDA
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

# --- 3) Embeddings (compact, good quality) ---
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(docs, batch_size=64, show_progress_bar=True)

# --- 4) Configure BERTopic (reproducible UMAP; moderate cluster size) ---
umap_model = UMAP(
    n_neighbors=15, n_components=5, min_dist=0.0,
    metric="cosine", random_state=42
)
hdbscan_model = HDBSCAN(
    min_cluster_size=25, metric="euclidean",
    cluster_selection_method="eom", prediction_data=True
)

topic_model = BERTopic(
    vectorizer_model=vectorizer,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    calculate_probabilities=True,
    verbose=False
)

# --- 5) Fit initial topics ---
topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)

# --- 6) Reduce to exactly 4 topics (API differences handled) ---
res = topic_model.reduce_topics(docs, nr_topics=4)
if isinstance(res, tuple):
    topic_model, mapping = res
else:
    topic_model, mapping = res, None

# --- 7) Recompute assignments after reduction ---
new_topics, new_probs = topic_model.transform(docs, embeddings=embeddings)

# Most-likely topic per doc and confidence (keep BERTopic's own -1 outliers)
assigned_topic = np.array(new_topics)          # may contain -1 (outliers)
best_conf = new_probs.max(axis=1)

threshold = 0.35
assigned_topic = np.where(best_conf >= threshold, assigned_topic, -1)

# --- 8) Top words per topic (c-TF-IDF) ---
def get_top_words(model, topn=15):
    out = {}
    for tid in model.get_topic_info().Topic.unique():
        if tid == -1:
            continue
        out[int(tid)] = [w for w, _ in model.get_topic(tid)[:topn]]
    return out

topic_topwords = get_top_words(topic_model, topn=15)

# --- 9) Attach results to DataFrame ---
NYT_Data_Base["bert_topic_id"]    = assigned_topic
NYT_Data_Base["bert_topic_conf"]  = best_conf
NYT_Data_Base["bert_topic_words"] = [
    ", ".join(topic_topwords.get(int(t), [])) if t != -1 else ""
    for t in assigned_topic
]

# Optional: map IDs to human-readable names AFTER inspecting top words
# (Set these once per run; IDs can change between runs.)
topic_map = {
    # 0: "Digital Surveillance",
    # 1: "Migration Policy",
    # 2: "Economic Inequality",
    # 3: "Climate Change",
    -1: "Unidentified",
}
NYT_Data_Base["bert_topic_name"] = NYT_Data_Base["bert_topic_id"].map(topic_map).fillna("Unassigned")

# --- 10) Quick summary ---
print("Assignment counts (incl. -1 Outliers):")
print(NYT_Data_Base["bert_topic_id"].value_counts(dropna=False).sort_index())
print("\nTop words per topic:")
for k, ws in topic_topwords.items():
    print(f"Topic {k}: {', '.join(ws)}")


#%% BERT and HDBSCAN mit 4 topics
# BERTopic: clean version, force 4 topics, confidence threshold -> -1
# Requirements: pip install bertopic sentence-transformers umap-learn hdbscan spacy
# python -m spacy download en_core_web_sm
# BERTopic: clean version, force 4 topics, confidence threshold -> -1
# Requirements: pip install bertopic sentence-transformers umap-learn hdbscan spacy
# python -m spacy download en_core_web_sm

import re
import numpy as np
import spacy
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN

# -----------------------------
# Lemmatization with POS filter
# -----------------------------
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
    NYT_Data_Base["plain_text_hard"].fillna("").apply(to_lemmas_nouns_adjs)
)

# -----------------------------
# Documents
# -----------------------------
docs = NYT_Data_Base["plain_text_lemma"].tolist()

# -----------------------------
# Vectorizer
# -----------------------------
vectorizer = CountVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=3,
    max_df=1.0
)

# -----------------------------
# Embeddings
# -----------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(docs, batch_size=64, show_progress_bar=True)

# -----------------------------
# BERTopic config
# -----------------------------
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

# -----------------------------
# Fit, then reduce to 4 topics
# -----------------------------
topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)
res = topic_model.reduce_topics(docs, nr_topics=4)
topic_model = res[0] if isinstance(res, tuple) else res

# -----------------------------
# Recompute assignments
# -----------------------------
new_topics, new_probs = topic_model.transform(docs, embeddings=embeddings)

best_topic = np.array(new_topics)          # <--- richtige np.array
best_conf  = new_probs.max(axis=1)

threshold = 0.35
assigned = np.where(best_conf >= threshold, best_topic, -1)

# -----------------------------
# Top words
# -----------------------------
def get_top_words(model, topn=15):
    out = {}
    for tid in model.get_topic_info().Topic.unique():
        if tid == -1:
            continue
        out[int(tid)] = [w for w, _ in model.get_topic(int(tid))[:topn]]
    return out

topic_topwords = get_top_words(topic_model, topn=15)

# -----------------------------
# Attach results
# -----------------------------
NYT_Data_Base["bert_topic_id"]    = assigned
NYT_Data_Base["bert_topic_conf"]  = best_conf
NYT_Data_Base["bert_topic_words"] = [
    ", ".join(topic_topwords.get(int(t), [])) if t != -1 else ""
    for t in assigned
]

# -----------------------------
# Quick summary
# -----------------------------
print("Assignment counts (incl. -1 by threshold):")
print(NYT_Data_Base["bert_topic_id"].value_counts(dropna=False).sort_index())
print("\nTop words per topic:")
for k, ws in topic_topwords.items():
    print(f"Topic {k}: {', '.join(ws)}")
