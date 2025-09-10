
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
NYT_Data_Base = pd.read_csv('/Paper_GenAI_SoSe25/Data/NYT_Dataset_Base.csv')



#%% LDA with SK-Learn // Run LDA with n=4 topics // Assign most likely topic to each case // Label Topic based on Top Words to related Search Term

# 1
texts = NYT_Data_Base["plain_text_hard"].fillna("")
vectorizer = CountVectorizer(stop_words="english", min_df=5, max_df=0.8)
X = vectorizer.fit_transform(texts)

# 2
lda = LatentDirichletAllocation(
    n_components=4,
    max_iter=20,
    learning_method="batch",
    random_state=42,
)
lda.fit(X)

# 3
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

# 4
doc_topic = lda.transform(X)                  
best_topic = doc_topic.argmax(axis=1) 
best_conf  = doc_topic.max(axis=1)

threshold = 0.35                                
unidentified_label = -1                      
assigned_topic = np.where(best_conf >= threshold, best_topic, unidentified_label)

# 5
NYT_Data_Base["lda_topic_id"]   = assigned_topic
NYT_Data_Base["lda_topic_conf"] = best_conf
NYT_Data_Base["lda_topic_words"] = [
    ", ".join(topic_topwords[t]) if t != unidentified_label else ""
    for t in assigned_topic
]

# 6
print("\nAssignment counts (including unidentified = -1):")
print(NYT_Data_Base["lda_topic_id"].value_counts(dropna=False).sort_index())

# 7
topic_map = {
    0: "Digital Surveillance",
    1: "Migration Policy",
    2: "Economic Inequality",
    3: "Climate Change",
    -1: "Unidentified",   # keep for thresholded cases
}

# Create a new column with names
NYT_Data_Base["lda_topic_name"] = NYT_Data_Base["lda_topic_id"].map(topic_map).fillna("Unidentified")

# optional new col
order = ["Climate Change", "Digital Surveillance", "Economic Inequality", "Migration Policy", "Unidentified"]
NYT_Data_Base["lda_topic_name"] = pd.Categorical(NYT_Data_Base["lda_topic_name"], categories=order, ordered=False)

# Quick check
print(NYT_Data_Base["lda_topic_name"].value_counts(dropna=False))

