#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 17:06:32 2025

@author: fabiangi
"""
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

# Results and Empirics
NYT_Data_Base = pd.read_csv("/NYT_Dataset_w_Topics_v2.csv")

#%% Matches of search term and lda results
matches = NYT_Data_Base["lda_topic_name"] == NYT_Data_Base["search_term"]

# Anzahl Übereinstimmungen
count_matches = matches.sum()
total_rows = len(NYT_Data_Base)

print(f"Übereinstimmungen: {count_matches} von {total_rows} ({count_matches/total_rows:.2%})")

#%% Matches of search term and BERTopic results

matches = NYT_Data_Base["bert_topic_name"] == NYT_Data_Base["search_term"]

# Anzahl matches
count_matches = matches.sum()
total_rows = len(NYT_Data_Base)

print(f"Übereinstimmungen: {count_matches} von {total_rows} ({count_matches/total_rows:.2%})")

#%% Confusion Matrix

# Evaluate topic agreement for NYT_Topics


# 1) Load
PATH = "/NYT_Dataset_w_Topics_v2.csv"
df = pd.read_csv(PATH)

# 2) Normalize strings (trim, keep case as-is)
def norm(s):
    return "" if pd.isna(s) else str(s).strip()

df["true"] = df["search_term"].map(norm)

# Try to auto-detect the predicted label columns
bert_col = next((c for c in df.columns if "bert" in c.lower() and "name" in c.lower()), "bert_topic_name")
lda_col  = next((c for c in df.columns if  "lda" in c.lower() and "name" in c.lower()),  "lda_topic_name")

df["bert"] = df[bert_col].map(norm)
df["lda"]  = df[lda_col].map(norm)

# 3)
TARGET = ["Climate Change", "Migration Policy", "Economic Inequality", "Digital Surveillance"]

# 4) 
acc_bert_all = (df["bert"] == df["true"]).mean()
acc_lda_all  = (df["lda"]  == df["true"]).mean()
print(f"BERT Overall Accuracy: {acc_bert_all:.2%}")
print(f"LDA  Overall Accuracy: {acc_lda_all:.2%}")

# 5) 
mask_bert_4 = df["true"].isin(TARGET) & df["bert"].isin(TARGET)
mask_lda_4  = df["true"].isin(TARGET) & df["lda"].isin(TARGET)

acc_bert_4 = (df.loc[mask_bert_4, "bert"] == df.loc[mask_bert_4, "true"]).mean() if mask_bert_4.any() else np.nan
acc_lda_4  = (df.loc[mask_lda_4,  "lda"]  == df.loc[mask_lda_4,  "true"]).mean() if mask_lda_4.any() else np.nan
print(f"BERT Accuracy (4 target topics only): {acc_bert_4:.2%}")
print(f"LDA  Accuracy (4 target topics only): {acc_lda_4:.2%}")

# 6) Confusion matrices 
def confusion_subset(pred_col):
    sub = df[df["true"].isin(TARGET) & df[pred_col].isin(TARGET)].copy()
    labels = TARGET
    cm = pd.crosstab(sub["true"], sub[pred_col], dropna=False)\
           .reindex(index=labels, columns=labels, fill_value=0)
    return cm

cm_bert = confusion_subset("bert")
cm_lda  = confusion_subset("lda")

print("\nConfusion Matrix (BERT):")
print(cm_bert)
print("\nConfusion Matrix (LDA):")
print(cm_lda)

# 7) 
def per_class_recall(pred_col):
    g = (df.assign(correct=df[pred_col] == df["true"])
           .groupby("true")["correct"].mean()
           .reindex(TARGET))
    return g

print("\nPer-class recall (BERT):")
print(per_class_recall("bert").apply(lambda x: f"{x:.2%}"))
print("\nPer-class recall (LDA):")
print(per_class_recall("lda").apply(lambda x: f"{x:.2%}"))

#  classification report
sub_bert = df[df["true"].isin(TARGET) & df["bert"].isin(TARGET)]
sub_lda  = df[df["true"].isin(TARGET) & df["lda"].isin(TARGET)]
print("\nClassification report (BERT, 4 topics):")
print(classification_report(sub_bert["true"], sub_bert["bert"], labels=TARGET, digits=3))
print("\nClassification report (LDA, 4 topics):")
print(classification_report(sub_lda["true"], sub_lda["lda"], labels=TARGET, digits=3))

# 8)  checks for 0%-recall cases
print("\nPredicted label counts (BERT):")
print(df["bert"].value_counts(dropna=False))
print("\nPredicted label counts (LDA):")
print(df["lda"].value_counts(dropna=False))

# Did BERT ever predict 'Economic Inequality'?
print("\nBERT predicts 'Economic Inequality' at least once?:",
      (df["bert"] == "Economic Inequality").any())

if "bert_topic_id" in df.columns:
    maj = (df[df["bert_topic_id"] != -1]
             .groupby("bert_topic_id")["true"]
             .agg(lambda s: s.value_counts().idxmax()))
    print("\nMajority true label per BERT topic_id:")
    print(maj)

#%%  why is Economic Inequality not covered in BERTopic?
df[df["search_term"]=="Economic Inequality"]["bert"].value_counts() #they mainly go to Digital Surveillance

# Was steckt in den Outliers?
df[df.get("bert_topic_id", -1) == -1]["search_term"].value_counts()

#%% Tables and Output Results as Reports
#  4-class evaluation + HTML report for LDA, BERTopic, LLM

import os
import re
import webbrowser
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix

# load daa
PATH = "/NYT_Dataset_w_Topics_v2_with_LLM.csv"

TARGET = ["Climate Change", "Migration Policy", "Economic Inequality", "Digital Surveillance"]

def norm(s): 
    return "" if pd.isna(s) else str(s).strip()

def evaluate_to_dfs(df, pred_col, target=TARGET):
    """Return (metrics_df, cm_df, rows_used, total_rows) for strict 4-class eval."""
    if "search_term" not in df.columns:
        raise ValueError("Column 'search_term' not found in df.")
    if pred_col not in df.columns:
        raise ValueError(f"Column '{pred_col}' not found in df.")

    y_true_full = df["search_term"].map(norm)
    y_pred_full = df[pred_col].map(norm)

    mask = y_true_full.isin(target) & y_pred_full.isin(target)
    y_true = y_true_full[mask]
    y_pred = y_pred_full[mask]

    if len(y_true) == 0:
        raise ValueError(f"No rows where both gold & '{pred_col}' are in TARGET.")

    # Per-class metrics
    rows = []
    for cls in target:
        tp = ((y_true == cls) & (y_pred == cls)).sum()
        fp = ((y_true != cls) & (y_pred == cls)).sum()
        fn = ((y_true == cls) & (y_pred != cls)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = (2*prec*rec)/(prec+rec) if (prec+rec) > 0 else 0.0
        sup  = (y_true == cls).sum()
        rows.append({"Class": cls, "Precision": prec, "Recall": rec, "F1": f1, "Support": int(sup)})

    # Macro-F1 über die 4 Klassen
    f1_macro = f1_score(y_true, y_pred, labels=target, average="macro")

    # Accuracy
    acc = (y_true == y_pred).mean()

    # Macro avg + Accuracy als rows
    rows.append({"Class": "Macro avg", "Precision": None, "Recall": None, "F1": f1_macro, "Support": int(len(y_true))})
    rows.append({"Class": "Accuracy",  "Precision": None, "Recall": None, "F1": acc,       "Support": int(len(y_true))})

    metrics_df = pd.DataFrame(rows, columns=["Class", "Precision", "Recall", "F1", "Support"])
    for c in ["Precision", "Recall", "F1"]:
        metrics_df[c] = metrics_df[c].astype(float).round(3)

    cm = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=target),
        index=[f"true:{t}" for t in target],
        columns=[f"pred:{t}" for t in target]
    )

    return metrics_df, cm, int(len(y_true)), int(len(df))

def html_report(model_name, metrics_df, cm_df, rows_used, total_rows, out_dir):
    """Save minimal HTML report and open in browser."""
    css = """
    <style>
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; }
      h1 { font-size: 20px; margin-bottom: 8px; }
      .subtitle { color: #666; margin-bottom: 18px; }
      table { border-collapse: collapse; margin-bottom: 22px; }
      th, td { border: 1px solid #ddd; padding: 6px 10px; text-align: right; }
      th:first-child, td:first-child { text-align: left; }
      thead th { background: #f7f7f7; }
      .small { color: #555; font-size: 13px; }
    </style>
    """
    html = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Eval Report</title>",
        css, "</head><body>"
    ]
    html.append(f"<h1>{model_name} — Strict 4-class Evaluation</h1>")
    html.append(f"<div class='subtitle'>Rows used: {rows_used} / {total_rows}</div>")
    html.append("<h3>Metrics</h3>")
    html.append(metrics_df.to_html(index=False))
    html.append("<h3>Confusion Matrix (rows=true, cols=pred)</h3>")
    html.append(cm_df.to_html())
    html.append("<p class='small'>Targets: " + ", ".join(TARGET) + "</p>")
    html.append("</body></html>")
    html_str = "\n".join(html)

    fname = re.sub(r'[^A-Za-z0-9_]+', '_', model_name.strip()) + "_eval.html"
    out_path = os.path.join(out_dir, fname)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_str)
    try:
        webbrowser.open("file://" + out_path)
    except Exception:
        pass
    return out_path

# reload
df = pd.read_csv(PATH)
out_dir = os.path.dirname(PATH)

# model define
models = [
    ("LDA", "lda_topic_name"),
    ("BERTopic", "bert_topic_name"),
    ("LLM (Llama)", "llm_topic_name"),
]

# Reports (Results)
for name, col in models:
    if col not in df.columns:
        print(f"\n[{name}] column '{col}' not found. Skipping.")
        continue
    metrics, cm, used, total = evaluate_to_dfs(df, col, TARGET)
    # Konsolen-Output
    print(f"\n=== {name} ===")
    print(f"Rows used (strict 4-class): {used} / {total}")
    print(metrics.to_string(index=False))
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(cm)
    # HTML-Report


    #%% Check cases where all models failed to predict the right label

import pandas as pd

# Falls df noch nicht geladen ist:
# df = pd.read_csv("NYT_Data_Final.csv")

mask = (
    (df["lda_topic_name"]  != df["search_term"]) &
    (df["bert_topic_name"] != df["search_term"]) &
    (df["llm_topic_name"]  != df["search_term"])
)

tri_df = df.loc[mask, [
    "search_term",
    "header",
    "plain_text",
    "lda_topic_name",
    "bert_topic_name",
    "llm_topic_name"
]]

# Optional speichern
# tri_df.to_csv("tri_misclassified_subset.csv", index=False)

print(tri_df.head())

# Verteilung der wahren Topics
distribution = tri_df["search_term"].value_counts()

print(distribution)

    report_path = html_report(name, metrics, cm, used, total, out_dir)
    print(f"[{name}] HTML report: {report_path}")


