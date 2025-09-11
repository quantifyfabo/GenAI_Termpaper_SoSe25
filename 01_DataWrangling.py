
"""
Created on Fri Aug 29 09:42:53 2025

@author: fabiangi
"""

# Packages
import os
import glob
import pandas as pd
import pypandoc
from pathlib import Path
import re
from striprtf.striprtf import rtf_to_text

#%% Loading Data into Dataframe

root = Path("/Paper_GenAI_SoSe25/Data/Data_250_Raw")

folders = [
    "ClimateChange_NYT_250",
    "DigitalSurveillance_NYT_250",
    "EconomicInequality_NYT_250",
    "MigrationPolicy_NYT_250",
]

def to_search_term(folder_name: str) -> str:
    """Derive search term from folder name before '_NYT_' and add spaces between capitals."""
    prefix = folder_name.split("_NYT_")[0]  
    return re.sub(r'(?<!^)([A-Z])', r' \1', prefix).strip()  

rows = []
for name in folders:
    topic_dir = root / name
    term = to_search_term(name)
    files = list(topic_dir.rglob("*.rtf")) + list(topic_dir.rglob("*.RTF"))
    for f in files:
        raw = f.read_text(errors="ignore")
        text = rtf_to_text(raw).strip()
        rows.append({"text": text, "search_term": term})

df = pd.DataFrame(rows, columns=["text", "search_term"])

print("Done. Rows:", len(df))
print(df["search_term"].value_counts())



# Extract header from the existing 'text' column
def extract_header(fulltext: str) -> str:
    if not isinstance(fulltext, str):
        return ""
    parts = fulltext.split("The New York Times", 1)
    return parts[0].strip() if len(parts) == 2 else ""

df["header"] = df["text"].apply(extract_header)




#%% Data Processing and Wrangling

def clean_middle(text: str) -> str:
    t = text.replace("\r\n", "\n").replace("\r", "\n")

    m_body_line = re.search(r'(?mi)^\s*Body\s*$', t)
    if m_body_line:
        start_idx = m_body_line.end()
    else:
        m_body_any = re.search(r'\bBody\b', t, flags=re.IGNORECASE)
        start_idx = m_body_any.end() if m_body_any else 0

    end_markers = []
    m_graphic = re.search(r'(?mi)^\s*Graphic\s*$', t)
    if m_graphic: end_markers.append(m_graphic.start())
    m_photos = re.search(r'(?mi)^\s*PHOTOS?:', t)
    if m_photos: end_markers.append(m_photos.start())
    m_loaddate = re.search(r'(?mi)^\s*Load-Date:', t)
    if m_loaddate: end_markers.append(m_loaddate.start())
    end_idx = min(end_markers) if end_markers else len(t)

    return t[start_idx:end_idx].strip()

# Apply cleaning function
df["text_clean"] = df["text"].apply(clean_middle)

# Preview and basic checks
print("Rows:", len(df))
print("Empty after cleaning:", (df["text_clean"].str.len() == 0).sum())
print(df["text_clean"].str.slice(0, 200).head(3))

#%% remove dates and links 
def remove_dates_and_links(text: str) -> str:
    text = re.sub(r'\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b', '', text)
    text = re.sub(r'\b\d{4}[./-]\d{1,2}[./-]\d{1,2}\b', '', text)
    months = ("January|February|March|April|May|June|July|August|September|October|November|December|"
              "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec")
    text = re.sub(r'\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d{1,2}(st|nd|rd|th)?\s+(?:' + months + r')(\s+\d{4})?\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:' + months + r')\s+\d{1,2}(st|nd|rd|th)?(,\s*\d{4})?\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(19|20)\d{2}\b', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

df["text_clean"] = df["text_clean"].apply(remove_dates_and_links)

# Preview cleaned results
print(df["text_clean"].str.slice(0, 200).head(3))


#%% Make text ready for tokenization. Lower, Punctuation, Underscores, Symbols, Double Whitespaces

def to_plain_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'[_]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Create plain_text column for tokenization
df["plain_text"] = df["text_clean"].apply(to_plain_text)


# create plain_text_hard
def remove_search_terms_words(text: str, term: str) -> str:
    words = term.lower().split()
    for w in words:
        text = re.sub(r'\b' + re.escape(w) + r'\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply function: create plain_text_hard
df["plain_text_hard"] = df.apply(
    lambda row: remove_search_terms_words(row["plain_text"], row["search_term"]),
    axis=1
)

# Preview
print(df[["search_term", "plain_text", "plain_text_hard"]].head(5))


#%% create check to see how much words got lost by each text reduction step

cols_to_check = ["text", "text_clean", "plain_text", "plain_text_hard"]
def word_count(s: str) -> int:
    if not isinstance(s, str):
        return 0
    return len(re.findall(r"\b\w+\b", s))

# Compute average word count for each column
avg_counts = {}
for col in cols_to_check:
    if col in df.columns:
        avg_counts[col] = df[col].apply(word_count).mean()

# Show as Series for easy comparison
avg_counts_series = pd.Series(avg_counts, name="avg_word_count")
print(avg_counts_series)



