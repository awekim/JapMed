"""Export full selected article texts for human review of frame disagreements.

Run on the Windows machine holding E:/Data_for_Practice/JapMedia.
Uses paired_frame_comparison.csv and frame_review_inputs.json already produced
by Topic_Frame_Coding_With_Articles.ipynb. Does not send data anywhere.
"""
from pathlib import Path
import json
import pandas as pd

BASE_DIR = Path(r"E:/Data_for_Practice/JapMedia")
VALIDATION = BASE_DIR / "validation"
ARTICLE_DIR = BASE_DIR / "data" / "embed_figure"
COMPARISON = VALIDATION / "paired_frame_comparison.csv"
INPUTS = VALIDATION / "frame_review_inputs.json"
OUTPUT = VALIDATION / "topic_audit_full_articles.csv"

if not COMPARISON.is_file() or not INPUTS.is_file():
    raise FileNotFoundError("Run the paired notebook first; comparison CSV and inputs JSON are required.")

comp = pd.read_csv(COMPARISON, encoding="utf-8-sig")
items = json.loads(INPUTS.read_text(encoding="utf-8"))
for col in ["source_file", "topic", "old_frame", "keyword_frame", "article_frame", "paired_difference"]:
    if col not in comp:
        raise ValueError(f"Comparison file is missing {col}")

# Include paired differences and original-versus-article differences, including
# the two cases which were not flagged by paired_difference alone.
flagged = comp.loc[
    comp["paired_difference"].fillna(False).astype(bool)
    | comp["old_frame"].ne(comp["article_frame"])
].copy()
keys = {(str(r.source_file), int(r.topic)) for r in flagged.itertuples()}
selected = [item for item in items if (item["source_file"], int(item["topic"])) in keys]
if len(selected) != len(keys):
    missing = keys - {(i["source_file"], int(i["topic"])) for i in selected}
    raise ValueError(f"Missing topic selection records: {missing}")

output = []
for source_file in sorted({i["source_file"] for i in selected}):
    path = ARTICLE_DIR / source_file
    if not path.is_file():
        raise FileNotFoundError(path)
    subset = [i for i in selected if i["source_file"] == source_file]
    columns = pd.read_csv(path, encoding="utf-8-sig", nrows=0).columns.tolist()
    bodies = {i["body_column"] for i in subset}
    if not bodies.issubset(columns):
        raise ValueError(f"Missing article body column in {path.name}: {bodies - set(columns)}")
    optional = [c for c in ["제목", "제목_en", "title", "Title", "날짜", "date", "Date", "URL", "url"] if c in columns]
    chosen = list({"topic", "topic_label"}.intersection(columns) | bodies | set(optional))
    df = pd.read_csv(path, encoding="utf-8-sig", usecols=chosen, low_memory=False)
    print(f"Reading {path.name}: {len(df)} rows; {len(subset)} flagged topics")
    for item in subset:
        frame_row = flagged.loc[(flagged.source_file == source_file) & (flagged.topic == int(item["topic"]))].iloc[0]
        for article in item["articles"]:
            idx = int(article["original_row"])
            if idx >= len(df):
                raise IndexError(f"{path.name}: selected row {idx} is beyond CSV length {len(df)}")
            row = df.iloc[idx]
            if "topic" in df and int(row.topic) != int(item["topic"]):
                raise ValueError(f"{path.name}, row {idx}: topic mismatch; check whether the article CSV changed")
            output.append({
                "source_file":source_file,
                "newspaper":item["newspaper"],
                "year":item["year"],
                "topic":item["topic"],
                "topic_keywords":item["topic_keywords"],
                "article_original_row":idx,
                "selection":article["selection"],
                "article_body":row[item["body_column"]],
                **{c:row[c] for c in optional},
                "old_frame":frame_row.old_frame,
                "keyword_frame":frame_row.keyword_frame,
                "article_frame":frame_row.article_frame,
                "coder_id":"",
                "climate_relevance":"",
                "problem_definition":"",
                "causal_interpretation":"",
                "evaluation":"",
                "proposed_response":"",
                "article_frame_human":"",
                "evidence_note":"",
                "topic_decision_after_review":"",
            })

result = pd.DataFrame(output)
result.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
print(f"Saved {len(result)} full articles across {len(keys)} topics to {OUTPUT}")
print("Check license terms before sharing article texts outside your research team.")