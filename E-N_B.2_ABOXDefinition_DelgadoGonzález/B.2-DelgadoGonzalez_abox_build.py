#!/usr/bin/env python
# -------------------------------------------------------------------
#   b2_abox_build.py   –   create ont/research_abox.ttl from 13 CSV
# -------------------------------------------------------------------
import pandas as pd
from rdflib import Graph, Namespace, RDF, Literal, XSD
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent          
DATA = BASE / "data"
ONT  = BASE / "ont" / "research_abox.ttl"

RES  = Namespace("http://example.org/res/")
INST = Namespace("http://example.org/inst/")

g = Graph()
g.bind("res", RES)
g.bind("inst", INST)

# ---------- helpers --------------------------------------------------
def uri(cls, _id):         # inst:Class_<id>
    return INST[f"{cls}_{int(_id)}"]

def split_ids(val):
    return [int(x) for x in str(val).split("|") if x not in ("nan", "")]

# ---------- 1 · Upload CSV -------------------------------------------
csv = {p.stem: pd.read_csv(p, dtype=str) for p in DATA.glob("*.csv")}

# — cast Numeric where applicable
for df in csv.values():
    for c in df.columns:
        if df[c].str.fullmatch(r"\d+(\.\d+)?").all():
            df[c] = df[c].astype(float).astype("Int64")

# ---------- 2 · Organization ----------------------------------------
for _, row in csv["organizations"].iterrows():
    org = uri("Organization", row.id)
    g += [
        (org, RDF.type, RES.Organization),
        (org, RES.orgName, Literal(row.organization)),
        (org, RES.orgType, Literal(row.type)),
    ]

# ---------- 3 · Author / Reviewer -----------------------------------
for _, row in csv["authors"].iterrows():
    a = uri("Author", row.id)
    g += [
        (a, RDF.type, RES.Author),
        (a, RES.fullName, Literal(row.author)),
    ]

# reviewers.csv: id == paper_id
for _, row in csv["reviewers"].iterrows():
    for aid in split_ids(row.author_ids):
        g.add((uri("Author", aid), RDF.type, RES.Reviewer))

# ---------- 4 · Topic -----------------------------------------------
for _, row in csv["keywords"].iterrows():
    t = uri("Topic", row.id)
    g += [
        (t, RDF.type, RES.Topic),
        (t, RES.title, Literal(row.keyword)),
    ]

# ---------- 5 · Conference & Editions -------------------------------
for _, row in csv["conferences"].iterrows():
    cs = uri("ConferenceSeries", row.id)
    g += [
        (cs, RDF.type, RES.ConferenceSeries),
        (cs, RES.title, Literal(row.conference)),
    ]

for _, row in csv["editions"].iterrows():
    ed  = uri("ConferenceEdition", row.id)
    ser = uri("ConferenceSeries", row.conference_id)
    g += [
        (ed, RDF.type, RES.ConferenceEdition),
        (ed, RES.belongsToSeries, ser),
        (ed, RES.year, Literal(int(row.year), datatype=XSD.gYear)),
        (ed, RES.city, Literal(row.city)),
    ]

# ---------- 6 · Journal & Volumes -----------------------------------
for _, row in csv["journals"].iterrows():
    j = uri("Journal", row.id)
    g += [
        (j, RDF.type, RES.Journal),
        (j, RES.title, Literal(row.journal)),
    ]

for _, row in csv["volumes"].iterrows():
    v = uri("JournalVolume", row.id)
    j = uri("Journal", row.journal_id)
    g += [
        (v, RDF.type, RES.JournalVolume),
        (v, RES.volume, Literal(int(row.volume))),
        (v, RES.year, Literal(int(row.year), datatype=XSD.gYear)),
        (v, RES.partOfJournal, j),
    ]

# ---------- 7 · Paper & Authorship ----------------------------------
for _, row in csv["papers"].iterrows():
    p = uri("Paper", row.id)
    g += [
        (p, RDF.type, RES.Paper),
        (p, RES.title,    Literal(row.title)),
        (p, RES.pages,    Literal(row.pages)),
        (p, RES.abstract, Literal(row.abstract)),
        (p, RES.doi,      Literal(row.doi)),
    ]

    if not pd.isna(row.volume_id):
        g.add((p, RES.publishedIn, uri("JournalVolume", row.volume_id)))
    if not pd.isna(row.edition_id):
        g.add((p, RES.publishedIn, uri("ConferenceEdition", row.edition_id)))

    for kid in split_ids(row.keyword_ids):
        g.add((p, RES.aboutTopic, uri("Topic", kid)))

    # Authorships (one per couple paper–author)
    for aid in split_ids(row.author_ids):
        auth = uri("Authorship", f"{row.id}_{aid}")
        g += [
            (auth, RDF.type, RES.Authorship),
            (auth, RES.authorOf, uri("Author", aid)),
            (auth, RES.paperOf,  p),
        ]

# Corresponding author (functional property)
for _, row in csv["corr_authors"].iterrows():
    auth = uri("Authorship", f"{row.id}_{row.author_id}")
    g.add((auth, RES.hasCorrespondingAuthor,
           uri("Author", row.author_id)))

# ---------- 8 · Affiliations ----------------------------------------
for _, row in csv["affiliations"].iterrows():
    g.add((uri("Author", row.id),
           RES.affiliatedTo,
           uri("Organization", row.organization_id)))

# ---------- 9 · Citations -------------------------------------------
for _, row in csv["citations"].iterrows():
    g.add((uri("Paper", row.citer_id),
           RES.cites,
           uri("Paper", row.cited_id)))

# ---------- 10 · ReviewEvents  (exactly 3 reviewers) ------------
reviews_df = csv["reviews"]

for pid, group in reviews_df.groupby("article_id"):
    ev  = uri("ReviewEvent", pid)      # One per article
    pap = uri("Paper", pid)

    g += [
        (ev, RDF.type, RES.ReviewEvent),
        (ev, RES.reviewedPaper, pap),
    ]

    # We added the 3 reviewers and their texts / decisions as literals.
    for _, row in group.iterrows():
        rev = uri("Author", row.author_id)
        g.add((ev, RES.reviewerOf, rev))
        g.add((ev, RES.reviewText, Literal(row.content)))
        g.add((ev, RES.decision,   Literal(row.decision)))

# ---------- 11 · Serialize -----------------------------------------
ONT.parent.mkdir(exist_ok=True)
g.serialize(ONT, format="turtle")
print(f"ABOX serializada en {ONT} – {len(g)} triples")

# --- Optional Validation --------------------------
for pid, group in csv["reviews"].groupby("article_id"):
    if len(group) != 3:
        print(f"⚠ Paper {pid} tiene {len(group)} revisores (≠3)")
