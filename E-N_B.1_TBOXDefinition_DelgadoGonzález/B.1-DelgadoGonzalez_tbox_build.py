#!/usr/bin/env python
# -------------------------------------------------------------------
#   b1_tbox_build.py   –   generate ont/research_tbox.owl (OWL 2 DL)
# -------------------------------------------------------------------
from rdflib import Graph, Namespace, RDF, RDFS, OWL, XSD, BNode, Literal
from pathlib import Path

RES = Namespace("http://example.org/res/")

g = Graph()
g.bind("res", RES)
g.bind("owl", OWL)
g.bind("xsd", XSD)

# ------------------------------------------------------------------ #
# 1 · CLASSES                                                        #
# ------------------------------------------------------------------ #
CLASSES = """
    Author Reviewer Organization
    Paper Topic Publication
    ConferenceSeries ConferenceEdition
    Journal JournalVolume
    Authorship ReviewEvent
""".split()

for c in CLASSES:
    g.add((RES[c], RDF.type, OWL.Class))

g += [
    (RES.Reviewer,          RDFS.subClassOf, RES.Author),
    (RES.ConferenceEdition, RDFS.subClassOf, RES.Publication),
    (RES.JournalVolume,     RDFS.subClassOf, RES.Publication),
    # Disjoint: ConferenceEdition ⊥ JournalVolume
    (RES.ConferenceEdition, OWL.disjointWith, RES.JournalVolume),
]

# ------------------------------------------------------------------ #
# 2 · OBJECT PROPERTIES                                              #
# ------------------------------------------------------------------ #
OBJ = {
    "affiliatedTo"      : ("Author","Organization"),
    "writes"            : ("Author","Paper"),
    "aboutTopic"        : ("Paper","Topic"),
    "cites"             : ("Paper","Paper"),
    "publishedIn"       : ("Paper","Publication"),
    "belongsToSeries"   : ("ConferenceEdition","ConferenceSeries"),
    "partOfJournal"     : ("JournalVolume","Journal"),
    "authorOf"          : ("Authorship","Author"),
    "paperOf"           : ("Authorship","Paper"),
    "hasCorrespondingAuthor": ("Authorship","Author"),
    "reviewerOf"        : ("ReviewEvent","Reviewer"),
    "reviewedPaper"     : ("ReviewEvent","Paper"),
    # inverse we add it later…
}

for p,(dom,rng) in OBJ.items():
    g += [
        (RES[p], RDF.type, OWL.ObjectProperty),
        (RES[p], RDFS.domain, RES[dom]),
        (RES[p], RDFS.range,  RES[rng]),
    ]

# Reverse cites ↔ isCitedBy
g += [
    (RES.isCitedBy, RDF.type, OWL.ObjectProperty),
    (RES.isCitedBy, OWL.inverseOf, RES.cites),
    (RES.isCitedBy, RDFS.domain, RES.Paper),
    (RES.isCitedBy, RDFS.range,  RES.Paper),
]

# Functional: an Authorship has **at most one** correspondingAuthor
g.add((RES.hasCorrespondingAuthor, RDF.type, OWL.FunctionalProperty))

# Special properties
g += [
    (RES.cites,        RDF.type, OWL.IrreflexiveProperty),
    (RES.cites,        RDF.type, OWL.AsymmetricProperty),
]

# ------------------------------------------------------------------ #
# 3 · DATA PROPERTIES                                                #
# ------------------------------------------------------------------ #
DATA = {
    "fullName"  : ("Author",          XSD.string),
    "orgName"   : ("Organization",    XSD.string),
    "orgType"   : ("Organization",    XSD.string),
    "title"     : ("Paper",           XSD.string),
    "pages"     : ("Paper",           XSD.string),
    "abstract"  : ("Paper",           XSD.string),
    "doi"       : ("Paper",           XSD.string),
    "year"      : ("Publication",     XSD.gYear),
    "volume"    : ("JournalVolume",   XSD.integer),
    "city"      : ("ConferenceEdition", XSD.string),
    "score"     : ("ReviewEvent",     XSD.integer),
    "decision"  : ("ReviewEvent",     XSD.string),
    "reviewText": ("ReviewEvent",     XSD.string),
}
for p,(dom,dt) in DATA.items():
    g += [
        (RES[p], RDF.type, OWL.DatatypeProperty),
        (RES[p], RDFS.domain, RES[dom]),
        (RES[p], RDFS.range , dt),
    ]

# ------------------------------------------------------------------ #
# 4 ·   CARDINALITY CONSTRAINTS                                      #
# ------------------------------------------------------------------ #
def cardinal(cls, prop, *, exact=None, min_=None, max_=None):
    """
    Add an OWL constraint on `prop` in the `cls` class.
    Use exactly one of:
        exact owl:cardinality
        min_  owl:minCardinality
        max_  owl:maxCardinality
    """
    r = BNode()
    # Type Restriction + on Property
    g.add((r, RDF.type, OWL.Restriction))
    g.add((r, OWL.onProperty, RES[prop]))

    if exact is not None:
        g.add((r, OWL.cardinality,
               Literal(exact, datatype=XSD.nonNegativeInteger)))
    if min_ is not None:
        g.add((r, OWL.minCardinality,
               Literal(min_, datatype=XSD.nonNegativeInteger)))
    if max_ is not None:
        g.add((r, OWL.maxCardinality,
               Literal(max_, datatype=XSD.nonNegativeInteger)))

    # Finally we link the constraint to the class
    g.add((RES[cls], RDFS.subClassOf, r))


# Authorship: exactly 1 paper, **minimum** 1 author, maximum 1 corresponding author
cardinal("Authorship", "paperOf", exact=1)
cardinal("Authorship", "authorOf", min_=1)
cardinal("Authorship", "hasCorrespondingAuthor", max_=1)

#ReviewEvent: exactly 3 reviewers, exactly 1 paper reviewed
cardinal("ReviewEvent", "reviewerOf", exact=3)
cardinal("ReviewEvent", "reviewedPaper", exact=1)

# Paper: at least 1 Reverse Authorship authorOf
rev = BNode()
g += [
    (rev, RDF.type, OWL.Restriction),
    (rev, OWL.onProperty, RES.authorOf),
    (rev, OWL.minCardinality,
        Literal(1, datatype=XSD.nonNegativeInteger)),
    (RES.Paper, RDFS.subClassOf, rev),
]

# ------------------------------------------------------------------ #
# 5 · SERIALIZE (Option B · RDF/XML with .owl extension)             #
# ------------------------------------------------------------------ #
Path("ont").mkdir(exist_ok=True)
TARGET = "ont/research_tbox.owl"          # ← keeps .owl extension
g.serialize(TARGET, format="pretty-xml")  # "xml" is also valid
print(f"TBOX actualizada → {TARGET}")

