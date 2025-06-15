# file: describe_triples.py
"""
Prints number of triples, entities, relations for each split
(train / validation / test) and the union.
"""
from pykeen.triples import TriplesFactory
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
experiments_dir = Path(__file__).resolve().parent.parent / "experiments"
splits = {
    "train": DATA / "train.tsv",
    "validation": DATA / "val.tsv",
    "test": DATA / "test.tsv",
}

tf_all = TriplesFactory.from_path(experiments_dir / "triples_raw.tsv")
print(f"Full KG: {len(tf_all.triples):,} triples | "
      f"{len(tf_all.entity_to_id):,} entities | "
      f"{len(tf_all.relation_to_id):,} relations")

for tag, path in splits.items():
    tf = TriplesFactory.from_path(path)
    print(f"{tag.capitalize():<11}: {len(tf.triples):,} triples")
