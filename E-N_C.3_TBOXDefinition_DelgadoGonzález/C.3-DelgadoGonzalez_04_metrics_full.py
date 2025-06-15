# ─── src/metrics_full.py  (works with the oldest API) ───────────────
from pathlib import Path
import json, time, torch
from pykeen.evaluation import RankBasedEvaluator
from pykeen.triples    import TriplesFactory

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS   = ["TransE", "TransH", "RotatE", "ComplEx"]

evaluator = RankBasedEvaluator()
evaluator.k = [1, 3, 5, 10]           # ← set ks here

tf_train = TriplesFactory.from_path(DATA_DIR / "train.tsv")
tf_val   = TriplesFactory.from_path(DATA_DIR / "val.tsv")
tf_test  = TriplesFactory.from_path(DATA_DIR / "test.tsv")

def load_model(folder):
    pt  = folder / "best_model.pt"
    pkl = folder / "trained_model.pkl"
    if pt.exists():
        from pykeen.models import Model
        return Model.from_checkpoint(pt)
    if pkl.exists():
        return torch.load(pkl, map_location="cpu", weights_only=False)
    raise FileNotFoundError(folder)

def reevaluate(name):
    folder = DATA_DIR / f"{name}_model"
    print(f"\n▶ Re-evaluating {name}")
    model = load_model(folder)

    t0 = time.time()
    metrics = evaluator.evaluate(
        model,
        tf_test.mapped_triples,
        additional_filter_triples=[
            tf_train.mapped_triples,
            tf_val.mapped_triples,
        ],
    )
    print(f"   finished in {(time.time()-t0)/60:.1f} min")

    with open(folder / "metrics_full.json", "w") as fh:
        json.dump(metrics.to_dict(), fh, indent=2)

for m in MODELS:
    reevaluate(m)

print("\n✓ metrics_full.json written for every model")
