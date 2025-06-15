# kge_comparative_analysis.py

import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, pairwise_distances
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from pykeen.models import TransE, RotatE, ComplEx
from pathlib import Path

# ------------------ CONFIGURACION ------------------
data_dir = Path(__file__).resolve().parent.parent / "data"
experiments_dir = Path(__file__).resolve().parent.parent / "experiments"
figures_dir = Path(__file__).resolve().parent.parent / "figures_C2"
os.makedirs(figures_dir, exist_ok=True)

models = {
    'TransE': TransE,
    'RotatE': RotatE,
    'ComplEx': ComplEx
}

# ------------------ CARGAR TRIPLES ------------------
tf_train = TriplesFactory.from_path(data_dir / "train.tsv")
tf_test = TriplesFactory.from_path(data_dir / "test.tsv")
tf_all = TriplesFactory.from_path(experiments_dir / "triples_raw.tsv")

# ------------------ FUNCIONES AUXILIARES ------------------
def infer_entity_types(triples):
    types = {}
    for h, r, t in triples:
        if 'writes' in r:
            types[h] = 'author'
            types[t] = 'paper'
        elif 'cites' in r:
            types[h] = 'paper'
            types[t] = 'paper'
    return types

def compute_embedding_dataframe(model_result, entity_to_id):
    model = model_result.model
    embs = model.entity_representations[0]().detach().numpy()
    if np.iscomplexobj(embs):
        real = np.real(embs)
        imag = np.imag(embs)
        emb_concat = np.concatenate([real, imag], axis=1)
        df = pd.DataFrame(emb_concat, index=list(entity_to_id.keys()))
    else:
        df = pd.DataFrame(embs, index=list(entity_to_id.keys()))
    df.index.name = 'entity'
    return df

def plot_embeddings(df, types, model_name):
    df['type'] = df.index.map(lambda x: types.get(x, 'unknown'))
    pca = PCA(n_components=2).fit_transform(df.drop(columns='type'))
    tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(df.drop(columns='type'))

    for method, embedding in zip(['pca', 'tsne'], [pca, tsne]):
        plot_df = pd.DataFrame(embedding, columns=['x', 'y'], index=df.index)
        plot_df['type'] = df['type']
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=plot_df, x='x', y='y', hue='type', s=20)
        plt.title(f"{model_name} - {method.upper()} projection")
        plt.tight_layout()
        plt.savefig(figures_dir / f"{model_name.lower()}_{method}.png", dpi=600)
        plt.close()

def calculate_metrics(df):
    X = df.drop(columns='type').values
    labels = df['type'].astype('category').cat.codes
    silhouette = silhouette_score(X, labels)
    intra = []
    inter = []
    for t in df['type'].unique():
        sub = df[df['type'] == t].drop(columns='type')
        if len(sub) > 1:
            dists = pairwise_distances(sub)
            intra.append(np.mean(dists[np.triu_indices_from(dists, 1)]))
    for t1 in df['type'].unique():
        for t2 in df['type'].unique():
            if t1 != t2:
                d1 = df[df['type'] == t1].drop(columns='type')
                d2 = df[df['type'] == t2].drop(columns='type')
                d = pairwise_distances(d1, d2)
                inter.append(np.mean(d))
    return {
        'silhouette_score': silhouette,
        'avg_intra_distance': np.mean(intra),
        'avg_inter_distance': np.mean(inter)
    }

def predict_citation_and_author(model_result, entity_to_id, relation_to_id, types):
    model = model_result.model

    # Entidades y relaciones
    h_label = [k for k, v in types.items() if v == 'paper'][0]  # primer paper
    r_label = [r for r in relation_to_id if 'cites' in r][0]

    w_candidates = [r for r in relation_to_id if 'write' in r.lower() or 'author' in r.lower()]
    if not w_candidates:
        raise ValueError("No se encontró una relación que represente 'writes' o 'author'. Revisa tus datos.")
    w_label = w_candidates[0]

    # IDs
    h_id = entity_to_id[h_label]
    r_id = relation_to_id[r_label]
    w_id = relation_to_id[w_label]

    # Embeddings
    h_emb = model.entity_representations[0](torch.tensor([h_id]))
    r_emb = model.relation_representations[0](torch.tensor([r_id]))
    pred_t = h_emb + r_emb

    all_embs = model.entity_representations[0]()

    # --- manejar embeddings complejos ---
    def to_real(x):
        x_np = x.detach().numpy()
        if np.iscomplexobj(x_np):
            return np.hstack([x_np.real, x_np.imag])
        return x_np

    pred_t_real = to_real(pred_t)
    all_embs_real = to_real(all_embs)

    # predecir paper citado
    distances = pairwise_distances(pred_t_real, all_embs_real)
    predicted_paper = list(entity_to_id.keys())[distances.argmin()]

    # predecir autor
    paper_id = entity_to_id[predicted_paper]
    paper_emb = model.entity_representations[0](torch.tensor([paper_id]))
    w_emb = model.relation_representations[0](torch.tensor([w_id]))
    pred_author = paper_emb - w_emb

    pred_author_real = to_real(pred_author)
    author_distances = pairwise_distances(pred_author_real, all_embs_real)
    predicted_author = list(entity_to_id.keys())[author_distances.argmin()]

    return h_label, predicted_paper, predicted_author


# ------------------ ENTRENAMIENTO Y ANALISIS ------------------
entity_types = infer_entity_types(tf_all.triples)
results_table = []
prediction_table = []

for model_name, model_cls in models.items():
    print(f"Entrenando {model_name}...")
    result = pipeline(
        model=model_name,
        training=tf_train,
        testing=tf_test,
        random_seed=42,
        model_kwargs=dict(embedding_dim=50),
        training_kwargs=dict(num_epochs=50, batch_size=128),
        device='cpu'
    )
    result.save_to_directory(data_dir / f"{model_name}_model_C2")

    emb_df = compute_embedding_dataframe(result, tf_train.entity_to_id)
    emb_df['type'] = emb_df.index.map(lambda x: entity_types.get(x, 'unknown'))

    plot_embeddings(emb_df.copy(), entity_types, model_name)

    metrics = calculate_metrics(emb_df)
    metrics['model'] = model_name
    results_table.append(metrics)

    h, t, a = predict_citation_and_author(result, tf_train.entity_to_id, tf_train.relation_to_id, entity_types)
    prediction_table.append({
        'model': model_name,
        'input_paper': h,
        'predicted_cited_paper': t,
        'predicted_author': a
    })

# ------------------ GUARDAR RESULTADOS ------------------
results_df = pd.DataFrame(results_table)
pred_df = pd.DataFrame(prediction_table)

results_df.to_csv(data_dir / "embedding_comparison_metrics_C2.csv", index=False)
pred_df.to_csv(data_dir / "embedding_predictions_C2.csv", index=False)

print("\nResumen de métricas por modelo:")
print(results_df)

print("\nPredicciones de cita y autor por modelo:")
print(pred_df)

# ------------------ OPCIONAL: MOSTRAR COMO TABLAS ------------------
from IPython.display import display
try:
    display(results_df)
    display(pred_df)
except:
    pass