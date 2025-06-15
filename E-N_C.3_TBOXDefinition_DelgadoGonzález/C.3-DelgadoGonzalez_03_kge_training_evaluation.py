#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------
# C3 – Entrenamiento y comparación de modelos KGE con PyKEEN
# ---------------------------------------------------------------

import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from pykeen.pipeline import pipeline, PipelineResult
from pykeen.triples import TriplesFactory
from pykeen.predict import predict_target

# ------------------------------------------------------------------
# 0. Directorios base
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]          # repo raíz
data_dir = ROOT / "data"
experiments_dir = Path(__file__).resolve().parent.parent / "experiments"
figures_dir = ROOT / "figures"
os.makedirs(figures_dir, exist_ok=True)

# ------------------------------------------------------------------
# 1. Cargar triples (train / test / completo)
# ------------------------------------------------------------------
tf_train = TriplesFactory.from_path(data_dir / "train.tsv")
tf_test  = TriplesFactory.from_path(data_dir / "test.tsv")
tf_all   = TriplesFactory.from_path(experiments_dir / "triples_raw.tsv") 

# ------------------------------------------------------------------
# 2. Diccionarios auxiliares
# ------------------------------------------------------------------
# Citas ya existentes
existing_citations = {
    (h, r, t) for h, r, t in tf_all.triples
    if r == '<http://example.org/res/cites>'
}

# Títulos de papers
titles_dict = {
    h: t for h, r, t in tf_all.triples
    if r == '<http://example.org/res/title>'
}

# ------------------------------------------------------------------
# 3. Configuraciones de modelos
# ------------------------------------------------------------------
model_configs = {
    "TransE":  dict(model_kwargs={"embedding_dim": 100},
                    training_kwargs={"num_epochs": 100, "batch_size": 128},
                    optimizer_kwargs={"lr": 0.001},
                    negative_sampler_kwargs={"num_negs_per_pos": 10}),
    "TransH":  dict(model_kwargs={"embedding_dim": 100},
                    training_kwargs={"num_epochs": 100, "batch_size": 128},
                    optimizer_kwargs={"lr": 0.001},
                    negative_sampler_kwargs={"num_negs_per_pos": 10}),
    "RotatE":  dict(model_kwargs={"embedding_dim": 100},
                    training_kwargs={"num_epochs": 100, "batch_size":  128},
                    optimizer_kwargs={"lr": 0.0005},
                    negative_sampler_kwargs={"num_negs_per_pos": 15}),
    "ComplEx": dict(model_kwargs={"embedding_dim": 100},
                    training_kwargs={"num_epochs": 100, "batch_size": 128},
                    optimizer_kwargs={"lr": 0.0007},
                    negative_sampler_kwargs={"num_negs_per_pos": 10}),
}

# ------------------------------------------------------------------
# 4. Entrenamiento + visualización
# ------------------------------------------------------------------
def entrenar_y_visualizar(model_name: str):
    """Entrena un modelo KGE, guarda métricas y genera figuras."""
    print(f"\n🟢 Entrenando modelo {model_name} ...")
    cfg = model_configs[model_name]

    result = pipeline(
        model=model_name,
        training=tf_train,
        testing=tf_test,
        random_seed=42,
        **cfg,
        device="cpu",                      
    )

    # -------- guardar resultado completo
    result.save_to_directory(data_dir / f"{model_name}_model")

        # -------- Loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(result.losses, marker='o')
    plt.title(f'Training Loss Curve ({model_name})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig(figures_dir / f"c3_loss_{model_name}.png")
    plt.close()

    # -------- Histogram and embedding projections
    emb = result.model.entity_representations[0]().detach().cpu().numpy()
    emb_real = emb.real if np.iscomplexobj(emb) else emb

    sns.histplot(emb_real.flatten(), bins=50, kde=True)
    plt.title(f'Embedding Distribution ({model_name})')
    plt.xlabel('Embedding Value')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(figures_dir / f"c3_emb_hist_{model_name}.png")
    plt.close()

    # PCA projection
    pca_coords = PCA(n_components=2).fit_transform(emb_real)
    sns.scatterplot(x=pca_coords[:, 0], y=pca_coords[:, 1], s=10, alpha=0.7)
    plt.title(f'PCA (2-D) – {model_name}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.tight_layout()
    plt.savefig(figures_dir / f"c3_pca_{model_name}.png")
    plt.close()

    # t-SNE projection
    tsne_coords = TSNE(n_components=2, perplexity=30,
                    n_iter=1000, random_state=42).fit_transform(emb_real)
    sns.scatterplot(x=tsne_coords[:, 0], y=tsne_coords[:, 1], s=10, alpha=0.7)
    plt.title(f't-SNE (2-D) – {model_name}')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.tight_layout()
    plt.savefig(figures_dir / f"c3_tsne_{model_name}.png")
    plt.close()


    # --- Inferencia: sugerir 5 posibles nuevas citas ---
    paper_uri    = "<http://example.org/inst/Paper_0>"
    relation_uri = "<http://example.org/res/cites>"

    preds = predict_target(
        model=result.model,
        head=paper_uri,
        relation=relation_uri,
        triples_factory=tf_train,
    )

    df = preds.df

    # 1) quitar ya citados
    already = {t for h, r, t in existing_citations if h == paper_uri}
    df = df[~df['tail_label'].isin(already)]

    # 2) quitar el propio Paper_0
    df = df[df['tail_label'] != paper_uri]

    # 3) filtrar solo inst/Paper_
    df = df[df['tail_label'].str.contains('inst/Paper_')]

    top5 = (df.head(5)
            .assign(title=lambda d: d['tail_label'].map(titles_dict),
                    model=model_name))

    top5.to_csv(data_dir / f"predictions_top5_{model_name}.csv", index=False)

    print("Top-5 candidatos de citación:")
    for _, row in top5.iterrows():
        print(f"  • {row.tail_label} | {row.title}")


    return result

# ------------------------------------------------------------------
# 5. Comparar métricas entre modelos entrenados
# ------------------------------------------------------------------
def comparar_metricas(modelos):
    filas = []
    for m in modelos:
        try:
            metrics_path = data_dir / f"{m}_model" / "results.json"
            with open(metrics_path) as fh:
                metrics = json.load(fh)["metrics"]["both"]["realistic"]
            filas.append({
                "Modelo": m,
                "MRR":    metrics.get("mean_reciprocal_rank"),
                "Hits@1": metrics.get("hits_at_1"),
                "Hits@3": metrics.get("hits_at_3"),
                "Hits@10": metrics.get("hits_at_10"),
            })
        except FileNotFoundError:
            print(f"No se encontró {metrics_path}")
        except Exception as e:
            print(f"Error leyendo métricas de {m}: {e}")

    if not filas:                # evita el KeyError cuando está vacío
        print("No se cargaron métricas de ningún modelo")
        return

    dfm = pd.DataFrame(filas).set_index("Modelo")
    print("\nComparación de métricas:\n", dfm.round(4))
    dfm.to_csv(figures_dir / "c3_comparacion_metricas.csv")

    # Barras MRR y Hits@10
    b = (dfm[["MRR", "Hits@10"]]
           .reset_index()
           .melt(id_vars="Modelo", var_name="Métrica", value_name="Valor"))
    sns.barplot(data=b, x="Modelo", y="Valor", hue="Métrica")
    plt.ylim(0, 1)
    plt.title("MRR vs Hits@10")
    plt.tight_layout()
    plt.savefig(figures_dir / "c3_metricas_barplot.png")
    plt.close()

    # Línea Hits@K
    l = (dfm[["Hits@1", "Hits@3", "Hits@10"]]
           .reset_index()
           .melt(id_vars="Modelo", var_name="Métrica", value_name="Valor"))
    sns.lineplot(data=l, x="Modelo", y="Valor", hue="Métrica",
                 marker="o", linewidth=2.5)
    plt.ylim(0, 1)
    plt.title("Hits@K por modelo")
    plt.tight_layout()
    plt.savefig(figures_dir / "c3_metricas_hits_line.png")
    plt.close()

# ------------------------------------------------------------------
# 6. Ejecución principal
# ------------------------------------------------------------------
if __name__ == "__main__":
    modelos = list(model_configs.keys())
    for m in modelos:
        entrenar_y_visualizar(m)

    comparar_metricas(modelos)

    # Consolidar las predicciones top-5 de todos los modelos
    csv_files = list(data_dir.glob("predictions_top5_*.csv"))
    if csv_files:
        pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True) \
          .to_csv(data_dir / "predictions_top5_all_models.csv", index=False)
        print("\nArchivo 'predictions_top5_all_models.csv' generado.")
