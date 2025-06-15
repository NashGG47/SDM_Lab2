import pandas as pd
from pykeen.triples import TriplesFactory
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# Obtener la raíz del proyecto desde la ubicación del script
ROOT = Path(__file__).resolve().parents[1]
triples_path = ROOT / "experiments" / "triples_raw.tsv"

# Verificar existencia del archivo
if not triples_path.exists():
    raise FileNotFoundError(f"Archivo no encontrado: {triples_path}")

print(f"Archivo encontrado en: {triples_path}")

# Crear carpetas si no existen
os.makedirs(ROOT / "figures", exist_ok=True)
os.makedirs(ROOT / "data", exist_ok=True)

# Cargar datos
df = pd.read_csv(triples_path, sep='\t', header=None, names=['h', 'r', 't'])

# Estadísticas de predicados
stats = df['r'].value_counts()
stats.index = stats.index.str.extract(r'/([^/>]+)>?$')[0]

# Gráfico de distribución de predicados
plt.figure(figsize=(10, 6))
sns.barplot(x=stats.index, y=stats.values)
plt.xticks(rotation=90)
plt.title("Distribución de predicados (resumen)")
plt.xlabel("Predicado")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.savefig(ROOT / "figures" / "c1_predicate_distribution_short.png")
plt.close()

# Copia del DataFrame para no afectar el original
df_plot = df.copy()

# Extraer parte final de la URI del predicado
df_plot['r_short'] = df_plot['r'].str.extract(r'/([^/>]+)>?$')[0]

# Figura de distribución de predicados usando etiquetas cortas
plt.figure(figsize=(12, 6))
sns.countplot(data=df_plot, y='r_short', order=df_plot['r_short'].value_counts().index)
plt.title('Distribución de Predicados (ABOX)')
plt.xlabel('Frecuencia')
plt.ylabel('Predicado')
plt.tight_layout()
plt.savefig(ROOT / "figures" / "c1_predicados_short.png")
plt.close()


# Crear TriplesFactory y hacer split
tf = TriplesFactory.from_path(str(triples_path))  # convertir a string por compatibilidad
train, val, test = tf.split([0.8, 0.1, 0.1], random_state=42, method='coverage')

# Guardar splits
pd.DataFrame(train.triples).to_csv(ROOT / "data" / "train.tsv", sep='\t', header=False, index=False)
pd.DataFrame(val.triples).to_csv(ROOT / "data" / "val.tsv", sep='\t', header=False, index=False)
pd.DataFrame(test.triples).to_csv(ROOT / "data" / "test.tsv", sep='\t', header=False, index=False)
