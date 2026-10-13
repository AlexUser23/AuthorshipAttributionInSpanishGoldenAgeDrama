from pathlib import Path
from transformers import AutoTokenizer 
import unicodedata
import re
import os
from tqdm import tqdm
import random
import pandas as pd


# Resolve repository paths independently of the working directory.
REPO_ROOT = Path(__file__).resolve().parents[2]

# --------------------- CONFIG ---------------------
MODEL_NAME = "BSC-LT/mRoBERTa"
INPUT_DIR = str(REPO_ROOT / "data/rhythmic_corpus_100plays")
OUTPUT_DIR = str(REPO_ROOT / "data/crossval_12folds_mroberta_tokenized_regenerated")
CHUNK_SIZE = 510
STRIDE = 0
ENCODING = "utf-8"
SEED = 44 #42
N_FOLDS = 12
# --------------------------------------------------

random.seed(SEED)

# cargar tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def normalizar_texto(texto: str) -> str: 
    """
    Normalización Unicode sin alterar saltos de línea lógicos,
    colapsa múltiples espacios en uno y elimina espacios inmediatamente
    después (y antes) de un salto de línea.
    """
    # Normalización Unicode y unificar finales de línea
    texto = unicodedata.normalize("NFKC", texto)
    texto = texto.replace("\r\n", "\n").replace("\r", "\n")

    # 1) Eliminar espacios/tabs justo antes de un salto de línea (evita "línea ␣\n")
    texto = re.sub(r'[ \t]+(?=\n)', '', texto)

    # 2) Eliminar espacios/tabs justo después de un salto de línea (evita "\n␣␣texto")
    texto = re.sub(r'\n[ \t]+', '\n', texto)

    # 3) Colapsar múltiples espacios/tabs consecutivos en un único espacio (no afecta a \n)
    texto = re.sub(r'[ \t]{2,}', ' ', texto)

    # 4) Opcional: quitar espacios iniciales/finales globales
    texto = texto.strip()

    return texto


def tokenizar_modelo(texto: str):
    return tokenizer.encode(texto, add_special_tokens=False)


def hacer_chunks_texto_original(texto: str, chunk_size: int, stride: int):
    encoding = tokenizer(texto, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoding["offset_mapping"]

    chunks = []
    start_idx = 0

    while start_idx < len(offsets):
        end_idx = min(start_idx + chunk_size, len(offsets))
        char_start = offsets[start_idx][0]
        char_end = offsets[end_idx - 1][1]
        chunk_texto = texto[char_start:char_end]
        chunks.append(chunk_texto)

        if end_idx == len(offsets):
            break

        start_idx += (chunk_size - stride)

    return chunks


# Obtener lista de archivos .txt
archivos = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".txt")]
if not archivos:
    raise ValueError(f"No se han encontrado archivos .txt en {INPUT_DIR}")

# Agrupar archivos por autor
autor_to_files = {}
for nombre_archivo in archivos:
    autor = nombre_archivo.split("_")[0]
    autor_key = autor if not autor.lower().startswith("desconocido") else "UNLABELED"
    ruta = os.path.join(INPUT_DIR, nombre_archivo)
    comedia = os.path.splitext(nombre_archivo)[0]
    autor_to_files.setdefault(autor_key, []).append({
        "filename": nombre_archivo,
        "filepath": ruta,
        "comedia": comedia
    })

# ----------------------------
# Asignación a N_FOLDS folds
# ----------------------------
folds = {i: [] for i in range(N_FOLDS)}

for autor, files in autor_to_files.items():
    files_copy = list(files)
    random.shuffle(files_copy)
    for idx, f in enumerate(files_copy):
        fold_idx = idx % N_FOLDS
        folds[fold_idx].append({**f, "autor": autor})

# Imprimir distribución simple
total_asignados = sum(len(v) for v in folds.values())
print(f"Autores encontrados: {len(autor_to_files)}")
print(f"Total archivos (comunmente 100): {len(archivos)}")
for i in range(N_FOLDS):
    print(f"Fold {i:02d}: {len(folds[i])} comedias")

# Función que procesa una lista de archivos (crea chunks y filas)
def procesar_lista_archivos(file_list):
    filas = []
    for meta in tqdm(file_list, desc="Chunking archivos"):
        ruta = meta["filepath"]
        autor = meta["autor"]
        comedia = meta["comedia"]

        with open(ruta, "r", encoding=ENCODING) as f:
            texto = normalizar_texto(f.read())

        trozos = hacer_chunks_texto_original(texto, CHUNK_SIZE, STRIDE)

        for i, chunk in enumerate(trozos):
            num_tokens = len(tokenizar_modelo(chunk))

            filas.append({
                "texto": chunk,
                "autor": autor,
                "comedia": comedia,
                "posicion": i,
                "num_tokens": num_tokens,
            })
    return filas

# Crear directorio de salida
os.makedirs(OUTPUT_DIR, exist_ok=True)

# NUEVAS VARIABLES: contador global de chunks y lista para desglose por fold
total_chunks = 0                     # <-- variable añadida
chunks_por_fold = []                 # <-- opcional: para guardar desglose por fold

# Procesar cada fold por separado (chunking *después* de la asignación de comedias)
resumen_asignacion = []
for i in range(N_FOLDS):
    print(f"Procesando FOLD {i} ...")
    file_list = folds[i]
    fold_rows = procesar_lista_archivos(file_list)
    df_fold = pd.DataFrame(fold_rows)

    path_fold = os.path.join(OUTPUT_DIR, f"fold_{i:02d}.csv")
    df_fold.to_csv(path_fold, index=False, encoding="utf-8")
    print(f"   Guardado: {path_fold}  (chunks: {len(df_fold)})")

    # ACTUALIZAR CONTADOR GLOBAL con los chunks de este fold
    total_chunks += len(df_fold)      # <-- línea añadida
    chunks_por_fold.append({"fold": i, "n_chunks": len(df_fold)})  # <-- opcional

    # añadir al resumen cuales comedias fueron asignadas a este fold
    for meta in file_list:
        resumen_asignacion.append({
            "fold": i,
            "autor": meta["autor"],
            "comedia": meta["comedia"],
            "filename": meta["filename"]
        })

# Guardar resumen de asignación por autor/comedia/fold
df_resumen_folds = pd.DataFrame(resumen_asignacion)
path_resumen_folds = os.path.join(OUTPUT_DIR, "resumen_asignacion_folds.csv")
df_resumen_folds.to_csv(path_resumen_folds, index=False, encoding="utf-8")
print("Resumen de asignación por fold guardado en:", path_resumen_folds)

# (Opcional) guardar desglose de chunks por fold
df_chunks_fold = pd.DataFrame(chunks_por_fold)
path_chunks_fold = os.path.join(OUTPUT_DIR, "chunks_por_fold.csv")
df_chunks_fold.to_csv(path_chunks_fold, index=False, encoding="utf-8")

# (Opcional) guardar estadísticas globales (incluye el total de chunks)
stats = {
    "total_chunks": total_chunks,
    "total_archivos": len(archivos),
    "n_folds": N_FOLDS,
    "chunk_size": CHUNK_SIZE,
    "stride": STRIDE
}
df_stats = pd.DataFrame([stats])
path_stats = os.path.join(OUTPUT_DIR, "dataset_stats.csv")
df_stats.to_csv(path_stats, index=False, encoding="utf-8")

# (Opcional) imprimir el total final
print(f"\n Proceso completado. CSVs de cada fold en: {OUTPUT_DIR}")
print(f"Total de chunks en el dataset completo: {total_chunks}")
print("Estadísticas guardadas en:", path_stats)

# (Opcional) guardar un pequeño resumen por autor de cuántas comedias fueron asignadas
resumen = []
for autor, files in autor_to_files.items():
    n_total = len(files)
    n_assigned = sum(1 for row in resumen_asignacion if row["autor"] == autor)
    resumen.append({"autor": autor, "total_comedias": n_total, "assigned_to_folds": n_assigned})

df_resumen = pd.DataFrame(resumen)
path_resumen = os.path.join(OUTPUT_DIR, "resumen_por_autor.csv")
df_resumen.to_csv(path_resumen, index=False, encoding="utf-8")
print("Resumen por autor guardado en:", path_resumen)