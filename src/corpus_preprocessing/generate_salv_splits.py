from pathlib import Path
from transformers import AutoTokenizer
import unicodedata
import re
import os
from tqdm import tqdm
import random
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

# Configuration
MODEL_NAME = "BSC-LT/mRoBERTa"
INPUT_DIR = str(REPO_ROOT / "data/salv_expanded_corpus")
OUTPUT_DIR = str(REPO_ROOT / "data/salv_splits_seed44")
CHUNK_SIZE = 510
STRIDE = 0
ENCODING = "utf-8"
SEED = 44


N_TEST_ALARCON  = 8
N_TEST_BELMONTE = 3
N_TEST_CLARAMONTE = 3
N_TEST_LOPE     = 8
N_TEST_TIRSO    = 8

# Author keys match filename prefixes.
N_TEST_PER_AUTHOR = {
    "Alarcón":    N_TEST_ALARCON,
    "Belmonte":   N_TEST_BELMONTE,
    "Claramonte": N_TEST_CLARAMONTE,
    "Lope":       N_TEST_LOPE,
    "Tirso":      N_TEST_TIRSO,
}

N_VALID_PER_AUTHOR = 2
DUDA_AUTHOR_KEY = "DUDA"


random.seed(SEED)


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if not tokenizer.is_fast:
    raise ValueError("El chunking requiere un tokenizador rápido con offsets.")


def normalizar_texto(texto: str) -> str:
    """Apply NFKC normalization and unify line endings."""

    texto = unicodedata.normalize("NFKC", texto)
    texto = texto.replace("\r\n", "\n").replace("\r", "\n")

    return texto


def tokenizar_modelo(texto: str):
    return tokenizer.encode(texto, add_special_tokens=False)


def hacer_chunks_texto_original(texto: str, chunk_size: int, stride: int):
    """Slice text using token offsets without special tokens."""
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


# Sort filenames before seeded sampling.
archivos = sorted(f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".txt"))
if not archivos:
    raise ValueError(f"No se han encontrado archivos .txt en {INPUT_DIR}")


# Filenames use the author_title.txt convention.
autor_to_files = {}
duda_files = []

for nombre_archivo in archivos:
    autor = nombre_archivo.split("_")[0]
    autor_key = autor if not autor.lower().startswith("desconocido") else "UNLABELED"
    ruta = os.path.join(INPUT_DIR, nombre_archivo)
    comedia = os.path.splitext(nombre_archivo)[0]

    entry = {
        "filename": nombre_archivo,
        "filepath": ruta,
        "comedia": comedia,
        "autor": autor_key
    }


    if autor_key == DUDA_AUTHOR_KEY:
        duda_files.append(entry)
    else:
        autor_to_files.setdefault(autor_key, []).append(entry)


train_files = []
val_files = []
test_files = []

for autor, files in autor_to_files.items():
    files_copy = list(files)
    n_available = len(files_copy)

    # Select test plays first; unlisted authors receive no test plays.

    n_test_autor = N_TEST_PER_AUTHOR.get(autor, 0)
    k_test = min(n_test_autor, n_available)
    seleccion_test = random.sample(files_copy, k=k_test) if k_test > 0 else []
    seleccion_test_set = {f["filename"] for f in seleccion_test}


    remaining_after_test = [f for f in files_copy if f["filename"] not in seleccion_test_set]

    # Select validation plays from the remaining texts.
    k_val = min(N_VALID_PER_AUTHOR, len(remaining_after_test))
    seleccion_val = random.sample(remaining_after_test, k=k_val) if k_val > 0 else []
    seleccion_val_set = {f["filename"] for f in seleccion_val}


    for f in files_copy:
        if f["filename"] in seleccion_test_set:
            test_files.append(f)
        elif f["filename"] in seleccion_val_set:
            val_files.append(f)
        else:
            train_files.append(f)


print(f"Autores encontrados (excl. DUDA): {len(autor_to_files)}")
print(f"Total archivos (excl. DUDA): {len(archivos) - len(duda_files)}")
print(f"Archivos con autor DUDA: {len(duda_files)}")
print(f"Archivos en train: {len(train_files)}")
print(f"Archivos en val:   {len(val_files)}")
print(f"Archivos en test:  {len(test_files)}")
print(f"Nota: n_test por autor → {N_TEST_PER_AUTHOR}; val por autor → {N_VALID_PER_AUTHOR}")


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


# Fragment partitions separately; keep disputed plays independent.
print("Procesando TRAIN...")
train_rows = procesar_lista_archivos(train_files)

print("Procesando VALIDACIÓN...")
val_rows = procesar_lista_archivos(val_files)

print("Procesando TEST...")
test_rows = procesar_lista_archivos(test_files)

print("Procesando DUDA...")
duda_rows = procesar_lista_archivos(duda_files)


df_train = pd.DataFrame(train_rows)
df_val = pd.DataFrame(val_rows)
df_test = pd.DataFrame(test_rows)
df_duda = pd.DataFrame(duda_rows)


os.makedirs(OUTPUT_DIR, exist_ok=True)


path_train = os.path.join(OUTPUT_DIR, "train.csv")
path_val = os.path.join(OUTPUT_DIR, "val.csv")
path_test = os.path.join(OUTPUT_DIR, "test.csv")
path_duda = os.path.join(OUTPUT_DIR, "duda.csv")

df_train.to_csv(path_train, index=False, encoding="utf-8")
df_val.to_csv(path_val, index=False, encoding="utf-8")
df_test.to_csv(path_test, index=False, encoding="utf-8")
df_duda.to_csv(path_duda, index=False, encoding="utf-8")

print("✅ CSVs guardados en:", OUTPUT_DIR)
print(" - train:", path_train)
print(" - val:  ", path_val)
print(" - test: ", path_test)
print(" - duda: ", path_duda)


def chunks_por_autor(df):
    if df.empty:
        return {}
    return df.groupby("autor").size().to_dict()

train_chunks = chunks_por_autor(df_train)
val_chunks   = chunks_por_autor(df_val)
test_chunks  = chunks_por_autor(df_test)
duda_chunks  = chunks_por_autor(df_duda)


todos_los_autores = set(autor_to_files.keys()) | {DUDA_AUTHOR_KEY}

resumen = []
for autor in sorted(todos_los_autores):
    if autor == DUDA_AUTHOR_KEY:
        n_total   = len(duda_files)
        n_train   = 0
        n_val     = 0
        n_test    = 0
        n_duda    = len(duda_files)
        c_train   = 0
        c_val     = 0
        c_test    = 0
        c_duda    = duda_chunks.get(autor, 0)
    else:
        files_autor = autor_to_files.get(autor, [])
        n_total  = len(files_autor)
        n_test   = sum(1 for f in test_files  if f["autor"] == autor)
        n_val    = sum(1 for f in val_files   if f["autor"] == autor)
        n_train  = sum(1 for f in train_files if f["autor"] == autor)
        n_duda   = 0
        c_train  = train_chunks.get(autor, 0)
        c_val    = val_chunks.get(autor, 0)
        c_test   = test_chunks.get(autor, 0)
        c_duda   = 0

    resumen.append({
        "autor":            autor,
        "total_comedias":   n_total,
        "comedias_train":   n_train,
        "comedias_val":     n_val,
        "comedias_test":    n_test,
        "comedias_duda":    n_duda,
        "chunks_train":     c_train,
        "chunks_val":       c_val,
        "chunks_test":      c_test,
        "chunks_duda":      c_duda,
        "chunks_total":     c_train + c_val + c_test + c_duda,
    })

df_resumen = pd.DataFrame(resumen)
path_resumen = os.path.join(OUTPUT_DIR, "resumen_por_autor.csv")
df_resumen.to_csv(path_resumen, index=False, encoding="utf-8")
print("Resumen por autor guardado en:", path_resumen)


print("\n── Totales globales de chunks ──")
print(f"  train : {len(df_train)}")
print(f"  val   : {len(df_val)}")
print(f"  test  : {len(df_test)}")
print(f"  duda  : {len(df_duda)}")
print(f"  TOTAL : {len(df_train) + len(df_val) + len(df_test) + len(df_duda)}")
