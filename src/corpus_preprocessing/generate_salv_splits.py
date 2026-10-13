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
MODEL_NAME = "BSC-LT/mRoBERTa" # "dccuchile/bert-base-spanish-wwm-cased" # 
INPUT_DIR = str(REPO_ROOT / "data/salv_expanded_corpus")
OUTPUT_DIR = str(REPO_ROOT / "data/salv_splits_seed44")
CHUNK_SIZE = 510
STRIDE = 0
ENCODING = "utf-8"
SEED = 44 

# Número de comedias por autor que irán a test (una variable por autor)
N_TEST_ALARCON  = 8  # Alarcón
N_TEST_BELMONTE = 3  # Belmonte
N_TEST_CLARAMONTE = 3  # Claramonte
N_TEST_LOPE     = 8  # Lope
N_TEST_TIRSO    = 8  # Tirso

# Mapa autor_key → n_test (la clave debe coincidir con el prefijo del nombre de archivo)
N_TEST_PER_AUTHOR = {
    "Alarcón":    N_TEST_ALARCON,
    "Belmonte":   N_TEST_BELMONTE,
    "Claramonte": N_TEST_CLARAMONTE,
    "Lope":       N_TEST_LOPE,
    "Tirso":      N_TEST_TIRSO,
}

N_VALID_PER_AUTHOR = 2  # número de comedias por autor que irán a validación (después de sacar test)
DUDA_AUTHOR_KEY = "DUDA"  # clave del autor con atribución dudosa
# --------------------------------------------------

random.seed(SEED)

# Ejecutar con el entorno qlora3 y conservar la configuración del modelo.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if not tokenizer.is_fast:
    raise ValueError("El chunking requiere un tokenizador rápido con offsets.")


def normalizar_texto(texto: str) -> str:
    """
    Normalización Unicode sin alterar saltos de línea lógicos,
    colapsa múltiples espacios en uno y elimina espacios inmediatamente
    después (y antes) de un salto de línea.
    """
    # Normalización Unicode y unificar finales de línea
    texto = unicodedata.normalize("NFKC", texto)
    texto = texto.replace("\r\n", "\n").replace("\r", "\n")

    return texto


def tokenizar_modelo(texto: str):
    return tokenizer.encode(texto, add_special_tokens=False)


def hacer_chunks_texto_original(texto: str, chunk_size: int, stride: int):
    """
    Hace chunking en base a offsets de tokens (sin añadir special tokens).
    Devuelve lista de strings (chunks) sin recortar por comillas u otras cosas.
    """
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


# Orden estable para que la selección con semilla no dependa de os.listdir.
archivos = sorted(f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".txt"))
if not archivos:
    raise ValueError(f"No se han encontrado archivos .txt en {INPUT_DIR}")

# Agrupar archivos por autor
# Se asume que el nombre del archivo tiene la forma "AUTOR_restodelnombre.txt"
autor_to_files = {}
duda_files = []  # lista separada para comedias con autoría dudosa

for nombre_archivo in archivos:
    autor = nombre_archivo.split("_")[0]
    autor_key = autor if not autor.lower().startswith("desconocido") else "UNLABELED"
    ruta = os.path.join(INPUT_DIR, nombre_archivo)
    comedia = os.path.splitext(nombre_archivo)[0]  # nombre del archivo sin extensión

    entry = {
        "filename": nombre_archivo,
        "filepath": ruta,
        "comedia": comedia,
        "autor": autor_key
    }

    # Separar comedias de autor DUDA en su propia lista
    if autor_key == DUDA_AUTHOR_KEY:
        duda_files.append(entry)
    else:
        autor_to_files.setdefault(autor_key, []).append(entry)

# Crear listas de archivos para train/val/test
train_files = []
val_files = []
test_files = []

for autor, files in autor_to_files.items():
    files_copy = list(files)  # trabajar sobre copia
    n_available = len(files_copy)

    # 1) seleccionar para TEST usando el n_test específico del autor
    #    Si el autor no está en el mapa, se usa 0 por defecto
    n_test_autor = N_TEST_PER_AUTHOR.get(autor, 0)
    k_test = min(n_test_autor, n_available)
    seleccion_test = random.sample(files_copy, k=k_test) if k_test > 0 else []
    seleccion_test_set = {f["filename"] for f in seleccion_test}

    # remover los seleccionados para test de la lista disponible
    remaining_after_test = [f for f in files_copy if f["filename"] not in seleccion_test_set]

    # 2) seleccionar para VALIDACIÓN (hasta N_VALID_PER_AUTHOR) desde los restantes
    k_val = min(N_VALID_PER_AUTHOR, len(remaining_after_test))
    seleccion_val = random.sample(remaining_after_test, k=k_val) if k_val > 0 else []
    seleccion_val_set = {f["filename"] for f in seleccion_val}

    # 3) asignar a test, val y train
    for f in files_copy:
        if f["filename"] in seleccion_test_set:
            test_files.append(f)
        elif f["filename"] in seleccion_val_set:
            val_files.append(f)
        else:
            train_files.append(f)

# Aviso al usuario (impreso)
print(f"Autores encontrados (excl. DUDA): {len(autor_to_files)}")
print(f"Total archivos (excl. DUDA): {len(archivos) - len(duda_files)}")
print(f"Archivos con autor DUDA: {len(duda_files)}")
print(f"Archivos en train: {len(train_files)}")
print(f"Archivos en val:   {len(val_files)}")
print(f"Archivos en test:  {len(test_files)}")
print(f"Nota: n_test por autor → {N_TEST_PER_AUTHOR}; val por autor → {N_VALID_PER_AUTHOR}")

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
                "posicion": i,            # posición dentro de la comedia (reinicia por cada comedia)
                "num_tokens": num_tokens,
                #"archivo_origen": os.path.basename(ruta)
            })
    return filas


# Procesar train, val y test por separado (para evitar data leakage)
print("Procesando TRAIN...")
train_rows = procesar_lista_archivos(train_files)

print("Procesando VALIDACIÓN...")
val_rows = procesar_lista_archivos(val_files)

print("Procesando TEST...")
test_rows = procesar_lista_archivos(test_files)

print("Procesando DUDA...")
duda_rows = procesar_lista_archivos(duda_files)

# Crear dataframes
df_train = pd.DataFrame(train_rows)
df_val = pd.DataFrame(val_rows)
df_test = pd.DataFrame(test_rows)
df_duda = pd.DataFrame(duda_rows)

# Crear directorio de salida
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Guardar CSVs
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

# ---- Resumen por autor con chunks por conjunto ----

# Calcular chunks por autor por conjunto a partir de los dataframes generados
def chunks_por_autor(df):
    if df.empty:
        return {}
    return df.groupby("autor").size().to_dict()

train_chunks = chunks_por_autor(df_train)
val_chunks   = chunks_por_autor(df_val)
test_chunks  = chunks_por_autor(df_test)
duda_chunks  = chunks_por_autor(df_duda)

# Construir resumen unificado de todos los autores (incluyendo DUDA)
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

# Imprimir totales globales de chunks
print("\n── Totales globales de chunks ──")
print(f"  train : {len(df_train)}")
print(f"  val   : {len(df_val)}")
print(f"  test  : {len(df_test)}")
print(f"  duda  : {len(df_duda)}")
print(f"  TOTAL : {len(df_train) + len(df_val) + len(df_test) + len(df_duda)}")
