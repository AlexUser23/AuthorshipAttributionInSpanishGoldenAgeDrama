# Baselines MFW

Implementación de Classic Delta, Cosine Delta, SVM lineal y un baseline
mayoritario, integrada en AA2 y ejecutada en el entorno compartido `qlora3`.
Las dependencias se incluyen en el `environment.yml` de la raíz del repositorio.
Los resultados conservados en `outputs/baseline_experiments/mfw_baselines/results.json` se obtuvieron en `qlora3`,
como confirma el autor.

## Ejecución

Desde la raíz de AA2:

```sh
conda activate qlora3
python src/baseline2/baselines.py
```

El programa utiliza `data/crossval_12folds_mroberta_tokenized/fold_00.csv`
a `fold_11.csv` y `data/rhythmic_corpus_100plays/*.txt`, y guarda el informe
en `outputs/baseline_experiments/mfw_baselines/results.json`.
Las rutas predeterminadas se resuelven desde la ubicación del script,
independientemente del directorio de ejecución. Para otras ubicaciones:

```sh
python src/baseline2/baselines.py --corpus /ruta/data --output /ruta/results.json
```

Los identificadores y autores proceden de los CSV; nunca se usan como
características predictoras. Se utilizan los mismos datos y particiones
conservados que en los experimentos Transformer.

## Protocolo

- Se evalúan fragmentos originales de los CSV y obras completas originales de
  los TXT en dos experimentos independientes. Las obras heredan la partición
  de sus fragmentos. No se generan nuevas particiones ni fragmentaciones.
- Doce iteraciones: test `i`, validación `(i+1)%12` y entrenamiento en los diez
  folds restantes. Se comprueba que cada obra pertenece a un único fold y autor,
  que no faltan posiciones y que entrenamiento contiene todos los autores.
- Minúsculas y palabras Unicode mediante `\b[^\W\d_]+\b`: se conservan tildes
  y palabras de una letra; se excluyen números y puntuación. No se aplica
  normalización ortográfica ni una lista de palabras funcionales.
- Las 500 palabras con mayor número total de apariciones en entrenamiento,
  sin culling. Los empates se resuelven alfabéticamente. Cada frecuencia se
  divide por **todas** las palabras de la muestra, incluidas las que no están
  entre las MFW.
- Media y desviación poblacional (`ddof=0`) calculadas solo en entrenamiento.
  Se eliminan columnas constantes, sin reemplazarlas por otras palabras.
  Los tres modelos comparten los vectores estandarizados de cada iteración.
- Classic Delta asigna el autor del vecino con menor diferencia absoluta
  media. Cosine Delta usa la menor distancia coseno sobre esos mismos vectores.
  En empates gana la primera fila de entrenamiento (orden de folds y filas CSV;
  orden alfabético de ficheros para obras). Un vector nulo detiene Cosine Delta
  porque su distancia coseno no está definida.
- `SVC(kernel="linear")`: SVM de margen blando con pérdida hinge, intercepto,
  estrategia multiclase one-vs-one, sin ponderación de clases ni probabilidades,
  tolerancia `1e-4`, shrinking activo, caché de 200 MB y sin límite de iteraciones.
  No interviene aleatoriedad. Se prueban `C = 0.1, 1, 10` por accuracy de
  validación, por separado en cada fold y nivel; en empate gana el menor C.
  Se conserva ese modelo, sin reentrenar con validación. La falta de convergencia
  detiene el programa. Los Delta también reservan el fold de validación.
- El baseline mayoritario predice el autor con más muestras **de entrenamiento**:
  fragmentos en un experimento y obras en el otro. En empate gana el primero
  alfabéticamente. No utiliza texto, validación ni test para elegir el autor.
- Test solo se usa para evaluación final. Accuracy y macro-F1 se calculan
  una sola vez sobre todas las predicciones de test reunidas, con los nueve
  autores. No se promedian métricas por fold ni se calcula F1 micro, redundante
  con accuracy en esta clasificación monoclase.

La clasificación de obras completas utiliza su texto íntegro; no es una
agregación de predicciones de fragmentos. Los fragmentos se leen completos,
sin aplicar de nuevo el tokenizador ni el truncamiento de los transformers.
Las MFW pueden incluir palabras de contenido: este código no permite afirmar
que se usan exclusivamente palabras funcionales o que se elimina todo sesgo
temático.

## Resultados y comprobaciones

El informe conservado [results.json](../../outputs/baseline_experiments/mfw_baselines/results.json) contiene los ocho
resultados globales obtenidos en `qlora3`, los C seleccionados
en orden de fold 00 a 11 y el protocolo. Incluye un
SHA-256 del programa y uno del corpus: para cada CSV y después cada TXT, ambos
en orden alfabético, se incorpora la ruta relativa UTF-8 seguida de un byte
nulo y del digest binario SHA-256 de su contenido. Esto identifica los insumos
sin duplicarlos. No se guardan modelos, matrices, distancias ni predicciones.
La ruta de salida del programa integrado es
`outputs/baseline_experiments/mfw_baselines/results.json`. El programa solo
escribe ese archivo tras completar los dos experimentos correctamente.

Comprobaciones numéricas de tokenización, denominador, estandarización,
vocabulario de entrenamiento, distancias, empates, selección de C,
exclusión de validación y baseline mayoritario:

```sh
python -m unittest discover -s src/baseline2 -v
```

El informe original se ha trasladado a la ruta definitiva sin modificar sus bytes.
Su hash del programa corresponde a la versión histórica, anterior a la protección
contra sobrescritura. Si la salida ya existe, el programa exige otra ruta
`--output` o la opción explícita `--overwrite`.

Referencias: [Evert et al. (2017)](https://doi.org/10.1093/llc/fqx023) para Delta
y [documentación de SVC](https://scikit-learn.org/1.7/modules/generated/sklearn.svm.SVC.html)
para el clasificador.
