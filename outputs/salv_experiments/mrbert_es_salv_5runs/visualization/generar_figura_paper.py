"""Figura para publicación basada en las proporciones de votos guardadas."""
import csv
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib-duda')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from generar_visualizacion import COLORS

OUT = Path(__file__).resolve().parent
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
})
with (OUT / 'asignaciones_por_fragmento.csv').open(encoding='utf-8', newline='') as f:
    rows = sorted(csv.DictReader(f), key=lambda r: int(r['fragmento']))

# Ancho de 180 mm, adecuado para una figura a dos columnas.
fig = plt.figure(figsize=(180 / 25.4, 39 / 25.4), facecolor='white')
ax = fig.add_axes([.025, .43, .95, .36])
for i, row in enumerate(rows):
    left = i
    for author, color in COLORS.items():
        fraction = float(row[f'proporcion_{author}'])
        if fraction:
            ax.add_patch(Rectangle((left, 0), fraction, 1,
                                   facecolor=color, edgecolor='none'))
            left += fraction
    ax.plot([i, i], [0, 1], color='white', linewidth=.35)
ax.set(xlim=(0, len(rows)), ylim=(0, 1), yticks=[])
ticks = [1] + list(range(5, len(rows) + 1, 5)) + [len(rows)]
ax.set_xticks([n - .5 for n in ticks], [str(n) for n in ticks])
ax.tick_params(axis='x', length=2, width=.5, pad=3, labelsize=7)
ax.set_xlabel('Fragmento', fontsize=8, labelpad=4)
for spine in ax.spines.values():
    spine.set_visible(False)
fig.legend(handles=[Patch(facecolor=c, label=a) for a, c in COLORS.items()],
           loc='lower center', bbox_to_anchor=(.5, .015), ncol=5,
           frameon=False, fontsize=8, handlelength=1.2, handleheight=.8,
           columnspacing=1.5)
for ext in ('png', 'pdf', 'svg'):
    fig.savefig(OUT / f'linea_media_paper.{ext}', dpi=600, facecolor='white')
plt.close(fig)
(OUT / 'linea_media_paper_caption.txt').write_text(
    'Atribución por fragmento de Siempre ayuda la verdad. Cada bloque representa '
    'uno de los 49 fragmentos, en orden textual; los colores muestran la proporción '
    'de votos de cada autor entre cinco ejecuciones con distintas semillas. '
    'Todos los bloques tienen igual ancho, independientemente de su longitud '
    'textual. Las proporciones corresponden a votos, no a probabilidades de autoría.\n',
    encoding='utf-8')
print('Generados linea_media_paper.png, .pdf, .svg y _caption.txt')
