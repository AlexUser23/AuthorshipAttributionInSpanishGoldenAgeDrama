"""Representa las predicciones guardadas; ejecutar con python3 este archivo."""
import csv
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib-duda')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
COLORS = {'Alarcón': '#d7191c', 'Belmonte': '#009e73',
          'Claramonte': '#f0e442', 'Lope': '#0072b2', 'Tirso': '#cc79a7'}


def main():
    runs = []
    for seed in range(1, 6):
        with (ROOT / f'seed_{seed}/duda_predictions.csv').open(encoding='utf-8-sig', newline='') as f:
            rows = sorted(csv.DictReader(f), key=lambda r: int(r['posicion']))
        assert len({r['posicion'] for r in rows}) == len(rows), 'Posiciones duplicadas'
        assert all(r['predicted_autor'] in COLORS for r in rows)
        runs.append(rows)
    reference = [(r['comedia'], r['posicion'], r['texto'], r['num_tokens']) for r in runs[0]]
    assert reference and len({r[0] for r in reference}) == 1
    for rows in runs[1:]:
        assert [(r['comedia'], r['posicion'], r['texto'], r['num_tokens']) for r in rows] == reference, 'Fragmentos no alineados'
    n = len(reference)
    votes = [Counter(rows[i]['predicted_autor'] for rows in runs) for i in range(n)]
    winners = [[a for a in COLORS if v[a] == max(v.values())] for v in votes]
    title = reference[0][0].removeprefix('DUDA_')

    def draw(ax, distributions, label):
        for i, parts in enumerate(distributions):
            x = i
            for author, fraction in parts:
                ax.add_patch(Rectangle((x, 0), fraction, 1, facecolor=COLORS[author], edgecolor='none'))
                x += fraction
            ax.add_patch(Rectangle((i, 0), 1, 1, fill=False, edgecolor='white', linewidth=1))
        ax.set(xlim=(0, n), ylim=(-.05, 1.05), yticks=[])
        ax.set_ylabel(label, rotation=0, ha='right', va='center', labelpad=12, fontsize=10)
        ax.set_xticks([i + .5 for i in range(n)])
        ax.set_xticklabels([str(i + 1) for i in range(n)], fontsize=8)
        ax.tick_params(axis='x', length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    def save(fig, name):
        for ext in ('png', 'svg', 'pdf'):
            fig.savefig(OUT / f'{name}.{ext}', dpi=200, bbox_inches='tight', facecolor='white')
        plt.close(fig)

    legend = [Patch(facecolor=c, label=a) for a, c in COLORS.items()]
    fig, axes = plt.subplots(7, 1, figsize=(17, 7.5))
    for seed, rows in enumerate(runs, 1):
        parts = [[(r['predicted_autor'], 1)] for r in rows]
        draw(axes[seed-1], parts, f'Semilla {seed}')
        single, ax = plt.subplots(figsize=(17, 2.5))
        draw(ax, parts, f'Semilla {seed}')
        single.suptitle(title, fontsize=15)
        single.legend(handles=legend, loc='lower center', ncol=5, frameon=False)
        single.subplots_adjust(left=.09, right=.99, top=.7, bottom=.4)
        save(single, f'linea_seed_{seed}')
    # Media de indicadores de clase: proporción de votos de cada autor.
    mean_parts = [[(a, v[a]/5) for a in COLORS if v[a]] for v in votes]
    draw(axes[5], mean_parts, 'Media de votos')
    draw(axes[6], [[(a, 1/len(w)) for a in w] for w in winners], 'Más votado')
    for i, w in enumerate(winners):
        if len(w) > 1:
            axes[6].text(i+.5, .5, '=', ha='center', va='center', color='white', weight='bold')
    axes[-1].set_xlabel('Fragmento en orden textual (1–49; posición original en CSV: 0–48)', labelpad=12)
    fig.suptitle(f'{title}\nAsignación por fragmento y semilla', fontsize=17, y=.98)
    fig.legend(handles=legend, loc='lower center', bbox_to_anchor=(.55, .045), ncol=5, frameon=False)
    fig.text(.55, .015, 'Media: cada color ocupa su proporción de votos entre las 5 semillas.  Más votado: = indica empate.', ha='center', fontsize=10)
    fig.subplots_adjust(left=.11, right=.99, top=.87, bottom=.2, hspace=.9)
    save(fig, 'lineas_fragmentos_5_semillas_media')
    single, ax = plt.subplots(figsize=(17, 2.8))
    draw(ax, mean_parts, 'Media de votos')
    single.suptitle(f'{title} — media de las 5 semillas', fontsize=15)
    single.legend(handles=legend, loc='lower center', bbox_to_anchor=(.55, .08), ncol=5, frameon=False)
    single.text(.55, .025, 'Cada fragmento muestra la proporción de votos por autor; no son probabilidades de autoría.', ha='center', fontsize=10)
    single.subplots_adjust(left=.11, right=.99, top=.72, bottom=.43)
    save(single, 'linea_media')
    with (OUT / 'asignaciones_por_fragmento.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['fragmento', 'posicion_original', 'num_tokens'] + [f'seed_{s}' for s in range(1, 6)] + [f'proporcion_{a}' for a in COLORS] + ['autores_mas_votados', 'acuerdo_maximo'])
        for i, v in enumerate(votes):
            writer.writerow([i+1, reference[i][1], reference[i][3]] + [rows[i]['predicted_autor'] for rows in runs] + [v[a]/5 for a in COLORS] + [' / '.join(winners[i]), max(v.values())/5])
    (OUT / 'README.txt').write_text(
        'Fuente: seed_1...seed_5/duda_predictions.csv; campo predicted_autor.\n'
        'Fragmentos ordenados por posicion, alineados y comprobados por texto entre semillas.\n'
        'Cada segmento tiene igual ancho: representa un fragmento, no longitud exacta en tokens.\n'
        'Numeración visual 1–49, equivalente a posiciones originales 0–48.\n'
        'Media de votos: media de los indicadores de clase (votos / 5) por fragmento.\n'
        'Las subdivisiones de colores en la media representan votos, no subfragmentos textuales.\n'
        'Más votado: moda por fragmento; empates con colores compartidos y signo =.\n'
        'Los votos no representan probabilidades de autoría.\n'
        'Formatos PNG, SVG y PDF; CSV con asignaciones y proporciones.\n'
        'Reproducir: python3 generar_visualizacion.py (requiere matplotlib).\n', encoding='utf-8')
    print(f'Generados gráficos para {n} fragmentos y 5 semillas en {OUT}')
    print('Empates:', [i+1 for i, w in enumerate(winners) if len(w)>1])


if __name__ == '__main__':
    main()
