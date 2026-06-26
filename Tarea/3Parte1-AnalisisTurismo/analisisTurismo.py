import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# CARGA DE DATOS
# ============================================================
# Cargar la muestra de 2000 filas generada en la Parte 1
BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(
    BASE_DIR.parent / "muestra_2000.csv",
    sep=",",
    quotechar='"',
    encoding="latin1"
)

# Mostrar columnas disponibles para identificar las de Gasto
print("=== COLUMNAS DISPONIBLES ===")
print(df.columns.tolist())
print(f"\nShape: {df.shape}")
print(df.head(3))


# IDENTIFICAR COLUMNAS DE GASTO
# ============================================================
gasto_cols = [c for c in df.columns if 'gasto' in c.lower() or 'Gasto' in c]
print(f"\n=== COLUMNAS DE GASTO IDENTIFICADAS ===")
print(gasto_cols)

# Convertir a numérico (algunos valores pueden venir como string)
for col in gasto_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Estadísticos descriptivos de las variables de gasto
print("\n=== ESTADÍSTICOS DESCRIPTIVOS ===")
print(df[gasto_cols].describe().round(2))

# PALETA DE COLORES
# ============================================================
COLORS = ['#1a2744', '#2563eb', '#0ea5e9', '#38bdf8', '#7dd3fc',
          '#93c5fd', '#bfdbfe', '#dbeafe']
NAVY   = '#1a2744'
BLUE   = '#2563eb'
LIGHT  = '#dbeafe'
RED    = '#dc2626'
GREEN  = '#16a34a'

# PARTE 1 — INTERVALOS DE CONFIANZA PARA LAS MEDIAS
# ============================================================
print("\n" + "="*65)
print("PARTE 1 — INTERVALOS DE CONFIANZA AL 95%")
print("="*65)

resultados_ic = {}
confianza = 0.95
alpha = 1 - confianza

for col in gasto_cols:
    datos = df[col].dropna()
    n    = len(datos)
    xbar = datos.mean()
    s    = datos.std(ddof=1)
    se   = s / np.sqrt(n)
    t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
    E    = t_crit * se
    li   = xbar - E
    ls   = xbar + E

    resultados_ic[col] = {
        'n': n, 'media': xbar, 'desvio': s,
        'error_std': se, 't_critico': t_crit,
        'margen_error': E, 'LI': li, 'LS': ls
    }

    print(f"\n--- {col} ---")
    print(f"  n                 = {n}")
    print(f"  Media muestral    = {xbar:.2f}")
    print(f"  Desvío muestral   = {s:.2f}")
    print(f"  Error estándar    = {se:.4f}")
    print(f"  t crítico (gl={n-1}, α/2={alpha/2}) = {t_crit:.4f}")
    print(f"  Margen de error   = {E:.4f}")
    print(f"  IC 95%            = ({li:.2f} ; {ls:.2f})")

# FIGURA 1 — INTERVALOS DE CONFIANZA (gráfico de bosque)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Intervalos de Confianza al 95% — Variables de Gasto',
             fontsize=14, fontweight='bold', color=NAVY, y=1.01)

# ── Subplot izquierdo: Forest plot ──────────────────────────
ax1 = axes[0]
cols_list = list(resultados_ic.keys())
n_vars = len(cols_list)
y_pos  = np.arange(n_vars)

for i, col in enumerate(cols_list):
    r = resultados_ic[col]
    ax1.errorbar(r['media'], i,
                 xerr=r['margen_error'],
                 fmt='o', color=COLORS[i % len(COLORS)],
                 markersize=8, capsize=6, linewidth=2,
                 label=col)
    ax1.text(r['LS'] + 2, i,
             f"({r['LI']:.1f} ; {r['LS']:.1f})",
             va='center', fontsize=8.5, color='#374151')

ax1.set_yticks(y_pos)
ax1.set_yticklabels([c.replace('Gasto', 'Gasto\n') for c in cols_list],
                    fontsize=9)
ax1.set_xlabel('USD', fontsize=10)
ax1.set_title('Forest Plot — Medias e IC 95%', fontsize=11,
              fontweight='bold', color=NAVY)
ax1.axvline(0, color='gray', linestyle='--', alpha=0.4)
ax1.grid(axis='x', alpha=0.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# ── Subplot derecho: barras con IC ───────────────────────────
ax2 = axes[1]
medias = [resultados_ic[c]['media'] for c in cols_list]
errores = [resultados_ic[c]['margen_error'] for c in cols_list]
labels_short = [c.replace('Gasto_', '').replace('Gasto', '') for c in cols_list]

bars = ax2.bar(labels_short, medias, color=COLORS[:n_vars],
               edgecolor='white', linewidth=1.5, alpha=0.85)
ax2.errorbar(labels_short, medias, yerr=errores,
             fmt='none', color=NAVY, capsize=6, linewidth=2)

for bar, m, e in zip(bars, medias, errores):
    ax2.text(bar.get_x() + bar.get_width()/2,
             m + e + max(medias)*0.01,
             f'{m:.1f}', ha='center', va='bottom',
             fontsize=8.5, fontweight='bold', color=NAVY)

ax2.set_ylabel('Gasto promedio (USD)', fontsize=10)
ax2.set_title('Gasto Promedio con Margen de Error (IC 95%)',
              fontsize=11, fontweight='bold', color=NAVY)
ax2.tick_params(axis='x', rotation=30)
ax2.grid(axis='y', alpha=0.3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(BASE_DIR / 'ic_gastos_Parte1.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[✓] Figura 1 guardada como 'ic_gastos_Parte1.png'")

# PARTE 2 — TEST DE HIPÓTESIS: µ_alojamiento < 350
# ============================================================
print("\n" + "="*65)
print("PARTE 2 — TEST DE HIPÓTESIS PARA GASTO EN ALOJAMIENTO")
print("="*65)

# Identificar columna de alojamiento
col_aloj = [c for c in gasto_cols if 'aloj' in c.lower()]
if not col_aloj:
    col_aloj = [gasto_cols[0]]   # fallback: primera columna de gasto
col_aloj = col_aloj[0]

print(f"\nVariable utilizada: '{col_aloj}'")

datos_aloj = df[col_aloj].dropna()
n_aloj   = len(datos_aloj)
xbar_aloj = datos_aloj.mean()
s_aloj   = datos_aloj.std(ddof=1)
mu0      = 350
alpha_h  = 0.05

# H0: µ >= 350   H1: µ < 350  (test unilateral izquierdo)
t_stat, p_valor_bilateral = stats.ttest_1samp(datos_aloj, mu0)
p_valor = p_valor_bilateral / 2  # unilateral izquierdo
if t_stat > 0:
    p_valor = 1 - p_valor_bilateral / 2

t_crit_h = stats.t.ppf(alpha_h, df=n_aloj-1)   # valor crítico unilateral izq

print(f"\n  H₀: µ_alojamiento ≥ 350 USD")
print(f"  H₁: µ_alojamiento < 350 USD  (unilateral izquierdo)")
print(f"  α = {alpha_h}")
print(f"\n  n                 = {n_aloj}")
print(f"  Media muestral    = {xbar_aloj:.4f}")
print(f"  Desvío muestral   = {s_aloj:.4f}")
print(f"  µ₀                = {mu0}")
print(f"\n  Estadístico T     = {t_stat:.4f}")
print(f"  Valor crítico t   = {t_crit_h:.4f}  (gl={n_aloj-1}, α={alpha_h})")
print(f"  P-valor           = {p_valor:.6f}")
print(f"\n  Región crítica    = T < {t_crit_h:.4f}")
print(f"  ¿T cae en RC?     = {t_stat < t_crit_h}")
print(f"  ¿P-valor < α?     = {p_valor < alpha_h}")

if p_valor < alpha_h:
    conclusion = (f"Se RECHAZA H₀. Con un nivel de significación del 5%, "
                  f"hay evidencia estadística suficiente para afirmar que "
                  f"los uruguayos gastan en promedio MENOS de $350 USD "
                  f"en alojamiento cuando viajan al exterior.")
else:
    conclusion = (f"No se rechaza H₀. Con un nivel de significación del 5%, "
                  f"no hay evidencia estadística suficiente para afirmar que "
                  f"los uruguayos gastan en promedio menos de $350 USD "
                  f"en alojamiento cuando viajan al exterior.")

print(f"\n  ► CONCLUSIÓN: {conclusion}")

# FIGURA 2 — VISUALIZACIÓN DEL TEST DE HIPÓTESIS
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f'Test de Hipótesis — Gasto en Alojamiento < $350 USD  (α = 0.05)',
             fontsize=13, fontweight='bold', color=NAVY)

# ── Subplot izq: distribución t con región crítica ──────────
ax = axes[0]
gl = n_aloj - 1
x  = np.linspace(stats.t.ppf(0.0001, gl), stats.t.ppf(0.9999, gl), 500)
y  = stats.t.pdf(x, gl)

ax.plot(x, y, color=NAVY, linewidth=2, label=f't({gl} gl)')
ax.fill_between(x, y, where=(x < t_crit_h),
                color=RED, alpha=0.35, label=f'Región crítica (T < {t_crit_h:.3f})')
ax.fill_between(x, y, where=(x >= t_crit_h),
                color=LIGHT, alpha=0.6)

ax.axvline(t_stat, color=BLUE, linewidth=2.5, linestyle='-',
           label=f'T observado = {t_stat:.3f}')
ax.axvline(t_crit_h, color=RED, linewidth=1.8, linestyle='--',
           label=f't crítico = {t_crit_h:.3f}')

ax.annotate(f'T = {t_stat:.3f}',
            xy=(t_stat, stats.t.pdf(t_stat, gl)),
            xytext=(t_stat - 1.5, stats.t.pdf(t_stat, gl) + 0.04),
            arrowprops=dict(arrowstyle='->', color=BLUE),
            color=BLUE, fontsize=9, fontweight='bold')

ax.set_xlabel('Estadístico t', fontsize=10)
ax.set_ylabel('Densidad', fontsize=10)
ax.set_title('Distribución t bajo H₀', fontsize=11, fontweight='bold', color=NAVY)
ax.legend(fontsize=8, loc='upper right')
ax.grid(alpha=0.25)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ── Subplot der: histograma de datos + µ0 ───────────────────
ax2 = axes[1]
ax2.hist(datos_aloj, bins=40, color=BLUE, alpha=0.6, edgecolor='white',
         density=True, label='Distribución muestral')
ax2.axvline(xbar_aloj, color=NAVY, linewidth=2.5, linestyle='-',
            label=f'x̄ = {xbar_aloj:.1f}')
ax2.axvline(mu0, color=RED, linewidth=2.5, linestyle='--',
            label=f'µ₀ = {mu0}')

# IC 95% para µ
ic_aloj = resultados_ic.get(col_aloj)
if ic_aloj:
    ax2.axvspan(ic_aloj['LI'], ic_aloj['LS'],
                alpha=0.15, color=GREEN,
                label=f"IC 95%: ({ic_aloj['LI']:.1f}; {ic_aloj['LS']:.1f})")

ax2.set_xlabel('Gasto en Alojamiento (USD)', fontsize=10)
ax2.set_ylabel('Densidad', fontsize=10)
ax2.set_title('Distribución del Gasto en Alojamiento', fontsize=11,
              fontweight='bold', color=NAVY)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.25)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Recuadro de resultado
color_box = '#fef2f2' if p_valor >= alpha_h else '#f0fdf4'
color_txt = RED if p_valor >= alpha_h else GREEN
resultado_txt = "No se rechaza H₀" if p_valor >= alpha_h else "Se rechaza H₀"
ax2.text(0.97, 0.97,
         f"P-valor = {p_valor:.4f}\n{resultado_txt}",
         transform=ax2.transAxes,
         ha='right', va='top', fontsize=9, fontweight='bold',
         color=color_txt,
         bbox=dict(boxstyle='round,pad=0.4', facecolor=color_box,
                   edgecolor=color_txt, alpha=0.9))

plt.tight_layout()
plt.savefig(BASE_DIR / 'test_hipotesis_alojamiento_Parte2.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[✓] Figura 2 guardada como 'test_hipotesis_alojamiento_Parte2.png'")

# ============================================================
# RESUMEN FINAL
# ============================================================
print("\n" + "="*65)
print("RESUMEN EJECUTIVO")
print("="*65)
print("\n[IC 95%]")
for col, r in resultados_ic.items():
    print(f"  {col:35s}: ({r['LI']:.2f} ; {r['LS']:.2f})  |  x̄={r['media']:.2f}")

print(f"\n[Test de hipótesis — {col_aloj}]")
print(f"  H₀: µ ≥ 350   vs   H₁: µ < 350")
print(f"  T = {t_stat:.4f}  |  p-valor = {p_valor:.6f}  |  α = {alpha_h}")
print(f"  Decisión: {'RECHAZAR H₀' if p_valor < alpha_h else 'NO RECHAZAR H₀'}")