# Archivo: plot3.py
"""
Gráficas ligeras a partir del Monitor de SB3.
Genera PNGs en plots/ y no abre ventanas (sin show).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend no interactivo
import matplotlib.pyplot as plt

LOGS_DIR = os.getenv("LOGS_DIR", "logs")
MODEL_NAME = os.getenv("MODEL_NAME", "ppo_robobo_v2")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "plots")

os.makedirs(OUTPUT_DIR, exist_ok=True)
monitor_file = os.path.join(LOGS_DIR, f"{MODEL_NAME}_monitor.monitor.csv")

if not os.path.exists(monitor_file):
    print(f"No se encontró {monitor_file}")
    if os.path.isdir(LOGS_DIR):
        print("Archivos en logs:", ", ".join(os.listdir(LOGS_DIR)))
    raise SystemExit(1)

try:
    df = pd.read_csv(monitor_file, skiprows=1)
except Exception as e:
    print(f"Error leyendo monitor: {e}")
    raise SystemExit(1)

if len(df) == 0:
    print("Monitor vacío, nada que graficar.")
    raise SystemExit(0)

# --- Recompensa por episodio ---
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(df.index, df["r"], alpha=0.35, label="Recompensa")
window = min(50, max(5, len(df)//10))
if window > 1:
    ax.plot(df.index, df["r"].rolling(window=window).mean(),
            linewidth=2, label=f"Media móvil ({window})")
ax.set_xlabel("Episodio"); ax.set_ylabel("Recompensa")
ax.set_title("Evolución de Recompensa"); ax.grid(True, alpha=0.3); ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "recompensa_por_episodio.png"),
                                dpi=180, bbox_inches="tight"); plt.close()

# --- Duración ---
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(df.index, df["l"], alpha=0.35, label="Duración (pasos)")
if window > 1:
    ax.plot(df.index, df["l"].rolling(window=window).mean(),
            linewidth=2, label=f"Media móvil ({window})")
ax.set_xlabel("Episodio"); ax.set_ylabel("Pasos")
ax.set_title("Duración de Episodios"); ax.grid(True, alpha=0.3); ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "duracion_episodios.png"),
                                dpi=180, bbox_inches="tight"); plt.close()

# --- Resumen compacto ---
n_episodes = len(df)
means = []
quartiles = np.array_split(df, min(4, n_episodes))
for q in quartiles:
    means.append(float(q["r"].mean()))

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar([f"Q{i+1}" for i in range(len(means))], means, alpha=0.85)
ax.set_title("Recompensa media por cuartil"); ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "resumen_entrenamiento.png"),
                                dpi=180, bbox_inches="tight"); plt.close()

print("Gráficas guardadas en:", OUTPUT_DIR)
