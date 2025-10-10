# Archivo: plot_training_results.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ==================== CONFIGURACIÓN ====================
LOGS_DIR = "logs/"
MODEL_NAME = "ppo_robobo_v2"
OUTPUT_DIR = "plots/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== LEER DATOS DEL MONITOR ====================
monitor_file = os.path.join(LOGS_DIR, f"{MODEL_NAME}_monitor.monitor.csv")

print("="*60)
print("GENERANDO GRÁFICAS DE ENTRENAMIENTO")
print("="*60)

if not os.path.exists(monitor_file):
    print(f"\n❌ No se encontró el archivo: {monitor_file}")
    print(f"\n💡 Buscando archivos en {LOGS_DIR}...")
    
    if os.path.exists(LOGS_DIR):
        print(f"\n📁 Archivos disponibles:")
        for f in os.listdir(LOGS_DIR):
            print(f"   - {f}")
    else:
        print(f"   ❌ El directorio {LOGS_DIR} no existe")
    
    print("\n⚠️  Asegúrate de haber ejecutado train_improved.py primero")
    exit(1)

# Leer CSV (saltar primera línea que es metadata)
try:
    df = pd.read_csv(monitor_file, skiprows=1)
    print(f"\n✅ Datos cargados: {len(df)} episodios")
    print(f"Columnas: {df.columns.tolist()}")
except Exception as e:
    print(f"\n❌ Error al leer el archivo CSV: {e}")
    exit(1)

# Verificar que hay suficientes datos
if len(df) < 5:
    print(f"\n⚠️  Advertencia: Solo {len(df)} episodios. Se necesitan más datos para gráficas significativas.")

# ==================== GRÁFICA 1: RECOMPENSA POR EPISODIO ====================
print("\nGenerando gráfica de recompensas...")
fig, ax = plt.subplots(figsize=(12, 6))

# Recompensa por episodio
ax.plot(df.index, df['r'], alpha=0.3, label='Recompensa por episodio', color='blue')

# Media móvil (ventana adaptativa)
window = min(50, max(5, len(df) // 10))
if window > 1:
    rolling_mean = df['r'].rolling(window=window, center=False).mean()
    ax.plot(df.index, rolling_mean, label=f'Media móvil ({window} eps)', 
            color='red', linewidth=2)

ax.set_xlabel('Episodio', fontsize=12)
ax.set_ylabel('Recompensa Total', fontsize=12)
ax.set_title('Evolución de la Recompensa durante el Entrenamiento', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'recompensa_por_episodio.png'), dpi=300, bbox_inches='tight')
print(f"✅ Guardado: {OUTPUT_DIR}recompensa_por_episodio.png")
plt.close()

# ==================== GRÁFICA 2: DURACIÓN DE EPISODIOS ====================
print("Generando gráfica de duración...")
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(df.index, df['l'], alpha=0.3, label='Duración (pasos)', color='green')

if window > 1:
    rolling_mean = df['l'].rolling(window=window, center=False).mean()
    ax.plot(df.index, rolling_mean, label=f'Media móvil ({window} eps)', 
            color='darkgreen', linewidth=2)

ax.set_xlabel('Episodio', fontsize=12)
ax.set_ylabel('Duración (pasos)', fontsize=12)
ax.set_title('Duración de los Episodios', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'duracion_episodios.png'), dpi=300, bbox_inches='tight')
print(f"✅ Guardado: {OUTPUT_DIR}duracion_episodios.png")
plt.close()

# ==================== GRÁFICA 3: ESTADÍSTICAS FINALES ====================
print("Generando resumen estadístico...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Estadísticas por cuartiles
n_episodes = len(df)
n_quartiles = min(4, n_episodes)  # Usar menos cuartiles si hay pocos datos
quartiles = np.array_split(df, n_quartiles)

quartile_stats = []
for i, q in enumerate(quartiles):
    quartile_stats.append({
        'Cuartil': f'{i+1}',
        'Media': q['r'].mean(),
        'Std': q['r'].std(),
        'Max': q['r'].max(),
        'Min': q['r'].min()
    })

quartile_df = pd.DataFrame(quartile_stats)

# a) Histograma de recompensas
bins = min(30, max(10, len(df) // 10))
axes[0, 0].hist(df['r'], bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
axes[0, 0].axvline(df['r'].mean(), color='red', linestyle='--', linewidth=2, 
                    label=f'Media: {df["r"].mean():.2f}')
axes[0, 0].set_xlabel('Recompensa Total', fontsize=10)
axes[0, 0].set_ylabel('Frecuencia', fontsize=10)
axes[0, 0].set_title('Distribución de Recompensas', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# b) Boxplot por cuartiles
if n_quartiles > 1:
    axes[0, 1].boxplot([q['r'].values for q in quartiles], 
                        labels=[f'Q{i+1}' for i in range(n_quartiles)])
    axes[0, 1].set_xlabel('Cuartil del Entrenamiento', fontsize=10)
    axes[0, 1].set_ylabel('Recompensa', fontsize=10)
    axes[0, 1].set_title('Recompensa por Cuartil de Entrenamiento', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
else:
    axes[0, 1].text(0.5, 0.5, 'Pocos datos\npara cuartiles', 
                    ha='center', va='center', fontsize=12)
    axes[0, 1].set_title('Recompensa por Cuartil', fontweight='bold')

# c) Mejora progresiva
colors = ['#ff9999', '#ffcc99', '#99ccff', '#99ff99'][:n_quartiles]
axes[1, 0].bar(quartile_df['Cuartil'], quartile_df['Media'], 
               alpha=0.7, color=colors)
axes[1, 0].set_xlabel('Cuartil', fontsize=10)
axes[1, 0].set_ylabel('Recompensa Media', fontsize=10)
axes[1, 0].set_title('Mejora Progresiva por Cuartil', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# d) Tabla de resumen
axes[1, 1].axis('off')

mejora = quartile_df.iloc[-1]["Media"] - quartile_df.iloc[0]["Media"] if n_quartiles > 1 else 0

table_data = [
    ['Métrica', 'Valor'],
    ['Episodios totales', f'{n_episodes}'],
    ['Recompensa media', f'{df["r"].mean():.2f} ± {df["r"].std():.2f}'],
    ['Recompensa máxima', f'{df["r"].max():.2f}'],
    ['Recompensa mínima', f'{df["r"].min():.2f}'],
    ['Duración media', f'{df["l"].mean():.1f} pasos'],
    ['Mejora Q1→Q{}'.format(n_quartiles), f'{mejora:.2f}'],
]
table = axes[1, 1].table(cellText=table_data, cellLoc='left', loc='center',
                         colWidths=[0.5, 0.5])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Estilo para la tabla
for i, cell in table.get_celld().items():
    if i[0] == 0:  # Header
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white')
    else:
        cell.set_facecolor('#f0f0f0' if i[0] % 2 == 0 else 'white')

plt.suptitle('Resumen del Entrenamiento', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig(os.path.join(OUTPUT_DIR, 'resumen_entrenamiento.png'), dpi=300, bbox_inches='tight')
print(f"✅ Guardado: {OUTPUT_DIR}resumen_entrenamiento.png")
plt.close()

# ==================== ESTADÍSTICAS EN CONSOLA ====================
print(f"\n{'='*60}")
print(f"RESUMEN DEL ENTRENAMIENTO")
print(f"{'='*60}")
print(f"📊 Episodios totales: {n_episodes}")
print(f"🎯 Recompensa media: {df['r'].mean():.2f} ± {df['r'].std():.2f}")
print(f"📈 Recompensa máxima: {df['r'].max():.2f}")
print(f"📉 Recompensa mínima: {df['r'].min():.2f}")
print(f"⏱️  Duración media: {df['l'].mean():.1f} pasos")
if n_quartiles > 1:
    print(f"📊 Mejora Q1→Q{n_quartiles}: {mejora:.2f}")

# Análisis de aprendizaje
print(f"\n{'='*60}")
print("ANÁLISIS DE APRENDIZAJE")
print(f"{'='*60}")

if n_episodes < 50:
    print("⚠️  Pocos episodios para análisis concluyente")
elif mejora > 0:
    print("✅ El modelo muestra MEJORA progresiva")
    if mejora > 5:
        print("   👍 Mejora significativa - buen progreso")
    else:
        print("   📊 Mejora leve - considera más entrenamiento")
else:
    print("⚠️  Sin mejora clara - revisa hiperparámetros o recompensas")

# Recomendaciones
if df['r'].mean() < 0:
    print("\n💡 Recomendación: Recompensa media negativa")
    print("   - Considera recompensas más generosas")
    print("   - O más timesteps de entrenamiento")
elif df['r'].mean() < 5:
    print("\n💡 Recomendación: Aprendizaje inicial")
    print("   - Aumenta timesteps para mejor resultado")
    print("   - El modelo está comenzando a aprender")
else:
    print("\n✅ El modelo ha aprendido comportamientos útiles")

print(f"{'='*60}\n")

print("📊 Todas las gráficas guardadas en:", OUTPUT_DIR)
print("\n💡 Para visualizar métricas detalladas en TensorBoard:")
print(f"   tensorboard --logdir {LOGS_DIR}")

plt.show()