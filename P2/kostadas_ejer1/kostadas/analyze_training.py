# analyze_training.py
"""
Script para analizar en detalle los resultados del entrenamiento NEAT.
Genera gráficas adicionales y estadísticas avanzadas.
"""

import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_training_data(results_dir='results_neat'):
    """Carga los datos del entrenamiento"""
    
    stats_path = os.path.join(results_dir, 'generation_stats.pkl')
    genome_path = os.path.join(results_dir, 'best_genome.pkl')
    
    if not os.path.exists(stats_path):
        print(f"❌ No se encuentra {stats_path}")
        return None, None
    
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    
    genome = None
    if os.path.exists(genome_path):
        with open(genome_path, 'rb') as f:
            genome = pickle.load(f)
    
    return stats, genome


def plot_fitness_statistics(stats, output_dir='results_neat'):
    """Crea gráficas detalladas de estadísticas de fitness"""
    
    generations = [s['generation'] for s in stats]
    max_fit = [s['max_fitness'] for s in stats]
    mean_fit = [s['mean_fitness'] for s in stats]
    min_fit = [s['min_fitness'] for s in stats]
    std_fit = [s['std_fitness'] for s in stats]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Análisis Detallado del Entrenamiento NEAT', 
                fontsize=16, fontweight='bold')
    
    # 1. Fitness con desviación estándar
    ax = axes[0, 0]
    ax.plot(generations, max_fit, 'g-', linewidth=2, label='Máximo', marker='o')
    ax.plot(generations, mean_fit, 'b-', linewidth=2, label='Media', marker='s')
    ax.plot(generations, min_fit, 'r-', linewidth=2, label='Mínimo', marker='^')
    
    # Banda de desviación estándar
    mean_array = np.array(mean_fit)
    std_array = np.array(std_fit)
    ax.fill_between(generations, 
                     mean_array - std_array, 
                     mean_array + std_array,
                     alpha=0.3, color='blue', label='±1 σ')
    
    ax.set_xlabel('Generación', fontsize=11)
    ax.set_ylabel('Fitness', fontsize=11)
    ax.set_title('Fitness con Desviación Estándar', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 2. Mejora generacional (delta fitness)
    ax = axes[0, 1]
    max_delta = np.diff([0] + max_fit)
    mean_delta = np.diff([0] + mean_fit)
    
    ax.bar(generations, max_delta, alpha=0.7, color='green', label='Δ Máximo')
    ax.bar(generations, mean_delta, alpha=0.7, color='blue', label='Δ Media')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    
    ax.set_xlabel('Generación', fontsize=11)
    ax.set_ylabel('Mejora en Fitness', fontsize=11)
    ax.set_title('Mejora Generacional', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 3. Evolución de la diversidad (std)
    ax = axes[1, 0]
    ax.plot(generations, std_fit, 'purple', linewidth=2.5, marker='o')
    ax.fill_between(generations, 0, std_fit, alpha=0.3, color='purple')
    
    ax.set_xlabel('Generación', fontsize=11)
    ax.set_ylabel('Desviación Estándar del Fitness', fontsize=11)
    ax.set_title('Diversidad de la Población', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Anotar momentos de alta diversidad
    high_diversity_threshold = np.percentile(std_fit, 75)
    for i, (gen, std) in enumerate(zip(generations, std_fit)):
        if std > high_diversity_threshold and i % 5 == 0:
            ax.annotate(f'Gen {gen}', xy=(gen, std),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.7)
    
    # 4. Número de especies por generación
    ax = axes[1, 1]
    num_species = [s['num_species'] for s in stats]
    
    colors_species = ['green' if n > 1 else 'red' for n in num_species]
    ax.bar(generations, num_species, color=colors_species, alpha=0.7)
    ax.axhline(np.mean(num_species), color='blue', linewidth=2, 
              linestyle='--', label=f'Media: {np.mean(num_species):.1f}')
    
    ax.set_xlabel('Generación', fontsize=11)
    ax.set_ylabel('Número de Especies', fontsize=11)
    ax.set_title('Especiación a lo Largo del Tiempo', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'analisis_detallado.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfica guardada: {filepath}")
    plt.close()


def analyze_genome_complexity(genome, output_dir='results_neat'):
    """Analiza la complejidad del mejor genoma"""
    
    # Contar elementos
    num_connections = len([c for c in genome.connections.values() if c.enabled])
    num_nodes = len(genome.nodes)
    
    # Analizar pesos
    weights = [c.weight for c in genome.connections.values() if c.enabled]
    
    if not weights:
        print("⚠️ No hay conexiones habilitadas en el genoma")
        return
    
    # Crear figura
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Análisis del Mejor Genoma', fontsize=14, fontweight='bold')
    
    # 1. Distribución de pesos
    ax = axes[0]
    ax.hist(weights, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(weights), color='red', linewidth=2, 
              linestyle='--', label=f'Media: {np.mean(weights):.2f}')
    ax.axvline(np.median(weights), color='green', linewidth=2, 
              linestyle='--', label=f'Mediana: {np.median(weights):.2f}')
    
    ax.set_xlabel('Peso de la Conexión', fontsize=11)
    ax.set_ylabel('Frecuencia', fontsize=11)
    ax.set_title('Distribución de Pesos', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Información del genoma
    ax = axes[1]
    ax.axis('off')
    
    info_text = f"""
ESTRUCTURA DEL MEJOR GENOMA
{'='*40}
Fitness: {genome.fitness:.2f}

Complejidad:
  • Nodos: {num_nodes}
  • Conexiones activas: {num_connections}
  
Estadísticas de Pesos:
  • Media: {np.mean(weights):.3f}
  • Mediana: {np.median(weights):.3f}
  • Desv. Estándar: {np.std(weights):.3f}
  • Mínimo: {min(weights):.3f}
  • Máximo: {max(weights):.3f}
  
Pesos por magnitud:
  • Fuertes (|w| > 2): {sum(abs(w) > 2 for w in weights)}
  • Medios (1 < |w| ≤ 2): {sum(1 < abs(w) <= 2 for w in weights)}
  • Débiles (|w| ≤ 1): {sum(abs(w) <= 1 for w in weights)}
  
Pesos por signo:
  • Positivos: {sum(w > 0 for w in weights)}
  • Negativos: {sum(w < 0 for w in weights)}
{'='*40}
"""
    
    ax.text(0.1, 0.95, info_text, transform=ax.transAxes, 
           fontsize=10, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'analisis_genoma.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfica guardada: {filepath}")
    plt.close()


def print_summary_statistics(stats, genome):
    """Imprime estadísticas resumidas en consola"""
    
    print("\n" + "=" * 60)
    print("RESUMEN ESTADÍSTICO DEL ENTRENAMIENTO")
    print("=" * 60)
    
    # Fitness
    max_fits = [s['max_fitness'] for s in stats]
    mean_fits = [s['mean_fitness'] for s in stats]
    
    print(f"\n📊 FITNESS:")
    print(f"   Fitness final (mejor): {max_fits[-1]:.2f}")
    print(f"   Fitness inicial: {max_fits[0]:.2f}")
    print(f"   Mejora total: {max_fits[-1] - max_fits[0]:.2f} "
          f"({((max_fits[-1]/max_fits[0])-1)*100:.1f}%)")
    print(f"   Mejor fitness alcanzado: {max(max_fits):.2f} "
          f"(Gen {max_fits.index(max(max_fits))})")
    
    # Convergencia
    print(f"\n⚡ CONVERGENCIA:")
    # Detectar estancamientos
    stagnant_periods = []
    current_stagnant = 0
    for i in range(1, len(max_fits)):
        if abs(max_fits[i] - max_fits[i-1]) < 0.1:
            current_stagnant += 1
        else:
            if current_stagnant >= 3:
                stagnant_periods.append(current_stagnant)
            current_stagnant = 0
    
    if stagnant_periods:
        print(f"   Periodos de estancamiento: {len(stagnant_periods)}")
        print(f"   Estancamiento más largo: {max(stagnant_periods)} generaciones")
    else:
        print(f"   Sin periodos significativos de estancamiento")
    
    # Diversidad
    print(f"\n🧬 DIVERSIDAD:")
    num_species = [s['num_species'] for s in stats]
    print(f"   Especies iniciales: {num_species[0]}")
    print(f"   Especies finales: {num_species[-1]}")
    print(f"   Máximo de especies: {max(num_species)} "
          f"(Gen {num_species.index(max(num_species))})")
    print(f"   Promedio de especies: {np.mean(num_species):.1f}")
    
    # Genoma
    if genome:
        print(f"\n🧠 MEJOR GENOMA:")
        num_connections = len([c for c in genome.connections.values() if c.enabled])
        num_nodes = len(genome.nodes)
        print(f"   Nodos: {num_nodes}")
        print(f"   Conexiones: {num_connections}")
        print(f"   Ratio conexiones/nodo: {num_connections/num_nodes:.2f}")
    
    print("=" * 60 + "\n")


def main():
    """Función principal"""
    
    print("=" * 60)
    print("ANÁLISIS AVANZADO DEL ENTRENAMIENTO NEAT")
    print("=" * 60 + "\n")
    
    RESULTS_DIR = 'results_neat'
    
    # Cargar datos
    print("Cargando datos del entrenamiento...")
    stats, genome = load_training_data(RESULTS_DIR)
    
    if stats is None:
        print("❌ No se pudieron cargar los datos")
        print("Asegúrate de haber ejecutado train_neat.py primero")
        return
    
    print(f"✓ Datos cargados: {len(stats)} generaciones\n")
    
    # Imprimir estadísticas
    print_summary_statistics(stats, genome)
    
    # Crear gráficas
    print("Generando gráficas de análisis...")
    plot_fitness_statistics(stats, RESULTS_DIR)
    
    if genome:
        analyze_genome_complexity(genome, RESULTS_DIR)
    
    print(f"\n{'=' * 60}")
    print(f"✅ ANÁLISIS COMPLETADO")
    print(f"✅ Gráficas guardadas en: {RESULTS_DIR}/")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
