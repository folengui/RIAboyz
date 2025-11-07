# analyze_training_3.py
"""
Análisis detallado del entrenamiento NEAT para el informe.
Genera estadísticas y tablas para documentación.
"""

import pickle
import numpy as np
from datetime import datetime


def analyze_neat_results(stats_path="results_neat_obstacle_3/generation_stats.pkl",
                         genome_path="results_neat_obstacle_3/best_genome.pkl"):
    """
    Analiza y reporta estadísticas del entrenamiento NEAT.
    """
    print("="*70)
    print("ANÁLISIS DE RESULTADOS NEAT - Escenario 2.3")
    print("="*70 + "\n")
    
    # Cargar datos
    try:
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
        with open(genome_path, 'rb') as f:
            best_genome = pickle.load(f)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("   Ejecuta primero: python train_neat_obstacle_3.py\n")
        return
    
    # 1. INFORMACIÓN GENERAL
    print("1. CONFIGURACIÓN DEL ENTRENAMIENTO")
    print("-" * 70)
    num_generations = len(stats.most_fit_genomes)
    print(f"   Generaciones ejecutadas: {num_generations}")
    print(f"   Población: 20 individuos")
    print(f"   Entradas: 5 (blob_size, pan_angle, ir_front, ir_left, ir_right)")
    print(f"   Salidas: 5 (PAN×2, movimiento×3)")
    print()
    
    # 2. EVOLUCIÓN DEL FITNESS
    print("2. EVOLUCIÓN DEL FITNESS")
    print("-" * 70)
    fitness_values = [g.fitness for g in stats.most_fit_genomes]
    fitness_mean = stats.get_fitness_mean()
    fitness_std = stats.get_fitness_stdev()
    
    print(f"   Fitness inicial: {fitness_values[0]:.2f}")
    print(f"   Fitness final: {fitness_values[-1]:.2f}")
    print(f"   Mejor fitness alcanzado: {max(fitness_values):.2f} (Gen {fitness_values.index(max(fitness_values))})")
    print(f"   Mejora total: {fitness_values[-1] - fitness_values[0]:+.2f} ({(fitness_values[-1]/fitness_values[0] - 1)*100:+.1f}%)")
    
    # Convergencia
    if fitness_values[-1] >= 100:
        first_convergence = next((i for i, f in enumerate(fitness_values) if f >= 100), None)
        print(f"   ✅ Convergió en generación {first_convergence} (fitness > 100)")
    else:
        print(f"   ⚠️ No convergió completamente (fitness final < 100)")
    print()
    
    # 3. COMPLEJIDAD DE LA RED
    print("3. COMPLEJIDAD DEL MEJOR GENOMA")
    print("-" * 70)
    print(f"   Fitness: {best_genome.fitness:.2f}")
    print(f"   Nodos totales: {len(best_genome.nodes)}")
    print(f"     - Nodos ocultos: {len(best_genome.nodes) - 5}")  # 5 entradas
    print(f"   Conexiones totales: {len(best_genome.connections)}")
    
    active_connections = sum(1 for c in best_genome.connections.values() if c.enabled)
    print(f"   Conexiones activas: {active_connections}")
    print(f"   Conexiones deshabilitadas: {len(best_genome.connections) - active_connections}")
    
    # Estadísticas de pesos
    weights = [c.weight for c in best_genome.connections.values() if c.enabled]
    if weights:
        print(f"\n   Pesos de conexiones:")
        print(f"     - Promedio: {np.mean(weights):.3f}")
        print(f"     - Desv. estándar: {np.std(weights):.3f}")
        print(f"     - Rango: [{min(weights):.3f}, {max(weights):.3f}]")
    print()
    
    # 4. ESPECIACIÓN
    print("4. ESPECIACIÓN")
    print("-" * 70)
    species_sizes = stats.get_species_sizes()
    num_species_per_gen = [len(sizes) for sizes in species_sizes]
    
    print(f"   Especies iniciales: {num_species_per_gen[0]}")
    print(f"   Especies finales: {num_species_per_gen[-1]}")
    print(f"   Máximo de especies: {max(num_species_per_gen)} (Gen {num_species_per_gen.index(max(num_species_per_gen))})")
    print(f"   Promedio de especies: {np.mean(num_species_per_gen):.1f}")
    print()
    
    # 5. TABLA RESUMEN POR GENERACIÓN (cada 10 gen)
    print("5. PROGRESO DEL ENTRENAMIENTO (cada 10 generaciones)")
    print("-" * 70)
    print(f"{'Gen':<6} {'Mejor Fit':<12} {'Fit Medio':<12} {'Especies':<10} {'Nodos':<8} {'Conex':<8}")
    print("-" * 70)
    
    for i in range(0, num_generations, 10):
        gen = i
        best_fit = fitness_values[i]
        mean_fit = fitness_mean[i]
        species = num_species_per_gen[i]
        nodes = len(stats.most_fit_genomes[i].nodes)
        conns = sum(1 for c in stats.most_fit_genomes[i].connections.values() if c.enabled)
        
        print(f"{gen:<6} {best_fit:<12.2f} {mean_fit:<12.2f} {species:<10} {nodes:<8} {conns:<8}")
    
    # Última generación
    if (num_generations - 1) % 10 != 0:
        gen = num_generations - 1
        best_fit = fitness_values[-1]
        mean_fit = fitness_mean[-1]
        species = num_species_per_gen[-1]
        nodes = len(best_genome.nodes)
        conns = active_connections
        print(f"{gen:<6} {best_fit:<12.2f} {mean_fit:<12.2f} {species:<10} {nodes:<8} {conns:<8}")
    
    print()
    
    # 6. RECOMENDACIONES
    print("6. RECOMENDACIONES")
    print("-" * 70)
    
    if fitness_values[-1] < 80:
        print("   ⚠️ Fitness bajo. Sugerencias:")
        print("      - Aumentar generaciones (150-200)")
        print("      - Aumentar población (30-40)")
        print("      - Revisar diseño de fitness")
    elif fitness_values[-1] < 100:
        print("   ⚠️ Convergencia parcial. Sugerencias:")
        print("      - Entrenar 20-30 generaciones más")
        print("      - Ajustar fitness_threshold a valor actual")
    else:
        print("   ✅ Entrenamiento exitoso")
        print("      - Fitness suficiente para validación")
        print("      - Proceder con validate_hybrid_3.py")
    
    print()
    print("="*70 + "\n")


def print_latex_table(stats_path="results_neat_obstacle_3/generation_stats.pkl"):
    """
    Genera tabla en formato LaTeX para el informe.
    """
    print("\n" + "="*70)
    print("TABLA LATEX PARA INFORME")
    print("="*70 + "\n")
    
    try:
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
    except FileNotFoundError:
        print("❌ No se encontraron estadísticas")
        return
    
    fitness_values = [g.fitness for g in stats.most_fit_genomes]
    fitness_mean = stats.get_fitness_mean()
    species_sizes = stats.get_species_sizes()
    num_species_per_gen = [len(sizes) for sizes in species_sizes]
    num_generations = len(fitness_values)
    
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{|c|c|c|c|}")
    print("\\hline")
    print("\\textbf{Generación} & \\textbf{Mejor Fitness} & \\textbf{Fitness Medio} & \\textbf{Especies} \\\\")
    print("\\hline")
    
    for i in range(0, num_generations, 20):
        print(f"{i} & {fitness_values[i]:.2f} & {fitness_mean[i]:.2f} & {num_species_per_gen[i]} \\\\")
    
    if (num_generations - 1) % 20 != 0:
        i = num_generations - 1
        print(f"{i} & {fitness_values[i]:.2f} & {fitness_mean[i]:.2f} & {num_species_per_gen[i]} \\\\")
    
    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{Evolución del entrenamiento NEAT}")
    print("\\label{tab:neat_training}")
    print("\\end{table}")
    print()


if __name__ == "__main__":
    analyze_neat_results()
    
    # Descomentar si necesitas tabla LaTeX
    # print_latex_table()
