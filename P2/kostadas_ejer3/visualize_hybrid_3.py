# visualize_hybrid_3.py
"""
Visualización de resultados del entrenamiento NEAT
y del sistema híbrido completo.
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
import neat
import graphviz


def visualize_neat_training(stats_path="results_neat_obstacle_3/generation_stats.pkl",
                            output_path="results_neat_obstacle_3"):
    """
    Genera gráficas del entrenamiento NEAT.
    """
    print("Generando visualizaciones del entrenamiento NEAT...")
    
    # Cargar estadísticas
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    
    # 1. Evolución del fitness
    generation = range(len(stats.most_fit_genomes))
    best_fitness = [g.fitness for g in stats.most_fit_genomes]
    avg_fitness = stats.get_fitness_mean()
    
    plt.figure(figsize=(12, 6))
    plt.plot(generation, best_fitness, 'b-', linewidth=2, label='Mejor Fitness')
    plt.plot(generation, avg_fitness, 'r--', linewidth=1.5, label='Fitness Promedio')
    plt.xlabel('Generación', fontsize=12)
    plt.ylabel('Fitness', fontsize=12)
    plt.title('Evolución del Fitness - NEAT', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_path}/fitness_evolution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ {output_path}/fitness_evolution.png")
    
    # 2. Estadísticas de genomas
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Número de especies
    species_sizes = stats.get_species_sizes()
    num_species = [len(sizes) for sizes in species_sizes]
    
    ax1.plot(generation, num_species, 'g-', linewidth=2)
    ax1.set_xlabel('Generación', fontsize=12)
    ax1.set_ylabel('Número de Especies', fontsize=12)
    ax1.set_title('Especiación', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Complejidad (nodos y conexiones del mejor)
    best_nodes = [len(g.nodes) for g in stats.most_fit_genomes]
    best_connections = [len(g.connections) for g in stats.most_fit_genomes]
    
    ax2.plot(generation, best_nodes, 'b-', linewidth=2, label='Nodos')
    ax2.plot(generation, best_connections, 'r-', linewidth=2, label='Conexiones')
    ax2.set_xlabel('Generación', fontsize=12)
    ax2.set_ylabel('Cantidad', fontsize=12)
    ax2.set_title('Complejidad del Mejor Genoma', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_path}/genome_stats.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ {output_path}/genome_stats.png")
    
    print("✅ Visualizaciones completadas\n")


def visualize_best_genome(genome_path="results_neat_obstacle_3/best_genome.pkl",
                          config_path="config-neat-obstacle-3",
                          output_path="results_neat_obstacle_3"):
    """
    Visualiza la red neuronal del mejor genoma.
    """
    print("Generando visualización de la red neuronal...")
    
    # Cargar genoma y config
    with open(genome_path, 'rb') as f:
        genome = pickle.load(f)
    
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )
    
    # Crear grafo
    dot = graphviz.Digraph(comment='Red NEAT', format='png')
    dot.attr(rankdir='LR')
    dot.attr('node', shape='circle', style='filled')
    
    # Nombres de nodos
    input_names = ['blob_size', 'pan_angle', 'ir_front', 'ir_left', 'ir_right']
    output_names = ['PAN_izq', 'PAN_der', 'Avanzar', 'Gira_izq', 'Gira_der']
    
    # Nodos de entrada
    for i in range(config.genome_config.num_inputs):
        dot.node(str(i), label=input_names[i], fillcolor='lightblue')
    
    # Nodos de salida
    for i in range(config.genome_config.num_outputs):
        node_id = str(-i - 1)
        dot.node(node_id, label=output_names[i], fillcolor='lightgreen')
    
    # Nodos ocultos
    for node_id in genome.nodes.keys():
        if node_id >= config.genome_config.num_inputs:
            dot.node(str(node_id), label=f'H{node_id}', fillcolor='lightyellow')
    
    # Conexiones
    for cg in genome.connections.values():
        if cg.enabled:
            weight = cg.weight
            color = 'red' if weight < 0 else 'blue'
            width = str(0.5 + abs(weight) / 5.0)
            dot.edge(str(cg.key[0]), str(cg.key[1]), 
                    label=f'{weight:.2f}', color=color, penwidth=width)
    
    # Renderizar
    output_file = f"{output_path}/best_genome_network"
    dot.render(output_file, cleanup=True)
    print(f"   ✅ {output_file}.png")
    
    # Info
    print(f"\nInformación del mejor genoma:")
    print(f"   Fitness: {genome.fitness:.2f}")
    print(f"   Nodos: {len(genome.nodes)}")
    print(f"   Conexiones activas: {sum(1 for c in genome.connections.values() if c.enabled)}")
    print("✅ Visualización completada\n")


def plot_validation_summary(validation_dir="validation_hybrid_3"):
    """
    Genera resumen visual de la validación híbrida.
    """
    print("Generando resumen de validación...")
    
    # Aquí podrías leer resultados si los guardas en archivo
    # Por simplicidad, solo creo placeholder
    
    print("   ℹ️  Revisa las trayectorias individuales en validation_hybrid_3/\n")


if __name__ == "__main__":
    print("="*70)
    print("VISUALIZACIÓN DE RESULTADOS - Sistema Híbrido NEAT + PPO")
    print("="*70 + "\n")
    
    # Entrenamientos NEAT
    try:
        visualize_neat_training()
    except FileNotFoundError:
        print("⚠️ No se encontraron estadísticas de entrenamiento NEAT")
        print("   Ejecuta primero: python train_neat_obstacle_3.py\n")
    
    # Mejor genoma
    try:
        visualize_best_genome()
    except FileNotFoundError:
        print("⚠️ No se encontró el mejor genoma NEAT")
        print("   Ejecuta primero: python train_neat_obstacle_3.py\n")
    
    # Validación
    plot_validation_summary()
    
    print("="*70)
    print("✅ Visualizaciones completadas")
    print("="*70)
