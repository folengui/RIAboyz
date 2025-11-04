# visualize_best_genome.py
"""
Visualiza la topología de la red neuronal del mejor genoma NEAT
Genera una imagen mostrando nodos y conexiones
"""

import pickle
import neat
import os
import graphviz
import matplotlib.pyplot as plt

def visualize_network(genome, config, filename='network'):
    """
    Visualiza la arquitectura de la red neuronal usando graphviz
    """
    # Obtener nombres de inputs y outputs
    node_names = {
        -1: 'Blob Size\nEfectivo',
        -2: 'Dist Centro',
        -3: 'IR Front',
        0: 'Recto',
        1: 'Izquierda',
        2: 'Derecha',
        3: 'Lento'
    }
    
    # Crear grafo dirigido
    dot = graphviz.Digraph(comment='Red Neuronal NEAT')
    dot.attr(rankdir='LR')  # Left to Right
    dot.attr('node', shape='circle', style='filled', fontsize='10')
    
    # Inputs (verde)
    with dot.subgraph(name='cluster_inputs') as c:
        c.attr(label='Inputs', fontsize='12', fontweight='bold')
        c.attr(color='lightgreen', style='filled', fillcolor='lightgreen')
        for i in range(config.genome_config.num_inputs):
            node_id = -(i + 1)
            c.node(str(node_id), node_names.get(node_id, f'Input {i}'),
                  fillcolor='lightgreen')
    
    # Outputs (azul)
    with dot.subgraph(name='cluster_outputs') as c:
        c.attr(label='Outputs', fontsize='12', fontweight='bold')
        c.attr(color='lightblue', style='filled', fillcolor='lightblue')
        for i in range(config.genome_config.num_outputs):
            c.node(str(i), node_names.get(i, f'Output {i}'),
                  fillcolor='lightblue')
    
    # Hidden nodes (amarillo)
    hidden_nodes = []
    for node_id in genome.nodes.keys():
        if node_id >= config.genome_config.num_outputs:
            hidden_nodes.append(node_id)
            dot.node(str(node_id), f'Hidden\n{node_id}',
                    fillcolor='lightyellow')
    
    # Conexiones
    for cg in genome.connections.values():
        if cg.enabled:
            # Color según el peso
            weight = cg.weight
            if weight > 0:
                color = 'green'
                penwidth = str(min(3.0, abs(weight)))
            else:
                color = 'red'
                penwidth = str(min(3.0, abs(weight)))
            
            dot.edge(str(cg.key[0]), str(cg.key[1]),
                    label=f'{weight:.2f}',
                    color=color,
                    penwidth=penwidth,
                    fontsize='8')
    
    return dot


def plot_genome_stats(genome, config, output_dir):
    """
    Genera estadísticas visuales del genoma
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Análisis del Mejor Genoma NEAT', fontsize=16, fontweight='bold')
    
    # 1. Distribución de pesos
    weights = [c.weight for c in genome.connections.values() if c.enabled]
    axes[0, 0].hist(weights, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Distribución de Pesos de Conexiones', fontweight='bold')
    axes[0, 0].set_xlabel('Peso')
    axes[0, 0].set_ylabel('Frecuencia')
    axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.5)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Conexiones activas vs inactivas
    enabled = sum(1 for c in genome.connections.values() if c.enabled)
    disabled = sum(1 for c in genome.connections.values() if not c.enabled)
    axes[0, 1].bar(['Activas', 'Inactivas'], [enabled, disabled],
                   color=['green', 'red'], alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Estado de Conexiones', fontweight='bold')
    axes[0, 1].set_ylabel('Cantidad')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. Tipo de nodos
    num_inputs = config.genome_config.num_inputs
    num_outputs = config.genome_config.num_outputs
    num_hidden = len([n for n in genome.nodes.keys() if n >= num_outputs])
    
    axes[1, 0].bar(['Inputs', 'Outputs', 'Hidden'],
                   [num_inputs, num_outputs, num_hidden],
                   color=['lightgreen', 'lightblue', 'lightyellow'],
                   edgecolor='black', alpha=0.7)
    axes[1, 0].set_title('Composición de la Red', fontweight='bold')
    axes[1, 0].set_ylabel('Número de Nodos')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. Resumen textual
    axes[1, 1].axis('off')
    info_text = f"""
    INFORMACIÓN DEL GENOMA
    
    Fitness: {genome.fitness:.2f}
    
    ESTRUCTURA:
    • Nodos Entrada: {num_inputs}
    • Nodos Salida: {num_outputs}
    • Nodos Ocultos: {num_hidden}
    • Total Nodos: {num_inputs + num_outputs + num_hidden}
    
    CONEXIONES:
    • Activas: {enabled}
    • Inactivas: {disabled}
    • Total: {enabled + disabled}
    
    PESOS:
    • Máximo: {max(weights):.3f}
    • Mínimo: {min(weights):.3f}
    • Promedio: {sum(weights)/len(weights):.3f}
    """
    axes[1, 1].text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center',
                   family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'genome_stats.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Estadísticas guardadas: {filepath}")
    plt.close()


def main():
    print("=" * 60)
    print("VISUALIZACIÓN DEL MEJOR GENOMA NEAT")
    print("=" * 60)
    
    # Configuración
    RESULTS_DIR = "results_neat_fast"
    GENOME_PATH = os.path.join(RESULTS_DIR, "best_genome.pkl")
    CONFIG_PATH = "config-neat-fast"
    
    # Verificar archivos
    if not os.path.exists(GENOME_PATH):
        print(f"❌ ERROR: No se encuentra {GENOME_PATH}")
        print("Ejecuta primero train_neat_fast.py")
        return
    
    if not os.path.exists(CONFIG_PATH):
        print(f"❌ ERROR: No se encuentra {CONFIG_PATH}")
        return
    
    # Cargar configuración
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH
    )
    
    # Cargar genoma
    print(f"\nCargando mejor genoma desde: {GENOME_PATH}")
    with open(GENOME_PATH, 'rb') as f:
        genome = pickle.load(f)
    
    print(f"✓ Genoma cargado | Fitness: {genome.fitness:.2f}\n")
    
    # Información básica
    num_hidden = len([n for n in genome.nodes.keys() if n >= config.genome_config.num_outputs])
    num_connections = len([c for c in genome.connections.values() if c.enabled])
    
    print(f"Estructura de la red:")
    print(f"  • Inputs: {config.genome_config.num_inputs}")
    print(f"  • Outputs: {config.genome_config.num_outputs}")
    print(f"  • Hidden: {num_hidden}")
    print(f"  • Conexiones activas: {num_connections}\n")
    
    # Generar visualización de la red
    print("Generando visualización de la topología...")
    try:
        dot = visualize_network(genome, config, 'best_genome_network')
        output_path = os.path.join(RESULTS_DIR, 'best_genome_network')
        dot.render(output_path, format='png', cleanup=True)
        print(f"✓ Topología guardada: {output_path}.png")
    except Exception as e:
        print(f"⚠️ Error generando topología con graphviz: {e}")
        print("   Asegúrate de tener graphviz instalado: pip install graphviz")
        print("   Y el ejecutable de graphviz en el PATH del sistema")
    
    # Generar estadísticas
    print("\nGenerando estadísticas del genoma...")
    plot_genome_stats(genome, config, RESULTS_DIR)
    
    print("\n" + "=" * 60)
    print("✅ VISUALIZACIÓN COMPLETADA")
    print(f"✅ Archivos guardados en: {RESULTS_DIR}/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
