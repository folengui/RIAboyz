# train_neat_ultrafast_3.py
"""
Entrenamiento ULTRARRÁPIDO para desarrollo y pruebas - Práctica 2.3
⚡ FILOSOFÍA: Sin checkpoints, mínimo debug, máxima velocidad
Población pequeña (12), pocas generaciones (15), episodios cortos (80 pasos).
"""

import neat
import pickle
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import graphviz

from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
from env_neat_obstacle_3 import RoboboNEATObstacleEnv


# ============================================
# CONFIGURACIÓN RÁPIDA
# ============================================
ROBOBO_IP = "localhost"
ROBOT_ID = 0
OBJECT_ID = "CYLINDERMIDBALL"
CONFIG_FILE = "config-neat-obstacle-3-fast"
RESULTS_DIR = "results"  # Directorio para estadísticas y gráficas
MODELS_DIR = "models"    # Directorio para modelos entrenados
GENERATIONS = 25  # ← REDUCIDO de 100 a 15
MAX_STEPS_PER_EPISODE = 50  # ← REDUCIDO de 150 a 90

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Contadores globales para debug
episode_counter = 0
current_generation = 0

# Variables globales para conexiones (se inicializan en run_neat)
rob = None
rbsim = None
env = None


def eval_genome(genome, config):
    """Evalúa un genoma con episodio más corto."""
    global episode_counter, current_generation, env
    
    episode_counter += 1
    
    # Crear la red neuronal desde el genoma
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    # Resetear environment (NO crear uno nuevo)
    obs, info = env.reset()
    done = False
    steps = 0
    
    # Contador de acciones
    action_counts = [0, 0, 0]  # 3 acciones: Recto, Izq, Der
    
    while not done and steps < MAX_STEPS_PER_EPISODE:
        # Obtener acción de la red neuronal
        output = net.activate(obs)
        action = output.index(max(output))
        action_counts[action] += 1
        
        # Ejecutar acción y actualizar observación
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
    
    # El fitness es el acumulado del episodio
    fitness = env.get_cumulative_fitness()
    
    # CRÍTICO: Asegurar que fitness es un número válido
    if fitness is None:
        print(f"    ⚠️ Fitness es None, usando 0.0")
        fitness = 0.0
    else:
        # Convertir a float nativo de Python (por si es numpy type)
        try:
            fitness = float(fitness)
        except (TypeError, ValueError):
            print(f"    ⚠️ Fitness no convertible a float ({type(fitness).__name__}: {fitness}), usando 0.0")
            fitness = 0.0
        
        # Verificar que sea finito (no inf, no nan)
        if not np.isfinite(fitness):
            print(f"    ⚠️ Fitness no finito (inf/nan: {fitness}), usando 0.0")
            fitness = 0.0
    
    # Debug: mostrar info del episodio
    status = "✓ TERMINADO" if done else "⏱️ TRUNCADO"
    action_names = ["Recto", "Izq", "Der"]
    actions_str = ", ".join([f"{action_names[i]}:{action_counts[i]}" for i in range(3)])
    
    # Info detallada
    goal_reached = info.get('goal_reached', False)
    collided = info.get('collided', False)
    exploration_dist = info.get('exploration_distance', 0)
    
    result_icon = "🎯" if goal_reached else ("💥" if collided else "⏱️")
    
    print(f"  {result_icon} Episodio {episode_counter} | Gen {current_generation} | Genoma {genome.key} | "
          f"Fitness: {fitness:.2f} | Steps: {steps}/{MAX_STEPS_PER_EPISODE} | {status}")
    print(f"    Acciones: {actions_str} | Exploración: {exploration_dist:.1f}")
    
    return fitness


def eval_genomes(genomes, config):
    """Evalúa todos los genomas."""
    global current_generation
    
    print(f"\n{'='*80}")
    print(f"🧬 Evaluando Generación {current_generation} - {len(genomes)} genomas")
    print(f"{'='*80}")
    
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)
    
    # Estadísticas de la generación
    fitnesses = [g.fitness for g in [genome for _, genome in genomes]]
    max_fit = max(fitnesses)
    mean_fit = sum(fitnesses) / len(fitnesses)
    min_fit = min(fitnesses)
    
    print(f"\n📊 Generación {current_generation} completada:")
    print(f"   Mejor: {max_fit:.2f} | Media: {mean_fit:.2f} | Peor: {min_fit:.2f}")
    
    current_generation += 1


def plot_fitness_evolution(stats, results_dir):
    """Gráfica de aprendizaje: evolución del fitness por generación."""
    generation = range(len(stats.most_fit_genomes))
    best_fitness = [c.fitness for c in stats.most_fit_genomes]
    avg_fitness = np.array(stats.get_fitness_mean())
    
    plt.figure(figsize=(12, 6))
    plt.plot(generation, best_fitness, 'g-', linewidth=2, label='Mejor Fitness', marker='o')
    plt.plot(generation, avg_fitness, 'b-', linewidth=2, label='Fitness Medio', alpha=0.7)
    
    plt.fill_between(generation, avg_fitness, best_fitness, alpha=0.2, color='green')
    
    plt.xlabel('Generación', fontsize=12, fontweight='bold')
    plt.ylabel('Fitness', fontsize=12, fontweight='bold')
    plt.title('Curva de Aprendizaje - Evolución del Fitness', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    
    # Anotación del mejor fitness
    best_gen = best_fitness.index(max(best_fitness))
    best_fit = max(best_fitness)
    plt.annotate(f'Mejor: {best_fit:.1f}\n(Gen {best_gen})',
                xy=(best_gen, best_fit),
                xytext=(10, -30),
                textcoords='offset points',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    filepath = os.path.join(results_dir, 'fitness_evolution_3.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  ✓ Gráfica de aprendizaje: {filepath}")
    plt.close()


def plot_species_evolution(stats, results_dir):
    """Gráfica de especies obtenidas a lo largo del entrenamiento."""
    species_sizes = stats.get_species_sizes()
    num_generations = len(species_sizes)
    
    # Preparar datos
    species_ids = list(species_sizes.keys())
    generations = range(num_generations)
    
    plt.figure(figsize=(12, 6))
    
    # Crear matriz para stacked area
    species_data = {}
    for sid in species_ids:
        species_data[sid] = []
        for gen in generations:
            if gen < len(species_sizes[sid]):
                species_data[sid].append(species_sizes[sid][gen])
            else:
                species_data[sid].append(0)
    
    # Stacked area chart
    bottom = np.zeros(num_generations)
    colors = plt.cm.tab20(np.linspace(0, 1, len(species_ids)))
    
    for idx, (sid, sizes) in enumerate(species_data.items()):
        plt.fill_between(generations, bottom, bottom + sizes, 
                        label=f'Especie {sid}', alpha=0.7, color=colors[idx])
        bottom += sizes
    
    plt.xlabel('Generación', fontsize=12, fontweight='bold')
    plt.ylabel('Número de Individuos', fontsize=12, fontweight='bold')
    plt.title('Evolución de Especies - Especiación NEAT', fontsize=14, fontweight='bold')
    plt.legend(fontsize=9, loc='upper right', ncol=2)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filepath = os.path.join(results_dir, 'species_evolution_3.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  ✓ Gráfica de especies: {filepath}")
    plt.close()


def draw_net(config, genome, view=False, filename=None, node_names=None, show_disabled=False, prune_unused=False,
             node_colors=None, fmt='png'):
    """Dibuja la red neuronal del genoma."""
    
    if node_names is None:
        node_names = {}
    
    assert type(node_names) is dict
    
    if node_colors is None:
        node_colors = {}
    
    assert type(node_colors) is dict
    
    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'
    }
    
    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)
    
    inputs = set()
    for k in config.genome_config.input_keys:
        inputs.add(k)
        name = node_names.get(k, str(k))
        input_attrs = {'style': 'filled', 'shape': 'box', 'fillcolor': 'lightblue'}
        dot.node(name, _attributes=input_attrs)
    
    outputs = set()
    for k in config.genome_config.output_keys:
        outputs.add(k)
        name = node_names.get(k, str(k))
        output_attrs = {'style': 'filled', 'fillcolor': 'lightgreen'}
        dot.node(name, _attributes=output_attrs)
    
    used_nodes = set(genome.nodes.keys())
    for n in used_nodes:
        if n in inputs or n in outputs:
            continue
        attrs = {'style': 'filled', 'fillcolor': node_colors.get(n, 'white')}
        dot.node(str(n), _attributes=attrs)
    
    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            input_node, output_node = cg.key
            a = node_names.get(input_node, str(input_node))
            b = node_names.get(output_node, str(output_node))
            style = 'solid' if cg.enabled else 'dotted'
            color = 'green' if cg.weight > 0 else 'red'
            width = str(0.1 + abs(cg.weight / 5.0))
            dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})
    
    dot.render(filename, view=view, cleanup=True)
    
    return dot


def plot_network_topology(config, genome, results_dir):
    """Gráfica de configuración de red neuronal."""
    node_names = {
        -1: 'IR Front',
        -2: 'IR Left', 
        -3: 'IR Right',
        0: 'Recto',
        1: 'Izq',
        2: 'Der'
    }
    
    filename = os.path.join(results_dir, 'network_topology_3')
    draw_net(config, genome, view=False, filename=filename, node_names=node_names, fmt='png')
    print(f"  ✓ Topología de red: {filename}.png")


def plot_2d_solution(results_dir):
    """Gráfica conceptual del plano 2D de la solución."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Robot posición inicial
    robot_start = (2, 2)
    ax.plot(robot_start[0], robot_start[1], 'bo', markersize=20, label='Robot (inicio)')
    ax.annotate('ROBOT\n(inicio)', xy=robot_start, xytext=(10, 10), 
               textcoords='offset points', fontsize=10, fontweight='bold')
    
    # Obstáculo (bloque)
    obstacle_rect = plt.Rectangle((4, 1.5), 1, 1, color='gray', alpha=0.7, label='Obstáculo')
    ax.add_patch(obstacle_rect)
    ax.text(4.5, 2, 'BLOCK', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Cilindro rojo (objetivo)
    cylinder = plt.Circle((7, 2), 0.4, color='red', alpha=0.7, label='Cilindro (objetivo)')
    ax.add_patch(cylinder)
    ax.text(7, 2, 'RED', ha='center', va='center', fontsize=8, color='white', fontweight='bold')
    
    # Estrategia: camino de navegación
    # Camino 1: Recto hasta detectar obstáculo
    ax.arrow(2, 2, 1.5, 0, head_width=0.15, head_length=0.2, fc='blue', ec='blue', alpha=0.5)
    
    # Camino 2: Girar para esquivar
    arc_x = np.linspace(3.5, 4, 20)
    arc_y = 2 + 0.8 * np.sin(np.linspace(0, np.pi, 20))
    ax.plot(arc_x, arc_y, 'b--', linewidth=2, alpha=0.5)
    
    # Camino 3: Navegar alrededor
    nav_x = np.linspace(4, 6, 20)
    nav_y = 2.8 - 0.3 * (nav_x - 4)
    ax.plot(nav_x, nav_y, 'b--', linewidth=2, alpha=0.5)
    
    # Camino 4: Aproximación final
    ax.arrow(6, 2.5, 0.6, -0.3, head_width=0.15, head_length=0.2, fc='green', ec='green', alpha=0.5)
    ax.text(5, 3.2, 'Estrategia NEAT:\n1. Explorar\n2. Esquivar obstáculo\n3. Detectar cilindro', 
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=9)
    
    # Campo visual (PAN barrido)
    from matplotlib.patches import Wedge
    fov = Wedge(robot_start, 2, -30, 30, alpha=0.2, color='yellow', label='Campo visual (barrido PAN)')
    ax.add_patch(fov)
    
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (unidades simulación)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Z (unidades simulación)', fontsize=11, fontweight='bold')
    ax.set_title('Plano 2D - Estrategia de Solución (Ejercicio 2.3)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    
    plt.tight_layout()
    filepath = os.path.join(results_dir, 'solution_2d_plan_3.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  ✓ Plano 2D solución: {filepath}")
    plt.close()


def run_neat():
    """Ejecuta el entrenamiento NEAT ultrarrápido."""
    global rob, rbsim, env
    
    print("="*70)
    print("⚡ ENTRENAMIENTO ULTRARRÁPIDO - NEAT (Ejercicio 2.3)")
    print("="*70)
    print(f"Población: 12 | Generaciones: {GENERATIONS} | Pasos/episodio: {MAX_STEPS_PER_EPISODE}")
    print(f"Config: {CONFIG_FILE} | Tiempo estimado: 20-30 min")
    print("="*70 + "\n")
    
    # Conectar (UNA SOLA VEZ al inicio)
    print("Conectando con RoboboSim...")
    rob = Robobo(ROBOBO_IP)
    rbsim = RoboboSim(ROBOBO_IP)
    
    # Crear environment compartido (UNA SOLA VEZ)
    env = RoboboNEATObstacleEnv(rob, rbsim, ROBOT_ID, OBJECT_ID, max_steps=MAX_STEPS_PER_EPISODE)
    
    print("✓ Conexiones establecidas")
    print(f"✓ Environment creado (se reutilizará para todos los genomas)\n")
    
    # Configuración NEAT
    # Cargar con encoding correcto para evitar UnicodeDecodeError en Windows
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, CONFIG_FILE)
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )
    
    # Población (o cargar desde checkpoint si existe)
    checkpoint_prefix = os.path.join(RESULTS_DIR, "neat-checkpoint-")
    
    # Buscar checkpoints existentes
    checkpoint_files = [f for f in os.listdir(RESULTS_DIR) if f.startswith("neat-checkpoint-")]
    if checkpoint_files:
        # Ordenar y obtener el más reciente
        # Manejar tanto "neat-checkpoint-3" como "neat-checkpoint-emergency-gen3"
        def get_checkpoint_gen(filename):
            if "emergency" in filename:
                # Formato: neat-checkpoint-emergency-gen3
                return int(filename.split("gen")[-1])
            else:
                # Formato: neat-checkpoint-3
                return int(filename.split("-")[-1])
        
        checkpoint_files.sort(key=get_checkpoint_gen)
        latest_checkpoint = os.path.join(RESULTS_DIR, checkpoint_files[-1])
        print(f"📂 Checkpoint encontrado: {checkpoint_files[-1]}")
        response = input("¿Continuar desde checkpoint? (s/n): ").lower()
        if response == 's':
            print(f"🔄 Cargando desde checkpoint: {latest_checkpoint}")
            p = neat.Checkpointer.restore_checkpoint(latest_checkpoint)
            print(f"✓ Checkpoint cargado. Generación actual: {p.generation}")
        else:
            print("🆕 Iniciando entrenamiento desde cero")
            p = neat.Population(config)
    else:
        print("🆕 No hay checkpoints previos. Iniciando desde cero")
        p = neat.Population(config)
    
    # Reporters
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    # CHECKPOINT automático cada 3 generaciones
    checkpointer = neat.Checkpointer(
        generation_interval=3,  # Guardar cada 3 generaciones
        time_interval_seconds=None,
        filename_prefix=checkpoint_prefix
    )
    p.add_reporter(checkpointer)
    print(f"💾 Checkpoints automáticos activados (cada 3 generaciones)")
    print(f"   Ubicación: {RESULTS_DIR}/neat-checkpoint-X")
    
    # Entrenar
    print(f"🚀 Entrenando {GENERATIONS} generaciones...\n")
    start_time = time.time()
    
    try:
        winner = p.run(eval_genomes, GENERATIONS)
    except KeyboardInterrupt:
        print("\n⚠️ Interrumpido por el usuario")
        # Guardar checkpoint de emergencia usando el checkpointer
        emergency_checkpoint = f"{checkpoint_prefix}emergency-gen{p.generation}"
        checkpointer.save_checkpoint(config, p.population, p.species, p.generation)
        # Renombrar el último checkpoint a "emergency"
        import shutil
        last_checkpoint = f"{checkpoint_prefix}{p.generation}"
        if os.path.exists(last_checkpoint):
            shutil.move(last_checkpoint, emergency_checkpoint)
            print(f"💾 Checkpoint de emergencia guardado: {emergency_checkpoint}")
        winner = stats.best_genome()
    except Exception as e:
        print(f"\n❌ Error durante entrenamiento: {e}")
        # Guardar checkpoint de emergencia usando el checkpointer
        emergency_checkpoint = f"{checkpoint_prefix}emergency-gen{p.generation}"
        try:
            checkpointer.save_checkpoint(config, p.population, p.species, p.generation)
            import shutil
            last_checkpoint = f"{checkpoint_prefix}{p.generation}"
            if os.path.exists(last_checkpoint):
                shutil.move(last_checkpoint, emergency_checkpoint)
                print(f"💾 Checkpoint de emergencia guardado: {emergency_checkpoint}")
        except:
            print("⚠️ No se pudo guardar checkpoint de emergencia")
        raise
    
    elapsed_time = time.time() - start_time
    
    # Guardar
    print(f"\n{'='*70}")
    print("✅ ENTRENAMIENTO COMPLETADO")
    print(f"{'='*70}")
    print(f"⏱️  Tiempo total: {elapsed_time/60:.1f} minutos ({elapsed_time:.0f} segundos)")
    print(f"🏆 Mejor fitness: {winner.fitness:.2f}")
    print(f"🧬 Arquitectura: {len(winner.nodes)} nodos | {len(winner.connections)} conexiones")
    print(f"📊 Episodios totales evaluados: {episode_counter}")
    print(f"📈 Generaciones completadas: {current_generation}")
    
    # Guardar modelo y estadísticas
    print(f"\n💾 Guardando resultados...")
    with open(f"{MODELS_DIR}/best_genome_3.pkl", 'wb') as f:
        pickle.dump(winner, f)
    with open(f"{RESULTS_DIR}/generation_stats_3.pkl", 'wb') as f:
        pickle.dump(stats, f)
    with open(CONFIG_FILE, 'r') as f_in:
        with open(f"{RESULTS_DIR}/config-neat-used_3.txt", 'w') as f_out:
            f_out.write(f_in.read())
    
    print(f"  ✓ Modelo guardado en: {MODELS_DIR}/best_genome_3.pkl")
    print(f"  ✓ Estadísticas guardadas en: {RESULTS_DIR}/generation_stats_3.pkl")
    
    # Generar gráficas
    print(f"\n📊 Generando visualizaciones...")
    try:
        plot_fitness_evolution(stats, RESULTS_DIR)
        plot_species_evolution(stats, RESULTS_DIR)
        plot_network_topology(config, winner, RESULTS_DIR)
        plot_2d_solution(RESULTS_DIR)
        print(f"  ✓ Todas las gráficas generadas correctamente en: {RESULTS_DIR}/")
    except Exception as e:
        print(f"  ⚠️ Error generando algunas gráficas: {e}")
    
    # Evaluación
    print(f"\n{'='*70}")
    print("📋 EVALUACIÓN FINAL")
    print(f"{'='*70}")
    if winner.fitness >= 80:
        print("✅ Fitness OK (>80) → Ejecutar: python validate_hybrid_3.py")
    elif winner.fitness >= 50:
        print("⚠️ Fitness moderado (50-80) → Entrenar más o validar con cuidado")
    else:
        print("❌ Fitness bajo (<50) → Revisar config o usar train_neat_obstacle_3.py")
    
    print(f"{'='*70}\n")
    
    # Cerrar conexiones
    try:
        env.close()
    finally:
        try:
            rob.disconnect()
        except:
            pass
        try:
            rbsim.disconnect()
        except:
            pass


if __name__ == "__main__":
    run_neat()
