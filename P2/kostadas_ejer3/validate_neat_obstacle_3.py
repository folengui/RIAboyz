# validate_neat_obstacle_3.py
"""
Validación del mejor genoma NEAT entrenado para ESCENARIO CON OBSTÁCULO
Ejecuta múltiples episodios y genera estadísticas de detección + TRAYECTORIAS 2D
"""

import neat
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, Circle

from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
from env_neat_obstacle_3 import RoboboNEATObstacleEnv


def plot_trajectory_2d(robot_x, robot_z, episode_num, success=False, 
                       collided=False, wall_contacted=False, 
                       output_dir='results_neat_obstacle_3_fast/validation'):
    """Visualiza la trayectoria 2D del robot en el escenario con obstáculo"""
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    robot_x = np.array(robot_x)
    robot_z = np.array(robot_z)
    
    # Trayectoria del robot con gradiente de color
    points = np.array([robot_x, robot_z]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    colors = plt.cm.viridis(np.linspace(0, 1, len(robot_x)))
    
    for i in range(len(segments)):
        ax.plot(segments[i, :, 0], segments[i, :, 1], 
               color=colors[i], linewidth=2.5, alpha=0.7)
    
    # Inicio y fin del robot
    ax.plot(robot_x[0], robot_z[0], 'go', markersize=15, 
           label='Inicio Robot', zorder=5, 
           markeredgecolor='darkgreen', markeredgewidth=2)
    ax.plot(robot_x[-1], robot_z[-1], 'r*', markersize=20, 
           label='Fin Robot', zorder=5, 
           markeredgecolor='darkred', markeredgewidth=2)
    
    # VISUALIZAR OBSTÁCULO (aproximado - depende de tu escenario)
    # Ajusta estas coordenadas según tu escenario real
    obstacle_x = 0  # Centro del obstáculo
    obstacle_z = 80  # Aproximadamente
    obstacle_width = 100
    obstacle_height = 20
    
    obstacle = Rectangle(
        (obstacle_x - obstacle_width/2, obstacle_z - obstacle_height/2),
        obstacle_width, obstacle_height,
        color='gray', alpha=0.5, label='Obstáculo (aprox.)',
        edgecolor='black', linewidth=2, zorder=3
    )
    ax.add_patch(obstacle)
    
    # VISUALIZAR CILINDRO (objetivo a detectar - aproximado)
    cylinder_x = 0  # Ajustar según tu escenario
    cylinder_z = 120  # Detrás del obstáculo
    ax.plot(cylinder_x, cylinder_z, 'bs', markersize=15, 
           label='Cilindro (objetivo)', 
           markeredgecolor='darkblue', markeredgewidth=2, zorder=5)
    
    # Círculo alrededor del cilindro
    circle = Circle((cylinder_x, cylinder_z), 15, color='blue', 
                   fill=False, linewidth=2, linestyle='--', 
                   alpha=0.5, zorder=3)
    ax.add_patch(circle)
    
    # Flechas de dirección
    step_interval = max(1, len(robot_x) // 8)
    for i in range(0, len(robot_x) - 1, step_interval):
        dx = robot_x[i + 1] - robot_x[i]
        dz = robot_z[i + 1] - robot_z[i]
        if abs(dx) > 0.1 or abs(dz) > 0.1:
            ax.arrow(robot_x[i], robot_z[i], dx * 0.5, dz * 0.5,
                    head_width=5, head_length=3, 
                    fc=colors[i], ec=colors[i],
                    alpha=0.6, linewidth=0.5)
    
    # Título
    status = "✓ ÉXITO" if success else "✗ FALLO"
    color_title = 'green' if success else 'red'
    
    extra_info = []
    if collided:
        extra_info.append("💥 Colisión")
    if wall_contacted:
        extra_info.append("🧱 Contactó pared")
    extra_str = " | ".join(extra_info) if extra_info else ""
    
    title = (f'Episodio {episode_num} - {status}\n'
            f'Pasos: {len(robot_x)} | {extra_str}\n'
            f'Posición Final: ({robot_x[-1]:.1f}, {robot_z[-1]:.1f})')
    
    ax.set_title(title, fontsize=12, fontweight='bold', color=color_title)
    ax.set_xlabel('Posición X', fontsize=12)
    ax.set_ylabel('Posición Z', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Barra de color temporal
    sm = plt.cm.ScalarMappable(cmap='viridis', 
                               norm=plt.Normalize(vmin=0, vmax=len(robot_x)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Paso de tiempo', fontsize=10)
    
    plt.tight_layout()
    filename = os.path.join(output_dir, f'trayectoria_episodio_{episode_num}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Guardada: {filename}")
    plt.close()


def plot_validation_summary(all_episodes_data, output_dir='results_neat_obstacle_3_fast/validation'):
    """Crea resumen visual de todos los episodios CON TRAYECTORIAS"""
    
    num_episodes = len(all_episodes_data)
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # Panel 1: TODAS LAS TRAYECTORIAS COMBINADAS
    ax = axes[0, 0]
    colors_episodes = plt.cm.tab10(np.linspace(0, 1, num_episodes))
    
    for i, ep_data in enumerate(all_episodes_data):
        robot_x = np.array(ep_data['robot_x'])
        robot_z = np.array(ep_data['robot_z'])
        success = ep_data['success']
        linestyle = '-' if success else '--'
        ax.plot(robot_x, robot_z, linestyle=linestyle, 
               linewidth=2, color=colors_episodes[i], alpha=0.7, 
               label=f"Ep {i+1}")
        ax.plot(robot_x[0], robot_z[0], 'o', color=colors_episodes[i], 
               markersize=8, markeredgecolor='black', markeredgewidth=1)
    
    # Visualizar obstáculo (aproximado)
    obstacle_x = 0
    obstacle_z = 80
    obstacle_width = 100
    obstacle_height = 20
    obstacle = Rectangle(
        (obstacle_x - obstacle_width/2, obstacle_z - obstacle_height/2),
        obstacle_width, obstacle_height,
        color='gray', alpha=0.3, label='Obstáculo',
        edgecolor='black', linewidth=1.5
    )
    ax.add_patch(obstacle)
    
    # Cilindro
    cylinder_x = 0
    cylinder_z = 120
    ax.plot(cylinder_x, cylinder_z, 'bs', markersize=12, 
           label='Cilindro', markeredgecolor='darkblue', markeredgewidth=1.5)
    
    ax.set_title('Todas las Trayectorias\n(continua=éxito, punteada=fallo)', 
                fontsize=12, fontweight='bold')
    ax.set_xlabel('Posición X', fontsize=11)
    ax.set_ylabel('Posición Z', fontsize=11)
    ax.legend(fontsize=7, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Panel 2: Tasa de éxito
    ax = axes[0, 1]
    num_success = sum(1 for ep in all_episodes_data if ep['success'])
    success_rate = (num_success / num_episodes) * 100
    
    colors = ['green' if ep['success'] else 'red' for ep in all_episodes_data]
    episodes = [f"Ep {i+1}" for i in range(num_episodes)]
    success_vals = [1 if ep['success'] else 0 for ep in all_episodes_data]
    
    ax.bar(episodes, success_vals, color=colors, alpha=0.7, edgecolor='black')
    ax.set_title(f'Tasa de Éxito: {success_rate:.1f}% ({num_success}/{num_episodes})', 
                fontsize=14, fontweight='bold')
    ax.set_ylabel('Éxito (1) / Fallo (0)', fontsize=12)
    ax.set_ylim([0, 1.2])
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=45)
    
    # Panel 3: Steps hasta detección
    ax = axes[1, 0]
    steps_list = [ep['steps'] for ep in all_episodes_data]
    colors_steps = ['green' if ep['success'] else 'red' for ep in all_episodes_data]
    
    ax.bar(episodes, steps_list, color=colors_steps, alpha=0.7, edgecolor='black')
    ax.set_title('Pasos por Episodio', fontsize=14, fontweight='bold')
    ax.set_ylabel('Número de Pasos', fontsize=12)
    ax.axhline(y=np.mean(steps_list), color='blue', linestyle='--', 
               linewidth=2, label=f'Promedio: {np.mean(steps_list):.1f}')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=45)
    
    # Panel 4: Fitness acumulado
    ax = axes[1, 1]
    fitness_list = [ep['fitness'] for ep in all_episodes_data]
    colors_fit = ['green' if ep['success'] else 'red' for ep in all_episodes_data]
    
    ax.bar(episodes, fitness_list, color=colors_fit, alpha=0.7, edgecolor='black')
    ax.set_title('Fitness Acumulado', fontsize=14, fontweight='bold')
    ax.set_ylabel('Fitness', fontsize=12)
    ax.axhline(y=np.mean(fitness_list), color='blue', linestyle='--', 
               linewidth=2, label=f'Promedio: {np.mean(fitness_list):.1f}')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    filename = os.path.join(output_dir, 'resumen_validacion.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Guardada: {filename}")
    plt.close()


def plot_detailed_statistics(all_episodes_data, output_dir='results_neat_obstacle_3_fast/validation'):
    """Crea panel adicional con estadísticas detalladas"""
    
    num_episodes = len(all_episodes_data)
    num_success = sum(1 for ep in all_episodes_data if ep['success'])
    success_rate = (num_success / num_episodes) * 100
    
    steps_list = [ep['steps'] for ep in all_episodes_data]
    fitness_list = [ep['fitness'] for ep in all_episodes_data]
    avg_steps = np.mean(steps_list)
    avg_fitness = np.mean(fitness_list)
    avg_steps_success = np.mean([ep['steps'] for ep in all_episodes_data if ep['success']]) if num_success > 0 else 0
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    stats_text = f"""
RESUMEN DE VALIDACIÓN - NEAT OBSTACLE
{'='*45}
Episodios totales: {num_episodes}
Éxitos: {num_success}
Fallos: {num_episodes - num_success}
Tasa de éxito: {success_rate:.1f}%

PROMEDIOS GENERALES:
  • Pasos promedio: {avg_steps:.1f}
  • Fitness promedio: {avg_fitness:.1f}

PROMEDIOS SOLO ÉXITOS:
  • Pasos promedio: {avg_steps_success:.1f}
{'='*45}

DETALLES POR EPISODIO:
"""
    for i, ep in enumerate(all_episodes_data, 1):
        status = "✓ ÉXITO" if ep['success'] else "✗ FALLO"
        collided = "💥" if ep.get('collided', False) else ""
        wall_contacted = "🧱" if ep.get('wall_contacted', False) else ""
        stats_text += (f"\n Ep {i}: {status} {collided}{wall_contacted} | "
                      f"{ep['steps']} pasos | Fit: {ep['fitness']:.1f}")
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
           fontsize=9, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    filename = os.path.join(output_dir, 'estadisticas_detalladas.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Guardada: {filename}")
    plt.close()


def main():
    """Función principal de validación"""
    
    print("=" * 70)
    print("VALIDACIÓN DEL MODELO NEAT - ESCENARIO CON OBSTÁCULO")
    print("=" * 70)
    
    # Configuración
    ROBOBO_IP = 'localhost'
    RESULTS_DIR = "results_neat_obstacle_3_fast"
    GENOME_PATH = os.path.join(RESULTS_DIR, "best_genome_3.pkl")
    CONFIG_PATH = "config-neat-obstacle-3-fast"
    VALIDATION_DIR = os.path.join(RESULTS_DIR, "validation")
    N_TEST_EPISODES = 10  # Más episodios para estadísticas robustas
    
    os.makedirs(VALIDATION_DIR, exist_ok=True)
    
    # Verificar archivos
    if not os.path.exists(GENOME_PATH):
        print(f"❌ ERROR: No se encuentra {GENOME_PATH}")
        print("Ejecuta primero train_neat_ultrafast_3.py")
        return
    
    if not os.path.exists(CONFIG_PATH):
        print(f"❌ ERROR: No se encuentra {CONFIG_PATH}")
        return
    
    # Cargar configuración y genoma
    print(f"\nCargando configuración desde: {CONFIG_PATH}")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH
    )
    
    print(f"Cargando mejor genoma desde: {GENOME_PATH}")
    with open(GENOME_PATH, 'rb') as f:
        winner = pickle.load(f)
    
    print(f"✓ Genoma cargado | Fitness entrenamiento: {winner.fitness:.2f}\n")
    
    # Crear red neuronal
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    
    # Conexiones
    print("Conectando con RoboboSim...")
    rob = Robobo(ROBOBO_IP)
    rbsim = RoboboSim(ROBOBO_IP)
    
    idrobot = 0
    idobject = "CYLINDERMIDBALL"
    env = RoboboNEATObstacleEnv(rob, rbsim, idrobot, idobject)
    
    # Validación
    all_episodes_data = []
    
    print(f"\n{'='*70}")
    print(f"Ejecutando {N_TEST_EPISODES} episodios de validación...")
    print(f"{'='*70}\n")
    
    for episode in range(N_TEST_EPISODES):
        print(f"--- Episodio {episode + 1}/{N_TEST_EPISODES} ---")
        
        obs, info = env.reset()
        done = False
        
        # Inicializar tracking de trayectoria
        rpos = info.get("robot_pos", {"x": 0.0, "z": 0.0})
        robot_x = [rpos["x"]]
        robot_z = [rpos["z"]]
        
        steps = 0
        success = False
        
        while not done:
            # Predecir acción con la red
            output = net.activate(obs)
            action = np.argmax(output)
            
            # Ejecutar
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Guardar posición actual
            rpos = info.get("robot_pos", {"x": 0.0, "z": 0.0})
            robot_x.append(rpos["x"])
            robot_z.append(rpos["z"])
            
            steps += 1
            if info.get('goal_reached', False):
                success = True
        
        # Guardar datos del episodio
        episode_data = {
            'robot_x': robot_x,
            'robot_z': robot_z,
            'steps': steps,
            'success': success,
            'fitness': info.get('cumulative_fitness', 0),
            'collided': info.get('collided', False),
            'wall_contacted': info.get('wall_contacted', False),
        }
        all_episodes_data.append(episode_data)
        
        # Mostrar resultado
        status = "✓ ÉXITO" if success else "✗ FALLO"
        status_color = '\033[92m' if success else '\033[91m'  # Verde/Rojo
        reset_color = '\033[0m'
        
        extra_info = []
        if info.get('collided', False):
            extra_info.append("💥 Colisión")
        if info.get('wall_contacted', False):
            extra_info.append("🧱 Contactó pared")
        
        extra_str = " | ".join(extra_info) if extra_info else ""
        
        print(f" {status_color}{status}{reset_color} | "
              f"Pasos: {steps:2d} | "
              f"Fitness: {info.get('cumulative_fitness', 0):6.1f} | "
              f"{extra_str}")
        
        # Generar gráfico de trayectoria individual
        plot_trajectory_2d(
            robot_x, robot_z, 
            episode + 1, 
            success=success,
            collided=info.get('collided', False),
            wall_contacted=info.get('wall_contacted', False),
            output_dir=VALIDATION_DIR
        )
    
    # Generar resumen
    print(f"\n{'='*70}")
    print(f"Creando resumen de validación...")
    plot_validation_summary(all_episodes_data, VALIDATION_DIR)
    plot_detailed_statistics(all_episodes_data, VALIDATION_DIR)
    
    # Estadísticas finales en consola
    num_success = sum(1 for ep in all_episodes_data if ep['success'])
    success_rate = (num_success / N_TEST_EPISODES) * 100
    avg_steps = np.mean([ep['steps'] for ep in all_episodes_data])
    avg_fitness = np.mean([ep['fitness'] for ep in all_episodes_data])
    
    print(f"\n{'='*70}")
    print(f"✅ VALIDACIÓN COMPLETADA")
    print(f"{'='*70}")
    print(f"Tasa de éxito: {success_rate:.1f}% ({num_success}/{N_TEST_EPISODES})")
    print(f"Pasos promedio: {avg_steps:.1f}")
    print(f"Fitness promedio: {avg_fitness:.1f}")
    print(f"Gráficas guardadas en: {VALIDATION_DIR}/")
    print(f"{'='*70}\n")
    
    # Cerrar
    try:
        env.close()
    finally:
        try:
            rob.disconnect()
        except:
            pass


if __name__ == "__main__":
    main()
