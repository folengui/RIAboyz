# validate_neat.py
"""
Validación del mejor genoma NEAT entrenado
Ejecuta múltiples episodios y genera visualizaciones de trayectorias
"""

import neat
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle

from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
from env_ej2 import RoboboNEATEnv


def plot_trajectory_2d(robot_x, robot_z, target_x, target_z, 
                       episode_num, success=False, output_dir='results_neat/validation'):
    """Visualiza la trayectoria 2D del robot"""
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
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
    
    # Cilindro (INMÓVIL)
    ax.plot(target_x, target_z, 'bs', markersize=15, 
           label='Cilindro (inmóvil)', 
           markeredgecolor='darkblue', markeredgewidth=2, zorder=5)
    
    # Círculo alrededor del objetivo
    circle = Circle((target_x, target_z), 20, color='blue', 
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
    
    # Distancia final
    final_dist = np.sqrt((robot_x[-1] - target_x) ** 2 + 
                        (robot_z[-1] - target_z) ** 2)
    
    # Título
    status = "✓ ÉXITO" if success else "✗ FALLO"
    color_title = 'green' if success else 'red'
    title = (f'Episodio {episode_num} - {status}\n'
            f'Pasos: {len(robot_x)} | Distancia Final: {final_dist:.1f}\n'
            f'Robot: ({robot_x[-1]:.1f}, {robot_z[-1]:.1f}) | '
            f'Cilindro: ({target_x:.1f}, {target_z:.1f})')
    
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


def plot_validation_summary(all_episodes_data, output_dir='results_neat/validation'):
    """Crea resumen visual de todos los episodios"""
    
    num_episodes = len(all_episodes_data)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Panel izquierdo: Todas las trayectorias
    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, num_episodes))
    
    for i, ep_data in enumerate(all_episodes_data):
        robot_x = np.array(ep_data['robot_x'])
        robot_z = np.array(ep_data['robot_z'])
        success = ep_data['success']
        linestyle = '-' if success else '--'
        ax.plot(robot_x, robot_z, linestyle=linestyle, 
               linewidth=2, color=colors[i], alpha=0.7, 
               label=f"Ep {i+1}")
        ax.plot(robot_x[0], robot_z[0], 'o', color=colors[i], 
               markersize=8, markeredgecolor='black', markeredgewidth=1)
    
    ax.set_title('Todas las Trayectorias\n(continua=éxito, punteada=fallo)', 
                fontsize=12, fontweight='bold')
    ax.set_xlabel('Posición X', fontsize=11)
    ax.set_ylabel('Posición Z', fontsize=11)
    ax.legend(fontsize=7, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Panel derecho: Estadísticas
    ax = axes[1]
    ax.axis('off')
    
    num_success = sum(1 for ep in all_episodes_data if ep['success'])
    success_rate = (num_success / num_episodes) * 100
    avg_steps = np.mean([ep['steps'] for ep in all_episodes_data])
    avg_final_dist = np.mean([ep['final_distance'] for ep in all_episodes_data])
    avg_fitness = np.mean([ep['fitness'] for ep in all_episodes_data])
    
    stats_text = f"""
RESUMEN DE VALIDACIÓN - NEAT
{'='*40}
Episodios totales: {num_episodes}
Éxitos: {num_success}
Fallos: {num_episodes - num_success}
Tasa de éxito: {success_rate:.1f}%
Pasos promedio: {avg_steps:.1f}
Distancia final promedio: {avg_final_dist:.1f}
Fitness promedio: {avg_fitness:.1f}
{'='*40}
Detalles por episodio:
"""
    for i, ep in enumerate(all_episodes_data, 1):
        status = "✓" if ep['success'] else "✗"
        stats_text += (f"\n Ep {i}: {status} | {ep['steps']} pasos | "
                      f"Dist: {ep['final_distance']:.1f} | "
                      f"Fit: {ep['fitness']:.1f}")
    
    ax.text(0.1, 0.95, stats_text, transform=ax.transAxes, 
           fontsize=10, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    filename = os.path.join(output_dir, 'resumen_validacion.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Guardada: {filename}")
    plt.close()


def main():
    """Función principal de validación"""
    
    print("=" * 60)
    print("VALIDACIÓN DEL MODELO NEAT")
    print("=" * 60)
    
    # Configuración
    ROBOBO_IP = 'localhost'
    RESULTS_DIR = "results"  # Cambiado para usar resultados del entrenamiento rápido
    GENOME_PATH = os.path.join(RESULTS_DIR, "best_genome.pkl")
    CONFIG_PATH = "config.txt"  # Usar la config del entrenamiento rápido
    VALIDATION_DIR = os.path.join(RESULTS_DIR, "validation")
    N_TEST_EPISODES = 5
    
    os.makedirs(VALIDATION_DIR, exist_ok=True)
    
    # Verificar archivos
    if not os.path.exists(GENOME_PATH):
        print(f"❌ ERROR: No se encuentra {GENOME_PATH}")
        print("Ejecuta primero train_neat.py")
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
    idobject = "CYLINDERBALL"
    env = RoboboNEATEnv(rob, rbsim, idrobot, idobject)
    
    # Validación
    all_episodes_data = []
    
    for episode in range(N_TEST_EPISODES):
        print(f"--- Episodio {episode + 1}/{N_TEST_EPISODES} ---")
        
        obs, info = env.reset()
        done = False
        
        # Posiciones
        rpos = info["position"]
        tpos = info["target_position"]
        
        robot_x = [rpos["x"]]
        robot_z = [rpos["z"]]
        target_x = tpos["x"]
        target_z = tpos["z"]
        
        steps = 0
        success = False
        
        while not done:
            # Predecir acción con la red
            output = net.activate(obs)
            action = np.argmax(output)
            
            # Ejecutar
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Guardar posición
            rpos = info["robot_pos"]
            robot_x.append(rpos["x"])
            robot_z.append(rpos["z"])
            
            steps += 1
            if info.get('goal_reached', False):
                success = True
        
        # Calcular distancia final
        final_distance = np.sqrt((robot_x[-1] - target_x) ** 2 + 
                                (robot_z[-1] - target_z) ** 2)
        
        episode_data = {
            'robot_x': robot_x,
            'robot_z': robot_z,
            'target_x': target_x,
            'target_z': target_z,
            'steps': steps,
            'success': success,
            'final_distance': final_distance,
            'fitness': info.get('cumulative_fitness', 0),
            'blob_size_efectivo': info.get('blob_size_efectivo', 0),
            'dist_from_center': info.get('dist_from_center', 0)
        }
        all_episodes_data.append(episode_data)
        
        status = "✓ ÉXITO" if success else "✗ FALLO"
        print(f" {status} | Pasos: {steps} | Distancia: {final_distance:.1f} | "
              f"Fitness: {info.get('cumulative_fitness', 0):.1f}")
        
        # Visualizar trayectoria
        plot_trajectory_2d(robot_x, robot_z, target_x, target_z, 
                          episode + 1, success, VALIDATION_DIR)
    
    # Resumen
    print(f"\nCreando resumen de validación...")
    plot_validation_summary(all_episodes_data, VALIDATION_DIR)
    
    print(f"\n{'=' * 60}")
    print(f"✅ VALIDACIÓN COMPLETADA")
    print(f"✅ Gráficas guardadas en: {VALIDATION_DIR}/")
    print(f"{'=' * 60}\n")
    
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
