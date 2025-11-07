# validate_hybrid_neat_ppo.py
"""
VALIDACIÓN HÍBRIDA: NEAT (Fase 1) + PPO (Fase 2)

FASE 1 (NEAT): Esquivar obstáculo y DETECTAR cilindro
  - Usa política genética entrenada (best_genome_3.pkl)
  - Objetivo: Navegar alrededor del obstáculo hasta detectar el cilindro
  - Termina cuando: blob_size > threshold (cilindro detectado)

FASE 2 (PPO): Acercamiento al cilindro
  - Usa política de refuerzo entrenada (ppo_robobo_final2.zip)
  - Objetivo: Acercarse al cilindro detectado
  - Termina cuando: IR_front > threshold (cerca del cilindro) o colisión
"""

import neat
import pickle
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from stable_baselines3 import PPO

from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim

# Importar entornos (ambos en el mismo directorio)
from env_neat_obstacle_3 import RoboboNEATObstacleEnv
from env2_validation import RoboboEnv as RoboboPPOEnv


# ============================================
# CONFIGURACIÓN
# ============================================
ROBOBO_IP = 'localhost'

# NEAT (Fase 1)
NEAT_RESULTS_DIR = "results_neat_obstacle_3_fast"
NEAT_GENOME_PATH = os.path.join(NEAT_RESULTS_DIR, "best_genome_3.pkl")
NEAT_CONFIG_PATH = "config-neat-obstacle-3-fast"
NEAT_MAX_STEPS = 50  # Máximo de pasos para fase NEAT

# PPO (Fase 2)
PPO_MODEL_PATH = "ppo_robobo_final2.zip"
PPO_MAX_STEPS = 100  # Máximo de pasos para fase PPO

# Validación
VALIDATION_DIR = os.path.join(NEAT_RESULTS_DIR, "validation_hybrid")
N_TEST_EPISODES = 5

# Thresholds de transición
BLOB_DETECTION_THRESHOLD = 0.0  # Tamaño mínimo de blob para considerar "detectado" en NEAT
IR_SUCCESS_THRESHOLD = 150  # Threshold IR para considerar éxito en PPO


# ============================================
# FUNCIONES DE VISUALIZACIÓN
# ============================================
def plot_hybrid_trajectory(phase1_data, phase2_data, episode_num, 
                          success=False, output_dir=VALIDATION_DIR):
    """
    Visualiza la trayectoria completa del sistema híbrido:
    - Fase 1 (NEAT): Rojo
    - Fase 2 (PPO): Azul
    """
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # FASE 1 - NEAT (navegación con obstáculo)
    if len(phase1_data['x']) > 0:
        x1 = np.array(phase1_data['x'])
        z1 = np.array(phase1_data['z'])
        
        # Trayectoria NEAT (rojo)
        ax.plot(x1, z1, 'r-', linewidth=3, alpha=0.7, label='Fase 1: NEAT (Esquivar)')
        ax.plot(x1[0], z1[0], 'go', markersize=15, label='Inicio', 
               zorder=5, markeredgecolor='darkgreen', markeredgewidth=2)
        
        # Punto de transición NEAT→PPO
        if len(x1) > 0:
            ax.plot(x1[-1], z1[-1], 'y*', markersize=20, 
                   label='Transición NEAT→PPO', zorder=5,
                   markeredgecolor='orange', markeredgewidth=2)
    
    # FASE 2 - PPO (acercamiento)
    if len(phase2_data['x']) > 0:
        x2 = np.array(phase2_data['x'])
        z2 = np.array(phase2_data['z'])
        
        # Trayectoria PPO (azul)
        ax.plot(x2, z2, 'b-', linewidth=3, alpha=0.7, label='Fase 2: PPO (Acercamiento)')
        ax.plot(x2[-1], z2[-1], 'r*', markersize=20, label='Fin', 
               zorder=5, markeredgecolor='darkred', markeredgewidth=2)
    
    # OBSTÁCULO (aproximado)
    obstacle_x = 0
    obstacle_z = 80
    obstacle_width = 100
    obstacle_height = 20
    obstacle = Rectangle(
        (obstacle_x - obstacle_width/2, obstacle_z - obstacle_height/2),
        obstacle_width, obstacle_height,
        color='gray', alpha=0.5, label='Obstáculo',
        edgecolor='black', linewidth=2, zorder=3
    )
    ax.add_patch(obstacle)
    
    # CILINDRO (objetivo)
    cylinder_x = 0
    cylinder_z = 120
    ax.plot(cylinder_x, cylinder_z, 'bs', markersize=15, 
           label='Cilindro (objetivo)', 
           markeredgecolor='darkblue', markeredgewidth=2, zorder=5)
    circle = Circle((cylinder_x, cylinder_z), 15, color='blue', 
                   fill=False, linewidth=2, linestyle='--', alpha=0.5, zorder=3)
    ax.add_patch(circle)
    
    # Título
    status = "✓ ÉXITO TOTAL" if success else "✗ FALLO"
    color_title = 'green' if success else 'red'
    
    phase1_steps = len(phase1_data['x'])
    phase2_steps = len(phase2_data['x'])
    total_steps = phase1_steps + phase2_steps
    
    title = (f'Episodio {episode_num} - {status}\n'
            f'Fase 1 (NEAT): {phase1_steps} pasos | '
            f'Fase 2 (PPO): {phase2_steps} pasos | '
            f'Total: {total_steps} pasos')
    
    ax.set_title(title, fontsize=12, fontweight='bold', color=color_title)
    ax.set_xlabel('Posición X', fontsize=12)
    ax.set_ylabel('Posición Z', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    filename = os.path.join(output_dir, f'trayectoria_hibrida_ep{episode_num}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Guardada: {filename}")
    plt.close()


def plot_hybrid_summary(all_episodes_data, output_dir=VALIDATION_DIR):
    """Resumen visual de todos los episodios híbridos"""
    
    num_episodes = len(all_episodes_data)
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # Panel 1: Trayectorias combinadas
    ax = axes[0, 0]
    colors = plt.cm.tab10(np.linspace(0, 1, num_episodes))
    
    for i, ep_data in enumerate(all_episodes_data):
        # Fase 1 (NEAT)
        x1 = np.array(ep_data['phase1_x'])
        z1 = np.array(ep_data['phase1_z'])
        if len(x1) > 0:
            ax.plot(x1, z1, '-', color=colors[i], linewidth=2, alpha=0.6)
        
        # Fase 2 (PPO)
        x2 = np.array(ep_data['phase2_x'])
        z2 = np.array(ep_data['phase2_z'])
        if len(x2) > 0:
            ax.plot(x2, z2, '--', color=colors[i], linewidth=2, alpha=0.6, 
                   label=f"Ep {i+1}")
    
    # Obstáculo y cilindro
    obstacle = Rectangle((-50, 70), 100, 20, color='gray', alpha=0.3, 
                        edgecolor='black', linewidth=1.5)
    ax.add_patch(obstacle)
    ax.plot(0, 120, 'bs', markersize=12, markeredgecolor='darkblue', markeredgewidth=1.5)
    
    ax.set_title('Todas las Trayectorias Híbridas\n(continua=NEAT, punteada=PPO)', 
                fontsize=12, fontweight='bold')
    ax.set_xlabel('Posición X')
    ax.set_ylabel('Posición Z')
    ax.legend(fontsize=8, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Panel 2: Tasa de éxito
    ax = axes[0, 1]
    successes = [ep['success'] for ep in all_episodes_data]
    success_rate = (sum(successes) / num_episodes) * 100
    colors_bar = ['green' if s else 'red' for s in successes]
    episodes = [f"Ep {i+1}" for i in range(num_episodes)]
    
    ax.bar(episodes, successes, color=colors_bar, alpha=0.7, edgecolor='black')
    ax.set_title(f'Tasa de Éxito: {success_rate:.1f}% ({sum(successes)}/{num_episodes})', 
                fontsize=14, fontweight='bold')
    ax.set_ylabel('Éxito (1) / Fallo (0)')
    ax.set_ylim([0, 1.2])
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=45)
    
    # Panel 3: Pasos por fase
    ax = axes[1, 0]
    phase1_steps = [ep['phase1_steps'] for ep in all_episodes_data]
    phase2_steps = [ep['phase2_steps'] for ep in all_episodes_data]
    
    x_pos = np.arange(num_episodes)
    width = 0.35
    
    ax.bar(x_pos - width/2, phase1_steps, width, label='Fase 1 (NEAT)', 
           color='salmon', alpha=0.8, edgecolor='black')
    ax.bar(x_pos + width/2, phase2_steps, width, label='Fase 2 (PPO)', 
           color='skyblue', alpha=0.8, edgecolor='black')
    
    ax.set_title('Pasos por Fase', fontsize=14, fontweight='bold')
    ax.set_ylabel('Número de Pasos')
    ax.set_xlabel('Episodio')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"Ep {i+1}" for i in range(num_episodes)])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 4: Estadísticas textuales
    ax = axes[1, 1]
    ax.axis('off')
    
    avg_phase1 = np.mean(phase1_steps)
    avg_phase2 = np.mean(phase2_steps)
    avg_total = np.mean([ep['total_steps'] for ep in all_episodes_data])
    
    stats_text = f"""
RESUMEN VALIDACIÓN HÍBRIDA
{'='*50}
Episodios totales: {num_episodes}
Éxitos: {sum(successes)}
Fallos: {num_episodes - sum(successes)}
Tasa de éxito: {success_rate:.1f}%

PASOS PROMEDIO:
  • Fase 1 (NEAT - Esquivar): {avg_phase1:.1f}
  • Fase 2 (PPO - Acercamiento): {avg_phase2:.1f}
  • Total: {avg_total:.1f}

{'='*50}
DETALLES POR EPISODIO:
"""
    for i, ep in enumerate(all_episodes_data, 1):
        status = "✓ ÉXITO" if ep['success'] else "✗ FALLO"
        detected = "🎯" if ep['cylinder_detected'] else "❌"
        stats_text += (f"\n Ep {i}: {status} | Detect:{detected} | "
                      f"NEAT:{ep['phase1_steps']} + PPO:{ep['phase2_steps']} = {ep['total_steps']}")
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
           fontsize=9, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    filename = os.path.join(output_dir, 'resumen_hibrido.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Guardada: {filename}")
    plt.close()


# ============================================
# VALIDACIÓN HÍBRIDA
# ============================================
def run_hybrid_episode(neat_net, ppo_model, neat_env, rob, rbsim, idrobot, idobject, episode_num):
    """
    Ejecuta un episodio completo con sistema híbrido NEAT→PPO
    
    Returns:
        dict con datos del episodio (trayectorias, pasos, éxito, etc.)
    """
    
    print(f"\n{'='*70}")
    print(f"EPISODIO {episode_num}")
    print(f"{'='*70}")
    
    # ============================================
    # FASE 1: NEAT - Navegación con obstáculo
    # ============================================
    print(f"\n🧬 FASE 1: NEAT (Esquivar obstáculo y detectar cilindro)")
    print(f"{'─'*70}")
    
    obs_neat, info_neat = neat_env.reset()
    
    phase1_x = []
    phase1_z = []
    phase1_steps = 0
    cylinder_detected = False
    
    # Obtener posición inicial
    rpos = info_neat.get("robot_pos", {"x": 0.0, "z": 0.0})
    phase1_x.append(rpos["x"])
    phase1_z.append(rpos["z"])
    
    done_neat = False
    
    while not done_neat and phase1_steps < NEAT_MAX_STEPS:
        # Predecir acción con NEAT
        output = neat_net.activate(obs_neat)
        action = np.argmax(output)
        
        # Ejecutar
        obs_neat, reward, terminated, truncated, info_neat = neat_env.step(action)
        done_neat = terminated or truncated
        
        # Trackear posición
        rpos = info_neat.get("robot_pos", {"x": 0.0, "z": 0.0})
        phase1_x.append(rpos["x"])
        phase1_z.append(rpos["z"])
        
        phase1_steps += 1
        
        # ¿Detectó el cilindro?
        if info_neat.get('goal_reached', False):
            cylinder_detected = True
            print(f"    🎯 ¡CILINDRO DETECTADO en paso {phase1_steps}!")
            break
        
        # DEBUG cada 10 pasos
        if phase1_steps % 10 == 0:
            print(f"    Step {phase1_steps}: IR={obs_neat} | Wall contact: {info_neat.get('wall_contacted', False)}")
    
    print(f"\n📊 Fase 1 completada:")
    print(f"    • Pasos: {phase1_steps}")
    print(f"    • Cilindro detectado: {'✓' if cylinder_detected else '✗'}")
    print(f"    • Posición final: ({phase1_x[-1]:.1f}, {phase1_z[-1]:.1f})")
    
    # ============================================
    # FASE 2: PPO - Acercamiento al cilindro
    # ============================================
    phase2_x = []
    phase2_z = []
    phase2_steps = 0
    success = False
    
    if cylinder_detected:
        print(f"\n🤖 FASE 2: PPO (Acercamiento al cilindro)")
        print(f"{'─'*70}")
        
        # CREAR el entorno PPO ahora que el cilindro fue detectado
        print(f"    🔧 Creando entorno PPO...")
        ppo_env = RoboboPPOEnv(rob, rbsim, idrobot, idobject)
        
        # IMPORTANTE: NO resetear el entorno, continuar desde la posición actual
        # Crear observación inicial para PPO basada en el estado actual
        obs_ppo, info_ppo = ppo_env.reset()  # Reset para inicializar estado interno
        
        # Trackear posición inicial de fase 2
        rpos_ppo = info_ppo.get("robot_pos", {"x": 0.0, "z": 0.0})
        phase2_x.append(rpos_ppo["x"])
        phase2_z.append(rpos_ppo["z"])
        
        done_ppo = False
        
        while not done_ppo and phase2_steps < PPO_MAX_STEPS:
            # Predecir acción con PPO
            action_ppo, _states = ppo_model.predict(obs_ppo, deterministic=True)
            
            # Ejecutar
            obs_ppo, reward_ppo, terminated_ppo, truncated_ppo, info_ppo = ppo_env.step(action_ppo)
            done_ppo = terminated_ppo or truncated_ppo
            
            # Trackear posición
            rpos_ppo = info_ppo.get("robot_pos", {"x": 0.0, "z": 0.0})
            phase2_x.append(rpos_ppo["x"])
            phase2_z.append(rpos_ppo["z"])
            
            phase2_steps += 1
            
            # ¿Llegó al objetivo?
            if info_ppo.get('goal_reached', False):
                success = True
                print(f"    🎉 ¡OBJETIVO ALCANZADO en paso {phase2_steps}!")
                break
            
            # ¿Colisionó?
            if info_ppo.get('collided', False):
                print(f"    💥 Colisión en paso {phase2_steps}")
                break
        
        print(f"\n📊 Fase 2 completada:")
        print(f"    • Pasos: {phase2_steps}")
        print(f"    • Éxito: {'✓' if success else '✗'}")
        print(f"    • Posición final: ({phase2_x[-1]:.1f}, {phase2_z[-1]:.1f})")
        
        # Cerrar entorno PPO después de usarlo
        try:
            ppo_env.close()
        except:
            pass
    else:
        print(f"\n⚠️  FASE 2 OMITIDA: No se detectó el cilindro en Fase 1")
    
    # ============================================
    # RESUMEN DEL EPISODIO
    # ============================================
    total_steps = phase1_steps + phase2_steps
    
    print(f"\n{'='*70}")
    print(f"RESUMEN EPISODIO {episode_num}:")
    print(f"  • Cilindro detectado: {'✓' if cylinder_detected else '✗'}")
    print(f"  • Objetivo alcanzado: {'✓' if success else '✗'}")
    print(f"  • Fase 1 (NEAT): {phase1_steps} pasos")
    print(f"  • Fase 2 (PPO): {phase2_steps} pasos")
    print(f"  • Total: {total_steps} pasos")
    print(f"{'='*70}")
    
    return {
        'phase1_x': phase1_x,
        'phase1_z': phase1_z,
        'phase2_x': phase2_x,
        'phase2_z': phase2_z,
        'phase1_steps': phase1_steps,
        'phase2_steps': phase2_steps,
        'total_steps': total_steps,
        'cylinder_detected': cylinder_detected,
        'success': success,
    }


# ============================================
# MAIN
# ============================================
def main():
    print("=" * 80)
    print("VALIDACIÓN HÍBRIDA: NEAT (Fase 1) + PPO (Fase 2)")
    print("=" * 80)
    
    os.makedirs(VALIDATION_DIR, exist_ok=True)
    
    # ============================================
    # VERIFICAR ARCHIVOS
    # ============================================
    if not os.path.exists(NEAT_GENOME_PATH):
        print(f"❌ ERROR: No se encuentra {NEAT_GENOME_PATH}")
        print("Ejecuta primero train_neat_ultrafast_3.py")
        return
    
    if not os.path.exists(NEAT_CONFIG_PATH):
        print(f"❌ ERROR: No se encuentra {NEAT_CONFIG_PATH}")
        return
    
    if not os.path.exists(PPO_MODEL_PATH):
        print(f"❌ ERROR: No se encuentra {PPO_MODEL_PATH}")
        return
    
    # ============================================
    # CARGAR MODELOS
    # ============================================
    print(f"\n📦 Cargando modelos...")
    
    # NEAT
    print(f"  • NEAT config: {NEAT_CONFIG_PATH}")
    neat_config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        NEAT_CONFIG_PATH
    )
    
    print(f"  • NEAT genome: {NEAT_GENOME_PATH}")
    with open(NEAT_GENOME_PATH, 'rb') as f:
        neat_genome = pickle.load(f)
    print(f"    ✓ Fitness entrenamiento: {neat_genome.fitness:.2f}")
    
    neat_net = neat.nn.FeedForwardNetwork.create(neat_genome, neat_config)
    
    # PPO
    print(f"  • PPO model: {PPO_MODEL_PATH}")
    ppo_model = PPO.load(PPO_MODEL_PATH)
    print(f"    ✓ Modelo cargado")
    
    # ============================================
    # CONECTAR CON ROBOBO
    # ============================================
    print(f"\n🤖 Conectando con RoboboSim...")
    rob = Robobo(ROBOBO_IP)
    rbsim = RoboboSim(ROBOBO_IP)
    
    idrobot = 0
    idobject = "CYLINDERMIDBALL"
    
    # Crear solo el entorno NEAT por ahora (PPO se creará cuando se necesite)
    neat_env = RoboboNEATObstacleEnv(rob, rbsim, idrobot, idobject, max_steps=NEAT_MAX_STEPS)
    
    print(f"    ✓ Entorno NEAT creado")
    
    # ============================================
    # EJECUTAR EPISODIOS DE VALIDACIÓN
    # ============================================
    all_episodes_data = []
    
    print(f"\n{'='*80}")
    print(f"Ejecutando {N_TEST_EPISODES} episodios híbridos...")
    print(f"{'='*80}")
    
    for episode in range(N_TEST_EPISODES):
        episode_data = run_hybrid_episode(
            neat_net, ppo_model, 
            neat_env, rob, rbsim, idrobot, idobject,
            episode + 1
        )
        all_episodes_data.append(episode_data)
        
        # Generar visualización individual
        plot_hybrid_trajectory(
            {'x': episode_data['phase1_x'], 'z': episode_data['phase1_z']},
            {'x': episode_data['phase2_x'], 'z': episode_data['phase2_z']},
            episode + 1,
            success=episode_data['success'],
            output_dir=VALIDATION_DIR
        )
        
        # Pequeña pausa entre episodios
        time.sleep(1)
    
    # ============================================
    # GENERAR RESUMEN
    # ============================================
    print(f"\n{'='*80}")
    print(f"Generando resumen de validación...")
    plot_hybrid_summary(all_episodes_data, VALIDATION_DIR)
    
    # Estadísticas finales
    num_detected = sum(1 for ep in all_episodes_data if ep['cylinder_detected'])
    num_success = sum(1 for ep in all_episodes_data if ep['success'])
    detection_rate = (num_detected / N_TEST_EPISODES) * 100
    success_rate = (num_success / N_TEST_EPISODES) * 100
    
    print(f"\n{'='*80}")
    print(f"✅ VALIDACIÓN HÍBRIDA COMPLETADA")
    print(f"{'='*80}")
    print(f"Tasa de detección (Fase 1): {detection_rate:.1f}% ({num_detected}/{N_TEST_EPISODES})")
    print(f"Tasa de éxito total: {success_rate:.1f}% ({num_success}/{N_TEST_EPISODES})")
    print(f"Visualizaciones guardadas en: {VALIDATION_DIR}/")
    print(f"{'='*80}\n")
    
    # ============================================
    # CERRAR CONEXIONES
    # ============================================
    try:
        neat_env.close()
    except:
        pass
    finally:
        try:
            rob.disconnect()
            rbsim.disconnect()
        except:
            pass


if __name__ == "__main__":
    main()
