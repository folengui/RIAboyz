# Archivo: validate.py

from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim 
from env3 import RoboboEnv  # ← Importar desde tu env3.py
from stable_baselines3 import PPO
import os
import matplotlib.pyplot as plt
import numpy as np

# ==================== CONFIGURACIÓN ====================
ROBOBO_IP = 'localhost'
MODEL_NAME = "ppo_robobo_v2_final"  # ← Debe coincidir con train_improved.py
MODELS_DIR = "models/"
MODEL_PATH = os.path.join(MODELS_DIR, f"{MODEL_NAME}.zip")
EPISODIOS_DE_PRUEBA = 5
OUTPUT_DIR = "validation_plots/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== CONEXIÓN Y CARGA ====================
print("="*60)
print("INICIANDO VALIDACIÓN DEL MODELO")
print("="*60)

print(f"\n1. Conectando con RoboboSim en {ROBOBO_IP}...")
rob = Robobo(ROBOBO_IP)
rbsim = RoboboSim(ROBOBO_IP)

idrobot = 0
idobject = "CYLINDERMIDBALL"

print("2. Creando entorno de validación...")
env = RoboboEnv(rob, rbsim, idrobot, idobject)

print(f"3. Cargando modelo desde: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    print(f"❌ ERROR: No se encontró el modelo en {MODEL_PATH}")
    print(f"\n💡 Buscando modelos disponibles...")
    
    # Buscar en directorio models
    if os.path.exists(MODELS_DIR):
        print(f"\n📁 Archivos en {MODELS_DIR}:")
        for f in os.listdir(MODELS_DIR):
            if f.endswith('.zip'):
                print(f"   ✓ {f}")
        
        # Buscar en checkpoints
        checkpoint_dir = os.path.join(MODELS_DIR, "checkpoints")
        if os.path.exists(checkpoint_dir):
            print(f"\n📁 Checkpoints disponibles:")
            for f in os.listdir(checkpoint_dir):
                if f.endswith('.zip'):
                    print(f"   ✓ checkpoints/{f}")
    else:
        print(f"   ❌ El directorio {MODELS_DIR} no existe")
    
    print("\n⚠️  Si acabas de entrenar, el modelo debería estar en:")
    print(f"    {MODEL_PATH}")
    exit(1)

try:
    model = PPO.load(MODEL_PATH, env=env)
    print("✅ Modelo cargado exitosamente!")
except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}")
    exit(1)

# ==================== VALIDACIÓN ====================
print(f"\n{'='*60}")
print(f"EJECUTANDO {EPISODIOS_DE_PRUEBA} EPISODIOS DE VALIDACIÓN")
print(f"{'='*60}\n")

# Estadísticas globales
all_rewards = []
all_lengths = []
successful_episodes = 0
collisions = 0

for episode in range(EPISODIOS_DE_PRUEBA):
    print(f"\n{'─'*60}")
    print(f"EPISODIO {episode + 1}/{EPISODIOS_DE_PRUEBA}")
    print(f"{'─'*60}")
    
    obs, info = env.reset()
    terminated = False
    truncated = False
    
    # Listas para trayectoria
    robot_path_x = [info["position"]["x"]]
    robot_path_y = [info["position"]["y"]]
    
    # Posición del objetivo
    target_pos = env.robosim.getObjectLocation(idobject)["position"]
    
    # Estadísticas del episodio
    episode_reward = 0
    steps = 0

    while not (terminated or truncated):
        # Predicción determinista (sin exploración)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Acumular estadísticas
        episode_reward += reward
        steps += 1
        
        # Guardar posición
        robot_path_x.append(info["robot_pos"]["x"])
        robot_path_y.append(info["robot_pos"]["y"])
        
        # Verificar éxito
        if info["goal_reached"]:
            successful_episodes += 1
        if info["collided"]:
            collisions += 1

    # Estadísticas globales
    all_rewards.append(episode_reward)
    all_lengths.append(steps)
    
    # Reporte del episodio
    print(f"\n{'─'*40}")
    print(f"RESULTADO EPISODIO {episode + 1}:")
    print(f"  Pasos: {steps}")
    print(f"  Recompensa total: {episode_reward:.2f}")
    print(f"  Objetivo alcanzado: {'✓ SÍ' if info.get('goal_reached', False) else '✗ NO'}")
    print(f"  Colisión: {'✓ SÍ' if info.get('collided', False) else '✗ NO'}")
    print(f"  Tamaño final del blob: {info.get('size', 0):.0f}")
    print(f"  IR frontal final: {info.get('ir_front', 0):.0f}")
    print(f"  Vio cilindro: {info.get('saw_cylinder', False)}")
    print(f"  Pasos desde última visión: {info.get('steps_since_saw', 999)}")
    if info.get('centered', -1) >= 0:
        print(f"  Distancia al centro: {info.get('centered', -1):.1f}")
    print(f"{'─'*40}")

    # ==================== VISUALIZACIÓN ====================
    plt.figure(figsize=(10, 10))
    
    # Trayectoria del robot
    plt.plot(robot_path_x, robot_path_y, 
             marker='o', markersize=4, linestyle='-', linewidth=2,
             color='blue', alpha=0.6, label='Trayectoria')
    
    # Punto inicial (verde)
    plt.scatter([robot_path_x[0]], [robot_path_y[0]], 
                s=200, c='green', marker='s', 
                edgecolors='black', linewidths=2,
                label='Inicio Robot', zorder=10)
    
    # Punto final (azul oscuro)
    plt.scatter([robot_path_x[-1]], [robot_path_y[-1]], 
                s=200, c='darkblue', marker='D', 
                edgecolors='black', linewidths=2,
                label='Final Robot', zorder=10)
    
    # Objetivo (rojo)
    plt.scatter([target_pos["x"]], [target_pos["y"]], 
                s=300, c='red', marker='X', 
                edgecolors='black', linewidths=2,
                label='Objetivo', zorder=10)
    
    # Círculo de éxito alrededor del objetivo
    circle = plt.Circle((target_pos["x"], target_pos["y"]), 
                        0.5, color='red', fill=False, 
                        linestyle='--', linewidth=2, alpha=0.5)
    plt.gca().add_patch(circle)
    
    # Configuración de la gráfica
    success_text = "✓ ÉXITO" if info.get("goal_reached", False) else "✗ FALLO"
    plt.title(f'Trayectoria - Episodio {episode + 1}\n'
              f'Recompensa: {episode_reward:.2f} | Pasos: {steps} | {success_text}',
              fontsize=14, fontweight='bold')
    plt.xlabel('Coordenada X (m)', fontsize=12)
    plt.ylabel('Coordenada Y (m)', fontsize=12)
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', fontsize=10)
    plt.axis('equal')
    
    # Guardar
    filename = os.path.join(OUTPUT_DIR, f'trayectoria_episodio_{episode + 1}.png')
    plt.savefig(filename, bbox_inches="tight", dpi=150)
    print(f"✅ Gráfica guardada: {filename}")
    plt.close()

# ==================== RESUMEN FINAL ====================
print(f"\n\n{'='*60}")
print("RESUMEN DE LA VALIDACIÓN")
print(f"{'='*60}")
print(f"📊 Episodios evaluados: {EPISODIOS_DE_PRUEBA}")
print(f"✓  Episodios exitosos: {successful_episodes} ({successful_episodes/EPISODIOS_DE_PRUEBA*100:.1f}%)")
print(f"✗  Colisiones: {collisions}")
print(f"\n📈 RECOMPENSAS:")
print(f"   Media: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
print(f"   Máxima: {np.max(all_rewards):.2f}")
print(f"   Mínima: {np.min(all_rewards):.2f}")
print(f"\n⏱️  DURACIÓN:")
print(f"   Media: {np.mean(all_lengths):.1f} pasos")
print(f"   Máxima: {np.max(all_lengths)} pasos")
print(f"   Mínima: {np.min(all_lengths)} pasos")
print(f"{'='*60}\n")

# ==================== GRÁFICA COMPARATIVA ====================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gráfica 1: Recompensas por episodio
axes[0].bar(range(1, EPISODIOS_DE_PRUEBA + 1), all_rewards, 
            color=['green' if r > 0 else 'red' for r in all_rewards],
            alpha=0.7, edgecolor='black')
axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axes[0].axhline(y=np.mean(all_rewards), color='blue', linestyle='--', 
                linewidth=2, label=f'Media: {np.mean(all_rewards):.2f}')
axes[0].set_xlabel('Episodio', fontsize=12)
axes[0].set_ylabel('Recompensa Total', fontsize=12)
axes[0].set_title('Recompensa por Episodio', fontweight='bold', fontsize=14)
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

# Gráfica 2: Duración de episodios
axes[1].bar(range(1, EPISODIOS_DE_PRUEBA + 1), all_lengths, 
            color='steelblue', alpha=0.7, edgecolor='black')
axes[1].axhline(y=np.mean(all_lengths), color='red', linestyle='--', 
                linewidth=2, label=f'Media: {np.mean(all_lengths):.1f}')
axes[1].set_xlabel('Episodio', fontsize=12)
axes[1].set_ylabel('Número de Pasos', fontsize=12)
axes[1].set_title('Duración de los Episodios', fontweight='bold', fontsize=14)
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.suptitle(f'Resumen de Validación - {successful_episodes}/{EPISODIOS_DE_PRUEBA} Éxitos', 
             fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])

summary_file = os.path.join(OUTPUT_DIR, 'resumen_validacion.png')
plt.savefig(summary_file, dpi=300, bbox_inches='tight')
print(f"✅ Resumen guardado: {summary_file}\n")
plt.show()

# ==================== CIERRE ====================
try:
    env.close()
finally:
    try:
        rob.disconnect()
    except:
        pass

print("✅ Validación completada!")