# Archivo: validate.py

from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim 
from env2 import RoboboEnv
from stable_baselines3 import PPO
import os
import matplotlib.pyplot as plt

# --- 1. CONFIGURACIÓN ---
ROBOBO_IP = 'localhost'
MODEL_NAME = "ppo_robobo_final"
MODELS_DIR = "models/"
MODEL_PATH = os.path.join(MODELS_DIR, f"{MODEL_NAME}.zip")
EPISODIOS_DE_PRUEBA = 5

# --- 2. CONEXIÓN Y CARGA ---
print("Conectando con RoboboSim...")
rob = Robobo(ROBOBO_IP)
rbsim = RoboboSim(ROBOBO_IP)
idrobot = 0
idobject = "CYLINDERMIDBALL"
env = RoboboEnv(rob,rbsim,idrobot,idobject)
print(f"Cargando modelo desde: {MODEL_PATH}")
model = PPO.load(MODEL_PATH, env=env)



# --- 3. BUCLE DE VALIDACIÓN ---
for episode in range(EPISODIOS_DE_PRUEBA):
    print(f"--- Iniciando Episodio de Prueba {episode + 1} ---")
    
    obs, info = env.reset()
    done = False
    
    # Listas para guardar la trayectoria
    robot_path_x = []
    robot_path_x.append(info["position"]["x"])
    robot_path_y = []
    robot_path_y.append(info["position"]["y"])
    # La posición del objetivo no cambia en el episodio
    target_pos = env.robosim.getObjectLocation(idobject)

    while not done:
        # Usamos deterministic=True para que el agente no explore, sino que elija la mejor acción
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        
        # Guardamos la posición actual para la gráfica
        robot_path_x.append(info['robot_pos']["x"])
        robot_path_y.append(info['robot_pos']["y"])

    print(f"Episodio {episode + 1} finalizado.")

    # --- 4. VISUALIZACIÓN DE LA TRAYECTORIA ---
    
    # --- 4. VISUALIZACIÓN DE LA TRAYECTORIA ---
    plt.figure(figsize=(8, 8))
    plt.plot(robot_path_x, robot_path_y, marker='o', linestyle='-', label='Trayectoria del Robot')
    plt.scatter([robot_path_x[0]], [robot_path_y[0]], s=150, c='green', label='Inicio Robot', zorder=5)
    plt.scatter([target_pos["x"]], [target_pos["y"]], s=150, c='red', marker='X', label='Objetivo', zorder=5)

    plt.title(f'Trayectoria del Robot - Episodio {episode + 1}')
    plt.xlabel('Coordenada X')
    plt.ylabel('Coordenada Y')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.grid(True)
    plt.legend()

    plt.savefig(f"trayectoria_episodio_{episode + 1}.png", bbox_inches="tight", dpi=150)
    print(f"Gráfica de la trayectoria guardada como 'trayectoria_episodio_{episode + 1}.png'")
    # --- 5. CIERRE ---
try:
    env.close()
finally:
    # Asegurar cierre de la conexión con Robobo
    try:
        rob.disconnect()
    except Exception:
        pass
