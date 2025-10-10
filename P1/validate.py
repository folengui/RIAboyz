# Archivo: validate.py

from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim 
from env2 import RoboboEnv
from stable_baselines3 import PPO
import os
import matplotlib.pyplot as plt
import numpy as np


def plot_path_xz(xs, zs, xt, zt,
                 title="Trayectoria X–Z",
                 figsize=(7, 7),
                 equal_aspect=True,
                 filename=None,
                 show=True):
    import numpy as np
    import matplotlib.pyplot as plt

    xs = np.asarray(xs, dtype=float)
    zs = np.asarray(zs, dtype=float)
    xt = np.asarray(xt, dtype=float)
    zt = np.asarray(zt, dtype=float)

    if xs.size == 0 or zs.size == 0:
        raise ValueError("xs y zs no pueden estar vacíos")
    if xt.size == 0 or zt.size == 0:
        raise ValueError("xt y zt no pueden estar vacíos")
    if xs.shape != zs.shape:
        raise ValueError(f"len(xs)={len(xs)} != len(zs)={len(zs)}")
    if xt.shape != zt.shape:
        raise ValueError(f"len(xt)={len(xt)} != len(zt)={len(zt)}")

    fig, ax = plt.subplots(figsize=figsize)

    # Solo líneas, sin marcadores
    ax.plot(xs, zs, linestyle='-', label='Trayectoria robot')
    ax.plot(xt, zt, linestyle='-', label='Trayectoria objeto')

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Límites usando ambas curvas
    all_x = np.concatenate([xs, xt])
    all_z = np.concatenate([zs, zt])
    min_x, max_x = float(all_x.min()), float(all_x.max())
    min_z, max_z = float(all_z.min()), float(all_z.max())
    pad_x = max(1e-3, 0.1 * max(1.0, max_x - min_x))
    pad_z = max(1e-3, 0.1 * max(1.0, max_z - min_z))
    ax.set_xlim(min_x - pad_x, max_x + pad_x)
    ax.set_ylim(min_z - pad_z, max_z + pad_z)

    if equal_aspect:
        ax.set_aspect('equal', adjustable='box')

    if filename:
        fig.savefig(filename, bbox_inches='tight', dpi=150)
    if show:
        plt.show()

    return fig, ax

# --- 1. CONFIGURACIÓN ---
ROBOBO_IP = 'localhost'
MODEL_NAME = "ppo_robobo_final2"
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
    robot_path_z = []
    robot_path_z.append(info["position"]["z"])
    object_path_x = []
    object_path_z = []
    while not done:
        # Usamos deterministic=True para que el agente no explore, sino que elija la mejor acción
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        
        # Guardamos la posición actual para la gráfica
        robot_path_x.append(info['robot_pos']["x"])
        robot_path_z.append(info['robot_pos']["z"])

        target_pos = env.robosim.getObjectLocation(idobject)["position"]
        object_path_x.append(target_pos["x"])
        object_path_z.append(target_pos["z"])

    print(f"Episodio {episode + 1} finalizado.")

    # --- 4. VISUALIZACIÓN DE LA TRAYECTORIA ---
    print(object_path_x)
    print(object_path_z)
    plot_path_xz(robot_path_x,robot_path_z,object_path_x,object_path_z)

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



