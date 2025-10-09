# Archivo: validate.py

from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim 
from env2 import RoboboEnv
from stable_baselines3 import PPO
import os
import matplotlib.pyplot as plt


def plot_path_xz(xs, zs, title="Trayectoria en el plano X–Z",
                 figsize=(7, 7), equal_aspect=True, filename=None, show=True):

    xs = np.asarray(xs, dtype=float)
    zs = np.asarray(zs, dtype=float)
    if xs.size == 0 or zs.size == 0:
        raise ValueError("Las listas xs y zs no pueden estar vacías.")
    if xs.shape != zs.shape:
        raise ValueError(f"Longitudes distintas: len(xs)={len(xs)} vs len(zs)={len(zs)}")

    fig, ax = plt.subplots(figsize=figsize)

    # Trayectoria (línea + puntos)
    ax.plot(xs, zs, linestyle='-', marker='o', label='Trayectoria')

    # Inicio y fin
    ax.scatter([xs[0]], [zs[0]], s=120, c='green', label='Inicio', zorder=5)
    ax.scatter([xs[-1]], [zs[-1]], s=120, c='red', marker='X', label='Fin', zorder=6)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Límites dinámicos con acolchado
    min_x, max_x = float(xs.min()), float(xs.max())
    min_z, max_z = float(zs.min()), float(zs.max())
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
    robot_path_z = []
    robot_path_z.append(info["position"]["z"])
    # La posición del objetivo no cambia en el episodio
    target_pos = env.robosim.getObjectLocation(idobject)["position"]

    while not done:
        # Usamos deterministic=True para que el agente no explore, sino que elija la mejor acción
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        
        # Guardamos la posición actual para la gráfica
        robot_path_x.append(info['robot_pos']["x"])
        robot_path_z.append(info['robot_pos']["z"])

    print(f"Episodio {episode + 1} finalizado.")

    # --- 4. VISUALIZACIÓN DE LA TRAYECTORIA ---
    
    plot_path_xz(robot_path_x,robot_path_z)

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



