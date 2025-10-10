import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
from env10 import RoboboEnv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

ROBOBO_IP = os.getenv("ROBOBO_IP", "localhost")

MODEL_PATH_ENV = os.getenv("MODEL_PATH", "").strip()
MODEL_NAME = os.getenv("MODEL_NAME", "ppo_robobo_v2_final").strip()

if MODEL_PATH_ENV:
    MODEL_PATH = os.path.normpath(os.path.join(PROJECT_ROOT, MODEL_PATH_ENV))
else:
    MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
    cleaned = MODEL_NAME.replace("\\", "/")
    if cleaned.startswith("models/"):
        cleaned = cleaned[len("models/"):]
    MODEL_PATH = os.path.join(MODELS_DIR, f"{cleaned}.zip")

EPISODES = int(os.getenv("VAL_EPISODES", "5"))
OUTPUT_DIR = os.path.join(BASE_DIR, "validation_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print(f"No se encontró el modelo: {MODEL_PATH}")
    models_dir = os.path.join(PROJECT_ROOT, "models")
    if os.path.isdir(models_dir):
        avail = [f for f in os.listdir(models_dir) if f.endswith(".zip")]
        print("Modelos en models/: " + (", ".join(avail) if avail else "(ninguno)"))
        ckpt_dir = os.path.join(models_dir, "checkpoints")
        if os.path.isdir(ckpt_dir):
            ckpts = [f"checkpoints/{f}" for f in os.listdir(ckpt_dir) if f.endswith(".zip")]
            if ckpts: print("Checkpoints: " + ", ".join(ckpts))
    raise SystemExit(1)

rob = Robobo(ROBOBO_IP); rbsim = RoboboSim(ROBOBO_IP)
env = RoboboEnv(rob, rbsim, 0, "CYLINDERMIDBALL")

try:
    model = PPO.load(MODEL_PATH, env=env, print_system_info=False)
except Exception as e:
    print(f"Error cargando modelo: {e}")
    env.close(); raise SystemExit(1)

all_rewards, all_lengths = [], []
successes = 0

print(f"Validando {EPISODES} episodios...")
for ep in range(1, EPISODES + 1):
    obs, info = env.reset()
    terminated = truncated = False
    xs = [info["position"]["x"]]; ys = [info["position"]["y"]]
    target_pos = env.robosim.getObjectLocation(env.objectID)["position"]
    ep_reward, steps = 0.0, 0

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += float(reward); steps += 1
        xs.append(info["robot_pos"]["x"]); ys.append(info["robot_pos"]["y"])

    all_rewards.append(ep_reward); all_lengths.append(steps)
    if info.get("goal_reached", False): successes += 1

    # Plot trayectoria con ejes auto
    xs_arr, ys_arr = np.array(xs), np.array(ys)
    pad = 0.5
    x_min, x_max = xs_arr.min() - pad, xs_arr.max() + pad
    y_min, y_max = ys_arr.min() - pad, ys_arr.max() + pad

    plt.figure(figsize=(8, 8))
    plt.plot(xs, ys, marker="o", markersize=3, linestyle="-", linewidth=1.5, alpha=0.8, label="Trayectoria")
    plt.scatter([xs[0]], [ys[0]], s=120, marker="s", label="Inicio")
    plt.scatter([xs[-1]], [ys[-1]], s=120, marker="D", label="Final")
    plt.scatter([target_pos["x"]], [target_pos["y"]], s=140, marker="X", label="Objetivo")
    plt.gca().add_patch(plt.Circle((target_pos["x"], target_pos["y"]), 0.5, fill=False, linestyle="--", linewidth=1.2))
    plt.title(f"Ep {ep} • R={ep_reward:.2f} • Pasos={steps} • "
              f"{'ÉXITO' if info.get('goal_reached', False) else 'FALLO'}")
    plt.xlabel("X"); plt.ylabel("Y"); plt.xlim(x_min, x_max); plt.ylim(y_min, y_max)
    plt.axis("equal"); plt.grid(True, alpha=0.3); plt.legend(loc="best", fontsize=9)
    out_png = os.path.join(OUTPUT_DIR, f"trayectoria_ep_{ep}.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close()

    print(f"Ep {ep}: R={ep_reward:.2f}, pasos={steps}, goal={'sí' if info.get('goal_reached', False) else 'no'}")

print(f"\nResumen: episodios={EPISODES}, éxitos={successes} ({successes/EPISODES*100:.1f}%), "
      f"R_media={np.mean(all_rewards):.2f}±{np.std(all_rewards):.2f}, "
      f"L_media={np.mean(all_lengths):.1f} pasos. PNGs en {OUTPUT_DIR}")

try:
    env.close()
finally:
    try: rob.disconnect()
    except Exception: pass
print("Listo.")
