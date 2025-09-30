# scripts/validate.py
from envs.robobo_env import RoboboEnv
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import os

MODEL_PATH = os.path.join("models","ppo_robobo.zip")

def run_episode(env, model, idx=1):
    obs, info = env.reset()
    xs = [info["robot_pos"]["x"]]; zs = [info["robot_pos"]["z"]]
    tx, ty, tz = env.sim.get_object_position("Cylinder")

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, rew, term, trunc, info = env.step(action)
        xs.append(info["robot_pos"]["x"]); zs.append(info["robot_pos"]["z"])
        done = term or trunc

    plt.figure(figsize=(6,6))
    plt.plot(xs, zs, marker="o", linewidth=1, label="Trayectoria robot")
    plt.scatter([xs[0]],[zs[0]], s=100, label="Inicio")
    plt.scatter([tx],[tz], s=120, marker="X", label="Objetivo")
    plt.title(f"Trayectoria 2D – Episodio {idx}")
    plt.xlabel("X"); plt.ylabel("Z"); plt.axis("equal"); plt.grid(True); plt.legend()
    out = os.path.join("logs", f"trayectoria_ep{idx}.png")
    plt.tight_layout(); plt.savefig(out, dpi=150)
    print("Guardado:", out)

def main():
    env = RoboboEnv(ip="127.0.0.1", action_mode="continuous")
    model = PPO.load(MODEL_PATH, env=env)
    for i in range(1, 4):
        run_episode(env, model, idx=i)
    env.close()

if __name__ == "__main__":
    main()
