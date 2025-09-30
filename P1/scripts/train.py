# scripts/train.py
from envs.robobo_env import RoboboEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import os, csv, time

LOG_DIR = "logs"; MODEL_DIR = "models"; MODEL_NAME = "ppo_robobo"
os.makedirs(LOG_DIR, exist_ok=True); os.makedirs(MODEL_DIR, exist_ok=True)

class CsvLogger(BaseCallback):
    def __init__(self, path, verbose=0):
        super().__init__(verbose); self.path = path; self.file=None; self.writer=None
        self.episodes = 0; self.buf = []; self.t0 = time.time()
    def _on_training_start(self):
        self.file = open(self.path,"w",newline=""); 
        self.writer = csv.DictWriter(self.file, fieldnames=["timesteps","episodes","ep_rew_mean","wallclock_s"])
        self.writer.writeheader()
    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episodes += 1
                self.buf.append(info["episode"]["r"])
                k = min(10, len(self.buf))
                mean_r = sum(self.buf[-k:]) / k
                self.writer.writerow({
                    "timesteps": self.num_timesteps,
                    "episodes": self.episodes,
                    "ep_rew_mean": mean_r,
                    "wallclock_s": int(time.time()-self.t0),
                })
                self.file.flush()
        return True
    def _on_training_end(self):
        if self.file: self.file.close()

def main():
    env = Monitor(RoboboEnv(ip="127.0.0.1", action_mode="continuous"))
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=os.path.join(LOG_DIR,"tb"),
                n_steps=1024, batch_size=256, learning_rate=3e-4, gamma=0.99,
                gae_lambda=0.95, ent_coef=0.0, clip_range=0.2, n_epochs=10)
    cb = CsvLogger(os.path.join(LOG_DIR,"training_metrics.csv"))
    model.learn(total_timesteps=200_000, callback=cb)
    model.save(os.path.join(MODEL_DIR, MODEL_NAME))
    env.close()
    print("Modelo guardado.")

if __name__ == "__main__":
    main()
