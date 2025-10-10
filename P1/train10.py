# Archivo: train3.py
"""
Entrenamiento eficiente con PPO sobre RoboboEnv.
- Verbose = 0
- Sin TensorBoard por defecto
- Progreso espaciado
- Checkpoints opcionales (desactívalos con SAVE_CKPT=0)
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
from env10 import RoboboEnv

# ==================== CONFIG ====================
ROBOBO_IP = os.getenv("ROBOBO_IP", "localhost")
MODEL_NAME = os.getenv("MODEL_NAME", "ppo_robobo_v2")
LOGS_DIR = os.getenv("LOGS_DIR", "logs")
MODELS_DIR = os.getenv("MODELS_DIR", "models")
TOTAL_TIMESTEPS = int(os.getenv("TOTAL_TIMESTEPS", "10000"))
REPORT_EVERY = int(os.getenv("REPORT_EVERY", "4000"))   # menos I/O
SAVE_CKPT = int(os.getenv("SAVE_CKPT", "1"))            # 0 = no guardar ckpts
CKPT_FREQ = int(os.getenv("CKPT_FREQ", "2000"))        # menos escrituras

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ==================== CALLBACK PROGRESO ====================
class TrainingProgressCallback(BaseCallback):
    """Resumen ligero cada REPORT_EVERY pasos."""
    def __init__(self, check_freq: int = REPORT_EVERY, verbose: int = 0):
        super().__init__(verbose); self.check_freq = check_freq
        self.rewards = []; self.lengths = []
    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                self.rewards.append(info["r"]); self.lengths.append(info["l"])
        if self.n_calls % self.check_freq == 0 and len(self.rewards) > 0:
            k = min(50, len(self.rewards))
            mean_r = float(np.mean(self.rewards[-k:]))
            max_r  = float(np.max(self.rewards[-k:]))
            mean_l = float(np.mean(self.lengths[-k:]))
            print(f"[{self.n_calls}/{TOTAL_TIMESTEPS}] eps={len(self.rewards)} "
                  f"meanR({k})={mean_r:.2f} maxR({k})={max_r:.2f} meanLen={mean_l:.1f}")
        return True

# ==================== ENTORNO ====================
print("Inicializando simulador y entorno...")
rob = Robobo(ROBOBO_IP); rbsim = RoboboSim(ROBOBO_IP)
env = Monitor(RoboboEnv(rob, rbsim, 0, "CYLINDERMIDBALL"),
              filename=os.path.join(LOGS_DIR, f"{MODEL_NAME}_monitor"))

# ==================== PPO ====================
model = PPO(
    "MlpPolicy", env,
    learning_rate=5e-4,
    n_steps=1024,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.02,
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=0,                 # silencioso
    tensorboard_log=None       # desactivado
)

callbacks = [TrainingProgressCallback(REPORT_EVERY)]
if SAVE_CKPT:
    callbacks.append(CheckpointCallback(
        save_freq=CKPT_FREQ,
        save_path=os.path.join(MODELS_DIR, "checkpoints"),
        name_prefix=MODEL_NAME,
        save_replay_buffer=False,
        save_vecnormalize=False
    ))

print(f"Entrenando: total_timesteps={TOTAL_TIMESTEPS:,} (ckpt cada {CKPT_FREQ} | SAVE_CKPT={SAVE_CKPT})")
try:
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        progress_bar=False,
        reset_num_timesteps=False
    )
    print("Entrenamiento finalizado.")
except KeyboardInterrupt:
    print("Entrenamiento interrumpido por el usuario.")
except Exception as e:
    print(f"ERROR en entrenamiento: {e}")
finally:
    # Guardar modelo final
    model_path = os.path.join(MODELS_DIR, f"{MODEL_NAME}_final")
    try:
        model.save(model_path)
        print(f"Modelo guardado en: {model_path}.zip")
    except Exception as e:
        print(f"Error guardando el modelo: {e}")
    try:
        env.close()
    except Exception:
        pass
print("Listo.")
