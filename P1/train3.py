# Archivo: train_improved.py
"""
Script de entrenamiento OPTIMIZADO para aprendizaje rápido.
Con 20,000 timesteps debería mostrar progreso claro en 30-60 minutos.
"""

from robobopy.Robobo import Robobo
from env3 import RoboboEnv
from robobosim.RoboboSim import RoboboSim
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
import os
import numpy as np

# ==================== CONFIGURACIÓN ====================
ROBOBO_IP = 'localhost'
MODEL_NAME = "ppo_robobo_v2"
LOGS_DIR = "logs/"
MODELS_DIR = "models/"
TOTAL_TIMESTEPS = 20_000  # ← OPTIMIZADO: 20k para ver progreso rápido

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ==================== CALLBACK ====================
class TrainingProgressCallback(BaseCallback):
    """Callback para mostrar progreso"""
    
    def __init__(self, check_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.successful_episodes = 0
        
    def _on_step(self) -> bool:
        # Capturar episodios completados
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                self.episode_rewards.append(info['r'])
                self.episode_lengths.append(info['l'])
        
        # Mostrar progreso
        if self.n_calls % self.check_freq == 0:
            if len(self.episode_rewards) > 0:
                recent = min(50, len(self.episode_rewards))
                mean_reward = np.mean(self.episode_rewards[-recent:])
                mean_length = np.mean(self.episode_lengths[-recent:])
                max_reward = np.max(self.episode_rewards[-recent:])
                
                print(f"\n{'='*60}")
                print(f"📊 PROGRESO: {self.n_calls}/{TOTAL_TIMESTEPS} steps ({self.n_calls/TOTAL_TIMESTEPS*100:.1f}%)")
                print(f"   Episodios: {len(self.episode_rewards)}")
                print(f"   Recompensa media (últimos {recent}): {mean_reward:.2f}")
                print(f"   Recompensa máxima (últimos {recent}): {max_reward:.2f}")
                print(f"   Duración media: {mean_length:.1f} pasos")
                print(f"{'='*60}\n")
        
        return True

# ==================== ENTORNO ====================
print("="*60)
print("CONFIGURACIÓN DE ENTRENAMIENTO RÁPIDO")
print("="*60)

print("\n1. Conectando con RoboboSim...")
rob = Robobo(ROBOBO_IP)
rbsim = RoboboSim(ROBOBO_IP)

idrobot = 0
idobject = "CYLINDERMIDBALL"

print("2. Creando entorno...")
env = Monitor(
    RoboboEnv(rob, rbsim, idrobot, idobject),
    filename=os.path.join(LOGS_DIR, f"{MODEL_NAME}_monitor")
)

# ==================== MODELO PPO OPTIMIZADO ====================
print("3. Creando modelo PPO (configuración rápida)...")

model = PPO(
    'MlpPolicy',
    env,
    learning_rate=5e-4,        # ← AUMENTADO: 5e-4 en lugar de 3e-4
    n_steps=1024,              # ← REDUCIDO: 1024 en lugar de 2048
    batch_size=64,             # Mantenido
    n_epochs=10,               # Mantenido
    gamma=0.99,                # Mantenido
    gae_lambda=0.95,           # Mantenido
    clip_range=0.2,            # Mantenido
    ent_coef=0.02,             # ← AUMENTADO: 0.02 para más exploración
    vf_coef=0.5,               # Mantenido
    max_grad_norm=0.5,         # Mantenido
    verbose=1,
    tensorboard_log=os.path.join(LOGS_DIR, MODEL_NAME)
)

print("✅ Modelo creado!")
print("\nHiperparámetros OPTIMIZADOS para aprendizaje rápido:")
print(f"  - Learning rate: 5e-4 (más alto → aprende más rápido)")
print(f"  - n_steps: 1024 (menos → actualiza más frecuente)")
print(f"  - Entropía: 0.02 (más exploración inicial)")
print(f"  - Episodios: máx 50 pasos (más cortos)")
print(f"  - Recompensas: más generosas")

# ==================== CALLBACKS ====================
# Checkpoint cada 5k steps
checkpoint_callback = CheckpointCallback(
    save_freq=5_000,
    save_path=os.path.join(MODELS_DIR, "checkpoints"),
    name_prefix=MODEL_NAME,
    save_replay_buffer=False,
    save_vecnormalize=False
)

progress_callback = TrainingProgressCallback(check_freq=1000)

# ==================== ENTRENAMIENTO ====================
print(f"\n{'='*60}")
print(f"INICIANDO ENTRENAMIENTO RÁPIDO")
print(f"{'='*60}")
print(f"Total timesteps: {TOTAL_TIMESTEPS:,}")
print(f"Tiempo estimado: 30-60 minutos")
print(f"Checkpoints cada: 5,000 steps")
print(f"Progreso cada: 1,000 steps")
print(f"{'='*60}\n")

try:
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, progress_callback],
        progress_bar=True,
        reset_num_timesteps=False
    )
    
    print("\n" + "="*60)
    print("✅ ¡ENTRENAMIENTO COMPLETADO!")
    print("="*60)
    
except KeyboardInterrupt:
    print("\n" + "="*60)
    print("⚠️  INTERRUMPIDO POR EL USUARIO")
    print("="*60)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    
finally:
    # Guardar modelo
    model_path = os.path.join(MODELS_DIR, f"{MODEL_NAME}_final")
    print(f"\nGuardando modelo en: {model_path}")
    
    try:
        model.save(model_path)
        print("✅ Modelo guardado")
    except Exception as e:
        print(f"❌ Error al guardar: {e}")
    
    # Cerrar
    print("\nCerrando entorno...")
    try:
        env.close()
        print("✅ Cerrado")
    except Exception as e:
        print(f"⚠️  Error: {e}")

# ==================== INSTRUCCIONES ====================
print(f"\n{'='*60}")
print("PRÓXIMOS PASOS")
print(f"{'='*60}")
print(f"\n1. Ver métricas:")
print(f"   tensorboard --logdir {LOGS_DIR}")
print(f"\n2. Graficar resultados:")
print(f"   python plot_training_results.py")
print(f"\n3. Validar modelo:")
print(f"   python validate.py")
print(f"\n💡 Si quieres mejor resultado, aumenta TOTAL_TIMESTEPS a 50k-100k")
print(f"{'='*60}\n")