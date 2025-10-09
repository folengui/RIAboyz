# train_basic.py — PPO mínimo para RoboboEnvBasic

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim

from basico import RoboboEnvBasic

# === 1. Instanciar Robobo y Sim ===
robobo = Robobo("localhost")
sim = RoboboSim("localhost")

# === 2. Crear entorno ===
env = RoboboEnvBasic(robobo, sim, robot_id=0, object_id="CYLINDERMIDBALL")

# (opcional) Chequea interfaz
check_env(env, warn=True)

# === 3. Modelo PPO ===
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    n_steps=512,        # algo más de señal por actualización
    batch_size=256,
    gamma=0.99,
    learning_rate=3e-4,
    ent_coef=0.02,
)

# === 4. Entrenar ===
model.learn(total_timesteps=30_000)

# === 5. Evaluar unas cuantas veces (on-policy)
obs, info = env.reset()
for i in range(15):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Eval step {i} -> a={action}, r={reward:.3f}, info={info}")
    if terminated or truncated:
        obs, info = env.reset()

# === 6. Cerrar ===
env.close()
