# Archivo: tra
from robobopy.Robobo import Robobo
from env2 import RoboboEnv
from robobosim.RoboboSim import RoboboSim 
from stable_baselines3 import PPO
import os

# --- 1. CONFIGURACIÓN ---
# IP del simulador (si está en la misma máquina, es '127.0.0.1')
ROBOBO_IP = 'localhost'
# Nombre para guardar los logs y el modelo
MODEL_NAME = "ppo_robobo_final2"
# Directorios para guardar logs y modelos
LOGS_DIR = "logs/"
MODELS_DIR = "models/"
# Pasos totales de entrenamiento
TOTAL_TIMESTEPS = 20000

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# --- 2. CONEXIÓN Y CREACIÓN DEL ENTORNO ---
print("Conectando con RoboboSim...")
rob = Robobo(ROBOBO_IP)
rbsim = RoboboSim(ROBOBO_IP)

print("Creando el entorno de entrenamiento...")
idrobot = 0
idobject = "CYLINDERMIDBALL"
env = RoboboEnv(rob,rbsim,idrobot,idobject)

# --- 3. CREACIÓN DEL MODELO DE APRENDIZAJE ---
print("Creando el modelo PPO...")
model = PPO('MlpPolicy', env, verbose=True, tensorboard_log=os.path.join(LOGS_DIR, MODEL_NAME))

# --- 4. ENTRENAMIENTO ---
print(f"Iniciando entrenamiento por {TOTAL_TIMESTEPS} pasos...")
model.learn(total_timesteps=TOTAL_TIMESTEPS)
print("¡Entrenamiento completado!")

# --- 5. GUARDADO DEL MODELO ---
model_path = os.path.join(MODELS_DIR, MODEL_NAME)
model.save(model_path)
print(f"Modelo guardado en: {model_path}")

# --- 6. CIERRE DEL ENTORNO ---
env.close()