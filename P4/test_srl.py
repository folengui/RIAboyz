
import time
from stable_baselines3 import PPO
from env_srl import RoboboEnvSRL
from feature_extractor import MobileNetV2Extractor

def test():
    # 1. Cargar Entorno
    env = RoboboEnvSRL(robobo_ip='10.20.28.7')
    
    # 2. Cargar Modelo
    model_path = "models/ppo_robobo_srl_mobilenet.zip"
    print(f"Cargando modelo de {model_path}...")
    
    try:
        # Importante: Pasar custom_objects si la clase no está en el path por defecto, 
        # aunque SB3 suele manejarlo si la clase está importada.
        model = PPO.load(model_path) # custom_objects not strictly needed if code is same
    except FileNotFoundError:
        print("❌ No se encuentra el modelo entrenado. Ejecuta train_srl.py primero.")
        env.close()
        return

    # 3. Loop de inferencia
    obs, _ = env.reset()
    print("Iniciando control neural visual...")
    
    try:
        while True:
            # Predecir acción (determinista)
            action, _states = model.predict(obs, deterministic=True)
            
            # Ejecutar paso
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Renderizar
            env.render()
            
            if terminated or truncated:
                print("Episodio terminado. Reiniciando...")
                obs, _ = env.reset()
                time.sleep(1.0)
                
    except KeyboardInterrupt:
        print("Deteniendo...")
    finally:
        env.close()

if __name__ == "__main__":
    test()
