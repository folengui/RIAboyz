
import os
import torch
from stable_baselines3 import PPO
from env_srl import RoboboEnvSRL
from feature_extractor import MobileNetV2Extractor

def train():
    # Asegurar directorio de modelos
    os.makedirs("models", exist_ok=True)
    
    # 1. Crear entorno
    # Ajustar IP si es necesario
    env = RoboboEnvSRL(robobo_ip='10.20.28.7')
    
    # 2. Configurar PPO con CnnPolicy y nuestro Extractor
    # features_dim debe coincidir con la salida de MobileNetV2Extractor (1280)
    policy_kwargs = dict(
        features_extractor_class=MobileNetV2Extractor,
        features_extractor_kwargs=dict(features_dim=1280),
        net_arch=dict(pi=[128, 64], vf=[128, 64]) # Reducción tras las features
    )
    
    print("Iniciando PPO con MobileNetV2 (SRL)...")
    model = PPO(
        "CnnPolicy", 
        env, 
        verbose=1,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        # tensorboard_log="./ppo_robobo_srl_tensorboard/"
    )
    
    # 3. Entrenar
    # Para probar funcionalidad: 1000 pasos. Para entrenar de verdad: >10000
    TOTAL_TIMESTEPS = 5000 
    print(f"Entrenando por {TOTAL_TIMESTEPS} pasos...")
    
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
        print("Entrenamiento finalizado.")
        
        # 4. Guardar
        save_path = "models/ppo_robobo_srl_mobilenet"
        model.save(save_path)
        print(f"Modelo guardado en {save_path}.zip")
        
    except KeyboardInterrupt:
        print("Entrenamiento interrumpido. Guardando lo que hay...")
        model.save("models/ppo_robobo_srl_interrupted")
        
    finally:
        env.close()

if __name__ == "__main__":
    train()
