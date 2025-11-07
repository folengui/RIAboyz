# list_checkpoints.py
"""
Utilidad para listar y gestionar checkpoints de NEAT.
"""

import os

RESULTS_DIR = "results_neat_obstacle_3_fast"

def list_checkpoints():
    """Lista todos los checkpoints disponibles."""
    if not os.path.exists(RESULTS_DIR):
        print(f"❌ Directorio {RESULTS_DIR} no existe")
        return
    
    checkpoint_files = [f for f in os.listdir(RESULTS_DIR) if f.startswith("neat-checkpoint-")]
    
    if not checkpoint_files:
        print("📂 No hay checkpoints disponibles")
        return
    
    # Ordenar por número de generación
    checkpoint_files.sort(key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else 0)
    
    print("="*60)
    print("📂 CHECKPOINTS DISPONIBLES")
    print("="*60)
    
    for i, checkpoint in enumerate(checkpoint_files, 1):
        filepath = os.path.join(RESULTS_DIR, checkpoint)
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        
        # Extraer generación
        if "emergency" in checkpoint:
            gen = checkpoint.split("gen")[-1]
            print(f"{i}. 🚨 {checkpoint:<40} ({size_mb:.2f} MB) - EMERGENCIA Gen {gen}")
        else:
            gen = checkpoint.split("-")[-1]
            print(f"{i}. ✅ {checkpoint:<40} ({size_mb:.2f} MB) - Generación {gen}")
    
    print("="*60)
    print(f"Total: {len(checkpoint_files)} checkpoints")
    print(f"Ubicación: {RESULTS_DIR}/")
    print("="*60)

if __name__ == "__main__":
    list_checkpoints()
