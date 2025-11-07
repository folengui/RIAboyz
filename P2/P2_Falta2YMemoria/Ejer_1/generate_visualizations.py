"""
Script para generar todas las visualizaciones necesarias para la entrega del ejercicio 1
"""
import pickle
import os
import shutil
import visualize

def load_data():
    """Carga los datos de entrenamiento NEAT"""
    import neat
    
    # Cargar estadísticas de generaciones (ahora es un neat.StatisticsReporter)
    with open('results/generation_stats.pkl', 'rb') as f:
        stats = pickle.load(f)
    
    # Cargar mejor genoma
    with open('models/best_genome.pkl', 'rb') as f:
        genome = pickle.load(f)
    
    # Cargar configuración
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        'config-neat-fast')
    
    return stats, genome, config

def generate_all_visualizations():
    """Genera todas las visualizaciones necesarias"""
    print("Cargando datos de entrenamiento...")
    stats, genome, config = load_data()
    
    output_dir = 'visualizations'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. Gráfica de aprendizaje (fitness evolution)
    print("Generando gráfica de aprendizaje...")
    visualize.plot_stats(stats, 
                        filename=os.path.join(output_dir, 'aprendizaje_fitness.png'))
    
    # 2. Gráfica de especies
    print("Generando gráfica de especies...")
    visualize.plot_species(stats,
                          filename=os.path.join(output_dir, 'especies_evolucion.png'))
    
    # 3. Configuración de red neuronal
    print("Generando topología de red neuronal...")
    node_names = {-1: 'Blob Size Ef.', -2: 'Dist Centro', -3: 'IR Front',
                  0: 'Acción 0', 1: 'Acción 1', 2: 'Acción 2', 3: 'Acción 3'}
    visualize.draw_net(config, genome, 
                      filename=os.path.join(output_dir, 'red_neuronal'),
                      node_names=node_names)
    
    print("\n✅ Visualizaciones generadas correctamente en la carpeta 'visualizations/'")

def copy_trajectory_plots():
    """Copia las gráficas de trayectorias de validación"""
    # Las trayectorias se generan al ejecutar validate_neat.py
    source_validation = 'results/validation'
    dest_dir = 'visualizations'
    
    print("\nCopiando trayectorias 2D de validación...")
    if os.path.exists(source_validation):
        copied = False
        for file in os.listdir(source_validation):
            if file.endswith('.png') and 'trayectoria' in file:
                shutil.copy2(os.path.join(source_validation, file),
                           os.path.join(dest_dir, file))
                print(f"  ✓ {file}")
                copied = True
        
        if not copied:
            print("  ⚠️  No se encontraron archivos de trayectoria.")
            print("  💡 Ejecuta primero: python validate_neat.py")
    else:
        print("  ⚠️  No existe la carpeta de validación.")
        print("  💡 Ejecuta primero: python validate_neat.py")

if __name__ == '__main__':
    print("="*60)
    print("GENERACIÓN DE VISUALIZACIONES - EJERCICIO 1")
    print("="*60)
    
    try:
        generate_all_visualizations()
        copy_trajectory_plots()
        
        print("\n" + "="*60)
        print("✅ VISUALIZACIONES GENERADAS CORRECTAMENTE")
        print("="*60)
        print("\nContenido generado:")
        print("  📊 aprendizaje_fitness.png      - Evolución del fitness")
        print("  📊 especies_evolucion.png       - Evolución de especies")
        print("  🧠 red_neuronal.svg             - Topología de la red neuronal")
        print("  🗺️  trayectoria_episodio_*.png - Trayectorias 2D del robot")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
