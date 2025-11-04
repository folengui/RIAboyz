# train_neat_fast.py
"""
Entrenamiento RÁPIDO de NEAT para Robobo - Práctica 2.1
⚡ CONFIGURACIÓN OPTIMIZADA PARA VELOCIDAD
Objetivo: Resultados decentes en ~15-20 minutos
"""

import neat
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
from env_neat import RoboboNEATEnv


class NEATTrainerFast:
    """Clase para entrenar y evaluar genomas de NEAT (versión rápida)"""
    
    def __init__(self, robobo_ip='localhost', max_steps=50):
        self.robobo_ip = robobo_ip
        self.max_steps = max_steps  # Reducido de 100 a 70
        
        # Crear directorios
        self.results_dir = "results_neat_fast"
        self.checkpoints_dir = os.path.join(self.results_dir, "checkpoints")
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        
        # Conexiones (se reutilizan para todos los genomas)
        print("Conectando con RoboboSim...")
        self.rob = Robobo(self.robobo_ip)
        self.rbsim = RoboboSim(self.robobo_ip)
        
        # Environment compartido con MENOS PASOS
        self.idrobot = 0
        self.idobject = "CYLINDERMIDBALL"
        self.env = RoboboNEATEnv(self.rob, self.rbsim, self.idrobot, self.idobject, max_steps=self.max_steps)
        
        # Tracking de estadísticas
        self.generation_stats = []
        self.best_genome = None
        self.best_fitness = -float('inf')
        self.episode_counter = 0  # Contador global de episodios
        self.current_generation = 0
        
        print(f"✓ Trainer inicializado (max_steps={self.max_steps} para velocidad)\n")

    def check_and_reconnect(self):
        """Verifica la conexión y reconecta si es necesario"""
        try:
            # Intentar una operación simple para verificar conexión
            self.rob.readIRSensor(0)
            return True
        except:
            print("⚠️ Conexión perdida. Reconectando...")
            try:
                import time
                self.rob.disconnect()
                self.rbsim.disconnect()
                time.sleep(2)
                
                self.rob = Robobo(self.robobo_ip)
                self.rbsim = RoboboSim(self.robobo_ip)
                self.env.robobo = self.rob
                self.env.robosim = self.rbsim
                
                print("✓ Reconexión exitosa")
                return True
            except Exception as e:
                print(f"❌ Error en reconexión: {e}")
                return False

    def eval_genome(self, genome, config):
        """
        Evalúa un genoma individual en el environment.
        Esta es la función de fitness.
        """
        try:
            # Incrementar contador de episodios
            self.episode_counter += 1
            
            # Crear la red neuronal desde el genoma
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            
            # Resetear environment
            obs, info = self.env.reset()
            done = False
            
            total_fitness = 0.0
            steps = 0
            max_steps = self.env.max_steps
            
            # Para debug: contador de acciones
            action_counts = [0, 0, 0, 0]
            
            while not done and steps < max_steps:
                # Obtener acción de la red neuronal usando la observación ACTUAL
                output = net.activate(obs)
                action = np.argmax(output)  # Seleccionar acción con mayor activación
                action_counts[action] += 1
                
                # Ejecutar acción y ACTUALIZAR la observación para el próximo paso
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                total_fitness += reward
                steps += 1
            
            # El fitness es el acumulado del episodio
            fitness = self.env.get_cumulative_fitness()
            
            # CRÍTICO: Asegurar que fitness es un número válido
            if fitness is None or not isinstance(fitness, (int, float)):
                print(f"    ⚠️ Fitness inválido ({fitness}), usando 0.0")
                fitness = 0.0
            elif not np.isfinite(fitness):  # Verificar que no sea inf o nan
                print(f"    ⚠️ Fitness no finito ({fitness}), usando 0.0")
                fitness = 0.0
            
            # Debug: mostrar info del episodio
            status = "✓ TERMINADO" if done else "⏱️ TRUNCADO"
            action_names = ["Recto", "Izq", "Der", "Lento"]
            actions_str = ", ".join([f"{action_names[i]}:{action_counts[i]}" for i in range(4)])
            print(f"  Episodio {self.episode_counter} acabado | Gen {self.current_generation} | Genoma {genome.key} | "
                  f"Fitness: {fitness:.2f} | Steps: {steps}/{max_steps} | {status}")
            print(f"    Acciones: {actions_str}")
            
            return fitness
            
        except Exception as e:
            print(f"⚠️ Error evaluando genoma {genome.key}: {e}")
            import traceback
            traceback.print_exc()
            return 0.0  # Fitness mínimo en caso de error

    def eval_genomes(self, genomes, config):
        """
        Evalúa todos los genomas de una generación.
        Esta función es llamada por NEAT en cada generación.
        """
        print(f"\n{'='*80}")
        print(f"🧬 Evaluando Generación {self.current_generation} - {len(genomes)} genomas")
        print(f"{'='*80}")
        
        for genome_id, genome in genomes:
            try:
                genome.fitness = self.eval_genome(genome, config)
                # Asegurar que el fitness nunca sea None
                if genome.fitness is None:
                    genome.fitness = 0.0
            except Exception as e:
                print(f"⚠️ Error evaluando genoma {genome_id}: {e}")
                genome.fitness = 0.0
        
        # Incrementar contador de generación para la próxima
        self.current_generation += 1
    
    def save_checkpoint(self, population, species_set, generation):
        """Guarda checkpoint completo para continuar entrenamiento"""
        try:
            checkpoint_data = {
                'generation': generation,
                'population': population,
                'species_set': species_set,
                'rndstate': np.random.get_state(),
                'generation_stats': self.generation_stats,
                'episode_counter': self.episode_counter
            }
            
            checkpoint_path = os.path.join(self.checkpoints_dir, f'checkpoint_gen_{generation}.pkl')
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Guardar mejor genoma por separado (para validar)
            best_genome = max(population.values(), key=lambda g: g.fitness)
            best_path = os.path.join(self.checkpoints_dir, f'best_genome_gen_{generation}.pkl')
            with open(best_path, 'wb') as f:
                pickle.dump(best_genome, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            print(f"💾 Checkpoint Gen {generation} guardado (fitness: {best_genome.fitness:.2f})")
        except Exception as e:
            print(f"⚠️ Error guardando checkpoint: {e}")
            print("Continuando entrenamiento sin guardar...")

    def load_checkpoint(self, checkpoint_path):
        """Carga checkpoint para continuar entrenamiento"""
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            self.current_generation = checkpoint_data['generation']
            self.generation_stats = checkpoint_data.get('generation_stats', [])
            self.episode_counter = checkpoint_data.get('episode_counter', 0)
            np.random.set_state(checkpoint_data['rndstate'])
            
            print(f"✅ Checkpoint Gen {self.current_generation} cargado ({len(checkpoint_data['population'])} genomas)")
            
            return checkpoint_data['population'], checkpoint_data['species_set']
            
        except Exception as e:
            print(f"❌ Error cargando checkpoint: {e}")
            return None, None
    
    def run_evolution(self, config_file, n_generations=15):
        """
        Ejecuta el proceso evolutivo completo (versión rápida).
        
        Args:
            config_file: Ruta al archivo de configuración NEAT
            n_generations: Número de generaciones a entrenar
        """
        # Cargar configuración
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_file
        )
        
        # Crear población
        p = neat.Population(config)
        
        # Añadir reporters para tracking
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        
        # Custom reporter solo para guardar estadísticas
        p.add_reporter(GenerationStatsReporter(self))
        
        print(f"⚡ ENTRENAMIENTO RÁPIDO - {n_generations} generaciones")
        print(f"Población: {config.pop_size} (reducida)")
        print(f"Max steps: {self.max_steps} (reducido)")
        print(f"Inputs: {config.genome_config.num_inputs}")
        print(f"Outputs: {config.genome_config.num_outputs}")
        print(f"⏱️ Tiempo estimado: ~{self.estimate_time(config.pop_size, n_generations, self.max_steps)} minutos")
        print("=" * 60)
        
        # Ejecutar evolución
        start_time = datetime.now()
        winner = p.run(self.eval_genomes, n_generations)
        end_time = datetime.now()
        
        # Guardar mejor genoma
        self.best_genome = winner
        self.best_fitness = winner.fitness
        
        duration = (end_time - start_time).total_seconds() / 60
        
        print("\n" + "=" * 60)
        print("¡EVOLUCIÓN COMPLETADA!")
        print(f"Mejor fitness: {winner.fitness:.2f}")
        print(f"⏱️ Tiempo real: {duration:.1f} minutos")
        print("=" * 60)
        
        # Guardar resultado final
        self.save_results(winner, stats, config)
        
        return winner, stats

    def estimate_time(self, pop_size, n_generations, max_steps):
        """Estima el tiempo de entrenamiento en minutos"""
        # ~0.3 segundos por paso en simulación
        seconds_per_episode = max_steps * 0.3
        seconds_per_generation = pop_size * seconds_per_episode
        total_seconds = n_generations * seconds_per_generation
        return int(total_seconds / 60)

    def save_results(self, winner, stats, config):
        """Guarda el mejor genoma y genera visualizaciones"""
        
        # Guardar mejor genoma
        winner_path = os.path.join(self.results_dir, 'best_genome.pkl')
        with open(winner_path, 'wb') as f:
            pickle.dump(winner, f)
        print(f"\n✓ Mejor genoma guardado: {winner_path}")
        
        # Guardar configuración usada
        config_backup = os.path.join(self.results_dir, 'config-neat-used.txt')
        import shutil
        shutil.copy('config-neat-fast', config_backup)
        
        # Guardar estadísticas
        stats_path = os.path.join(self.results_dir, 'generation_stats.pkl')
        with open(stats_path, 'wb') as f:
            pickle.dump(self.generation_stats, f)
        
        # Crear visualizaciones
        self.plot_training_progress()

    def plot_training_progress(self):
        """Gráfica de progreso del entrenamiento"""
        if not self.generation_stats:
            return
        
        generations = [s['generation'] for s in self.generation_stats]
        max_fitness = [s['max_fitness'] for s in self.generation_stats]
        mean_fitness = [s['mean_fitness'] for s in self.generation_stats]
        min_fitness = [s['min_fitness'] for s in self.generation_stats]
        
        plt.figure(figsize=(14, 6))
        
        plt.plot(generations, max_fitness, 'g-', linewidth=2, label='Mejor Fitness', marker='o')
        plt.plot(generations, mean_fitness, 'b-', linewidth=2, label='Fitness Medio', alpha=0.7)
        plt.plot(generations, min_fitness, 'r-', linewidth=1, label='Peor Fitness', alpha=0.5)
        
        plt.fill_between(generations, min_fitness, max_fitness, alpha=0.2)
        
        plt.xlabel('Generación', fontsize=12, fontweight='bold')
        plt.ylabel('Fitness', fontsize=12, fontweight='bold')
        plt.title('Evolución del Fitness (Entrenamiento Rápido)', 
                 fontsize=14, fontweight='bold')
        plt.legend(fontsize=11, loc='best')
        plt.grid(True, alpha=0.3)
        
        # Anotaciones
        best_gen = generations[max_fitness.index(max(max_fitness))]
        best_fit = max(max_fitness)
        plt.annotate(f'Mejor: {best_fit:.1f}\n(Gen {best_gen})',
                    xy=(best_gen, best_fit),
                    xytext=(10, 10),
                    textcoords='offset points',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        filepath = os.path.join(self.results_dir, 'fitness_evolution.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfica guardada: {filepath}")
        plt.close()

    def close(self):
        """Cierra las conexiones"""
        try:
            self.env.close()
        finally:
            try:
                self.rob.disconnect()
            except:
                pass
            try:
                self.rbsim.disconnect()
            except:
                pass


class GenerationStatsReporter(neat.reporting.BaseReporter):
    """Reporter personalizado para guardar estadísticas por generación"""
    
    def __init__(self, trainer):
        self.trainer = trainer
        self.generation = 0
    
    def post_evaluate(self, config, population, species, best_genome):
        """
        Se llama después de evaluar cada generación.
        Guarda solo estadísticas (sin checkpoints).
        """
        fitnesses = [g.fitness for g in population.values()]
        
        stats = {
            'generation': self.generation,
            'max_fitness': max(fitnesses),
            'mean_fitness': np.mean(fitnesses),
            'min_fitness': min(fitnesses),
            'std_fitness': np.std(fitnesses),
            'num_species': len(species.species)
        }
        
        self.trainer.generation_stats.append(stats)
        self.generation += 1


def main():
    """Función principal con soporte para continuar desde checkpoint"""
    import sys
    
    print("=" * 60)
    print("⚡ ENTRENAMIENTO RÁPIDO CON NEAT - Práctica 2.1")
    print("Objetivo: Resultados decentes en ~15-20 minutos")
    print("=" * 60 + "\n")
    
    # Configuración RÁPIDA
    CONFIG_FILE = 'config-neat-fast'
    N_GENERATIONS = 15      # Reducido de 50
    MAX_STEPS = 50          # Reducido de 100
    ROBOBO_IP = 'localhost'
    
    # Verificar que existe el archivo de configuración
    if not os.path.exists(CONFIG_FILE):
        print(f"❌ ERROR: No se encuentra el archivo {CONFIG_FILE}")
        print("Asegúrate de tener el archivo config-neat-fast en el directorio actual.")
        return
    
    # Crear trainer con configuración rápida
    trainer = NEATTrainerFast(robobo_ip=ROBOBO_IP, max_steps=MAX_STEPS)
    
    try:
        winner, stats = trainer.run_evolution(CONFIG_FILE, N_GENERATIONS)
        
        print("\n" + "=" * 60)
        print("✅ ENTRENAMIENTO COMPLETADO")
        print(f"   Mejor fitness: {winner.fitness:.2f}")
        print(f"   Resultados en: {trainer.results_dir}/")
        print(f"   Mejor genoma guardado: {trainer.results_dir}/best_genome.pkl")
        print("=" * 60 + "\n")
        
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
