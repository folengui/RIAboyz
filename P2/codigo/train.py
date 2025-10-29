import neat
import pickle
import os
import numpy as np
from environment import RoboboEnv
from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim

# ⚙️ Inicialización del entorno
MODELS_PATH = "models/"
IP_ENTORNO = 'localhost'
robobo = Robobo(IP_ENTORNO)
robobosim = RoboboSim(IP_ENTORNO)
idrobot = 0
idobject = "CYLINDERMIDBALL"
env = RoboboEnv(robobo, robobosim, idrobot=idrobot, idobject=idobject)

# 🎯 Función de evaluación de genomas
def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        obs, info = env.reset()
        total_reward = 0.0
        done = False
        step = 0

        while not done and step < 100:
            # Red -> acción
            output = net.activate(obs)
            action = np.argmax(output)  # Acción con mayor salida

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1

        genome.fitness = total_reward

# 🧬 Función principal de entrenamiento
def run_neat(config_file):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Entrenar durante N generaciones
    winner = p.run(eval_genomes, 30)

    print("\n🏆 Mejor genoma encontrado:\n", winner)
    return winner

def save_winner(winner, config_path, save_dir):

    os.makedirs(save_dir, exist_ok=True)

 
    genome_path = os.path.join(save_dir, "winner_genome.pkl")
    with open(genome_path, "wb") as f:
        pickle.dump(winner, f)
    print(f" Genoma guardado en: {genome_path}")


    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )
    net = neat.nn.FeedForwardNetwork.create(winner, config)

    network_path = os.path.join(save_dir, "winner_network.pkl")
    with open(network_path, "wb") as f:
        pickle.dump(net, f)
    print(f"Red neuronal guardada en: {network_path}")


if __name__ == "__main__":
    winner = run_neat("config.txt")
    save_winner(winner, config_path="config.txt", save_dir=MODELS_PATH)


    env.close()
