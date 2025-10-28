import neat
import numpy as np
from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
from environment import RoboboEnv

# Crear entorno
rob = Robobo("localhost")
rbsim = RoboboSim("localhost")
env = RoboboEnv(rob, rbsim, 0, "CYLINDERMIDBALL")

# Función de evaluación
def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        obs, info = env.reset()
        done, trunc = False, False
        total_reward = 0

        while not (done or trunc):
            action = net.activate(obs)          # salida de la red
            action = np.tanh(action)            # limitar a [-1, 1]
            obs, reward, done, trunc, info = env.step(action)
            total_reward += reward

        genome.fitness = total_reward

# Configurar NEAT
config_path = "config_neat.txt"  # ruta a tu archivo de configuración
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

# Crear población
pop = neat.Population(config)
pop.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
pop.add_reporter(stats)

# Entrenar
winner = pop.run(eval_genomes, n=50)  # 50 generaciones

print("\nMejor red encontrada:")
print(winner)

# Validar el mejor
net = neat.nn.FeedForwardNetwork.create(winner, config)
obs, info = env.reset()
done, trunc = False, False
while not (done or trunc):
    action = np.tanh(net.activate(obs))
    obs, reward, done, trunc, info = env.step(action)

env.close()
rob.disconnect()
rbsim.disconnect()
