import gymnasium as gym
import numpy as np
from gymnasium import spaces

from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim 

rbb = Robobo('localhost')
rbb.connect()

sim = RoboboSim('localhost')
sim.connect()   


#Devuelve lista de strings con IDS de los objetos
objects = sim.getObjects()
#Devuelve lista de ints con IDS de los robots
robots = sim.getRobots()

for object in objects:
    objeto = object
    print(objeto)
    #Devuelve un diccionario con : "position {"x","y","z"}" "rotation{"x","y","z"}"
    locObjeto = sim.getObjectLocation(objeto)
    print(f"Localizacion del cilindro: {locObjeto}")

for robobo in robots:
    robot = robobo
    print(robot)
    #Devuelve un diccionario con : "position {"x","y","z"}" "rotation{"x","y","z"}"
    locRobot = sim.getRobotLocation(robot)
    print(f"Localización robot: {locRobot}")


rbb.moveWheels(10,10)
rbb.wait(2)
rbb.stopMotors()

pollas = rbb.readAllIRSensor()

print(pollas)


