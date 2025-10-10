
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


pos = {"x": 0, "y": 0, "z":0}
for object in objects:
    objeto = object
    print(objeto)
    #Devuelve un diccionario con : "position {"x","y","z"}" "rotation{"x","y","z"}"
    rotoOB = sim.getObjectLocation(objeto)["position"]
    sim.setObjectLocation(objeto,pos,rotoOB)



