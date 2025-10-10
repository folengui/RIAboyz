# test_values.py
from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
from robobopy.utils.Color import Color
from robobopy.utils.IR import IR

rob = Robobo('localhost')
rob.connect()
rob.moveTiltTo(120, 5, True)

rbsim = RoboboSim('localhost')
rbsim.connect()

# Coloca el robot cerca del cilindro y observa
for i in range(20):
    blob = rob.readColorBlob(Color.RED)
    distancia = rob.readIRSensor(IR.FrontC)
    print(f"Paso {i+1}: size={blob.size}, posx={blob.posx}, distancia={distancia}")
    rob.moveWheelsByTime(10, 10, 0.5)  # Avanzar un poco

rob.disconnect()
rbsim.disconnect()