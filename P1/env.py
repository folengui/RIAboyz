# Archivo: robobo_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import time

# Importamos las librerías de Robobo
from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim 
from robobopy.utils.IR import IR
from robobopy.utils.Color import Color


# === CONSTANTES ===
# Umbral de los sensores IR para considerar un choque
IR_COLLISION_THRESHOLD = 400
# Distancia objetivo para considerar que el robot ha llegado
GOAL_DISTANCE_THRESHOLD = 0.5
# Máximo de pasos por episodio
MAX_EPISODE_STEPS = 100

ACTIONS = [
    (10,10,1),     #Continua recto 1s
    (10,10,0.5),   #Continua recto 0.5s
    (10,0,0.5),    #Gira izquierda poco
    (0,10,0.5),    #Gira derecha poco
    (10,0,1),    #Gira izquierda mucho
    (0,10,1)     #Gira derecha mucho

]

class RoboboEnv(gym.Env):
    """
    Entorno personalizado de Robobo para Gymnasium.
    El agente aprenderá a acercarse a un cilindro rojo evitando chocar con las paredes.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, robobo:Robobo,robobosim:RoboboSim, idrobot:int, idobject: str):
        super(RoboboEnv, self).__init__()
        
        self.roboboID = idrobot
        self.objectID = idobject
        self.objectColor = Color.RED

        # Guardamos la instancia de conexión con el robot
        self.robobo = robobo
        self.robobo.connect()
        self.robobo.moveTiltTo(90,5,True)

        # Creamos una instancia para usar las funciones específicas del simulador
        self.robosim = robobosim
        self.robosim.connect()

        self.locRobotinit = self.robosim.getRobotLocation(idrobot)
        self.posRobotinit = self.locRobotinit["position"]
        self.rotRobotinit = self.locRobotinit["rotation"]

        #self.posRobotinitX = self.posRobot["x"]
        #self.posRbotinitY = self.posRobot["y"]
        #self.posRobotinitZ = self.posRobot["z"]

        self.locObjectinit = self.robosim.getObjectLocation(idobject)
        self.posObjectinit = self.locObjectinit["position"]
        self.rotObjectinit = self.locObjectinit["rotation"]
        #self.posObjectinitX = self.posObject["x"]
        #self.posObjectinitY = self.posObject["y"]
        #self.posObjectinitZ = self.posObject["z"]

        # === 1. Espacio de Acciones ===
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.posX_blob_before = 0
        self.size_blob_before = 0

        # === 2. Espacio de Observaciones ===
        # Definimos lo que el agente "ve". Un array de 2 valores continuos:
        # obs[0] -> Distancia al objetivo
        # obs[1] -> Ángulo hacia el objetivo
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

        # === 3. Variables Internas del Entorno ===
        self.current_step = 0
        self.previous_distance = 0.0
        self.episode = 0
        self.reward_history = []

    def _get_obs(self):
        """
        Método privado para obtener la observación actual del simulador.
        """
        blob = self.robobo.readColorBlob(self.objectColor)

        size = blob.size
        pos_x = blob.posx

        # Devolvemos la observación como un array de NumPy
        return np.array([size, pos_x], dtype=np.float32)

    def reset(self, seed=None, options=None):
        """
        Reinicia el entorno para un nuevo episodio.
        """
        super().reset(seed=seed)
        
        # --- Reposicionamiento Aleatorio ---
        # Colocamos al robot y al cilindro en posiciones aleatorias para que el agente generalice
        self.robosim.setRobotLocation(self.roboboID, self.posRobotinit, self.rotRobotinit)
        self.robosim.setObjectLocation(self.objectID, self.posObjectinit,self.rotObjectinit)        
        self.robobo.moveTiltTo(90,5,True)
        self.current_step = 0
        self.episode += 1

        # Obtenemos la observación inicial
        observation = self._get_obs()
        self.previous_distance = observation[0]
        info = {"position": self.robosim.getRobotLocation(self.roboboID)["position"]}
        print("RESETING EPISODE...........")
        print(f"Num steps: {self.step}    Mean reward: {np.mean(self.reward_history)}")
        return observation, info

    def step(self, action):
        """
        Ejecuta un paso en el entorno a partir de una acción.
        """
        self.current_step += 1

        # --- 1. Traducir y Ejecutar Acción ---
        rw, lw, duration = ACTIONS[action]
        self.robobo.moveWheelsByTime(rw, lw, duration) # Movemos por 0.2s
        
        # --- 2. Obtener Nueva Observación ---
        observation = self._get_obs()
        size_blob_now = observation[0]
        posX_blob_now = observation[1]

        # --- 3. Calcular la Recompensa ---
        reward = 0
        terminated = False # Por defecto, el episodio no termina

        # Recompensa por acercarse: la diferencia de distancias
        reward =   (np.abs(self.posX_blob_before - 50) - np.abs(posX_blob_now - 50))
        reward = size_blob_now + (size_blob_now - self.size_blob_before) 

        # Actualizamos los blobs para el siguiente paso
        self.posX_blob_before = posX_blob_now
        self.size_blob_before = size_blob_now


        # Penalización grande por chocar
        if self.robobo.readIRSensor(IR.FrontC) > IR_COLLISION_THRESHOLD or self.robobo.readIRSensor(IR.FrontLL) > IR_COLLISION_THRESHOLD or self.robobo.readIRSensor(IR.FrontRR) > IR_COLLISION_THRESHOLD:
            terminated = True
            print("¡Colisión!")

        # --- 4. Comprobar Fin de Episodio por Tiempo ---
        if self.current_step >= MAX_EPISODE_STEPS:
            terminated = True
            print("Tiempo agotado.")

        # Devolvemos la información
        info = {'robot_pos': self.robosim.getRobotLocation(self.roboboID)["position"]}

        self.reward_history.append(reward)

        print(f"Episode-step: {self.episode}-{self.step} Observation: {observation}  Reward: {reward}   Terminated: {terminated}  ")

        return observation, reward, terminated, False, info # El 'False' es por 'truncated'

    def close(self):
        """
        Limpia el entorno al finalizar.
        """
        self.robobo.stopMotors()
        self.robobo.disconnect()
        self.robosim.disconnect()
        print("Entorno cerrado.")
