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


# === CONSTANTES ===
# Umbral de los sensores IR para considerar un choque
IR_COLLISION_THRESHOLD = 400
# Distancia objetivo para considerar que el robot ha llegado
GOAL_DISTANCE_THRESHOLD = 0.5
# Máximo de pasos por episodio
MAX_EPISODE_STEPS = 750

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

        # Guardamos la instancia de conexión con el robot
        self.robobo = robobo
        self.robobo.connect()

        # Creamos una instancia para usar las funciones específicas del simulador
        self.robosim = robobosim
        self.robosim.connect()

        self.locRobotinit = self.robosim.getRobotLocation(idrobot)
        self.posRobotinit = self.locRobot["position"]
        self.rotRobotinit = self.locRobot["rotation"]

        #self.posRobotinitX = self.posRobot["x"]
        #self.posRbotinitY = self.posRobot["y"]
        #self.posRobotinitZ = self.posRobot["z"]

        self.locObjectinit = self.robosim.getObjectLocation(idobject)
        self.posObjectinit = self.locObject["position"]
        self.rotObjectinit = self.locObject["rotation"]
        #self.posObjectinitX = self.posObject["x"]
        #self.posObjectinitY = self.posObject["y"]
        #self.posObjectinitZ = self.posObject["z"]

        # === 1. Espacio de Acciones ===
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # === 2. Espacio de Observaciones ===
        # Definimos lo que el agente "ve". Un array de 5 valores continuos:
        # obs[0] -> Distancia al objetivo
        # obs[1] -> Ángulo hacia el objetivo
        # obs[2] -> Valor del sensor IR Frontal Central
        # obs[3] -> Valor del sensor IR Frontal Izquierdo
        # obs[4] -> Valor del sensor IR Frontal Derecho
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

        # === 3. Variables Internas del Entorno ===
        self.current_step = 0
        self.previous_distance = 0.0
        self.episode = 0

    def _get_obs(self):
        """
        Método privado para obtener la observación actual del simulador.
        """
        # --- Posición y Orientación ---
        robot_loc = self.robosim.getRobotLocation(self.roboboID)
        robot_pos = robot_loc["position"]
        robot_rot = robot_loc["rotation"]
        target_loc = self.robosim.getObjectLocation(self.objectID)
        target_pos = target_loc["position"]

        # --- Cálculo de Distancia y Ángulo ---
        dist_to_target = math.sqrt((target_pos["x"] - robot_pos["x"])**2 + (target_pos["y"] - robot_pos["y"])**2)

        x_dist = robot_pos["x"] - target_pos["x"]
        y_dist = robot_pos["y"] - target_pos["y"]
        angle_diff = (np.arctan2(y_dist,x_dist) + np.pi) % (2*np.pi) - np.pi

        # --- Lectura de Sensores Infrarrojos ---
        ir_front_center = self.robobo.readIRSensor(IR.FrontC)
        ir_front_left = self.robobo.readIRSensor(IR.FrontLL)
        ir_front_right = self.robobo.readIRSensor(IR.FrontRR)
        
        # Devolvemos la observación como un array de NumPy
        return np.array([dist_to_target, angle_diff, ir_front_center, ir_front_left, ir_front_right], dtype=np.float32)

    def reset(self, seed=None, options=None):
        """
        Reinicia el entorno para un nuevo episodio.
        """
        super().reset(seed=seed)
        
        # --- Reposicionamiento Aleatorio ---
        # Colocamos al robot y al cilindro en posiciones aleatorias para que el agente generalice
        self.robosim.setRobotLocation(self.roboboID, self.posRobotinit, self.rotRobotinit)
        self.robosim.setObjectLocation(self.objectID, self.posObjectinit,self.rotObjectinit)        
        self.current_step = 0
        self.episode += 1

        # Obtenemos la observación inicial
        observation = self._get_obs()
        self.previous_distance = observation[0]
        
        return observation

    def step(self, action):
        """
        Ejecuta un paso en el entorno a partir de una acción.
        """
        self.current_step += 1

        # --- 1. Traducir y Ejecutar Acción ---
        left_wheel_speed = int(action[0] * 100)
        right_wheel_speed = int(action[1] * 100)
        self.rob.moveWheels(left_wheel_speed, right_wheel_speed, 200) # Movemos por 0.2s
        
        # --- 2. Obtener Nueva Observación ---
        observation = self._get_obs()
        dist_now = observation[0]
        ir_c, ir_l, ir_r = observation[2], observation[3], observation[4]

        # --- 3. Calcular la Recompensa ---
        reward = 0
        terminated = False # Por defecto, el episodio no termina

        # Recompensa por acercarse: la diferencia de distancias
        reward = (self.previous_distance - dist_now) * 10

        # Penalización pequeña por cada paso para incentivar la rapidez
        reward -= 0.1

        # Penalización grande por chocar
        if ir_c > IR_COLLISION_THRESHOLD or ir_l > IR_COLLISION_THRESHOLD or ir_r > IR_COLLISION_THRESHOLD:
            reward = -200
            terminated = True
            print("¡Colisión!")

        # Recompensa grande por llegar al objetivo
        if dist_now < GOAL_DISTANCE_THRESHOLD:
            reward = 300
            terminated = True
            print("¡Objetivo alcanzado!")

        # --- 4. Comprobar Fin de Episodio por Tiempo ---
        if self.current_step >= MAX_EPISODE_STEPS:
            terminated = True
            print("Tiempo agotado.")

        # Actualizamos la distancia previa para el siguiente paso
        self.previous_distance = dist_now
        
        # Devolvemos la información
        info = {'robot_pos': self.robosim.get_robot_position()}
        return observation, reward, terminated, False, info # El 'False' es por 'truncated'

    def close(self):
        """
        Limpia el entorno al finalizar.
        """
        self.rob.stop()
        print("Entorno cerrado.")
