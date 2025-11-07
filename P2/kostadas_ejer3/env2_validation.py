# Versión de env2.py con condiciones de éxito PERMISIVAS para validación


import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging
import matplotlib.pyplot as plt

from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim 
from robobopy.utils.IR import IR
from robobopy.utils.Color import Color

#VARIABLES GLOBALES 
# IR_COLLISION_THRESHOLD = 400
MAX_EPISODE_STEPS = 200
SIZE_SAW_CYLINDER_THRESHOLD = 1.0
SIZE_VISIBLE_MIN = 1.0

# ✨ CONDICIONES DE ÉXITO PERMISIVAS 
IR_SUCCESS_THRESHOLD = 150  
MAX_STEPS_WITHOUT_VISION = 3

# Variables globales del robot
SIZE_MAX = 600.0
POSX_MAX = 100.0
CENTER_X = 50.0

#CONJUNTO DE ACCIONES 
ACTIONS = [
    (10, 10, 0.5),   # 0: Avanzar
    (10, -10, 0.3),  # 1: Girar izquierda
    (-10, 10, 0.3),  # 2: Girar derecha
    (5, 5, 0.5),     # 3: Avanzar lento
]

logger = logging.getLogger(__name__)

class RoboboEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, robobo: Robobo, robobosim: RoboboSim, idrobot: int, idobject: str):
        super().__init__()
        self.roboboID = idrobot
        self.objectID = idobject
        self.objectColor = Color.RED

        self.robobo = robobo
        # Solo conectar si no está ya conectado
        if not hasattr(self.robobo, '_connected') or not self.robobo._connected:
            self.robobo.connect()
            self.robobo.moveTiltTo(120, 5, True)

        self.robosim = robobosim
        # Solo conectar si no está ya conectado
        if not hasattr(self.robosim, '_connected') or not self.robosim._connected:
            self.robosim.connect()

        loc_robot_init = self.robosim.getRobotLocation(idrobot)
        self.posRobotinit = loc_robot_init["position"]
        self.rotRobotinit = loc_robot_init["rotation"]

        loc_obj_init = self.robosim.getObjectLocation(idobject)
        if loc_obj_init is not None:
            self.posObjectinit = loc_obj_init["position"]
            self.rotObjectinit = loc_obj_init["rotation"]
        else:
            # Si no se puede obtener la posición del objeto, usar valores por defecto
            # Esto pasa cuando el entorno se crea después de NEAT
            self.posObjectinit = {"x": 0.0, "y": 0.0, "z": 120.0}
            self.rotObjectinit = {"x": 0.0, "y": 0.0, "z": 0.0}
            print("⚠️  No se pudo obtener posición inicial del objeto, usando valores por defecto")

        # Acción
        self.action_space = spaces.Discrete(len(ACTIONS))

        # Observación: [blob.size, blob.posx]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([SIZE_MAX, POSX_MAX], dtype=np.float32),
            dtype=np.float32
        )

        # Estado interno
        self.current_step = 0
        self.episode = 0
        self.reward_history = []

        self.ARENA = (-100,300)

        self.size_blob_before = 0.0
        self.posX_blob_before = CENTER_X  
        self.last_action = None
        self.action_repeat_count = 0
        self.recently_saw_cylinder = False
        self.steps_since_saw_cylinder = 999

    def _get_obs(self):
        blob = self.robobo.readColorBlob(self.objectColor)
        size = float(getattr(blob, "size", 0.0))
        pos_x = float(getattr(blob, "posx", CENTER_X))
        size = np.clip(size, self.observation_space.low[0], self.observation_space.high[0])
        pos_x = np.clip(pos_x, self.observation_space.low[1], self.observation_space.high[1])
        return np.array([size, pos_x], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.robosim.setRobotLocation(self.roboboID, self.posRobotinit, self.rotRobotinit)

        object_pos = {"x" : float(self.np_random.uniform(*self.ARENA)),
                      "y" : self.posObjectinit["y"],
                      "z" : float(self.np_random.uniform(*self.ARENA))
                      }
        self.robosim.setObjectLocation(self.objectID, object_pos , self.rotObjectinit)
        self.robobo.moveTiltTo(120, 5, True)

        self.current_step = 0
        self.episode += 1

        obs = self._get_obs()
        self.size_blob_before = float(obs[0])
        self.posX_blob_before = float(obs[1])

        info = {"position": self.robosim.getRobotLocation(self.roboboID)["position"]}
        self.reward_history = []
        return obs, info

    def step(self, action):
        self.current_step += 1

        # Ejecutar acción
        rw, lw, duration = ACTIONS[int(action)]
        self.robobo.moveWheelsByTime(rw, lw, duration)

        # Nueva observación
        obs = self._get_obs()
        size_norm, posx_norm = float(obs[0]), float(obs[1])
        size_now = size_norm
        posx_now = posx_norm
        ir_front = self.robobo.readIRSensor(IR.FrontC)

        if size_now > SIZE_SAW_CYLINDER_THRESHOLD:
            self.recently_saw_cylinder = True
            self.steps_since_saw_cylinder = 0
        else:
            self.steps_since_saw_cylinder += 1
            if self.steps_since_saw_cylinder > MAX_STEPS_WITHOUT_VISION:
                self.recently_saw_cylinder = False

        reward = 0.0
       
        # Recompensa por acercarse
        delta_size = size_now - self.size_blob_before
        if delta_size > 0:
            reward += 2.0 * (delta_size / SIZE_MAX)
        elif delta_size < 0:
            reward -= 0.2 * (abs(delta_size) / SIZE_MAX)
       
        # Recompensa por centrado
        if size_now >= SIZE_VISIBLE_MIN and self.size_blob_before >= SIZE_VISIBLE_MIN:
            dist_to_center_now = abs(posx_now - CENTER_X)
            dist_to_center_before = abs(self.posX_blob_before - CENTER_X)
            delta_centering = dist_to_center_before - dist_to_center_now
           
            if delta_centering > 0:
                reward += 0.5 * (delta_centering / CENTER_X)
            elif delta_centering < 0:
                reward -= 0.1 * (abs(delta_centering) / CENTER_X)
       
        # Bonus por proximidad
        if ir_front > 50 and self.recently_saw_cylinder:
            reward += 1.0
       
        # Bonus por ver el cilindro
        if size_now > SIZE_SAW_CYLINDER_THRESHOLD:
            reward += 0.1
 
        # Penalización por perder de vista
        if size_now < SIZE_VISIBLE_MIN and self.size_blob_before >= SIZE_VISIBLE_MIN:
            if ir_front > 100:
                reward -= 0
            else:
                reward -= 3.0
       
        # Penalización suave por no ver nada
        if size_now < SIZE_VISIBLE_MIN and ir_front < 300:
            reward -= 0.005
       
        self.last_action = action
        self.size_blob_before = size_now
        self.posX_blob_before = posx_now
 
        # CONDICIONES DE TERMINACIÓN SIMPLES
        terminated = False
        truncated = False
 
        #ÉXITO SIMPLE
        goal_reached = (ir_front > IR_SUCCESS_THRESHOLD) and self.recently_saw_cylinder
       
        if goal_reached:
            terminated = True
            reward = 30.0
            print(f" ¡OBJETIVO ALCANZADO! IR={ir_front}")
 
        # Colisión
        # collided_obstacle = (
        #     self.robobo.readIRSensor(IR.FrontLL) > IR_COLLISION_THRESHOLD or
        #     self.robobo.readIRSensor(IR.FrontRR) > IR_COLLISION_THRESHOLD
        # )
       
        # if collided_obstacle:
        #     terminated = True
        #     reward = -5.0
 
        # Límite de pasos
        if self.current_step >= MAX_EPISODE_STEPS:
            truncated = True
            terminated = True
            if ir_front > 200 and self.recently_saw_cylinder:
                reward += 5.0

        # Info
        info = {
            "robot_pos": self.robosim.getRobotLocation(self.roboboID)["position"],
            "goal_reached": goal_reached,
            # "collided": collided_obstacle,
            "size": size_now,
            "ir_front": ir_front,
            "saw_cylinder": self.recently_saw_cylinder,
            "steps_since_saw": self.steps_since_saw_cylinder,
            "centered": abs(posx_now - CENTER_X) if size_now >= SIZE_VISIBLE_MIN else -1,
            "delta_size": delta_size,
            "action_repeats": self.action_repeat_count
        }
 
        self.reward_history.append(reward)
       
        # Print periódico
        if self.current_step % 10 == 0 or terminated or truncated:
            if goal_reached:
                status = "✓GOAL"
            else:
                status = "..."
            dist_str = f"{abs(posx_now - CENTER_X):4.1f}" if size_now >= SIZE_VISIBLE_MIN else " N/A"
            vision_indicator = f"👁️-{self.steps_since_saw_cylinder}" if self.recently_saw_cylinder else ""
           
            print(f"Ep{self.episode}-S{self.current_step:3d}: "
                  f"size={size_now:5.0f}, IR={ir_front:4.0f}, dist={dist_str}, "
                  f"vis={vision_indicator}, R={reward:+6.2f} {status}")
       
        return obs, float(reward), terminated, truncated, info

    def close(self):
        try:
            self.robobo.stopMotors()
        finally:
            self.robobo.disconnect()
            self.robosim.disconnect()