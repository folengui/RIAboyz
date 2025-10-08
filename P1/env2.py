# Archivo: robobo_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging

from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim 
from robobopy.utils.IR import IR
from robobopy.utils.Color import Color

IR_COLLISION_THRESHOLD = 400
MAX_EPISODE_STEPS = 100

# Heurísticos: ajusta a tu simulador
SIZE_MAX = 20000.0  # tamaño de blob máximo razonable
POSX_MAX = 100.0    # posx suele ser [0,100]; ajusta si es distinto
CENTER_X = 50.0     # centro del campo de visión
GOAL_SIZE_THRESHOLD = 15000.0  # “suficientemente cerca” del cilindro
GOAL_POSX_MARGIN = 10.0        # centrado aceptable

ACTIONS = [
    (10, 10, 1.0),
    (10, 10, 0.5),
    (10, -10, 0.5),
    (-10, 10, 0.5),
    (10, -10, 1.0),
    (-10, 10, 1.0),
]

import matplotlib.pyplot as plt

def plot_rewards_bars(rewards, title="Recompensa por paso",
                      figsize=(10, 4), filename=None, show=True, stride_xticks=1):
    steps = list(range(1, len(rewards) + 1))

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(steps, rewards)

    ax.set_title(title)
    ax.set_xlabel("Paso")
    ax.set_ylabel("Recompensa")
    ax.grid(True, axis='y', alpha=0.3)

    # Reducir densidad de etiquetas si hay muchos pasos
    if stride_xticks > 1:
        ax.set_xticks(steps[::stride_xticks])
    elif len(steps) > 50:
        ax.set_xticks(steps[::max(1, len(steps)//20)])

    if filename:
        fig.savefig(filename, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    return fig, ax

logger = logging.getLogger(__name__)

class RoboboEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, robobo: Robobo, robobosim: RoboboSim, idrobot: int, idobject: str):
        super().__init__()
        self.roboboID = idrobot
        self.objectID = idobject
        self.objectColor = Color.RED

        self.robobo = robobo
        self.robobo.connect()
        self.robobo.moveTiltTo(90, 5, True)

        self.robosim = robobosim
        self.robosim.connect()

        loc_robot_init = self.robosim.getRobotLocation(idrobot)
        self.posRobotinit = loc_robot_init["position"]
        self.rotRobotinit = loc_robot_init["rotation"]

        loc_obj_init = self.robosim.getObjectLocation(idobject)
        self.posObjectinit = loc_obj_init["position"]
        self.rotObjectinit = loc_obj_init["rotation"]

        # 1) Acción
        self.action_space = spaces.Discrete(len(ACTIONS))

        # 2) Observación: [blob.size, blob.posx]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([SIZE_MAX, POSX_MAX], dtype=np.float32),
            dtype=np.float32
        )

        # Estado interno
        self.current_step = 0
        self.episode = 0
        self.reward_history = []

        self.size_blob_before = 0.0
        self.posX_blob_before = CENTER_X  # centro como neutro inicial

    def _get_obs(self):
        blob = self.robobo.readColorBlob(self.objectColor)
        size = float(getattr(blob, "size", 0.0))
        pos_x = float(getattr(blob, "posx", CENTER_X))
        # Clip por seguridad al espacio
        size = np.clip(size, self.observation_space.low[0], self.observation_space.high[0])
        pos_x = np.clip(pos_x, self.observation_space.low[1], self.observation_space.high[1])
        return np.array([size, pos_x], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Recolocación (puedes randomizar con self.np_random si quieres)
        self.robosim.setRobotLocation(self.roboboID, self.posRobotinit, self.rotRobotinit)
        self.robosim.setObjectLocation(self.objectID, self.posObjectinit, self.rotObjectinit)
        self.robobo.moveTiltTo(90, 5, True)

        self.current_step = 0
        self.episode += 1

        obs = self._get_obs()
        self.size_blob_before = float(obs[0])
        self.posX_blob_before = float(obs[1])

        info = {"position": self.robosim.getRobotLocation(self.roboboID)["position"]}
        print("RESETING EPISODE...........")
        print(f"Num steps: {self.step}    Mean reward: {np.mean(self.reward_history)}")

        plot_rewards_bars(self.reward_history)
        self.reward_history = []
        return obs, info

    def step(self, action):
        self.current_step += 1

        # Ejecutar acción
        rw, lw, duration = ACTIONS[int(action)]
        self.robobo.moveWheelsByTime(rw, lw, duration)

        # Nueva observación
        obs = self._get_obs()
        size_now, posx_now = float(obs[0]), float(obs[1])

        # ----------------
        # Recompensa (simple y consistente):
        # + progreso al acercarse (aumento de size)
        # + centrado (menor |posx-CENTER_X|)
        # ----------------
        delta_size = (size_now - self.size_blob_before)
        delta_center = abs(self.posX_blob_before - CENTER_X) - abs(posx_now - CENTER_X)
        reward = 0.001 * delta_size + 0.01 * delta_center

        # Actualiza para el próximo paso
        self.size_blob_before = size_now
        self.posX_blob_before = posx_now

        # Eventos de terminación
        terminated = False
        truncated = False

        # Colisión (si decides tratarla como final de episodio)
        collided = (
            self.robobo.readIRSensor(IR.FrontC) > IR_COLLISION_THRESHOLD or
            self.robobo.readIRSensor(IR.FrontLL) > IR_COLLISION_THRESHOLD or
            self.robobo.readIRSensor(IR.FrontRR) > IR_COLLISION_THRESHOLD
        )
        if collided:
            terminated = True
            reward -= 1.0  # penaliza fuerte la colisión

        # Objetivo alcanzado: cerca y centrado
        goal_reached = (size_now >= GOAL_SIZE_THRESHOLD) and (abs(posx_now - CENTER_X) <= GOAL_POSX_MARGIN)
        if goal_reached:
            terminated = True
            reward += 2.0

        # Límite de pasos → truncated
        if self.current_step >= MAX_EPISODE_STEPS:
            truncated = True

        info = {
            "robot_pos": self.robosim.getRobotLocation(self.roboboID)["position"],
            "goal_reached": goal_reached,
            "collided": collided
        }

        self.reward_history.append(reward)
        print(f"Episode-step: {self.episode}-{self.step} Observation: {obs}  Reward: {reward}   Terminated: {terminated}  ")
        return obs, float(reward), terminated, truncated, info

    def render(self):
        # opcional: integrar con visor del simulador si lo tienes
        pass

    def close(self):
        try:
            self.robobo.stopMotors()
        finally:
            self.robobo.disconnect()
            self.robosim.disconnect()


#Al igual util pa chekearlo
# from stable_baselines3.common.env_checker import check_env
# from stable_baselines3 import PPO  # o A2C/DQN según toque

# env = RoboboEnv(robobo, robobosim, idrobot, idobject)
# check_env(env, warn=True)  # te dirá si algo no cuadra

# # (opcional) envolver con TimeLimit en vez de controlar dentro
# # import gymnasium as gym
# # env = gym.wrappers.TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)

# model = PPO("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=50_000)
