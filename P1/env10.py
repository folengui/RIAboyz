# Archivo: env3.py
"""
Entorno de Gymnasium para entrenar a Robobo a acercarse y "quedarse" junto a un cilindro.
Modo anti-exploit (rápido y estable):
- Sin colisiones explícitas
- Solo primer "first-seen" bonificado; NO hay bonus por reacquire
- Ver solo bonifica si hay PROGRESO (acercamiento o centrado)
- Penalización suave por alternar giros sin progreso ("spin tax")
- Penaliza ver sin mejorar ("holding cost") para cortar bucles ver/no-progreso
- Éxito robusto: mantener proximidad algunos pasos (dwell)
- Truncado temprano por estancamiento con visión
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
from robobopy.utils.IR import IR
from robobopy.utils.Color import Color

# ==================== CONSTANTES ====================
IR_SUCCESS_THRESHOLD = 200           # éxito por proximidad frontal (IR central)
SIZE_SAW_CYLINDER_THRESHOLD = 1.0    # umbral mínimo de blob para "ver"
MAX_STEPS_WITHOUT_VISION = 5
MAX_EPISODE_STEPS = 100              # puedes subir a 150 si quieres episodios más largos

# Límites de observación
SIZE_MAX = 570.0
POSX_MAX = 100.0
CENTER_X = 50.0
SIZE_VISIBLE_MIN = 1.0

# Acciones (rueda derecha, izquierda, duración)
ACTIONS = [
    (10, 10, 0.5),   # 0: Avanzar
    (10, -10, 0.3),  # 1: Girar derecha
    (-10, 10, 0.3),  # 2: Girar izquierda
    (5, 5, 0.5),     # 3: Avanzar lento
]

# ==================== SHAPING ====================
SUCCESS_REWARD   = 30.0

APPROACH_GAIN    = 3.0     # prioriza acercamiento
CENTERING_GAIN   = 0.6     # boost moderado al centrado
SEE_BONUS        = 0.10    # goteo SOLO si hay progreso (ver abajo)
TIME_PENALTY     = -0.003  # coste por paso (impulsa a resolver)

# Anti-exploit
DELTA_SIZE_EPS   = 0.5     # progreso mínimo en tamaño del blob (px)
DELTA_CENTER_EPS = 0.5     # mejora mínima de centrado (px)
SPIN_TAX         = -0.08   # penaliza alternar giros sin progreso
HOLDING_COST     = -0.01   # coste por ver pero no mejorar (evita farm visual)

# Empuje a avanzar cuando está centrado
CENTER_BAND      = 8.0     # píxeles de tolerancia alrededor del centro
FORWARD_BONUS    = 0.02    # premio al avanzar centrado

# Éxito robusto por “mantener” proximidad
DWELL_STEPS      = 3       # nº de pasos consecutivos cerca para éxito

# Truncado por estancamiento con visión
STALL_WINDOW     = 18      # pasos recientes a vigilar
STALL_MIN_SUM    = 2.0     # suma mínima de deltas de tamaño en la ventana
STALL_PENALTY    = -3.0    # castigo al truncar por estancamiento

class RoboboEnv(gym.Env):
    """Entorno ligero con shaping anti-exploit."""
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, robobo: Robobo, robobosim: RoboboSim, idrobot: int, idobject: str):
        super().__init__()

        self.roboboID = idrobot
        self.objectID = idobject
        self.objectColor = Color.RED

        # Conexión
        self.robobo = robobo
        self.robobo.connect()
        self.robobo.moveTiltTo(120, 5, True)

        self.robosim = robobosim
        self.robosim.connect()

        # Posiciones iniciales
        loc_robot_init = self.robosim.getRobotLocation(idrobot)
        self.posRobotinit = loc_robot_init["position"]
        self.rotRobotinit = loc_robot_init["rotation"]
        loc_obj_init = self.robosim.getObjectLocation(idobject)
        self.posObjectinit = loc_obj_init["position"]
        self.rotObjectinit = loc_obj_init["rotation"]

        # Espacios
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Estado interno
        self.current_step = 0
        self.episode = 0
        self.reward_history = []
        self.size_blob_before = 0.0
        self.posX_blob_before = CENTER_X
        self.last_action = None
        self.recently_saw_cylinder = False
        self.steps_since_saw_cylinder = 999

        # Anti-exploit
        self._first_seen_in_ep = False
        self._delta_size_window = []  # últimos deltas de tamaño
        self._prev_turn = None        # 1 (der), -1 (izq), 0 (no giro)
        self._dwell = 0               # pasos consecutivos “cerca”

    def _get_obs(self):
        """Observación normalizada [size_norm, posx_norm]."""
        blob = self.robobo.readColorBlob(self.objectColor)
        try:
            size = float(blob.size); pos_x = float(blob.posx)
        except (AttributeError, TypeError):
            size = 0.0; pos_x = CENTER_X
        if size < SIZE_VISIBLE_MIN:
            pos_x = CENTER_X

        size_norm = np.clip(size / SIZE_MAX, 0.0, 1.0)
        pos_x_norm = np.clip(pos_x / POSX_MAX, 0.0, 1.0)
        return np.array([size_norm, pos_x_norm], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.robosim.setRobotLocation(self.roboboID, self.posRobotinit, self.rotRobotinit)
        self.robosim.setObjectLocation(self.objectID, self.posObjectinit, self.rotObjectinit)
        self.robobo.moveTiltTo(120, 5, True)

        self.current_step = 0
        self.episode += 1

        obs = self._get_obs()
        self.size_blob_before = float(obs[0]) * SIZE_MAX
        self.posX_blob_before = float(obs[1]) * POSX_MAX
        self.last_action = None
        self.recently_saw_cylinder = False
        self.steps_since_saw_cylinder = 999
        self.reward_history = []

        # anti-exploit
        self._first_seen_in_ep = False
        self._delta_size_window.clear()
        self._prev_turn = None
        self._dwell = 0

        info = {"position": self.robosim.getRobotLocation(self.roboboID)["position"]}
        return obs, info

    def step(self, action):
        self.current_step += 1

        # Ejecutar acción
        rw, lw, duration = ACTIONS[int(action)]
        self.robobo.moveWheelsByTime(rw, lw, duration)

        # Observación y sensores
        obs = self._get_obs()
        size_norm, posx_norm = float(obs[0]), float(obs[1])
        size_now = size_norm * SIZE_MAX
        posx_now = posx_norm * POSX_MAX
        ir_front = self.robobo.readIRSensor(IR.FrontC)

        # Visión previa
        prev_seen = self.recently_saw_cylinder
        prev_gap = self.steps_since_saw_cylinder

        # Actualizar visión
        if size_now > SIZE_SAW_CYLINDER_THRESHOLD:
            self.recently_saw_cylinder = True
            self.steps_since_saw_cylinder = 0
        else:
            self.steps_since_saw_cylinder += 1
            if self.steps_since_saw_cylinder > MAX_STEPS_WITHOUT_VISION:
                self.recently_saw_cylinder = False

        # ==================== RECOMPENSAS ====================
        reward = 0.0

        # 0) Time penalty
        reward += TIME_PENALTY

        # 1) Acercarse (∆size positivo)
        delta_size = size_now - self.size_blob_before
        if delta_size > 0:
            reward += APPROACH_GAIN * (delta_size / SIZE_MAX)

        # 2) Centrarse (mejora |posx - CENTER_X|) si se ve en ambos pasos
        cent_prog = 0.0
        if size_now >= SIZE_VISIBLE_MIN and self.size_blob_before >= SIZE_VISIBLE_MIN:
            dist_now = abs(posx_now - CENTER_X)
            dist_prev = abs(self.posX_blob_before - CENTER_X)
            delta_center = dist_prev - dist_now
            cent_prog = delta_center
            if delta_center > 0:
                reward += CENTERING_GAIN * (delta_center / CENTER_X)
        else:
            dist_now = abs(posx_now - CENTER_X)

        # 3) Bonus por ver SOLO si hay PROGRESO (evita farm)
        if size_now > SIZE_SAW_CYLINDER_THRESHOLD:
            if (delta_size > DELTA_SIZE_EPS) or (cent_prog > DELTA_CENTER_EPS):
                reward += SEE_BONUS
            else:
                # Holding cost: ver pero no mejorar nada
                reward += HOLDING_COST

        # 4) Solo primer "first-seen" del episodio (sin bonus por reacquire)
        if (not prev_seen) and (size_now > SIZE_SAW_CYLINDER_THRESHOLD) and (not self._first_seen_in_ep) and prev_gap >= 999:
            reward += 0.5   # pequeño empujón inicial
            self._first_seen_in_ep = True

        # 5) Spin tax: alternar giros sin progreso
        turn_dir = 0
        if int(action) == 1:   # derecha
            turn_dir = +1
        elif int(action) == 2: # izquierda
            turn_dir = -1
        if self._prev_turn is not None and turn_dir != 0:
            if (self._prev_turn == -turn_dir) and (delta_size <= DELTA_SIZE_EPS) and (cent_prog <= DELTA_CENTER_EPS):
                reward += SPIN_TAX
        if turn_dir != 0:
            self._prev_turn = turn_dir
        elif int(action) in (0, 3):
            self._prev_turn = 0

        # 6) Empuje a avanzar cuando está centrado
        if size_now > SIZE_SAW_CYLINDER_THRESHOLD and dist_now < CENTER_BAND and int(action) in (0, 3):
            reward += FORWARD_BONUS

        # Ventana de progreso para estancamiento
        self._delta_size_window.append(delta_size)
        if len(self._delta_size_window) > STALL_WINDOW:
            self._delta_size_window.pop(0)

        # Actualizar estado previo
        self.last_action = action
        self.size_blob_before = size_now
        self.posX_blob_before = posx_now

        # ==================== TERMINACIÓN ====================
        terminated = False
        truncated = False

        # Éxito robusto: mantener proximidad unos pasos
        close_enough = (ir_front > IR_SUCCESS_THRESHOLD) and self.recently_saw_cylinder
        self._dwell = self._dwell + 1 if close_enough else 0
        goal_reached = (self._dwell >= DWELL_STEPS)
        if goal_reached:
            terminated = True
            reward = SUCCESS_REWARD

        # Truncado por estancamiento con visión (corta loops)
        if (not terminated) and self.recently_saw_cylinder and len(self._delta_size_window) >= STALL_WINDOW:
            if sum(self._delta_size_window) < STALL_MIN_SUM:
                truncated = True
                reward += STALL_PENALTY

        # Límite de pasos
        if (not terminated) and self.current_step >= MAX_EPISODE_STEPS:
            truncated = True

        info = {
            "robot_pos": self.robosim.getRobotLocation(self.roboboID)["position"],
            "goal_reached": goal_reached,
            "collided": False,
            "size": size_now,
            "ir_front": ir_front,
            "saw_cylinder": self.recently_saw_cylinder,
            "steps_since_saw": self.steps_since_saw_cylinder,
            "centered": abs(posx_now - CENTER_X) if size_now >= SIZE_VISIBLE_MIN else -1,
            "delta_size": delta_size,
        }

        self.reward_history.append(reward)
        return obs, float(reward), terminated, truncated, info

    def render(self):
        pass

    def close(self):
        try:
            self.robobo.stopMotors()
        except Exception:
            pass
        finally:
            try:
                self.robobo.disconnect()
            except Exception:
                pass
            try:
                self.robosim.disconnect()
            except Exception:
                pass


# Verificación básica opcional
if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env
    rob = Robobo('localhost'); rbsim = RoboboSim('localhost')
    env = RoboboEnv(rob, rbsim, 0, "CYLINDERMIDBALL")
    try:
        check_env(env, warn=True)
        obs, info = env.reset()
        for _ in range(3):
            a = env.action_space.sample()
            obs, r, done, trunc, info = env.step(a)
            if done or trunc:
                break
    finally:
        env.close()
