# robobo_env_basic.py — v1
# - Conexión robusta (SIM -> esperar estado -> ROBOBO)
# - Copias profundas de poses iniciales (el cilindro SÍ resetea)
# - Auto-escalado de distancia (mm/cm -> "metros lógicos")
# - Recompensa con progreso (distancia y ángulo)
# - D_MAX efectivo por episodio (anti-saturación)
# - Telemetría útil en info e impresión fin de episodio

import time
import copy
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim

# ====== Constantes tunables globales ======
D_MAX = 10.0            # fallback / para sugerencias (se usa D por episodio)
MAX_STEPS = 150

# Recompensa base (suave): estado absoluto (no-progreso)
W_THETA_BASE = 0.3      # peso ángulo (|θ|/π)
W_DIST_BASE  = 0.7      # peso distancia (d/D)

# Recompensa de progreso entre pasos
PROG_D   = 1.0          # premio por reducción de distancia
PROG_TH  = 0.2          # premio por reducción de |θ|
STEP_PEN = 0.001        # pequeño coste temporal por paso

# Acciones (right, left) y duración (0: izq, 1: recto, 2: dcha)
FWD_SPEED = (12, 12)
TURN_L    = (-10, 10)
TURN_R    = (10, -10)
ACT_TIME  = 0.3
ACTIONS = [TURN_L, FWD_SPEED, TURN_R]

# Éxito
D_GOAL        = 1.0                     # “metros lógicos”
THETA_GOAL    = np.deg2rad(12.0)        # ±12°
SUCCESS_BONUS = 3.0

class RoboboEnvBasic(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, robobo: Robobo, robosim: RoboboSim, robot_id: int, object_id: str):
        super().__init__()
        self.robobo = robobo
        self.sim = robosim
        self.robot_id = robot_id
        self.object_id = object_id

        # 1) Conecta SIM primero
        self.sim.connect()

        # 2) Espera a que el estado esté disponible (evita None a la primera ejecución)
        def _wait_state(fn, name, timeout=5.0, dt=0.05):
            t0 = time.time()
            out = fn()
            while out is None and (time.time() - t0) < timeout:
                time.sleep(dt)
                out = fn()
            if out is None:
                raise RuntimeError(f"[RoboboEnvBasic] {name} no disponible tras {timeout}s")
            return out

        r0 = _wait_state(lambda: self.sim.getRobotLocation(self.robot_id), "getRobotLocation(robot)")
        o0 = _wait_state(lambda: self.sim.getObjectLocation(self.object_id), "getObjectLocation(object)")

        # 3) Ahora conecta Robobo (motores)
        self.robobo.connect()

        # 4) Guarda poses iniciales con copias profundas (para resetear bien)
        self.robot_pos0 = copy.deepcopy(r0["position"])
        self.robot_rot0 = copy.deepcopy(r0["rotation"])
        self.obj_pos0   = copy.deepcopy(o0["position"])
        self.obj_rot0   = copy.deepcopy(o0["rotation"])

        # Observaciones: [theta_norm ∈ [-1,1], d_norm ∈ [0,1]]
        self.observation_space = spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([ 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(ACTIONS))
        self._action_counts = np.zeros(self.action_space.n, dtype=int)

        # Estado interno
        self.steps = 0
        self._d_max_seen = 0.0       # mayor distancia RAW observada (para sugerir D_MAX)
        self._episode_d_max = None   # D efectivo del episodio (anti-saturación)
        self._prev_d = None          # para progreso de distancia (ya escalada)
        self._prev_abs_theta = None  # para progreso angular
        self._dist_scale = 1.0       # mm->m 0.001, cm->m 0.01, else 1.0

    # ---------- utilidades ----------
    def _pose_rel_raw(self):
        """Devuelve (d_raw, theta) sin escalar distancia. theta en rad, envuelto [-pi,pi]."""
        r = self.sim.getRobotLocation(self.robot_id)
        o = self.sim.getObjectLocation(self.object_id)
        rx, ry = r["position"]["x"], r["position"]["y"]
        ox, oy = o["position"]["x"], o["position"]["y"]

        # En este sim: yaw es rotation['y'] en GRADOS -> rad
        yaw_rad = np.deg2rad(float(r["rotation"]["y"]))

        dx, dy = ox - rx, oy - ry
        d_raw = float(np.hypot(dx, dy))
        theta = float(np.arctan2(dy, dx) - yaw_rad)
        theta = (theta + np.pi) % (2*np.pi) - np.pi

        if d_raw > self._d_max_seen:
            self._d_max_seen = d_raw
        return d_raw, theta

    def _pose_rel(self):
        """Devuelve (d_escalada, theta). d en 'metros lógicos' según _dist_scale."""
        d_raw, theta = self._pose_rel_raw()
        return d_raw * self._dist_scale, theta

    def _obs_from(self, d, theta, D_used):
        """Construye obs normalizada + valores normalizados auxiliares."""
        d_norm = float(np.clip(d / max(D_used, 1e-6), 0.0, 1.0))
        theta_norm = float(np.clip(theta / np.pi, -1.0, 1.0))
        obs = np.array([theta_norm, d_norm], dtype=np.float32)
        return obs, d_norm, theta_norm

    # ---------- Gym API ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Desalinear ±15° para forzar giros
        delta = float(self.np_random.uniform(-15.0, 15.0))
        rot = copy.deepcopy(self.robot_rot0)
        rot["y"] = self.robot_rot0["y"] + delta
        self.sim.setRobotLocation(self.robot_id, self.robot_pos0, rot)
        self.sim.setObjectLocation(self.object_id, self.obj_pos0, self.obj_rot0)

        # Detecta escala de distancia según d_raw inicial
        d0_raw, _ = self._pose_rel_raw()
        if d0_raw > 1000.0:
            self._dist_scale = 0.001   # mm -> m
        elif d0_raw > 10.0:
            self._dist_scale = 0.01    # cm -> m
        else:
            self._dist_scale = 1.0     # ya en m
        print(f"[RoboboEnvBasic] Escala de distancia = {self._dist_scale} (d0_raw={d0_raw:.1f})")

        # D efectivo del episodio: evita saturación (20% sobre d0; mínimo 0.5)
        d0_m, theta0 = self._pose_rel()
        self._episode_d_max = max(d0_m * 1.2, 0.5)

        # Reset de estado episodio
        self.steps = 0
        self._action_counts[:] = 0
        self._prev_d = None
        self._prev_abs_theta = None

        obs, _, _ = self._obs_from(d0_m, theta0, self._episode_d_max)
        return obs, {
            "d_max_seen_raw": round(self._d_max_seen, 3),
            "episode_d_max": round(self._episode_d_max, 3),
            "dist_scale": self._dist_scale
        }

    def step(self, action):
        self.steps += 1

        # Ejecuta acción
        rw, lw = ACTIONS[int(action)]
        self.robobo.moveWheelsByTime(rw, lw, ACT_TIME)

        # Lee estado y normaliza con el D del episodio
        d, theta = self._pose_rel()
        D_used = self._episode_d_max if self._episode_d_max is not None else D_MAX
        obs, d_norm, theta_norm = self._obs_from(d, theta, D_used)

        # -------- Recompensa --------
        abs_th = abs(theta)

        # (1) Termino base (estado absoluto, suave)
        reward = (W_THETA_BASE * (-(abs_th/np.pi)) +
                  W_DIST_BASE  * (-(d / max(D_used, 1e-6))))
        reward = float(reward)

        # (2) Progreso desde el paso anterior
        if self._prev_d is not None:
            reward += PROG_D * (self._prev_d - d) / max(D_used, 1e-6)
        if self._prev_abs_theta is not None:
            reward += PROG_TH * (self._prev_abs_theta - abs_th) / np.pi

        # (3) Coste temporal
        reward -= STEP_PEN

        # Actualiza previos
        self._prev_d = d
        self._prev_abs_theta = abs_th

        # -------- Terminaciones --------
        goal = (d <= D_GOAL) and (abs(theta) <= THETA_GOAL)
        terminated = bool(goal)
        if goal:
            reward += SUCCESS_BONUS

        truncated = self.steps >= MAX_STEPS

        # -------- Telemetría --------
        a = int(action)
        self._action_counts[a] += 1
        info = {
            "d": round(d, 3),
            "theta_deg": round(np.degrees(theta), 1),
            "d_norm": round(d_norm, 3),
            "goal": bool(goal),
            "action_counts": self._action_counts.copy(),
            "episode_d_max": round(D_used, 3),
            "suggested_D_MAX_from_raw": round(self._d_max_seen * 1.1, 3),
            "dist_scale": self._dist_scale
        }

        if terminated or truncated:
            print(f"Fin de episodio {self.steps} steps -> acciones {self._action_counts} | "
                  f"d={info['d']}, theta={info['theta_deg']}°, goal={goal}, "
                  f"D_used={info['episode_d_max']}, scale={self._dist_scale}")

        return obs, float(reward), terminated, truncated, info

    def close(self):
        try:
            print(f"[RoboboEnvBasic] d_raw_max={self._d_max_seen:.3f} -> "
                  f"sugerencia D_MAX_raw≈{self._d_max_seen*1.1:.3f} (margen 10%, antes de escalar).")
            self.robobo.stopMotors()
        finally:
            self.robobo.disconnect()
            self.sim.disconnect()
