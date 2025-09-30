# envs/robobo_env.py
import math
import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from robobopy.Robobo import Robobo
from robobopy.utils.IR import IR
from robobosim.RoboboSim import RoboboSim

IR_COLLISION_THRESHOLD = 400.0   # ajusta tras ver rangos reales de tus IR
GOAL_DISTANCE = 0.50             # metros (o unidades del mundo), ajusta a tu escena
MAX_STEPS = 750                  # limite por episodio
RESET_SETTLE_SECONDS = 0.3
STEP_SENSOR_DELAY = 0.1


def _wrap_pi(rad: float) -> float:
    return (rad + math.pi) % (2 * math.pi) - math.pi


class RoboboEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, ip: str = "127.0.0.1", action_mode: str = "continuous", robot_id: int = 0,
                 target_id: str = "Cylinder"):
        super().__init__()
        self.robot_id = robot_id
        self.target_id = target_id
        self.rob = Robobo(ip, robot_id=self.robot_id)
        self.sim = RoboboSim(ip)
        self.action_mode = action_mode

        self.robot_connected = False
        self.sim_connected = False

        if self.action_mode == "continuous":
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(5)

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

        self.step_count = 0
        self.prev_dist = None

        self.last_pose = (0.0, 0.0, 0.0)
        self.last_target = (0.0, 0.0)
        self.last_ir = (0.0, 0.0, 0.0)
        self._last_robot_raw = None
        self._last_target_raw = None

    def _ensure_connections(self):
        if not self.robot_connected:
            try:
                self.rob.connect()
                self.robot_connected = True
            except Exception as exc:  # pragma: no cover - dependiente del hardware
                raise RuntimeError("No se pudo conectar con el robot Robobo. Esta el simulador/robot encendido?") from exc
        if not self.sim_connected:
            try:
                self.sim.connect()
                self.sim.wait(RESET_SETTLE_SECONDS)
                self.sim_connected = True
            except Exception as exc:  # pragma: no cover - dependiente del hardware
                raise RuntimeError("No se pudo conectar con RoboboSim. Esta el simulador ejecutandose?") from exc

    def _poll_location(self, getter, cache_attr: str):
        loc = getter()
        if not loc:
            deadline = time.time() + 0.5
            while not loc and time.time() < deadline:
                self.sim.wait(0.05)
                loc = getter()
        setattr(self, cache_attr, loc)
        return loc

    def _get_pose(self):
        loc = self._poll_location(lambda: self.sim.getRobotLocation(self.robot_id), "_last_robot_raw")
        if not loc:
            return self.last_pose
        position = loc.get("position", {})
        rotation = loc.get("rotation", {})
        x = float(position.get("x", self.last_pose[0]))
        z = float(position.get("z", self.last_pose[1]))
        yaw_deg = float(rotation.get("y", rotation.get("z", math.degrees(self.last_pose[2]))))
        pose = (x, z, math.radians(yaw_deg))
        self.last_pose = pose
        return pose

    def _get_target(self):
        loc = self._poll_location(lambda: self.sim.getObjectLocation(self.target_id), "_last_target_raw")
        if not loc:
            return self.last_target
        position = loc.get("position", {})
        target = (float(position.get("x", self.last_target[0])), float(position.get("z", self.last_target[1])))
        self.last_target = target
        return target

    def _read_ir_sensor(self, sensor: IR) -> float:
        try:
            return float(self.rob.readIRSensor(sensor))
        except Exception:
            return 0.0

    def _read_irs(self):
        values = (
            self._read_ir_sensor(IR.FrontC),
            self._read_ir_sensor(IR.FrontL),
            self._read_ir_sensor(IR.FrontR),
        )
        self.last_ir = values
        return values

    def _sleep(self, seconds: float):
        try:
            self.rob.wait(seconds)
        except Exception:
            time.sleep(seconds)

    def _drive(self, left_speed: float, right_speed: float, duration: float = 0.2):
        rs = int(np.clip(right_speed, -100, 100))
        ls = int(np.clip(left_speed, -100, 100))
        try:
            self.rob.moveWheelsByTime(rs, ls, duration, wait=False)
        except Exception:
            self.rob.moveWheels(rs, ls)
        self._sleep(duration)

    def _apply_action(self, action):
        if self.action_mode == "continuous":
            left = np.clip(action[0], -1.0, 1.0) * 100
            right = np.clip(action[1], -1.0, 1.0) * 100
            self._drive(left, right)
        else:
            a = int(action)
            if a == 0:
                self._drive(80, 80)
            elif a == 1:
                self._drive(-40, 40)
            elif a == 2:
                self._drive(40, -40)
            elif a == 3:
                self._drive(-60, -60)
            else:
                self.rob.stopMotors()
                self._sleep(STEP_SENSOR_DELAY)

    def _get_obs(self):
        x, z, yaw = self.last_pose if self.step_count else (0.0, 0.0, 0.0)
        tx, tz = self.last_target if self.step_count else (0.0, 0.0)

        pose = self._get_pose()
        target = self._get_target()
        ir_fc, ir_fl, ir_fr = self._read_irs()

        x, z, yaw = pose
        tx, tz = target
        dx, dz = tx - x, tz - z
        dist = math.hypot(dx, dz)
        angle_to_target = math.atan2(dz, dx)
        angle_rel = _wrap_pi(angle_to_target - yaw)
        return np.array([dist, angle_rel, ir_fc, ir_fl, ir_fr], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._ensure_connections()

        if options is None or options.get("reset_sim", True):
            try:
                self.sim.resetSimulation()
            except Exception:
                pass
            self.sim.wait(RESET_SETTLE_SECONDS)

        robot_position = {
            "x": float(np.random.uniform(-4, 4)),
            "y": 0.0,
            "z": float(np.random.uniform(-4, 4)),
        }
        robot_rotation = {
            "x": 0.0,
            "y": float(np.random.uniform(0, 360)),
            "z": 0.0,
        }
        target_position = {
            "x": float(np.random.uniform(-3, 3)),
            "y": 0.2,
            "z": float(np.random.uniform(-3, 3)),
        }

        self.sim.setRobotLocation(self.robot_id, robot_position, robot_rotation)
        self.sim.setObjectLocation(self.target_id, target_position, None)
        self.sim.wait(RESET_SETTLE_SECONDS)

        self.step_count = 0
        self.prev_dist = None

        obs = self._get_obs()
        self.prev_dist = float(obs[0])

        info = {
            "robot_pos": self._last_robot_raw,
            "target_pos": self._last_target_raw,
        }
        return obs, info

    def step(self, action):
        self._ensure_connections()
        self.step_count += 1
        self._apply_action(action)
        self.sim.wait(STEP_SENSOR_DELAY)

        obs = self._get_obs()
        dist_now = float(obs[0])
        ir_fc = float(obs[2])
        ir_fl = float(obs[3])
        ir_fr = float(obs[4])

        if self.prev_dist is None:
            self.prev_dist = dist_now

        reward = (self.prev_dist - dist_now) * 10.0
        reward -= 0.05

        terminated = False
        if max(ir_fc, ir_fl, ir_fr) > IR_COLLISION_THRESHOLD:
            reward -= 200.0
            terminated = True

        if dist_now < GOAL_DISTANCE:
            reward += 300.0
            terminated = True

        truncated = self.step_count >= MAX_STEPS
        self.prev_dist = dist_now

        info = {
            "robot_pos": self._last_robot_raw,
            "target_pos": self._last_target_raw,
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        try:
            self.rob.stopMotors()
        except Exception:
            pass
        if self.robot_connected:
            try:
                self.rob.disconnect()
            except Exception:
                pass
            self.robot_connected = False
        if self.sim_connected:
            try:
                self.sim.disconnect()
            except Exception:
                pass
            self.sim_connected = False
        time.sleep(0.3)
