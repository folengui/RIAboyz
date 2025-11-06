# env_neat.py
"""
Entorno Gymnasium simplificado para NEAT.
- Cilindro rojo INMÓVIL
- Fitness basado en acercamiento y centrado
- Sin reward shaping complejo
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim 
from robobopy.utils.IR import IR
from robobopy.utils.Color import Color

# CONSTANTES
IR_COLLISION_THRESHOLD_FRONT = 200
IR_COLLISION_THRESHOLD_SIDE = 200
MAX_EPISODE_STEPS = 100
SIZE_SAW_CYLINDER_THRESHOLD = 1.0
SIZE_VISIBLE_MIN = 1.0

# Condiciones de éxito permisivas
IR_SUCCESS_THRESHOLD = 150 
MAX_EPISODE_STEPS_WITHOUT_VISION = 3

# Normalización
SIZE_MAX = 600.0
POSX_MAX = 100.0
CENTER_X = 50.0

# Acciones disponibles
ACTIONS = [
    (10, 10, 0.5),   # 0: Avanzar recto
    (5, -5, 0.3),  # 1: Girar izquierda
    (-5, 5, 0.3),  # 2: Girar derecha

]


class RoboboNEATEnv(gym.Env):
    """Environment optimizado para NEAT con cilindro inmóvil"""
    
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, robobo: Robobo, robobosim: RoboboSim, 
                 idrobot: int, idobject: str, max_steps: int = MAX_EPISODE_STEPS):
        super().__init__()
        
        self.roboboID = idrobot
        self.objectID = idobject
        self.objectColor = Color.RED
        self.max_steps = max_steps

        self.robobo = robobo
        self.robosim = robobosim
        
        # Conectar solo si no está ya conectado
        if not hasattr(self.robobo, '_connected') or not self.robobo._connected:
            self.robobo.connect()
            
        
        if not hasattr(self.robosim, '_connected') or not self.robosim._connected:
            self.robosim.connect()

        # Esperar a que el simulador esté listo
        import time
        time.sleep(1)  # Dar tiempo al simulador para inicializar

        # Guardar posiciones iniciales con verificación
        loc_robot_init = self.robosim.getRobotLocation(idrobot)
        if loc_robot_init is None:
            print("⚠️ Esperando a que RoboboSim cargue el robot...")
            time.sleep(2)
            loc_robot_init = self.robosim.getRobotLocation(idrobot)
            if loc_robot_init is None:
                raise RuntimeError("❌ ERROR: RoboboSim no devuelve la ubicación del robot. "
                                 "Asegúrate de que el robot esté cargado en el simulador.")
        
        self.posRobotinit = {'x': -1000.0, 'y':39.0, 'z': -400.0}
        self.rotRobotinit = {'x': 3.209609e-06, 'y': 89.98872, 'z': -3.280258e-06}

        loc_obj_init = self.robosim.getObjectLocation(idobject)
        if loc_obj_init is None:
            raise RuntimeError(f"❌ ERROR: No se encuentra el objeto '{idobject}' en RoboboSim. "
                             "Asegúrate de que el cilindro rojo esté en la escena.")
        
        self.posObjectinit = loc_obj_init["position"]
        self.rotObjectinit = loc_obj_init["rotation"]

        # Espacios
        self.action_space = spaces.Discrete(len(ACTIONS))
        
        # Observación MEJORADA: [blob_size_efectivo, distancia_al_centro, ir_front_normalized]
        # blob_size_efectivo = blob_size ponderado por centrado (evita ambigüedad)
        # distancia_al_centro = qué tan descentrado está (0=centrado, 1=muy descentrado)
        self.observation_space = spaces.Box(
            low=np.array([0.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Estado interno
        self.current_step = 0
        self.episode = 0
        self.cumulative_fitness = 0.0
        
        # Para tracking (estilo env2_validation)
        self.size_blob_before = 0.0
        self.posX_blob_before = CENTER_X
        self.saw_cylinder = False
        self.recently_saw_cylinder = False
        self.steps_since_saw_cylinder = 999
        self.steps_seeing_cylinder = 0
        
        # Para penalizar comportamiento repetitivo
        self.last_action = None
        self.action_repeat_count = 0
        
        # Rangos de spawn
        self.ARENA_X = (-100, 300)
        self.ARENA_Z = (-100, 300)
        
        # Constantes para rewards
        self.MAX_STEPS_WITHOUT_VISION = 3

    def _get_obs(self):
        """
        Obtiene observación normalizada con dirección del descentrado.
        
        Ahora el valor de 'distancia_al_centro' tiene signo:
        -1 → completamente a la izquierda
        0 → centrado
        +1 → completamente a la derecha
        """
        blob = self.robobo.readColorBlob(self.objectColor)
        if blob.size > 0:
            self.saw_cylinder = True
        
        size = float(getattr(blob, "size", 0.0))
        pos_x = float(getattr(blob, "posx", CENTER_X))
        ir_front = float(self.robobo.readIRSensor(IR.FrontC))
        
        # Calcular desplazamiento con signo
        # (derecha = positivo, izquierda = negativo)
        offset_from_center = pos_x - CENTER_X
        offset_norm = np.clip(offset_from_center / CENTER_X, -1.0, 1.0)

        # Distancia (magnitud) al centro (para el factor de centrado)
        dist_from_center_norm = abs(offset_norm)

        # Factor de centrado cuadrático (1 centrado, 0 muy ladeado)
        centering_factor = 1.0 - (dist_from_center_norm ** 2)
        
        # Blob size efectivo ponderado por centrado
        size_norm = np.clip(size / SIZE_MAX, 0.0, 1.0)
        blob_size_efectivo = size_norm * centering_factor
        
        # IR normalizado (0–1)
        ir_norm = np.clip(ir_front / 1000.0, 0.0, 1.0)
        
        # Retornar observación con rango [-1, 1]
        # blob_size_efectivo -> 0..1
        # offset_norm -> -1..1  (dirección)
        # ir_norm -> 0..1
        return np.array([blob_size_efectivo, offset_norm, ir_norm], dtype=np.float32)


    def reset(self, seed=None, options=None):
        """Resetea el episodio - robot y cilindro vuelven a posición inicial"""
        super().reset(seed=seed)
        
        import time
        
        # Detener motores primero
        self.robobo.stopMotors()
        
        # Resetear robot a posición inicial
        self.robosim.resetSimulation()
        time.sleep(0.1)
        self.robosim.setRobotLocation(self.roboboID, self.posRobotinit, self.rotRobotinit)
        time.sleep(0.1)
        self.robobo.moveTiltTo(120, 5, True)
        
        # Resetear cilindro a posición inicial fija (IMPORTANTE)
        self.robosim.setObjectLocation(self.objectID, self.posObjectinit, self.rotObjectinit)
        
        # Dar tiempo al simulador para aplicar los cambios
        time.sleep(0.05)
        
        # Estado interno
        self.current_step = 0
        self.episode += 1
        self.cumulative_fitness = 0.0
        self.recently_saw_cylinder = False
        self.steps_since_saw_cylinder = 999
        self.best_ir_seen = 0.0
        self.last_action = None
        self.action_repeat_count = 0
        self.saw_cylinder = False

        obs = self._get_obs()
        # Guardar estado previo para calcular deltas
        self.size_blob_before = float(obs[0]) * SIZE_MAX  # Desnormalizar
        self.posX_blob_before = float(obs[1]) * CENTER_X  # Desnormalizar
        
        info = {
            "position": self.robosim.getRobotLocation(self.roboboID)["position"],
            "target_position": self.posObjectinit
        }
        
        return obs, info

    def step(self, action):
        """Ejecuta una acción y retorna fitness incremental"""
        self.current_step += 1

        # Ejecutar acción
        rw, lw, duration = ACTIONS[int(action)]
        self.robobo.moveWheelsByTime(rw, lw, duration)

        # Observación
        obs = self._get_obs()
        blob_size_efectivo, dist_from_center_norm, ir_norm = obs

        # Desnormalizar para cálculos
        ir_front = ir_norm * 1000.0

        # Obtener valores reales del blob
        blob = self.robobo.readColorBlob(self.objectColor)
        size_now = float(getattr(blob, "size", 0.0))
        posx_now = float(getattr(blob, "posx", CENTER_X))

        # Actualizar tracking de visión
        if size_now > SIZE_SAW_CYLINDER_THRESHOLD:
            self.recently_saw_cylinder = True
            self.steps_since_saw_cylinder = 0
        else:
            self.steps_since_saw_cylinder += 1
            if self.steps_since_saw_cylinder > self.MAX_STEPS_WITHOUT_VISION:
                self.recently_saw_cylinder = False

        # ============================================
        # NUEVO SISTEMA DE FITNESS (enfocado en recompensas positivas)
        # ============================================
        fitness = 0.0
        fitness_details = []

        # ---- 1️⃣ RECOMPENSA POR CENTRADO Y ACERCAMIENTO ----
        delta_size = size_now - self.size_blob_before
        dist_to_center_now = abs(posx_now - CENTER_X)
        dist_to_center_before = abs(self.posX_blob_before - CENTER_X)

        # Acercarse al cilindro
        if delta_size > 0:
            acercamiento = 30.0 * (delta_size / SIZE_MAX)
            fitness += acercamiento
            fitness_details.append(f"Acerca:+{acercamiento:.2f}")

        # Mejor centrado (reducción del error lateral)
        delta_center = dist_to_center_before - dist_to_center_now
        if delta_center > 0:
            mejora_centro = 2.0 * (delta_center / CENTER_X)
            fitness += mejora_centro
            fitness_details.append(f"Centro:+{mejora_centro:.2f}")

        # ---- 2️⃣ RECOMPENSA POR PROXIMIDAD ----
        if ir_front > 50 and self.recently_saw_cylinder:
            prox_bonus = 3.0
            fitness += prox_bonus
            fitness_details.append(f"Prox:+{prox_bonus:.1f}")

        # ---- 3️⃣ BONUS: SI RECUPERA VISIÓN ----
        if (self.size_blob_before < SIZE_VISIBLE_MIN) and (size_now >= SIZE_VISIBLE_MIN):
            recupera_bonus = 10.0
            fitness += recupera_bonus
            fitness_details.append("Recupera:+5.0")
        
        if (self.size_blob_before > SIZE_VISIBLE_MIN) and (size_now < SIZE_VISIBLE_MIN):
            pierde_bonus = 5.0
            fitness -= pierde_bonus
            fitness_details.append("Pierde:-5.0")

        # ---- 4️⃣ RECOMPENSA POR MANTENER VISIÓN VARIOS PASOS ----
        if size_now >= SIZE_VISIBLE_MIN:
            fitness += 1  # por cada frame viendo el cilindro
            fitness_details.append("Ve:+1.0")
        else:
            fitness -= 0.1
            fitness_details.append("No ve:-0.1")

        # ---- 5️⃣ BONUS POR EXPLORAR (CAMBIAR DE ACCIÓN) ----
        if self.last_action is not None:
            if action != self.last_action:
                explorar_bonus = 0.5
                fitness += explorar_bonus
                fitness_details.append("Explora:+0.5")
            else:
                # pequeña recompensa neutra por consistencia
                fitness += 0.05

        # ---- Actualizar estado previo ----
        self.size_blob_before = size_now
        self.posX_blob_before = posx_now
        self.last_action = action
        self.cumulative_fitness += fitness

        # ---- DEBUG ----
        if self.current_step % 10 == 0 or abs(fitness) > 0.5:
            details_str = " ".join(fitness_details) if fitness_details else "0"
            print(f"Step {self.current_step}: R={fitness:+.2f} ({details_str}) | Total={self.cumulative_fitness:.2f}")



        # ============================================
        # CONDICIONES DE TERMINACIÓN Y BONUS FINAL
        # ============================================
        terminated = False
        truncated = False

        # Éxito: está viendo y está cerca
        goal_reached = (ir_front > IR_SUCCESS_THRESHOLD) and self.recently_saw_cylinder

        if goal_reached:
            print(f"🎯 ¡OBJETIVO ALCANZADO! IR={ir_front:.1f} en step {self.current_step}")
            terminated = True
            fitness = 100.0
            self.cumulative_fitness += fitness

        # Colisión lateral (más penalizada)
        collided = (
            not self.recently_saw_cylinder > 0 and
            (self.robobo.readIRSensor(IR.FrontLL) > IR_COLLISION_THRESHOLD_SIDE or
            self.robobo.readIRSensor(IR.FrontRR) > IR_COLLISION_THRESHOLD_SIDE or
            self.robobo.readIRSensor(IR.FrontC) > IR_COLLISION_THRESHOLD_FRONT)
        )

        if collided:
            print(f"⚠️ COLISIÓN detectada en step {self.current_step}")
            terminated = True
            fitness = -5.0
            self.cumulative_fitness += fitness

        # Límite de pasos
        if self.current_step >= self.max_steps:
            truncated = True
            if ir_front > 200 and self.recently_saw_cylinder:
                final_bonus = 10.0
                fitness += final_bonus
                self.cumulative_fitness += final_bonus

        # Info
        info = {
            "robot_pos": self.robosim.getRobotLocation(self.roboboID)["position"],
            "goal_reached": goal_reached,
            "collided": collided,
            "size": size_now,
            "posx": posx_now,
            "ir_front": ir_front,
            "cumulative_fitness": self.cumulative_fitness,
            "steps": self.current_step,
            "recently_saw_cylinder": self.recently_saw_cylinder,
            "steps_since_saw": self.steps_since_saw_cylinder
        }

        return obs, float(fitness), terminated, truncated, info


    def get_cumulative_fitness(self):
        """Retorna el fitness acumulado del episodio"""
        return self.cumulative_fitness

    def close(self):
        """Cierra las conexiones"""
        try:
            self.robobo.stopMotors()
        except:
            pass
