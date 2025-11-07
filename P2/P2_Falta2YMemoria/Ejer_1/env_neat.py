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
IR_COLLISION_THRESHOLD = 300
MAX_EPISODE_STEPS = 100
SIZE_SAW_CYLINDER_THRESHOLD = 1.0
SIZE_VISIBLE_MIN = 1.0

# Condiciones de éxito permisivas
IR_SUCCESS_THRESHOLD = 150 
MAX_EPISODE_STEPS_WITHOUT_VISION = 5

# Normalización
SIZE_MAX = 600.0
POSX_MAX = 100.0
CENTER_X = 50.0

# Acciones disponibles
ACTIONS = [
    (10, 10, 0.5),   # 0: Avanzar recto
    (10, -10, 0.3),  # 1: Girar izquierda
    (-10, 10, 0.3),  # 2: Girar derecha
    (5, 5, 0.5),     # 3: Avanzar lento
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
            self.robobo.moveTiltTo(120, 5, True)
        
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
        
        self.posRobotinit = loc_robot_init["position"]
        self.rotRobotinit = loc_robot_init["rotation"]

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
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
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
        self.recently_saw_cylinder = False
        self.steps_since_saw_cylinder = 999
        
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
        Obtiene observación normalizada con ponderación de blob.
        
        SOLUCIÓN AL PROBLEMA DE AMBIGÜEDAD:
        - blob_size crudo puede ser igual estando cerca+ladeado o lejos+centrado
        - Creamos blob_size_efectivo = blob_size * factor_centrado
        - Factor de centrado penaliza si está muy descentrado
        - Así la red recibe información no ambigua
        """
        blob = self.robobo.readColorBlob(self.objectColor)
        size = float(getattr(blob, "size", 0.0))
        pos_x = float(getattr(blob, "posx", CENTER_X))
        ir_front = float(self.robobo.readIRSensor(IR.FrontC))
        
        # Calcular qué tan descentrado está (0 = centrado, 1 = muy descentrado)
        dist_from_center = abs(pos_x - CENTER_X)
        dist_from_center_norm = np.clip(dist_from_center / CENTER_X, 0.0, 1.0)
        
        # Factor de centrado: 1.0 si está centrado, 0.0 si está muy descentrado
        # Usamos función cuadrática para penalizar más fuerte cuando está muy descentrado
        centering_factor = 1.0 - (dist_from_center_norm ** 2)
        
        # Blob size EFECTIVO: ponderado por centrado
        # Si está descentrado, el blob_size se reduce aunque sea grande
        # Esto elimina la ambigüedad: cerca+ladeado tendrá blob_efectivo bajo
        size_norm = np.clip(size / SIZE_MAX, 0.0, 1.0)
        blob_size_efectivo = size_norm * centering_factor
        
        # IR normalizado
        ir_norm = np.clip(ir_front / 1000.0, 0.0, 1.0)
        
        # Retornar: [blob_size_efectivo, distancia_al_centro, ir_front]
        return np.array([blob_size_efectivo, dist_from_center_norm, ir_norm], dtype=np.float32)

    def reset(self, seed=None, options=None):
        """Resetea el episodio - robot y cilindro vuelven a posición inicial"""
        super().reset(seed=seed)
        
        import time
        
        # Detener motores primero
        self.robobo.stopMotors()
        
        # RESETEAR SIMULACIÓN COMPLETA (resetea robot + objetos a posiciones iniciales)
        # Esto es más robusto que setRobotLocation manual
        self.robosim.resetSimulation()
        
        # Dar tiempo al simulador para aplicar los cambios
        time.sleep(0.3)  # Mayor tiempo para asegurar reset completo
        
        # Configurar cámara: centrar PAN y bajar TILT para ver el cilindro en el suelo
        self.robobo.movePanTo(0, 100)
        self.robobo.moveTiltTo(120, 100)  # 120 grados = cámara mirando hacia abajo
        time.sleep(0.2)
        
        # Estado interno
        self.current_step = 0
        self.episode += 1
        self.cumulative_fitness = 0.0
        self.recently_saw_cylinder = False
        self.steps_since_saw_cylinder = 999
        self.best_ir_seen = 0.0
        self.last_action = None
        self.action_repeat_count = 0

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
        
        # Actualizar tracking de visión (igual que env2_validation)
        if size_now > SIZE_SAW_CYLINDER_THRESHOLD:
            self.recently_saw_cylinder = True
            self.steps_since_saw_cylinder = 0
        else:
            self.steps_since_saw_cylinder += 1
            if self.steps_since_saw_cylinder > self.MAX_STEPS_WITHOUT_VISION:
                self.recently_saw_cylinder = False

        # ============================================
        # SISTEMA DE FITNESS (adaptado de env2_validation)
        # ============================================
        fitness = 0.0
        fitness_details = []
        
        # 1. Recompensa por acercarse (delta de tamaño)
        delta_size = size_now - self.size_blob_before
        if delta_size > 0:
            acercamiento = 2.0 * (delta_size / SIZE_MAX)
            fitness += acercamiento
            fitness_details.append(f"Acerca:+{acercamiento:.2f}")
        elif delta_size < 0:
            alejamiento = -0.2 * (abs(delta_size) / SIZE_MAX)
            fitness += alejamiento
            fitness_details.append(f"Aleja:{alejamiento:.2f}")
        
        # 2. Recompensa por centrado (solo si ve el cilindro ahora y antes)
        SIZE_VISIBLE_MIN = 1.0
        if size_now >= SIZE_VISIBLE_MIN and self.size_blob_before >= SIZE_VISIBLE_MIN:
            dist_to_center_now = abs(posx_now - CENTER_X)
            dist_to_center_before = abs(self.posX_blob_before - CENTER_X)
            delta_centering = dist_to_center_before - dist_to_center_now
            
            if delta_centering > 0:
                mejora_centrado = 0.5 * (delta_centering / CENTER_X)
                fitness += mejora_centrado
                fitness_details.append(f"Centro:+{mejora_centrado:.2f}")
            elif delta_centering < 0:
                peor_centrado = -0.1 * (abs(delta_centering) / CENTER_X)
                fitness += peor_centrado
                fitness_details.append(f"Descen:{peor_centrado:.2f}")
        
        # 3. Bonus por proximidad (IR) - solo si vio el cilindro recientemente
        if ir_front > 50 and self.recently_saw_cylinder:
            prox_bonus = 1.0
            fitness += prox_bonus
            fitness_details.append(f"Prox:+{prox_bonus:.1f}")
        
        # 4. Bonus pequeño por ver el cilindro
        if size_now > SIZE_SAW_CYLINDER_THRESHOLD:
            ver_bonus = 0.1
            fitness += ver_bonus
            fitness_details.append(f"Ver:+{ver_bonus:.1f}")
        
        # 5. Penalización por perder de vista
        if size_now < SIZE_VISIBLE_MIN and self.size_blob_before >= SIZE_VISIBLE_MIN:
            if ir_front > 100:
                # Si está cerca por IR, no penalizar tanto
                pass
            else:
                perdio_vista = -3.0
                fitness += perdio_vista
                fitness_details.append(f"Perdió:{perdio_vista:.1f}")
        
        # 6. Penalización suave por no ver nada
        if size_now < SIZE_VISIBLE_MIN and ir_front < 300:
            no_ve = -0.005
            fitness += no_ve
            # No añadir a details para no saturar
        
        # 7. Penalización por repetir la misma acción (evita girar en círculos)
        if action == self.last_action:
            self.action_repeat_count += 1
            # Penalización progresiva: más repeticiones = más penalización
            if self.action_repeat_count > 5:  # Permitir hasta 5 repeticiones
                repeticion_penalty = -0.04 * (self.action_repeat_count - 5)
                fitness += repeticion_penalty
                if self.action_repeat_count % 10 == 0:  # Mostrar cada 10
                    fitness_details.append(f"Repite:{repeticion_penalty:.2f}")
        else:
            self.action_repeat_count = 0
        
        # Actualizar estado previo
        self.size_blob_before = size_now
        self.posX_blob_before = posx_now
        self.last_action = action
        
        self.cumulative_fitness += fitness
        
        # DEBUG: Mostrar fitness cada 10 steps
        if self.current_step % 10 == 0 or fitness > 0.5 or fitness < -0.5:
            details_str = " ".join(fitness_details) if fitness_details else "0"
            print(f"      Step {self.current_step}: R={fitness:+.2f} ({details_str}) | Total={self.cumulative_fitness:.2f}")

        # CONDICIONES DE TERMINACIÓN
        terminated = False
        truncated = False
        
        # Definir éxito (adaptado de env2_validation) - más permisivo
        goal_reached = (ir_front > IR_SUCCESS_THRESHOLD) and self.recently_saw_cylinder
        
        # DEBUG: Mostrar valores cuando está cerca del cilindro
        if ir_front > 300 or size_now > 50:  # Si está relativamente cerca
            dist_center = abs(posx_now - CENTER_X)
            print(f"    [Step {self.current_step}] IR: {ir_front:.1f} (>{IR_SUCCESS_THRESHOLD}?) | "
                  f"Size: {size_now:.1f} | Centro: {dist_center:.1f} | "
                  f"VioReciente: {self.recently_saw_cylinder} | GOAL: {goal_reached}")
        
        # Éxito (adaptado de env2_validation)
        if goal_reached:
            print(f"    🎯 ¡OBJETIVO ALCANZADO! IR={ir_front:.1f} en step {self.current_step}")
            terminated = True
            # Recompensa grande por éxito (igual que env2_validation)
            fitness = 30.0  # Sobrescribir el fitness de este step
            self.cumulative_fitness += 30.0
        
    
        
        # Colisión lateral (menos penalizada)
        collided = ( 
            not self.recently_saw_cylinder and
            (self.robobo.readIRSensor(IR.FrontLL) > IR_COLLISION_THRESHOLD or
            self.robobo.readIRSensor(IR.FrontRR) > IR_COLLISION_THRESHOLD or 
            self.robobo.readIRSensor(IR.FrontC) > IR_COLLISION_THRESHOLD 
            )
        )
        
        if collided:
            print(f"    ⚠️ COLISIÓN detectada en step {self.current_step}")
            terminated = True
            # Penalización por colisión (igual que env2_validation)
            fitness = -5.0  # Sobrescribir
            self.cumulative_fitness += -5.0
        
        # Límite de pasos (SOLO truncated, NO terminated)
        if self.current_step >= self.max_steps:
            truncated = True
            # Bonus generoso si estaba cerca al acabar (adaptado de env2_validation)
            if ir_front > 200 and self.recently_saw_cylinder:
                final_bonus = 5.0
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
