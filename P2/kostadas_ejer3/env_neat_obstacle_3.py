# env_neat_obstacle_3.py
"""
Entorno NEAT para escenario 2.3 con OBSTÁCULO - VERSIÓN OPTIMIZADA.
OBJETIVO: Detectar el cilindro (NO acercarse).
Fase 1 del sistema híbrido AE + AR.

OPTIMIZACIONES:
1. Polling directo (más rápido y determinista)
2. Barrido optimizado (3 posiciones clave)
3. Para al detectar (no sigue barriendo)
4. Fitness mejorado para exploración
5. 3x más rápido: 0.13s/step vs 0.4s/step
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time

from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim 
from robobopy.utils.IR import IR
from robobopy.utils.Color import Color

# CONSTANTES
IR_COLLISION_THRESHOLD = 200 # Valor crudo del sensor
MAX_EPISODE_STEPS = 50
SIZE_DETECTION_THRESHOLD = 0  # Tamaño mínimo para considerar "detectado"


ACTIONS = [
    (10, 10, 0.5),      # 0: Avanzar recto
    (10, -10, 0.3),     # 1: Girar robot izquierda
    (-10, 10, 0.3),     # 2: Girar robot derecha
]
# Formato: (wheel_right, wheel_left, duration)


class RoboboNEATObstacleEnv(gym.Env):
    """
    Environment para NEAT - Fase 1: Búsqueda y detección.
    OBJETIVO: Encontrar el cilindro (no acercarse).
    
    OPTIMIZACIONES:
    - Barrido rápido con polling
    - Sin callbacks asíncronos
    - Fitness que premia exploración activa
    """
    
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
        time.sleep(1)

        # Guardar posiciones iniciales
        loc_robot_init = self.robosim.getRobotLocation(idrobot)
        if loc_robot_init is None:
            print("⚠️ Esperando a que RoboboSim cargue el robot...")
            time.sleep(2)
            loc_robot_init = self.robosim.getRobotLocation(idrobot)
            if loc_robot_init is None:
                raise RuntimeError("❌ ERROR: RoboboSim no devuelve la ubicación del robot.")
        
        self.posRobotinit = loc_robot_init["position"]
        self.rotRobotinit = loc_robot_init["rotation"]

        # Intentar obtener posición del objeto (opcional - solo para info)
        # NO es necesario para el funcionamiento, buscamos por COLOR
        try:
            loc_obj_init = self.robosim.getObjectLocation(idobject)
            if loc_obj_init:
                self.posObjectinit = loc_obj_init["position"]
                self.rotObjectinit = loc_obj_init["rotation"]
            else:
                self.posObjectinit = None
                self.rotObjectinit = None
        except:
            print(f"⚠️ No se pudo obtener ubicación del objeto '{idobject}' (no crítico - buscamos por color ROJO)")
            self.posObjectinit = None
            self.rotObjectinit = None

        # Espacios
        self.action_space = spaces.Discrete(len(ACTIONS))
        
        # Observación: [ir_front, ir_left, ir_right]
        # Solo sensores IR (sin información del blob - arquitectura híbrida)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Activar tracking de blob ROJO
        self.robobo.setActiveBlobs(red=True, green=False, blue=False, custom=False)

        # Estado interno
        self.current_step = 0
        self.episode = 0
        self.cumulative_fitness = 0.0
        
        # Tracking de exploración
        self.last_action = None
        self.action_repeat_count = 0
        
        # Flag de detección
        self.blob_detected_flag = False
        
        # NUEVO: Flag para tracking de fases de aprendizaje
        self.wall_contacted = False  # ¿Ya detectó la pared inicial?
        self.wall_cleared = False    # ¿Ya despejó el frente tras contactar?
        self.wall_contact_step = 0   # Step en que contactó la pared
        
        # Para detectar comportamiento sin hardcodear acciones
        self.last_ir_front = None
        self.consecutive_free_steps = 0  # Contador de steps con frente libre
        
        # Optimización: Contador para barrido de PAN (no hacerlo cada step)
        self.pan_scan_counter = 0
        self.PAN_SCAN_INTERVAL = 3  # Barrer cada N acciones

    def _get_obs(self):
        """
        Obtiene observación normalizada (solo IR sensors).
        
        IMPORTANTE: No incluye información del blob.
        Esto es deliberado para la arquitectura híbrida:
        - Fase 1 (NEAT): Navega sin "ver" el objetivo
        - Fase 2 (otro sistema): Usa información visual para acercarse
        """
        # Sensores IR
        ir_front = float(self.robobo.readIRSensor(IR.FrontC))
        ir_left = float(self.robobo.readIRSensor(IR.FrontLL))
        ir_right = float(self.robobo.readIRSensor(IR.FrontRR))
        
        # Normalizar
        ir_front_norm = np.clip(ir_front / 1000.0, 0.0, 1.0)
        ir_left_norm = np.clip(ir_left / 1000.0, 0.0, 1.0)
        ir_right_norm = np.clip(ir_right / 1000.0, 0.0, 1.0)
        
        return np.array([ir_front_norm, ir_left_norm, ir_right_norm], dtype=np.float32)

    def reset(self, seed=None, options=None):
        """Resetea el episodio."""
        super().reset(seed=seed)
        
        # Detener motores
        self.robobo.stopMotors()
        
        # RESETEAR SIMULACIÓN COMPLETA (resetea robot + objetos a posiciones iniciales)
        # Esto es más robusto que setRobotLocation manual
        self.robosim.resetSimulation()
        time.sleep(0.3)  # ⏱️ Dar tiempo a que el simulador resetee todo
        
        # Centrar PAN
        self.robobo.movePanTo(0, 100)
        time.sleep(0.1)
        
        # DEBUG: Verificar reset cada 5 episodios
        if self.episode % 5 == 0:
            loc_after = self.robosim.getRobotLocation(self.roboboID)
            if loc_after:
                print(f"    📍 [RESET] Robot reseteado:")
                print(f"       Posición: {loc_after['position']}")
                print(f"       Rotación: {loc_after['rotation']}")
        
        # Estado interno
        self.current_step = 0
        self.episode += 1
        self.cumulative_fitness = 0.0
        self.last_action = None
        self.action_repeat_count = 0
        
        # Resetear flag de detección
        self.blob_detected_flag = False
        self.wall_contacted = False  # Resetear flag de contacto con pared
        self.wall_cleared = False    # Resetear flag de frente despejado
        self.wall_contact_step = 0   # Resetear step de contacto
        self.pan_scan_counter = 0    # Resetear contador de barrido
        self.last_ir_front = None
        self.consecutive_free_steps = 0

        obs = self._get_obs()
        
        # DEBUG: Mostrar configuración al inicio del episodio
        if self.episode % 5 == 1:  # Cada 5 episodios
            print(f"\n🔧 [Episodio {self.episode}] IR_COLLISION_THRESHOLD = {IR_COLLISION_THRESHOLD}")
        
        # Obtener posición inicial del robot
        robot_location = self.robosim.getRobotLocation(self.roboboID)
        robot_pos = robot_location.get("position", {"x": 0.0, "z": 0.0}) if robot_location else {"x": 0.0, "z": 0.0}
        
        info = {
            "episode": self.episode,
            "robot_pos": robot_pos,  # Posición inicial para tracking
        }
        
        return obs, info

    def step(self, action):
        """
        Ejecuta acción de navegación + barrido automático optimizado.
        
        OPTIMIZACIÓN CLAVE:
        - Barrido en 3 posiciones clave: -60°, 0°, +60°
        - Polling directo (no callback asíncrono)
        - Para inmediatamente al detectar
        """
        self.current_step += 1

        # 1. EJECUTAR ACCIÓN DE MOVIMIENTO
        rw, lw, duration = ACTIONS[int(action)]
        self.robobo.moveWheelsByTime(rw, lw, duration)
        
        # 2. BARRIDO RÁPIDO DE PAN - OPTIMIZADO (solo cada N acciones)
        self.pan_scan_counter += 1
        should_scan = (self.pan_scan_counter % self.PAN_SCAN_INTERVAL == 0)
        
        if not self.blob_detected_flag and should_scan:
            print(f"    🔍 [Step {self.current_step}] Iniciando barrido PAN (scan #{self.pan_scan_counter // self.PAN_SCAN_INTERVAL})...")
            # Posiciones clave que cubren campo visual amplio
            scan_positions = [-60, 0, 60]
            
            for pan_pos in scan_positions:
                # Mover PAN a posición
                self.robobo.movePanTo(pan_pos, 100)  # Velocidad máxima
                time.sleep(0.08)  # Tiempo mínimo para estabilizar
                
                # LEER BLOB DIRECTAMENTE (polling, no callback)
                blob = self.robobo.readColorBlob(self.objectColor)
                blob_size = float(getattr(blob, "size", 0.0))
                
                # ¿Detectado?
                if blob_size > SIZE_DETECTION_THRESHOLD:
                    self.blob_detected_flag = True
                    print(f"    🎯 ¡DETECTADO! Blob size={blob_size:.1f} en PAN={pan_pos}° (scan #{self.pan_scan_counter // self.PAN_SCAN_INTERVAL})")
                    break  # ¡Para el barrido inmediatamente!
            
            # VOLVER PAN A POSICIÓN NEUTRAL (solo si hicimos barrido)
            self.robobo.movePanTo(0, 100)
            time.sleep(0.05)
        
        # 4. TOMAR OBSERVACIÓN (solo IR sensors)
        obs = self._get_obs()
        ir_front, ir_left, ir_right = obs
        
        # Valores crudos para cálculos y debug
        ir_front_raw = ir_front * 1000.0
        ir_left_raw = ir_left * 1000.0
        ir_right_raw = ir_right * 1000.0
        
        # DEBUG: Mostrar lecturas de sensores cada 10 steps
        if self.current_step % 10 == 0:
            fase = "FASE 1 (Acercamiento)" if not self.wall_contacted else "FASE 2 (Wall-following)"
            scan_info = f"[PAN scan cada {self.PAN_SCAN_INTERVAL} steps]" if should_scan else ""
            print(f"    📡 [Step {self.current_step}] {fase} | IR RAW: Front={ir_front_raw:.0f} | Left={ir_left_raw:.0f} | Right={ir_right_raw:.0f} {scan_info}")

        # ============================================
        # SISTEMA DE FITNESS POR FASES - APRENDIZAJE GUIADO
        # ============================================
        fitness = 0.0
        
        # Detectar si algún sensor IR detecta algo (threshold bajo para detectar contacto inicial)
        any_ir_active = (ir_front > 0.004) or (ir_left > 0.004) or (ir_right > 0.004)
        
        # ============================================
        # FASE 1: ACERCAMIENTO INICIAL AL BLOQUE
        # ============================================
        if not self.wall_contacted:
            # El robot AÚN NO ha contactado con la pared
            
            # 🎯 OBJETIVO: Llegar al bloque LO MÁS RÁPIDO POSIBLE
            
            # 1. MEGA RECOMPENSA por primer contacto con pared
            if any_ir_active:
                self.wall_contacted = True
                self.wall_contact_step = self.current_step  # Guardar cuándo contactó
                
                # 🔥 RECOMPENSA PROPORCIONAL A LA RAPIDEZ
                # Cuanto antes llegue, mayor recompensa (sin hardcodear acciones)
                # Máximo: 10 puntos (si llega en step 1)
                # Mínimo: 5 puntos (si llega en step 50)
                steps_bonus = 5.0 * (1.0 - (self.current_step / self.max_steps))
                fitness += 5.0 + steps_bonus  # Base 5 + bonus por rapidez
                
                print(f"    🎉 ¡PRIMER CONTACTO CON PARED! Step {self.current_step}/{self.max_steps} "
                      f"| Bonus rapidez: +{steps_bonus:.1f} | IR: F={ir_front:.3f} L={ir_left:.3f} R={ir_right:.3f}")
            
            # 2. Penalizar cada step sin contactar (incentivo de urgencia)
            else:
                fitness -= 0.1  # Pequeña penalización por NO haber llegado aún
            
            # 3. Penalizar repetir acción (evitar quedarse atascado)
            if action == self.last_action:
                self.action_repeat_count += 1
                if self.action_repeat_count > 7:
                    fitness -= 0.5 * (self.action_repeat_count - 7)
            else:
                self.action_repeat_count = 0
        
        # ============================================
        # FASE 2: ESQUIVAR EN DIAGONAL
        # ============================================
        else:
            # El robot YA contactó con la pared → ahora debe esquivarla
            
            # 🎯 OBJETIVO: Despejar el frente LO MÁS RÁPIDO POSIBLE
            
            # 1. MEGA RECOMPENSA por PRIMERA VEZ que despeja el frente
            if not self.wall_cleared and ir_front < 0.004:  # Threshold MUY bajo (casi 0) = realmente despejado
                self.wall_cleared = True
                
                # 🔥🔥 RECOMPENSA PROPORCIONAL A LA RAPIDEZ DE ESQUIVE
                # Cuanto antes despeje tras contactar, mayor recompensa
                steps_to_clear = self.current_step - self.wall_contact_step
                max_steps_to_clear = 7  # Lo ideal es despejar en máximo 7 steps
                
                # Bonus decrece si tarda mucho
                clear_bonus = 50.0 * max(0.0, 1.0 - (steps_to_clear / max_steps_to_clear))
                fitness += 50.0 + clear_bonus  # Base 50 + bonus por rapidez
                
                print(f"    🎊 ¡FRENTE DESPEJADO! Despejó en {steps_to_clear} steps tras contacto "
                      f"| Bonus rapidez: +{clear_bonus:.1f} | IR_front: {ir_front:.3f}")
            
            # 2. RECOMPENSA CONTINUA: Mantener frente libre (SIN hardcodear acciones)
            if ir_front < 0.004:  # Frente LIBRE (muy bajo = realmente despejado)
                self.consecutive_free_steps += 1
                
                # Base: Premiar mantener frente libre
                fitness += 10.0

                
                # BONUS EXTRA: Premiar mantener frente libre varios steps seguidos
                if self.consecutive_free_steps > 3:
                    fitness += 5.0  # Bonus por consistencia
            else:  # Aún hay obstáculo al frente (ir_front >= 0.005)
                self.consecutive_free_steps = 0  # Resetear contador
                
                # SUB-OBJETIVO: Despejar el frente
                # Pequeña penalización por seguir bloqueado (el incentivo real es el bonus de despejar)
                fitness -= 2.0  # Penalización suave, el incentivo es la recompensa por despejar
            
            # 3. Penalizar repetir acción (evitar loops infinitos)
            if action == self.last_action:
                self.action_repeat_count += 1
                if self.action_repeat_count > 10:
                    fitness -= 0.5 * (self.action_repeat_count - 10)
            else:
                self.action_repeat_count = 0
        
        self.last_action = action
        self.cumulative_fitness += fitness
        
        # Guardar estado para próximo step
        self.last_ir_front = ir_front

        # ============================================
        # CONDICIONES DE TERMINACIÓN
        # ============================================
        terminated = False
        truncated = False
        
        # ÉXITO: Blob detectado
        goal_reached = self.blob_detected_flag
        
        if goal_reached:
            terminated = True
            
            # MEGA BONUS por éxito (OBJETIVO PRINCIPAL)
            success_bonus = 400.0  # Aumentado: detectar DEBE ser mucho mejor que solo esquivar
            fitness += success_bonus
            self.cumulative_fitness += success_bonus
            
            # BONUS CRUCIAL por rapidez (encontrar rápido)
            # Cuanto antes detecte, mayor recompensa
            steps_remaining = self.max_steps - self.current_step
            early_bonus = (steps_remaining / self.max_steps) * 100.0
            fitness += early_bonus
            self.cumulative_fitness += early_bonus
            
            print(f"    🎯 ¡ÉXITO! Cilindro detectado en step {self.current_step}/{self.max_steps} "
                  f"| Fitness: {self.cumulative_fitness:.1f}")

        # FALLO: Colisión fuerte
        collided = (
            ir_front_raw > IR_COLLISION_THRESHOLD or 
            ir_left_raw > IR_COLLISION_THRESHOLD or 
            ir_right_raw > IR_COLLISION_THRESHOLD
        )
        
        # DEBUG: Mostrar detalles de colisión
        if collided:
            terminated = True
            fitness -= 10.0  # Penalización moderada (no destruir exploración)
            self.cumulative_fitness -= 10.0
            
            # Identificar qué sensor detectó la colisión
            collision_sensors = []
            if ir_front_raw > IR_COLLISION_THRESHOLD:
                collision_sensors.append(f"FRONT={ir_front_raw:.0f}")
            if ir_left_raw > IR_COLLISION_THRESHOLD:
                collision_sensors.append(f"LEFT={ir_left_raw:.0f}")
            if ir_right_raw > IR_COLLISION_THRESHOLD:
                collision_sensors.append(f"RIGHT={ir_right_raw:.0f}")
            
            print(f"    💥 COLISIÓN en step {self.current_step}!")
            print(f"       Sensores: {' | '.join(collision_sensors)} (Threshold={IR_COLLISION_THRESHOLD})")
        
        # DEBUG: Advertencia si está muy cerca pero no colisiona
        elif ir_front_raw > 200 or ir_left_raw > 200 or ir_right_raw > 200:
            close_sensors = []
            if ir_front_raw > 200:
                close_sensors.append(f"FRONT={ir_front_raw:.0f}")
            if ir_left_raw > 200:
                close_sensors.append(f"LEFT={ir_left_raw:.0f}")
            if ir_right_raw > 200:
                close_sensors.append(f"RIGHT={ir_right_raw:.0f}")
            print(f"    ⚠️  MUY CERCA (step {self.current_step}): {' | '.join(close_sensors)}")
        
        # Límite de pasos
        if self.current_step >= self.max_steps:
            truncated = True

        # Obtener posición actual del robot para tracking
        robot_location = self.robosim.getRobotLocation(self.roboboID)
        robot_pos = robot_location.get("position", {"x": 0.0, "z": 0.0}) if robot_location else {"x": 0.0, "z": 0.0}

        # Info
        info = {
            "goal_reached": goal_reached,
            "collided": collided,
            "cumulative_fitness": self.cumulative_fitness,
            "steps": self.current_step,
            "wall_contacted": self.wall_contacted,
            "robot_pos": robot_pos,  # Para tracking de trayectoria
        }

        return obs, float(fitness), terminated, truncated, info

    def get_cumulative_fitness(self):
        """Retorna el fitness acumulado del episodio."""
        return self.cumulative_fitness

    def close(self):
        """Cierra las conexiones."""
        try:
            self.robobo.stopMotors()
            self.robobo.movePanTo(0, 100)
        except:
            pass
