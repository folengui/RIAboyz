
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
from ultralytics import YOLO
from robobopy_videostream.RoboboVideo import RoboboVideo
from robobopy.Robobo import Robobo
from robobopy.utils.IR import IR

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
MAX_EPISODE_STEPS = 150
SIZE_SAW_OBJECT_THRESHOLD = 0.05  # 5% del area aprox
SIZE_VISIBLE_MIN = 0.01

# Recompensas
REWARD_GOAL = 30.0
REWARD_CLOSER = 1.0
REWARD_CENTERED = 0.5
PENALTY_LOST = -1.0

# Acciones (Igual que P3)
# (rw, lw, duration)
ACTIONS = [
    (15, 15, 0.4),    # 0: Avanzar
    (15, -15, 0.2),   # 1: Girar izquierda
    (-15, 15, 0.2),   # 2: Girar derecha
    (0, 0, 0.3),      # 3: Parar/Esperar (opcional)
]

# Dimensiones imagen para la red (MobileNetV2 espera 224x224 usualmente)
IMG_WIDTH = 224
IMG_HEIGHT = 224

class RoboboEnvSRL(gym.Env):
    """
    Entorno Robobo para P4 Option 1 (SRL).
    Observación: Imagen RGB (224x224, 3)
    Recompensa: Calculada internamente usando YOLO (posición botella), 
                pero el agente NO ve estas coordenadas, solo la imagen.
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, robobo_ip='10.20.28.7'):
        super().__init__()
        
        self.ip = robobo_ip
        
        # 1. Configurar Robobo
        print(f"Conectando Robobo en {self.ip}...")
        self.robobo = Robobo(self.ip)
        self.robobo.connect()
        self.robobo.moveTiltTo(90, 5, True) # Cámara mirando al frente
        
        # 2. Configurar Video
        print("Conectando Video Stream...")
        self.videoStream = RoboboVideo(self.ip)
        self.videoStream.connect()
        self.robobo.startStream()
        
        # 3. Modelo YOLO para calcular RECOMPENSAS (Oráculo)
        print("Cargando YOLO (Oráculo)...")
        self.yolo_model = YOLO("yolov8n.pt")
        self.target_class = "bottle"
        
        # 4. Espacios de acción y observación
        self.action_space = spaces.Discrete(len(ACTIONS))
        
        # Observación: Imagen uint8 channels last (H, W, C)
        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=(IMG_HEIGHT, IMG_WIDTH, 3), 
            dtype=np.uint8
        )
        
        # Estado interno
        self.current_step = 0
        self.last_dist_center = 0.0
        self.last_size = 0.0
        self.last_frame = None

    def _get_obs(self):
        """Captura frame, lo redimensiona y devuelve."""
        frame = self.videoStream.getImage()
        
        if frame is None:
            # Si falla, devolver imagen negra
            return np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
        
        # Guardar frame original para debug/render
        self.last_frame = frame.copy()
        
        # Redimensionar para la CNN
        obs_frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
        
        # Convertir BGR a RGB (Torchvision espera RGB)
        obs_frame = cv2.cvtColor(obs_frame, cv2.COLOR_BGR2RGB)
        
        return obs_frame

    def _get_oracle_info(self, frame):
        """
        Usa YOLO para saber dónde está la botella y calcular reward.
        Retorna: (detected, size_rel, center_x_rel)
        """
        if frame is None:
            return False, 0.0, 0.5
            
        results = self.yolo_model(frame, verbose=False, conf=0.4, classes=[39]) # 39=bottle
        
        best_box = None
        max_area = 0
        
        if len(results) > 0 and results[0].boxes:
            for box in results[0].boxes:
                # Ya hemos filtrado por clase y conf en la llamada
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                area = (x2-x1)*(y2-y1)
                if area > max_area:
                    max_area = area
                    best_box = box
                    
        if best_box:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
            
            # Area relativa 0.0 a 1.0 (aprox)
            size_rel = max_area / (w * h)
            
            # Centro relativo X (0.0 a 1.0)
            center_x = (x1 + x2) / 2
            center_x_rel = center_x / w
            
            return True, size_rel, center_x_rel
            
        return False, 0.0, 0.5

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # Resetear robot? A veces es mejor no moverlo mucho al resetear en real
        self.robobo.stopMotors()
        
        obs = self._get_obs()
        
        # Inicializar estado previo para reward shaping
        detectado, size, cx = self._get_oracle_info(self.last_frame) # last_frame actualizado en _get_obs
        self.last_size = size
        self.last_dist_center = abs(cx - 0.5)
        
        return obs, {}

    def step(self, action):
        self.current_step += 1
        
        # 1. Ejecutar Acción
        rw, lw, duration = ACTIONS[action]
        self.robobo.moveWheelsByTime(rw, lw, duration)
        # Esperar un poco para que el movimiento tenga efecto antes de la foto
        # (El moveWheelsByTime no bloquea todo el rato si duration es muy corta, pero asumimos sincrono en logica simple)
        # En la librería original suele ser asíncrono o bloqueante según implementación. 
        # Añadimos un pequeño sleep por seguridad si es necesario, pero moveWheelsByTime suele esperar.
        
        # 2. Obtener Nueva Observación
        obs = self._get_obs() # Actualiza self.last_frame
        
        # 3. Calcular Reward (ORÁCULO)
        # Usamos el frame original (mayor resolución) para YOLO
        detectado, size, cx = self._get_oracle_info(self.last_frame)
        ir_front = self.robobo.readIRSensor(IR.FrontC)
        
        reward = 0.0
        terminated = False
        truncated = False
        
        dist_center = abs(cx - 0.5)
        
        if detectado:
            # Recompensa por ver el objeto
            reward += 0.1
            
            # Recompensa por acercarse (size aumenta)
            if size > self.last_size:
                reward += REWARD_CLOSER * (size - self.last_size) * 10 
            
            # Recompensa por centrar
            if dist_center < self.last_dist_center:
                reward += REWARD_CENTERED
                
            # Penalización por estar muy descentrado
            if dist_center > 0.3:
                reward -= 0.1
                
        else:
            # Penalización por no ver nada
            reward += PENALTY_LOST
            
        # 4. Comprobar Finalización
        # Éxito: Objeto cerca (IR alto) y detectado visualmente (para asegurar que es el objeto y no una pared)
        if ir_front > 100 and detectado and size > SIZE_SAW_OBJECT_THRESHOLD:
            reward += REWARD_GOAL
            terminated = True
            print("🎉 ¡OBJETIVO ALCANZADO!")
            
        # Timeout
        if self.current_step >= MAX_EPISODE_STEPS:
            truncated = True
            
        # Actualizar estado previo
        self.last_size = size
        self.last_dist_center = dist_center
        
        info = {
            "detectado": detectado,
            "size": size,
            "center": cx
        }
        
        print(f"Step {self.current_step}: R={reward:.2f} | Det={detectado} Size={size:.3f} IR={ir_front}")
        
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.last_frame is not None:
            cv2.imshow("Robobo Visual SRL", self.last_frame)
            cv2.waitKey(1)

    def close(self):
        self.robobo.stopMotors()
        self.robobo.stopStream()
        self.videoStream.disconnect()
        self.robobo.disconnect()
        cv2.destroyAllWindows()
