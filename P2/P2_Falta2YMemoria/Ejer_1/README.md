# Ejercicio 1 - NEAT para Aproximación al Cilindro 

## 🚀 Uso

### 1. Entrenar el Modelo

```bash
python train_neat.py
```

**Tiempo estimado**: ~15-20 minutos (15 generaciones)

### 2. Validar Modelo NEAT

```bash
python validate_neat.py
```

Ejecuta validación completa del modelo entrenado y genera trayectorias 2D.

### 3. Generar Visualizaciones

```bash
python generate_visualizations.py
```

Genera en `visualizations/`:
- `aprendizaje_fitness.png` - Evolución del fitness NEAT
- `especies_evolucion.png` - Evolución de especies (¡con datos reales!)
- `red_neuronal.svg` - Topología red neuronal
- `trayectoria_episodio_*.png` - Trayectorias 2D

