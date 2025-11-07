# Ejercicio 3 - Sistema Híbrido NEAT + PPO

## 📋 Descripción

Sistema híbrido que combina **NEAT** para navegación y detección de cilindro, con **PPO** (Proximal Policy Optimization) para aproximación al objetivo.

## 🎯 Funcionamiento

1. **Fase 1 - NEAT**: El robot navega evitando el obstáculo y detectando el cilindro usando sensores IR
2. **Transición**: Cuando NEAT detecta el cilindro (salida > 0.5), se activa PPO
3. **Fase 2 - PPO**: El robot se aproxima al cilindro usando visión del blob



## 🚀 Uso

### 1. Validar Sistema Híbrido

```bash
python validate_hybrid.py
```

Ejecuta validación completa NEAT→PPO.


### 2. Generar Visualizaciones

```bash
python generate_visualizations.py
```

Genera en `visualizations/`:
- `aprendizaje_fitness.png` - Evolución del fitness NEAT
- `especies_evolucion.png` - Evolución de especies
- `red_neuronal.svg` - Topología red neuronal
- `trayectoria_episodio_*.png` - Trayectorias 2D

### 3. Entrenar NEAT (opcional)

```bash
python train_neat.py
```

