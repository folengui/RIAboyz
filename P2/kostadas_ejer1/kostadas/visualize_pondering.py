# visualize_pondering.py
"""
Script para visualizar el efecto de la ponderación en blob_size.
Genera gráficas que muestran cómo el blob_size_efectivo resuelve la ambigüedad.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch

# Configurar estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)


def calculate_effective_size(blob_size_norm, dist_from_center_norm):
    """
    Calcula el blob_size_efectivo según la fórmula implementada.
    """
    centering_factor = 1.0 - (dist_from_center_norm ** 2)
    return blob_size_norm * centering_factor


def plot_heatmap():
    """
    Crea un heatmap mostrando blob_size_efectivo para diferentes
    combinaciones de blob_size y distancia al centro.
    """
    # Crear grilla
    blob_sizes = np.linspace(0, 1, 100)
    distances = np.linspace(0, 1, 100)
    
    X, Y = np.meshgrid(blob_sizes, distances)
    Z = calculate_effective_size(X, Y)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Heatmap
    contour = ax.contourf(X, Y, Z, levels=20, cmap='RdYlGn')
    cbar = plt.colorbar(contour, ax=ax, label='Blob Size Efectivo')
    
    # Líneas de contorno
    contour_lines = ax.contour(X, Y, Z, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2f')
    
    # Ejemplos específicos
    examples = [
        (0.8, 0.0, "A: Cerca + Centrado\n(IDEAL)", 'green'),
        (0.8, 0.6, "B: Cerca + Ladeado\n(MALO)", 'red'),
        (0.3, 0.0, "C: Lejos + Centrado\n(REGULAR)", 'blue'),
        (0.3, 0.6, "D: Lejos + Ladeado\n(MUY MALO)", 'darkred'),
    ]
    
    for size, dist, label, color in examples:
        effective = calculate_effective_size(size, dist)
        ax.plot(size, dist, 'o', markersize=15, color=color, 
               markeredgecolor='black', markeredgewidth=2, zorder=10)
        ax.annotate(f'{label}\nEfectivo: {effective:.2f}', 
                   xy=(size, dist), xytext=(15, 15),
                   textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3',
                                 color='black', lw=2),
                   fontsize=9, fontweight='bold', color='white')
    
    ax.set_xlabel('Blob Size (normalizado)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Distancia al Centro (normalizado)\n0 = centrado, 1 = extremo', 
                 fontsize=13, fontweight='bold')
    ax.set_title('Heatmap: Blob Size Efectivo = Blob Size × Factor de Centrado\n' + 
                'Factor de Centrado = 1 - (distancia)²',
                fontsize=15, fontweight='bold', pad=20)
    
    # Anotación explicativa
    explanation = (
        "Interpretación:\n"
        "• Verde (>0.5): Buena situación, avanzar\n"
        "• Amarillo (0.3-0.5): Regular, ajustar\n"
        "• Rojo (<0.3): Mala situación, corregir"
    )
    ax.text(0.02, 0.98, explanation, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('results_neat/heatmap_ponderacion.png', dpi=300, bbox_inches='tight')
    print("✓ Guardado: results_neat/heatmap_ponderacion.png")
    plt.close()


def plot_comparison_scenarios():
    """
    Gráfica comparando escenarios específicos ANTES y DESPUÉS de ponderar.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comparación: Observaciones SIN vs CON Ponderación', 
                fontsize=16, fontweight='bold')
    
    # Escenarios de ejemplo
    scenarios = [
        {
            'name': 'Escenario A: CERCA + LADEADO',
            'blob_size': 0.75,
            'dist_center': 0.7,
            'description': 'Robot cerca pero muy descentrado\n(ve solo un lado del cilindro)'
        },
        {
            'name': 'Escenario B: LEJOS + CENTRADO',
            'blob_size': 0.75,  # ¡Mismo blob_size!
            'dist_center': 0.1,
            'description': 'Robot lejos pero bien alineado\n(ve todo el cilindro completo)'
        }
    ]
    
    # Panel 1: Sin ponderar (AMBIGUO)
    ax = axes[0, 0]
    ax.set_title('SIN Ponderación: AMBIGUO ❌', fontsize=13, fontweight='bold', color='red')
    
    x_pos = [0, 1]
    colors_without = ['orange', 'orange']  # ¡Mismo color porque son iguales!
    
    bars = ax.bar(x_pos, [scenarios[0]['blob_size'], scenarios[1]['blob_size']], 
                 color=colors_without, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Escenario A\n(Cerca+Ladeado)', 'Escenario B\n(Lejos+Centrado)'])
    ax.set_ylabel('Blob Size', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.axhline(0.75, color='red', linestyle='--', linewidth=2, alpha=0.5)
    
    # Anotación del problema
    ax.text(0.5, 0.85, '⚠️ PROBLEMA: Mismo valor (0.75)\npero situaciones MUY diferentes!', 
           ha='center', fontsize=11, fontweight='bold', color='red',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Con ponderación (CLARO)
    ax = axes[0, 1]
    ax.set_title('CON Ponderación: CLARO ✅', fontsize=13, fontweight='bold', color='green')
    
    effective_A = calculate_effective_size(scenarios[0]['blob_size'], scenarios[0]['dist_center'])
    effective_B = calculate_effective_size(scenarios[1]['blob_size'], scenarios[1]['dist_center'])
    
    colors_with = ['red', 'green']  # ¡Ahora son diferentes!
    
    bars = ax.bar(x_pos, [effective_A, effective_B], 
                 color=colors_with, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Escenario A\n(Cerca+Ladeado)', 'Escenario B\n(Lejos+Centrado)'])
    ax.set_ylabel('Blob Size Efectivo', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1)
    
    # Valores específicos
    for i, (val, color) in enumerate(zip([effective_A, effective_B], colors_with)):
        ax.text(i, val + 0.05, f'{val:.3f}', ha='center', fontsize=12, 
               fontweight='bold', color=color)
    
    # Anotación de la solución
    ax.text(0.5, 0.85, '✅ SOLUCIÓN: Valores diferentes\nreflejan situaciones diferentes!', 
           ha='center', fontsize=11, fontweight='bold', color='green',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Detalles Escenario A
    ax = axes[1, 0]
    ax.axis('off')
    
    text_A = f"""
{scenarios[0]['name']}
{'='*45}

{scenarios[0]['description']}

MEDICIONES:
  Blob Size (crudo):      {scenarios[0]['blob_size']:.2f}
  Dist. al Centro:        {scenarios[0]['dist_center']:.2f}
  
CÁLCULO:
  Factor Centrado = 1 - ({scenarios[0]['dist_center']:.2f})²
                  = 1 - {scenarios[0]['dist_center']**2:.2f}
                  = {1 - scenarios[0]['dist_center']**2:.2f}
  
  Blob Efectivo = {scenarios[0]['blob_size']:.2f} × {1 - scenarios[0]['dist_center']**2:.2f}
                = {effective_A:.3f}

INTERPRETACIÓN:
  ⚠️ Valor bajo ({effective_A:.3f}) indica que aunque
  ve algo grande, NO está en buena posición.
  
  Acción recomendada: GIRAR para centrar
{'='*45}
"""
    
    ax.text(0.05, 0.95, text_A, transform=ax.transAxes, 
           fontsize=10, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Panel 4: Detalles Escenario B
    ax = axes[1, 1]
    ax.axis('off')
    
    text_B = f"""
{scenarios[1]['name']}
{'='*45}

{scenarios[1]['description']}

MEDICIONES:
  Blob Size (crudo):      {scenarios[1]['blob_size']:.2f}
  Dist. al Centro:        {scenarios[1]['dist_center']:.2f}
  
CÁLCULO:
  Factor Centrado = 1 - ({scenarios[1]['dist_center']:.2f})²
                  = 1 - {scenarios[1]['dist_center']**2:.2f}
                  = {1 - scenarios[1]['dist_center']**2:.2f}
  
  Blob Efectivo = {scenarios[1]['blob_size']:.2f} × {1 - scenarios[1]['dist_center']**2:.2f}
                = {effective_B:.3f}

INTERPRETACIÓN:
  ✅ Valor alto ({effective_B:.3f}) indica buena
  alineación. El tamaño refleja distancia real.
  
  Acción recomendada: AVANZAR recto
{'='*45}
"""
    
    ax.text(0.05, 0.95, text_B, transform=ax.transAxes, 
           fontsize=10, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('results_neat/comparacion_escenarios.png', dpi=300, bbox_inches='tight')
    print("✓ Guardado: results_neat/comparacion_escenarios.png")
    plt.close()


def plot_centering_factor():
    """
    Muestra cómo varía el factor de centrado con la distancia.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    distances = np.linspace(0, 1, 100)
    
    # Comparar diferentes exponentes
    factors_linear = 1 - distances
    factors_quadratic = 1 - (distances ** 2)
    factors_cubic = 1 - (distances ** 3)
    
    # Panel 1: Comparación de funciones
    ax = axes[0]
    ax.plot(distances, factors_linear, 'b-', linewidth=2.5, label='Lineal: 1 - d')
    ax.plot(distances, factors_quadratic, 'g-', linewidth=2.5, label='Cuadrática: 1 - d² (USADA)')
    ax.plot(distances, factors_cubic, 'r-', linewidth=2.5, label='Cúbica: 1 - d³')
    
    ax.set_xlabel('Distancia al Centro (normalizada)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Factor de Centrado', fontsize=12, fontweight='bold')
    ax.set_title('Comparación de Funciones de Penalización', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Marcar punto de ejemplo
    example_dist = 0.5
    example_factor = 1 - (example_dist ** 2)
    ax.plot(example_dist, example_factor, 'go', markersize=15, markeredgecolor='black', markeredgewidth=2)
    ax.annotate(f'Ejemplo: d={example_dist:.1f}\nFactor={example_factor:.2f}',
               xy=(example_dist, example_factor), xytext=(20, -20),
               textcoords='offset points',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
               arrowprops=dict(arrowstyle='->', lw=2))
    
    # Panel 2: Interpretación
    ax = axes[1]
    ax.axis('off')
    
    interpretation = """
INTERPRETACIÓN DEL FACTOR DE CENTRADO
═══════════════════════════════════════════════════

La función cuadrática (1 - d²) es la elegida porque:

✅ VENTAJAS:
  • Permisiva cerca del centro (d < 0.3)
    → No penaliza pequeñas desviaciones
  
  • Estricta lejos del centro (d > 0.7)
    → Penaliza fuertemente estar muy ladeado
  
  • Balance entre lineal y cúbica
    → Ni muy suave ni muy agresiva

📊 EJEMPLOS NUMÉRICOS:

  Distancia  │ Lineal │ Cuadrática │ Cúbica
  ═══════════╪════════╪════════════╪════════
    0.0      │  1.00  │    1.00    │  1.00
    0.2      │  0.80  │    0.96    │  0.99  ← Permisivo
    0.5      │  0.50  │    0.75    │  0.88
    0.7      │  0.30  │    0.51    │  0.66
    1.0      │  0.00  │    0.00    │  0.00  ← Estricto

⚖️ BALANCE:
  • Si d < 0.3: Factor > 0.90 (casi sin penalizar)
  • Si d > 0.7: Factor < 0.50 (penalizar fuertemente)
  
🎯 EFECTO EN APRENDIZAJE:
  El robot aprende que:
  - Pequeñas desviaciones están OK
  - Grandes desviaciones son MUY malas
  - Debe priorizar centrado antes de acercarse
"""
    
    ax.text(0.05, 0.95, interpretation, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('results_neat/factor_centrado.png', dpi=300, bbox_inches='tight')
    print("✓ Guardado: results_neat/factor_centrado.png")
    plt.close()


def main():
    """Función principal"""
    
    print("=" * 60)
    print("VISUALIZACIÓN DE LA PONDERACIÓN")
    print("=" * 60)
    
    import os
    os.makedirs('results_neat', exist_ok=True)
    
    print("\nGenerando visualizaciones...")
    
    print("\n1. Heatmap de blob_size_efectivo...")
    plot_heatmap()
    
    print("\n2. Comparación de escenarios...")
    plot_comparison_scenarios()
    
    print("\n3. Análisis del factor de centrado...")
    plot_centering_factor()
    
    print("\n" + "=" * 60)
    print("✅ VISUALIZACIONES COMPLETADAS")
    print("=" * 60)
    print("\nGráficas guardadas en results_neat/:")
    print("  • heatmap_ponderacion.png")
    print("  • comparacion_escenarios.png")
    print("  • factor_centrado.png")
    print("\nEstas gráficas muestran claramente cómo la ponderación")
    print("resuelve el problema de ambigüedad identificado por tu profesor.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
