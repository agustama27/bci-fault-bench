"""bcibench — banco de experimentación Software-in-the-Loop para el TFG.

Componentes (según Métodos del Entregable 1):
- data:      carga de BCI Competition IV 2a (BNCI2014_001) vía MOABB
- decoder:   CSP + LDA (mne + scikit-learn), entrenado en sesión 1
- replay:    reproducción de una corrida como flujo LSL a su tasa original
- injector:  modelos de fallo sobre la interfaz de transporte
- consumer:  pipeline bajo prueba (ventanas, preprocesado, decodificación)
- telemetry: recolector de telemetría por ventana
- detectors: detector por umbrales y detector basado en modelo
- runner:    ejecutor de campañas (semillas, bloques, reanudación)
"""
__version__ = "0.1.0"
