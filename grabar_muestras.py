"""
grabar_muestras.py — Graba muestras del microfono para entrenar el modelo
USO: python grabar_muestras.py
"""

import os, time
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
import warnings
warnings.filterwarnings('ignore')

os.system('')

VERDE    = "\033[92m"
ROJO     = "\033[91m"
AMARILLO = "\033[93m"
CYAN     = "\033[96m"
GRIS     = "\033[90m"
BOLD     = "\033[1m"
RESET    = "\033[0m"

SAMPLE_RATE  = 16000
DURACION     = 1.5
DURACION_CURVA = 3.5
DATASET_DIR  = "dataset"
N_POR_SESION = 30

CLASES = {
    "adelante":  "Di claramente:  ADELANTE",
    "atras":     "Di claramente:  ATRAS",
    "izquierda": "Di claramente:  IZQUIERDA",
    "derecha":   "Di claramente:  DERECHA",
    "detener":   "Di claramente:  DETENER",
    "curva_izq": "Di claramente:  CURVA IZQUIERDA",
    "curva_der": "Di claramente:  CURVA DERECHA",
    "ruido":     "Haz ruido, golpea la mesa, habla cualquier cosa, o queda en silencio",
}

# Funciones auxiliares 
def contar(clase):
    carpeta = os.path.join(DATASET_DIR, clase)
    if not os.path.isdir(carpeta):
        return 0
    return len([f for f in os.listdir(carpeta)
                if f.lower().endswith(('.wav','.ogg','.mp3','.m4a','.flac'))])

# Detección de voz básica 
def tiene_voz(audio):
    rms = np.sqrt(np.mean(audio ** 2))
    return 20 * np.log10(rms + 1e-10) > -35

def grabar(duracion):
    audio = sd.rec(int(duracion * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype='float32')
    sd.wait()
    return audio.flatten()

def guardar(clase, audio):
    carpeta = os.path.join(DATASET_DIR, clase)
    os.makedirs(carpeta, exist_ok=True)
    ruta = os.path.join(carpeta, f"mic_{clase}_{int(time.time()*1000)}.wav")
    wavfile.write(ruta, SAMPLE_RATE, (audio * 32767).astype(np.int16))

def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"{BOLD}{VERDE}")
    print("  ╔══════════════════════════════════════════════╗")
    print("  ║   GRABAR MUESTRAS — Proyecto Final IA 2026  ║")
    print("  ╚══════════════════════════════════════════════╝")
    print(f"{RESET}")

    nombres = list(CLASES.keys())
    print(f"  {BOLD}Que clase quieres grabar?{RESET}\n")
    for i, clase in enumerate(nombres):
        n = contar(clase)
        color = VERDE if n >= 25 else (AMARILLO if n > 0 else ROJO)
        print(f"  {CYAN}  {i+1}{RESET}  {clase:12}  {color}{n} muestras{RESET}")
    print(f"\n  {CYAN}  0{RESET}  Salir\n")

    op = input(f"  Opcion: ").strip()
    if not op.isdigit() or int(op) == 0 or int(op) > len(nombres):
        print(f"\n  {AMARILLO}Saliendo.{RESET}\n")
        return

    clase = nombres[int(op) - 1]
    instruccion = CLASES[clase]
    duracion = DURACION_CURVA if clase in ("curva_izq", "curva_der") else DURACION

    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"\n  {BOLD}Clase: {VERDE}{clase.upper()}{RESET}\n")
    print(f"  {instruccion}\n")
    print(f"  Se grabaran {CYAN}{N_POR_SESION}{RESET} muestras de {CYAN}{duracion}s{RESET} cada una.")
    print(f"  Habla {BOLD}apenas aparezca GRABANDO{RESET} — tienes {duracion}s.\n")
    input(f"  {GRIS}Presiona ENTER cuando estes listo...{RESET}")
    print()

    grabadas = 0
    rechazadas = 0

    for i in range(N_POR_SESION):
        # Cuenta regresiva
        for n in [3, 2, 1]:
            print(f"\r  {GRIS}Muestra {i+1}/{N_POR_SESION}  —  {n}...{RESET}   ", end="", flush=True)
            time.sleep(0.6)

        print(f"\r  {BOLD}{VERDE}>>> GRABANDO <<<   muestra {i+1}/{N_POR_SESION}{RESET}   ", end="", flush=True)
        audio = grabar(duracion)

        if clase != "ruido" and not tiene_voz(audio):
            print(f"\r  {ROJO}✗ No se detecto voz — se repite{RESET}                    ")
            rechazadas += 1
            i -= 1  # no cuenta
            time.sleep(0.3)
            continue

        guardar(clase, audio)
        grabadas += 1
        print(f"\r  {VERDE}✓{RESET} Muestra {grabadas}/{N_POR_SESION} guardada                        ")
        time.sleep(0.1)

    print(f"\n  {VERDE}{BOLD}Listo!{RESET} {grabadas} muestras grabadas para '{clase}'.")
    if rechazadas:
        print(f"  {GRIS}({rechazadas} rechazadas por silencio){RESET}")

    print(f"\n  {GRIS}Pasos siguientes:{RESET}")
    print(f"  {CYAN}python grabar_muestras.py{RESET}  ← graba otra clase")
    print(f"  {CYAN}python modelo_voz.py{RESET}       ← re-entrena el modelo")
    print(f"  {CYAN}python inferencia.py{RESET}        ← prueba en tiempo real\n")

if __name__ == "__main__":
    main()
