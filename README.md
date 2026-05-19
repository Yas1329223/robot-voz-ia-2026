# Robot Voz — Proyecto Final IA 2026
**Universidad Rafael Landívar — Facultad de Ingeniería**

Sistema de control de robot móvil por comandos de voz en tiempo real.
Arquitectura: laptop (Python + modelo SVM) → ESP32 (WiFi) → Arduino Uno → motores stepper 28BYJ-48.

---

## Comandos reconocidos
`adelante` · `atras` · `izquierda` · `derecha` · `detener` · `curva_izq` · `curva_der`

---

## Estructura
```
esp32_firmware__1_/     Firmware del ESP32 (Arduino IDE)
  esp32_firmware__1_.ino
  web_page.h            Interfaz web del robot
  protocol.h            Protocolo de comunicación ESP32 → Arduino

dataset/                Carpetas vacías — grabar localmente con grabar_muestras.py
  adelante/
  atras/
  izquierda/
  derecha/
  detener/
  curva_izq/
  curva_der/
  ruido/

grabar_muestras.py      Graba muestras del micrófono de la laptop
modelo_voz.py           Entrena el modelo SVM/MLP con MFCC
inferencia.py           Inferencia en tiempo real + web en localhost:5050
```

---

## Instalación
```bash
pip install numpy sounddevice librosa scikit-learn joblib requests flask scipy
```

## Uso
```bash
# 1. Grabar dataset con el micrófono de la laptop
python grabar_muestras.py

# 2. Entrenar el modelo
python modelo_voz.py

# 3. Correr inferencia (abre http://localhost:5050 en el browser)
python inferencia.py
```

## Hardware
- Arduino Uno + 2× ULN2003 + 2× 28BYJ-48 stepper
- ESP32 (CH340) — recibe comandos por WiFi y los reenvía al Arduino por Serial
- Micrófono USB o integrado de laptop

## Configuración WiFi
Editar `esp32_firmware__1_/esp32_firmware__1_.ino`:
```cpp
const char* WIFI_SSID = "TU_RED";
const char* WIFI_PASS = "TU_PASSWORD";
```
Y actualizar la IP en `inferencia.py`:
```python
ESP32_IP = "IP_DEL_ESP32"
```
