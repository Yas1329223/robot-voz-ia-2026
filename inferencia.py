"""
=============================================================
                INFERENCIA EN TIEMPO REAL
=============================================================
Captura audio -> clasifica -> manda al ESP32 por WiFi
Web en tiempo real: http://localhost:5050

USO:
  pip install flask
  python inferencia.py
"""

import os, sys, time, warnings, threading, queue, json
import numpy as np
import sounddevice as sd
import librosa
import joblib
import torch
import torch.nn as nn
import requests
warnings.filterwarnings('ignore')

os.system('')  # habilitar ANSI en Windows 10+

# ── Colores ───────────────────────────────────────────────────────────────────
VERDE    = "\033[92m"
ROJO     = "\033[91m"
AMARILLO = "\033[93m"
CYAN     = "\033[96m"
GRIS     = "\033[90m"
BOLD     = "\033[1m"
RESET    = "\033[0m"


# ── Configuracion ─────────────────────────────────────────────────────────────
ESP32_IP    = "192.168.0.32"
ESP32_URL   = f"http://{ESP32_IP}/cmd"
SAMPLE_RATE = 16000
DURACION    = 1.5    # debe coincidir con modelo_voz.py
UMBRAL_CONF = 0.60
N_MFCC      = 13
SILENCIO_DB = 28     # threshold VAD: -28 dB requiere voz más clara, filtra ruido ambiente

COMANDOS = {
    "adelante":  "RECTA_10S",
    "atras":     "ATRAS_10S",
    "izquierda": "GIRO_90_IZQ",
    "derecha":   "GIRO_90_DER",
    "detener":   "STOP",
    "curva_izq": "CURVA_IZQ",
    "curva_der": "CURVA_DER",
    "ruido":     None,
}

EMOJIS = {
    "adelante":  "⬆",
    "atras":     "⬇",
    "izquierda": "⬅",
    "derecha":   "➡",
    "detener":   "⛔",
    "curva_izq": "↩",
    "curva_der": "↪",
    "ruido":     "~",
}

# ── Comandos compuestos (Módulo Secuencial LSTM) ───────────────────────────────
# Detecta secuencias de 2 palabras y ejecuta acción compuesta
COMPUESTOS = {
    ("adelante",  "detener"):   ("RECTA_10S",   "STOP",        "AVANZA Y PARA"),
    ("atras",     "detener"):   ("ATRAS_10S",   "STOP",        "RETROCEDE Y PARA"),
    ("izquierda", "adelante"):  ("GIRO_90_IZQ", "RECTA_10S",   "GIRA IZQ Y AVANZA"),
    ("derecha",   "adelante"):  ("GIRO_90_DER", "RECTA_10S",   "GIRA DER Y AVANZA"),
    ("izquierda", "derecha"):   ("GIRO_90_IZQ", "GIRO_90_DER", "GIRA IZQ Y DER"),
    ("derecha",   "izquierda"): ("GIRO_90_DER", "GIRO_90_IZQ", "GIRA DER Y IZQ"),
    ("adelante",  "izquierda"): ("RECTA_10S",   "GIRO_90_IZQ", "AVANZA Y GIRA IZQ"),
    ("adelante",  "derecha"):   ("RECTA_10S",   "GIRO_90_DER", "AVANZA Y GIRA DER"),
}
VENTANA_COMPUESTO = 6.0   # segundos máximos entre dos palabras

_seq_buffer: list = []    # [(clase, timestamp), ...]

def detectar_compuesto(clase, conf):
    """Registra la clase detectada y verifica si forma un comando compuesto."""
    ahora = time.time()
    # Limpiar entradas antiguas
    while _seq_buffer and (ahora - _seq_buffer[0][1]) > VENTANA_COMPUESTO:
        _seq_buffer.pop(0)
    # Solo acumula si la confianza es alta (≥70%) para evitar falsos compuestos
    if clase and clase != "ruido" and conf >= 0.70:
        _seq_buffer.append((clase, ahora))
    # Buscar patrón en las últimas 2 palabras
    if len(_seq_buffer) >= 2:
        seq = (_seq_buffer[-2][0], _seq_buffer[-1][0])
        if seq in COMPUESTOS:
            cmd1, cmd2, etiqueta = COMPUESTOS[seq]
            _seq_buffer.clear()
            return cmd1, cmd2, etiqueta
    return None, None, None

# ── Cola SSE ──────────────────────────────────────────────────────────────────
_sse: queue.Queue = queue.Queue(maxsize=60)

def _push(tipo, **kw):
    kw["tipo"] = tipo
    kw["ts"]   = time.strftime("%H:%M:%S")
    try:
        _sse.put_nowait(kw)
    except queue.Full:
        pass

# ── HTML embebido ─────────────────────────────────────────────────────────────
_HTML = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Robot Voz Live</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:sans-serif;background:#0a0a0a;color:#fff;min-height:100vh;padding:20px 16px}
h1{text-align:center;font-size:24px;font-weight:800;color:#27a058;padding:18px 0 4px}
.sub{text-align:center;font-size:12px;color:#444;font-family:monospace;margin-bottom:20px}
.dot-row{text-align:center;margin-bottom:16px}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;background:#27a058;margin-right:7px;animation:blink 1.4s infinite}
.dot-txt{font-size:13px;color:#27a058}
.dot-txt.err{color:#e04040}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.2}}

/* Tarjeta palabra actual */
.card{background:#111;border:1px solid #1a3a2a;border-radius:16px;padding:20px 24px;max-width:480px;margin:0 auto 20px;text-align:center}
.big-word{font-size:48px;font-weight:900;color:#27a058;letter-spacing:3px;min-height:62px;transition:color .2s}
.big-word.dim{color:#222}
.big-word.warn{color:#e08800}
.conf-track{margin:12px 0 8px;height:8px;background:#1a1a1a;border-radius:4px;overflow:hidden}
.conf-fill{height:100%;border-radius:4px;transition:width .25s,background .25s}
.conf-fill.hi{background:#27a058}
.conf-fill.mid{background:#e08800}
.conf-fill.lo{background:#e04040}
.cmd-label{font-size:13px;font-family:monospace;color:#444;min-height:18px}
.cmd-label b{color:#27a058}

/* Transcripcion fluida */
.transcript-box{max-width:480px;margin:0 auto 20px;background:#0f0f0f;border:1px solid #1a1a1a;border-radius:16px;padding:18px 20px;min-height:90px}
.transcript-title{font-size:11px;color:#333;font-family:monospace;margin-bottom:10px;letter-spacing:1px}
.transcript-text{font-size:22px;font-weight:600;line-height:1.7;word-wrap:break-word}
.transcript-text .palabra{display:inline;margin-right:8px;transition:color 1.5s}
.transcript-text .nueva{color:#27a058}
.transcript-text .reciente{color:#ccc}
.transcript-text .vieja{color:#333}
.transcript-text .ignorada{color:#222;font-size:16px}
.transcript-text .ruido-t{color:#1e1e1e;font-size:14px}
.cursor{display:inline-block;width:2px;height:22px;background:#27a058;margin-left:2px;vertical-align:middle;animation:cursor .9s infinite}
@keyframes cursor{0%,100%{opacity:1}50%{opacity:0}}

/* Historial */
.log{max-width:480px;margin:0 auto;display:flex;flex-direction:column;gap:5px;max-height:220px;overflow-y:auto}
.row{display:flex;align-items:center;gap:8px;padding:7px 12px;background:#111;border:1px solid #161616;border-radius:8px;font-size:12px;animation:pop .25s}
@keyframes pop{from{opacity:0;transform:translateY(3px)}to{opacity:1}}
.rt{color:#2a2a2a;font-family:monospace;font-size:11px;min-width:52px}
.em{font-size:15px;min-width:22px;text-align:center}
.wrd{font-weight:700;min-width:90px}
.wrd.ok{color:#27a058}.wrd.dim{color:#2a2a2a}.wrd.warn{color:#e08800}
.pct{color:#333;font-family:monospace;font-size:11px;min-width:38px}
.act{margin-left:auto;font-family:monospace;font-size:11px}
.act.ok{color:#27a058}.act.no{color:#2a2a2a}
</style>
</head>
<body>
<h1>🤖 Robot Voz</h1>
<p class="sub">Universidad Rafael Landívar &middot; Proyecto Final IA 2026</p>
<div class="dot-row">
  <span class="dot"></span>
  <span class="dot-txt" id="stxt">Conectado &mdash; escuchando...</span>
</div>

<div class="card">
  <div class="big-word dim" id="bw">&mdash;</div>
  <div class="conf-track"><div class="conf-fill" id="cf" style="width:0%"></div></div>
  <div class="cmd-label" id="cl">esperando...</div>
</div>

<div class="transcript-box">
  <div class="transcript-title">TRANSCRIPCION EN VIVO</div>
  <div class="transcript-text" id="transcript"><span class="cursor"></span></div>
</div>
<div id="sr-warn" style="display:none;text-align:center;font-size:12px;color:#e04040;margin-bottom:10px">Tu browser no soporta reconocimiento de voz</div>

<div class="log" id="log"></div>

<script>
const EM={adelante:'⬆',atras:'⬇',izquierda:'⬅',derecha:'➡',detener:'⛔',curva_izq:'↩',curva_der:'↪',ruido:'~'};
const es=new EventSource('/events');

// ── Web Speech API — transcripcion real de todo lo que se dice ──
const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
let srFinal='';
if(SR){
  const recog=new SR();
  recog.lang='es-GT';
  recog.continuous=true;
  recog.interimResults=true;
  recog.onresult=ev=>{
    let interim='';
    for(let i=ev.resultIndex;i<ev.results.length;i++){
      if(ev.results[i].isFinal) srFinal+=ev.results[i][0].transcript+' ';
      else interim+=ev.results[i][0].transcript;
    }
    // mantener solo las ultimas ~200 chars del texto final
    if(srFinal.length>200) srFinal=srFinal.slice(-200);
    const box=document.getElementById('transcript');
    box.innerHTML='<span class="vieja">'+srFinal+'</span>'
      +'<span class="nueva">'+interim+'</span>'
      +'<span class="cursor"></span>';
    box.scrollTop=box.scrollHeight;
  };
  recog.onend=()=>{try{recog.start();}catch(e){}};
  recog.start();
} else {
  document.getElementById('sr-warn').style.display='block';
}

es.onmessage=e=>{
  const d=JSON.parse(e.data);
  if(d.tipo==='ping')return;
  const pct=Math.round(d.conf*100);

  // Tarjeta principal
  const bw=document.getElementById('bw');
  const cf=document.getElementById('cf');
  const cl=document.getElementById('cl');
  bw.textContent=(EM[d.clase]||'')+' '+d.clase;
  bw.className='big-word'+(d.cmd?'':d.conf<0.5?' dim':' warn');
  cf.style.width=pct+'%';
  cf.className='conf-fill '+(d.conf>=0.7?'hi':d.conf>=0.5?'mid':'lo');
  cl.innerHTML=d.cmd?'&rarr; <b>'+d.cmd+'</b>':'<span style="color:#2a2a2a">ignorado ('+pct+'%)</span>';


  // Historial
  const log=document.getElementById('log');
  const r=document.createElement('div');
  r.className='row';
  const wc=d.cmd?'ok':d.conf<0.5?'dim':'warn';
  const ac=d.cmd?'ok':'no';
  const lat=d.lat_ms!=null?d.lat_ms+'ms':'';
  r.innerHTML='<span class="rt">'+d.ts+'</span>'
    +'<span class="em">'+(EM[d.clase]||'')+' </span>'
    +'<span class="wrd '+wc+'">'+d.clase+'</span>'
    +'<span class="pct">'+pct+'%</span>'
    +'<span class="pct" style="color:#555">'+lat+'</span>'
    +'<span class="act '+ac+'">'+(d.cmd||'&mdash;')+'</span>';
  log.prepend(r);
  if(log.children.length>40)log.removeChild(log.lastChild);
};
es.onerror=()=>{
  const t=document.getElementById('stxt');
  t.textContent='Reconectando...';t.className='dot-txt err';
};
es.onopen=()=>{
  const t=document.getElementById('stxt');
  t.textContent='Conectado — escuchando...';t.className='dot-txt';
};
</script>
</body>
</html>"""

# ── Flask server ──────────────────────────────────────────────────────────────
def _flask_thread():
    try:
        from flask import Flask, Response
        import logging
        logging.getLogger('werkzeug').setLevel(logging.ERROR)

        app = Flask(__name__)

        @app.route('/')
        def index():
            return _HTML

        @app.route('/events')
        def events():
            def gen():
                while True:
                    try:
                        ev = _sse.get(timeout=20)
                        yield f"data: {json.dumps(ev)}\n\n"
                    except queue.Empty:
                        yield 'data: {"tipo":"ping"}\n\n'
            return Response(gen(), mimetype='text/event-stream',
                            headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

        app.run(host='0.0.0.0', port=5050, threaded=True, use_reloader=False)
    except ImportError:
        pass  # Flask no instalado — continua sin web
    except Exception:
        pass

# ── Arquitectura LSTM (debe coincidir con modelo_lstm.py) ────────────────────
class LSTMVoz(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, n_clases, dropout=0.3):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden1, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.fc1   = nn.Linear(hidden2, 64)
        self.relu  = nn.ReLU()
        self.fc2   = nn.Linear(64, n_clases)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out    = self.drop1(out)
        out, _ = self.lstm2(out)
        out    = out[:, -1, :]
        out    = self.drop2(out)
        return self.fc2(self.relu(self.fc1(out)))

# ── Extraccion de features ────────────────────────────────────────────────────
def _normalizar_audio(audio):
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    n = int(SAMPLE_RATE * DURACION)
    return audio[:n] if len(audio) >= n else np.pad(audio, (0, n - len(audio)))

def extraer_mfcc(audio):
    audio = _normalizar_audio(audio)
    mfcc       = librosa.feature.mfcc(y=audio.astype(float), sr=SAMPLE_RATE,
                                       n_mfcc=N_MFCC, n_fft=512, hop_length=160)
    mfcc_mean  = np.mean(mfcc, axis=1)
    mfcc_std   = np.std(mfcc, axis=1)
    delta_mean = np.mean(librosa.feature.delta(mfcc), axis=1)
    return np.concatenate([mfcc_mean, mfcc_std, delta_mean])

def extraer_mfcc_seq(audio):
    audio = _normalizar_audio(audio)
    mfcc = librosa.feature.mfcc(y=audio.astype(float), sr=SAMPLE_RATE,
                                  n_mfcc=N_MFCC, n_fft=512, hop_length=160)
    return mfcc.T  # (timesteps, N_MFCC)

# ── VAD mejorado ──────────────────────────────────────────────────────────────
def tiene_voz(audio):
    rms_total = np.sqrt(np.mean(audio ** 2))
    if 20 * np.log10(rms_total + 1e-10) <= -SILENCIO_DB:
        return False
    # Verificar que al menos 20% del audio tenga energía real (evita clicks cortos)
    frame = int(SAMPLE_RATE * 0.02)
    frames_con_voz = sum(
        1 for i in range(0, len(audio) - frame, frame)
        if np.sqrt(np.mean(audio[i:i+frame]**2)) > 10**(-SILENCIO_DB/20)
    )
    total_frames = (len(audio) - frame) // frame
    return frames_con_voz / max(total_frames, 1) >= 0.20

# ── ESP32 ─────────────────────────────────────────────────────────────────────
def enviar(cmd):
    try:
        return requests.post(ESP32_URL, json={"cmd": cmd}, timeout=1.0).status_code == 200
    except Exception:
        return False

# ── Barra de confianza (terminal) ─────────────────────────────────────────────
def barra(v, w=18):
    n = int(v * w)
    c = VERDE if v >= 0.7 else (AMARILLO if v >= 0.5 else ROJO)
    return c + "█" * n + GRIS + "░" * (w - n) + RESET

# ── Header de terminal ────────────────────────────────────────────────────────
def header(clases):
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"{BOLD}{VERDE}")
    print("  ╔══════════════════════════════════════════════════════╗")
    print("  ║         ROBOT VOZ — Proyecto Final IA                ║")
    print("  ╚══════════════════════════════════════════════════════╝")
    print(f"{RESET}")
    print(f"  {GRIS}ESP32 :{RESET} {CYAN}{ESP32_IP}{RESET}   "
          f"{GRIS}Mín conf:{RESET} {CYAN}{int(UMBRAL_CONF*100)}%{RESET}   "
          f"{GRIS}Captura:{RESET} {CYAN}{DURACION}s{RESET}")
    print(f"  {GRIS}Web   :{RESET} {CYAN}http://localhost:5050{RESET}  "
          f"{GRIS}← abre en el browser{RESET}")
    print()
    print(f"  {GRIS}{'─'*66}{RESET}")
    print(f"  {GRIS}{'HORA':8}  {'OYÓ':13} {'CONFIANZA':22} {'LATENCIA':10} {'ACCIÓN'}{RESET}")
    print(f"  {GRIS}{'─'*66}{RESET}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Cargar modelo LSTM
    print(f"\n{BOLD}  Cargando modelo LSTM...{RESET}")
    try:
        scaler  = joblib.load("scaler_lstm.pkl")
        encoder = joblib.load("encoder_lstm.pkl")
        clases  = list(encoder.classes_)
        n_cls   = len(clases)
        lstm_model = LSTMVoz(N_MFCC, 128, 64, n_cls)
        lstm_model.load_state_dict(torch.load("modelo_lstm.pt", weights_only=True))
        lstm_model.eval()
        print(f"  {VERDE}✓{RESET} LSTM listo (98.6%) — {n_cls} clases: "
              f"{CYAN}{', '.join(clases)}{RESET}")
    except FileNotFoundError:
        print(f"  {ROJO}✗ No se encontro modelo_lstm.pt — ejecute primero: python modelo_lstm.py{RESET}")
        sys.exit(1)

    # Verificar ESP32
    print(f"  Verificando ESP32 {GRIS}({ESP32_IP}){RESET}...")
    try:
        esp_ok = requests.get(f"http://{ESP32_IP}/status", timeout=2.0).status_code == 200
    except Exception:
        esp_ok = False
    print(f"  {''+VERDE+'✓ ESP32 conectado' if esp_ok else AMARILLO+'⚠  ESP32 sin respuesta (modo demo)'}{RESET}")

    # Iniciar web server
    threading.Thread(target=_flask_thread, daemon=True).start()
    time.sleep(0.8)

    header(clases)

    ultimo_cmd    = None
    ultimo_tiempo = 0
    COOLDOWN      = 0.8

    while True:
        try:
            audio = sd.rec(int(DURACION * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                           channels=1, dtype='float32')
            sd.wait()
            audio = audio.flatten()

            if not tiene_voz(audio):
                print(f"  {GRIS}{time.strftime('%H:%M:%S')}  {'·':13}  {'[silencio]'}{RESET}           ",
                      end="\r")
                continue

            t_inicio  = time.perf_counter()
            seq       = extraer_mfcc_seq(audio)
            seq_norm  = scaler.transform(seq)
            xt        = torch.tensor(seq_norm[np.newaxis], dtype=torch.float32)
            with torch.no_grad():
                logits = lstm_model(xt)[0]
                probas = torch.softmax(logits, dim=0).numpy()
            pred      = int(probas.argmax())
            conf      = float(probas.max())
            clase     = encoder.inverse_transform([pred])[0]
            emoji     = EMOJIS.get(clase, "")
            cmd_http  = COMANDOS.get(clase)

            # Decidir acción
            cmd_enviado = None
            if conf < UMBRAL_CONF:
                accion      = f"{GRIS}ignorado ({int(conf*100)}%){RESET}"
                color_clase = GRIS
            elif clase == "ruido":
                accion      = f"{GRIS}ruido{RESET}"
                color_clase = GRIS
            else:
                ahora = time.time()
                if cmd_http == ultimo_cmd and (ahora - ultimo_tiempo) < COOLDOWN:
                    accion      = f"{GRIS}cooldown{RESET}"
                    color_clase = AMARILLO
                elif enviar(cmd_http):
                    cmd_enviado   = cmd_http
                    ultimo_cmd    = cmd_http
                    ultimo_tiempo = ahora
                    accion        = f"{VERDE}→ {cmd_http}{RESET}"
                    color_clase   = VERDE
                else:
                    accion      = f"{ROJO}✗ ESP32 sin respuesta{RESET}"
                    color_clase = ROJO

            t_ms = (time.perf_counter() - t_inicio) * 1000
            lat_color = VERDE if t_ms < 150 else (AMARILLO if t_ms < 300 else ROJO)
            lat_str = f"{lat_color}{t_ms:5.0f}ms{RESET}"

            print(f"\r  {GRIS}{time.strftime('%H:%M:%S')}{RESET}  "
                  f"{color_clase}{emoji} {clase:11}{RESET}  "
                  f"{barra(conf)} {int(conf*100):3}%  "
                  f"{lat_str}  "
                  f"{accion}                ")

            # ── Detección de comando compuesto (módulo secuencial) ─────────────
            if conf >= UMBRAL_CONF and clase != "ruido":
                c1, c2, etiqueta = detectar_compuesto(clase, conf)
                if c1:
                    print(f"\n  {BOLD}{AMARILLO}⚡ COMANDO COMPUESTO: {etiqueta}{RESET}")
                    print(f"     Ejecutando: {CYAN}{c1}{RESET} → {CYAN}{c2}{RESET}")
                    enviar(c1)
                    time.sleep(2.0)
                    enviar(c2)
                    cmd_enviado = f"{c1}+{c2}"
                    _push("pred", clase=f"[{etiqueta}]", conf=round(conf, 3),
                          cmd=cmd_enviado)
                    print(f"  {GRIS}{'─'*56}{RESET}")
                    continue

            _push("pred", clase=clase, conf=round(conf, 3), cmd=cmd_enviado,
                  lat_ms=round(t_ms))

        except KeyboardInterrupt:
            print(f"\n\n  {AMARILLO}Deteniendo...{RESET}")
            enviar("STOP")
            print(f"  {VERDE}✓ STOP enviado. Hasta luego.{RESET}\n")
            break
        except Exception as ex:
            print(f"\r  {ROJO}Error: {ex}{RESET}          ")
            time.sleep(0.3)

if __name__ == "__main__":
    main()
