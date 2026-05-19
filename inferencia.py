"""
=============================================================
 INFERENCIA EN TIEMPO REAL — PROYECTO FINAL IA 2026
 Universidad Rafael Landívar
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
ESP32_IP    = "10.202.168.49"
ESP32_URL   = f"http://{ESP32_IP}/cmd"
SAMPLE_RATE = 16000
DURACION    = 1.5    # debe coincidir con modelo_voz.py
UMBRAL_CONF = 0.50
N_MFCC      = 13
SILENCIO_DB = 35     # threshold VAD: -35 dB acepta voz normal de micrófono

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
.sub{text-align:center;font-size:12px;color:#444;font-family:monospace;margin-bottom:24px}
.card{background:#111;border:1px solid #1a3a2a;border-radius:16px;padding:24px;max-width:480px;margin:0 auto 20px;text-align:center}
.big-word{font-size:52px;font-weight:900;color:#27a058;letter-spacing:3px;min-height:68px;transition:color .2s}
.big-word.dim{color:#222}
.big-word.warn{color:#e08800}
.conf-track{margin:14px 0 10px;height:10px;background:#1a1a1a;border-radius:5px;overflow:hidden}
.conf-fill{height:100%;border-radius:5px;transition:width .25s,background .25s}
.conf-fill.hi{background:#27a058}
.conf-fill.mid{background:#e08800}
.conf-fill.lo{background:#e04040}
.cmd-label{font-size:13px;font-family:monospace;color:#444;min-height:20px}
.cmd-label b{color:#27a058}
.dot-row{text-align:center;margin-bottom:16px}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;background:#27a058;margin-right:7px;animation:blink 1.4s infinite}
.dot-txt{font-size:13px;color:#27a058}
.dot-txt.err{color:#e04040}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.2}}
.log{max-width:480px;margin:0 auto;display:flex;flex-direction:column;gap:5px;max-height:320px;overflow-y:auto}
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
<div class="log" id="log"></div>
<script>
const EM={adelante:'⬆',atras:'⬇',izquierda:'⬅',derecha:'➡',detener:'⛔',curva_izq:'↩',curva_der:'↪',ruido:'~'};
const es=new EventSource('/events');
es.onmessage=e=>{
  const d=JSON.parse(e.data);
  if(d.tipo==='ping')return;
  const pct=Math.round(d.conf*100);
  const bw=document.getElementById('bw');
  const cf=document.getElementById('cf');
  const cl=document.getElementById('cl');
  bw.textContent=(EM[d.clase]||'')+' '+d.clase;
  bw.className='big-word'+(d.cmd?'':d.conf<0.5?' dim':' warn');
  cf.style.width=pct+'%';
  cf.className='conf-fill '+(d.conf>=0.7?'hi':d.conf>=0.5?'mid':'lo');
  cl.innerHTML=d.cmd?'&rarr; <b>'+d.cmd+'</b>':'<span style="color:#2a2a2a">ignorado ('+pct+'%)</span>';
  const log=document.getElementById('log');
  const r=document.createElement('div');
  r.className='row';
  const wc=d.cmd?'ok':d.conf<0.5?'dim':'warn';
  const ac=d.cmd?'ok':'no';
  r.innerHTML='<span class="rt">'+d.ts+'</span>'
    +'<span class="em">'+(EM[d.clase]||'')+' </span>'
    +'<span class="wrd '+wc+'">'+d.clase+'</span>'
    +'<span class="pct">'+pct+'%</span>'
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

# ── Extraccion de features ────────────────────────────────────────────────────
def extraer_mfcc(audio):
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    n     = int(SAMPLE_RATE * DURACION)
    audio = audio[:n] if len(audio) >= n else np.pad(audio, (0, n - len(audio)))
    mfcc       = librosa.feature.mfcc(y=audio.astype(float), sr=SAMPLE_RATE,
                                       n_mfcc=N_MFCC, n_fft=512, hop_length=160)
    mfcc_mean  = np.mean(mfcc, axis=1)
    mfcc_std   = np.std(mfcc, axis=1)
    delta_mean = np.mean(librosa.feature.delta(mfcc), axis=1)
    return np.concatenate([mfcc_mean, mfcc_std, delta_mean])

# ── VAD ───────────────────────────────────────────────────────────────────────
def tiene_voz(audio):
    rms = np.sqrt(np.mean(audio ** 2))
    return 20 * np.log10(rms + 1e-10) > -SILENCIO_DB

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
    print("  ║     🤖  ROBOT VOZ — Proyecto Final IA 2026          ║")
    print("  ║         Universidad Rafael Landívar                  ║")
    print("  ╚══════════════════════════════════════════════════════╝")
    print(f"{RESET}")
    print(f"  {GRIS}ESP32 :{RESET} {CYAN}{ESP32_IP}{RESET}   "
          f"{GRIS}Mín conf:{RESET} {CYAN}{int(UMBRAL_CONF*100)}%{RESET}   "
          f"{GRIS}Captura:{RESET} {CYAN}{DURACION}s{RESET}")
    print(f"  {GRIS}Web   :{RESET} {CYAN}http://localhost:5050{RESET}  "
          f"{GRIS}← abre en el browser{RESET}")
    print()
    print(f"  {GRIS}{'─'*56}{RESET}")
    print(f"  {GRIS}{'HORA':8}  {'OYÓ':13} {'CONFIANZA':22} {'ACCIÓN'}{RESET}")
    print(f"  {GRIS}{'─'*56}{RESET}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Cargar modelo
    print(f"\n{BOLD}  Cargando modelo...{RESET}")
    try:
        modelo  = joblib.load("modelo_voz.pkl")
        scaler  = joblib.load("scaler_voz.pkl")
        encoder = joblib.load("encoder_voz.pkl")
        clases  = list(encoder.classes_)
        print(f"  {VERDE}✓{RESET} Modelo listo — {len(clases)} clases: "
              f"{CYAN}{', '.join(clases)}{RESET}")
    except FileNotFoundError:
        print(f"  {ROJO}✗ No se encontro modelo_voz.pkl — ejecute primero: python modelo_voz.py{RESET}")
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

            feat      = extraer_mfcc(audio)
            fs        = scaler.transform([feat])
            pred      = modelo.predict(fs)[0]
            probas    = modelo.predict_proba(fs)[0]
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

            print(f"\r  {GRIS}{time.strftime('%H:%M:%S')}{RESET}  "
                  f"{color_clase}{emoji} {clase:11}{RESET}  "
                  f"{barra(conf)} {int(conf*100):3}%  "
                  f"{accion}                ")

            _push("pred", clase=clase, conf=round(conf, 3), cmd=cmd_enviado)

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
