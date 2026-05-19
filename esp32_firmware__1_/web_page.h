#pragma once

const char HTML_PAGE[] PROGMEM = R"rawhtml(
<!DOCTYPE html>
<html lang="es">

<head>

<meta charset="UTF-8">

<meta name="viewport" content="width=device-width, initial-scale=1">

<title>Robot Control</title>

<style>

*{
  box-sizing:border-box;
  margin:0;
  padding:0
}

body{
  font-family:Arial,sans-serif;
  background:radial-gradient(circle at top,#163b2a,#050505 60%);
  color:white;
  min-height:100vh;
  display:flex;
  flex-direction:column;
  align-items:center;
  justify-content:center;
  padding:20px;
  gap:18px
}

h1{
  font-size:30px;
  color:#45ff9a;
  text-shadow:0 0 18px rgba(69,255,154,.4)
}

#estado{
  font-size:14px;
  background:#101010;
  color:#45ff9a;
  padding:10px 18px;
  border-radius:25px;
  border:1px solid #2b8f5d
}

.wrapper{
  display:flex;
  flex-direction:row;
  gap:22px;
  align-items:stretch;
  justify-content:center;
  flex-wrap:wrap
}

.panel,.micBox{
  background:rgba(15,15,15,.92);
  border:1px solid #252525;
  border-radius:28px;
  padding:22px;
  box-shadow:0 0 30px rgba(0,0,0,.45)
}

.pad{
  display:grid;
  grid-template-columns:repeat(3,95px);
  grid-template-rows:repeat(4,85px);
  gap:10px
}

.btn{
  border-radius:20px;
  border:2px solid #252525;
  background:#171717;
  color:white;
  font-size:32px;
  cursor:pointer;
  transition:.12s
}

.btn:active{
  transform:scale(.92);
  background:#123d28;
  border-color:#45ff9a
}

.stop-btn{
  background:#3a1111;
  color:#ff5555;
  border-color:#632020
}

.curva{
  background:#101d31;
  border-color:#256096
}

.empty{
  background:transparent;
  border:none
}

.micBox{
  width:320px;
  display:flex;
  flex-direction:column;
  align-items:center;
  justify-content:center;
  gap:12px
}

#mic-btn{
  width:95px;
  height:95px;
  border-radius:50%;
  border:3px solid #333;
  background:#151515;
  font-size:40px;
  cursor:pointer
}

#mic-btn.on{
  border-color:#45ff9a;
  background:#0b281a;
  box-shadow:0 0 25px rgba(69,255,154,.35)
}

#transcript{
  font-size:18px;
  color:#ffffff;
  text-align:center;
  min-height:30px;
  width:100%;
  word-wrap:break-word
}

#debug{
  font-size:12px;
  color:#888
}

.chips{
  display:flex;
  flex-wrap:wrap;
  justify-content:center;
  gap:6px
}

.chip{
  font-size:11px;
  background:#151515;
  color:#45ff9a;
  border:1px solid #2b8f5d;
  padding:6px 10px;
  border-radius:20px
}

</style>

</head>

<body>

<h1>Robot Control</h1>

<div id="estado">Conectando...</div>

<div class="wrapper">

  <div class="panel">

    <div class="pad">

      <div class="empty"></div>

      <button class="btn" onclick="adelante()">&#x2B06;&#xFE0F;</button>

      <div class="empty"></div>

      <button class="btn" onclick="giro('GIRO_90_IZQ')">&#x2B05;&#xFE0F;</button>

      <button class="btn stop-btn" onclick="par()">&#x25A0;</button>

      <button class="btn" onclick="giro('GIRO_90_DER')">&#x27A1;&#xFE0F;</button>

      <div class="empty"></div>

      <button class="btn" onclick="atras()">&#x2B07;&#xFE0F;</button>

      <div class="empty"></div>

      <button class="btn curva" onclick="curva('CURVA_IZQ')">&#x21A9;&#xFE0F;</button>

      <button class="btn curva" onclick="curva('CURVA_DER')">&#x21AA;&#xFE0F;</button>

    </div>

  </div>

  <div class="micBox">

    <button id="mic-btn" onclick="toggleMic()">&#x1F3A4;</button>

    <div id="transcript">Toca el micr&oacute;fono y habla</div>

    <div id="debug">mic: apagado</div>

    <div class="chips">

      <span class="chip">adelante</span>

      <span class="chip">reversa</span>

      <span class="chip">izquierda</span>

      <span class="chip">derecha</span>

      <span class="chip">curva izquierda</span>

      <span class="chip">curva derecha</span>

      <span class="chip">detener</span>

    </div>

  </div>

</div>

<script>

const VOCES={
  "adelante":"RECTA_10S",
  "avanzar":"RECTA_10S",
  "recto":"RECTA_10S",
  "atras":"ATRAS_10S",
  "reversa":"ATRAS_10S",
  "retrocede":"ATRAS_10S",
  "retroceder":"ATRAS_10S",
  "para atras":"ATRAS_10S",
  "hacia atras":"ATRAS_10S",
  "curva izquierda":"CURVA_IZQ",
  "curva derecha":"CURVA_DER",
  "izquierda":"GIRO_90_IZQ",
  "derecha":"GIRO_90_DER",
  "detener":"STOP",
  "parar":"STOP",
  "alto":"STOP",
  "stop":"STOP"
};

async function send(cmd){
  try{
    await fetch('/cmd',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({cmd})
    });
    document.getElementById('estado').textContent='-> '+cmd;
  }catch(e){
    document.getElementById('estado').textContent='Sin conexion';
  }
}

function adelante(){ send('RECTA_10S'); }
function atras()   { send('ATRAS_10S'); }
function giro(cmd) { send(cmd); }
function curva(cmd){ send(cmd); }
function par()     { send('STOP'); }

let rec=null, mic=false;
function dbg(t){ document.getElementById('debug').textContent=t; }

function limpiarTexto(txt){
  return txt.toLowerCase().normalize("NFD").replace(/[̀-ͯ]/g,"");
}

function ejecutarComando(c){
  if(c==='RECTA_10S')      adelante();
  else if(c==='ATRAS_10S') atras();
  else if(c==='STOP')      par();
  else if(c==='CURVA_IZQ'||c==='CURVA_DER') curva(c);
  else                     giro(c);
}

function toggleMic(){
  if(!mic){
    const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
    if(!SR){ alert('Usa Chrome'); return; }
    rec=new SR();
    rec.lang='es-GT';
    rec.continuous=true;
    rec.interimResults=true;
    rec.onstart=()=>dbg('Escuchando...');
    rec.onresult=e=>{
      const resultado=e.results[e.results.length-1];
      const original=resultado[0].transcript.trim();
      const t=limpiarTexto(original);
      document.getElementById('transcript').textContent=original;
      dbg('Oyo: '+original);
      if(!e.results[e.results.length-1].isFinal)return;
      let encontrado=false;
      for(const[f,c] of Object.entries(VOCES)){
        if(t.includes(limpiarTexto(f))){
          dbg('Comando: '+c);
          ejecutarComando(c);
          encontrado=true;
          break;
        }
      }
      if(!encontrado) dbg('No reconocido');
    };
    rec.onerror=e=>dbg('Error: '+e.error);
    rec.start();
    mic=true;
    document.getElementById('mic-btn').className='on';
  } else {
    rec.stop();
    mic=false;
    document.getElementById('mic-btn').className='';
    dbg('mic: apagado');
    par();
  }
}

</script>

</body>

</html>
)rawhtml";
