#include <WiFi.h>
#include <WebServer.h>
#include <ArduinoJson.h>
#include "protocol.h"
#include "web_page.h"

const char* WIFI_SSID = "URL-WIFI";
const char* WIFI_PASS = "";

WebServer server(80);

String currentCmd = "STOP";
String currentState = "STOPPED";

unsigned long lastSendMs = 0;
const int SEND_INTERVAL = 100;
const int CURVA_DURACION = 21000;

bool buildPacketFromCmd(const char* cmd, uint8_t* pkt){

  pkt[0] = MAGIC_BYTE;

  // ===== ADELANTE =====
  if(strcmp(cmd,"RECTA")==0){
    pkt[1]=50; pkt[2]=DIR_CCW;
    pkt[3]=50; pkt[4]=DIR_CW;
  }

  // ===== ATRAS =====
  else if(strcmp(cmd,"ATRAS")==0){
    pkt[1]=50; pkt[2]=DIR_CW;
    pkt[3]=50; pkt[4]=DIR_CCW;
  }

  // ===== IZQUIERDA =====
  else if(strcmp(cmd,"GIRO_90_IZQ")==0){
    pkt[1]=100; pkt[2]=DIR_CW;
    pkt[3]=100; pkt[4]=DIR_CW;
  }

  // ===== DERECHA =====
  else if(strcmp(cmd,"GIRO_90_DER")==0){
    pkt[1]=100; pkt[2]=DIR_CCW;
    pkt[3]=100; pkt[4]=DIR_CCW;
  }

  // ===== STOP =====
  else if(strcmp(cmd,"STOP")==0 || strcmp(cmd,"DETENTE")==0 || strcmp(cmd,"PARAR")==0){
    pkt[1]=0; pkt[2]=DIR_CCW;
    pkt[3]=0; pkt[4]=DIR_CW;
  }

  else return false;

  return true;
}

void sendCmd(const char* cmd){

  uint8_t pkt[PACKET_SIZE];

  if(buildPacketFromCmd(cmd,pkt)){
    Serial2.write(pkt,PACKET_SIZE);
  }
}

// ===== ADELANTE 10S =====
void adelante10s(){

  unsigned long inicio = millis();

  while(millis()-inicio < 10000){

    server.handleClient();

    sendCmd("RECTA");

    delay(120);

    if(currentCmd=="STOP") break;
  }

  sendCmd("STOP");
}

// ===== ATRAS 10S =====
void atras10s(){

  unsigned long inicio = millis();

  while(millis()-inicio < 10000){

    server.handleClient();

    sendCmd("ATRAS");

    delay(120);

    if(currentCmd=="STOP") break;
  }

  sendCmd("STOP");
}

// ===== CURVA IZQUIERDA =====
void curvaIzquierda(){

  unsigned long inicio = millis();

  while(millis()-inicio < CURVA_DURACION){

    server.handleClient();

    sendCmd("RECTA");
    delay(220);

    sendCmd("GIRO_90_IZQ");
    delay(130);

    if(currentCmd=="STOP") break;
  }

  sendCmd("STOP");
}

// ===== CURVA DERECHA =====
void curvaDerecha(){

  unsigned long inicio = millis();

  while(millis()-inicio < CURVA_DURACION){

    server.handleClient();

    sendCmd("RECTA");
    delay(220);

    sendCmd("GIRO_90_DER");
    delay(130);

    if(currentCmd=="STOP") break;
  }

  sendCmd("STOP");
}

void sendCORS(){

  server.sendHeader("Access-Control-Allow-Origin","*");
  server.sendHeader("Access-Control-Allow-Methods","GET,POST,OPTIONS");
  server.sendHeader("Access-Control-Allow-Headers","Content-Type");
}

void handleRoot(){

  sendCORS();

  server.send(200,"text/html",HTML_PAGE);
}

void handleStatus(){

  sendCORS();

  StaticJsonDocument<128> doc;

  doc["state"]=currentState;

  doc["last"]=currentCmd;

  String out;

  serializeJson(doc,out);

  server.send(200,"application/json",out);
}

void handleCmd(){

  sendCORS();

  if(server.method()==HTTP_OPTIONS){

    server.send(204);

    return;
  }

  if(!server.hasArg("plain")){

    server.send(400,"application/json","{\"error\":\"sin body\"}");

    return;
  }

  StaticJsonDocument<128> doc;

  if(deserializeJson(doc,server.arg("plain"))){

    server.send(400,"application/json","{\"error\":\"JSON invalido\"}");

    return;
  }

  const char* cmd=doc["cmd"];

  if(!cmd){

    server.send(400,"application/json","{\"error\":\"falta cmd\"}");

    return;
  }

  currentCmd=String(cmd);

  currentState=(strcmp(cmd,"STOP")==0)?"STOPPED":"NAVIGATING";

  lastSendMs=0;

  server.send(200,"application/json","{\"ok\":true}");
}

void setup(){

  Serial.begin(115200);

  Serial2.begin(115200,SERIAL_8N1,16,17);

  WiFi.begin(WIFI_SSID,WIFI_PASS);

  Serial.print("Conectando WiFi");

  while(WiFi.status()!=WL_CONNECTED){

    delay(500);

    Serial.print(".");
  }

  Serial.println();

  Serial.println("IP: "+WiFi.localIP().toString());

  server.on("/",HTTP_GET,handleRoot);

  server.on("/cmd",HTTP_POST,handleCmd);

  server.on("/cmd",HTTP_OPTIONS,handleCmd);

  server.on("/status",HTTP_GET,handleStatus);

  server.begin();

  Serial.println("Listo");
}

void loop(){

  server.handleClient();

  unsigned long now=millis();

  if(now-lastSendMs>=SEND_INTERVAL){

    lastSendMs=now;

    if(currentCmd=="RECTA_10S"){

      adelante10s();

      currentCmd="STOP";
    }
    else if(currentCmd=="ATRAS_10S"){

      atras10s();

      currentCmd="STOP";
    }
    else if(currentCmd=="CURVA_IZQ"){

      curvaIzquierda();

      currentCmd="STOP";
    }
    else if(currentCmd=="CURVA_DER"){

      curvaDerecha();

      currentCmd="STOP";
    }
    else if(currentCmd=="GIRO_90_IZQ"){

      sendCmd("GIRO_90_IZQ");

      delay(850);

      sendCmd("STOP");

      currentCmd="STOP";
    }
    else if(currentCmd=="GIRO_90_DER"){

      sendCmd("GIRO_90_DER");

      delay(850);

      sendCmd("STOP");

      currentCmd="STOP";
    }
    else{

      sendCmd(currentCmd.c_str());
    }
  }
}
