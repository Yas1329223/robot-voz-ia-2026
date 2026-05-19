#include <SoftwareSerial.h>

SoftwareSerial espSerial(10, 11);

// Pines ULN2003
const int L1=6, L2=7, L3=8, L4=9;
const int R1=2, R2=3, R3=4, R4=5;

#define MAGIC_BYTE  0xAA
#define PACKET_SIZE 5
#define DIR_CW      0x00
#define DIR_CCW     0x01

uint8_t rxBuf[PACKET_SIZE];
uint8_t rxCount = 0;
bool    synced  = false;

uint8_t leftSteps  = 0;
uint8_t leftDir    = DIR_CW;
uint8_t rightSteps = 0;
uint8_t rightDir   = DIR_CW;
bool    stopped    = true;

// Secuencia de pasos para 28BYJ-48
const int SEQ[8][4] = {
  {1,0,0,0},{1,1,0,0},{0,1,0,0},{0,1,1,0},
  {0,0,1,0},{0,0,1,1},{0,0,0,1},{1,0,0,1}
};

int stepL = 0;
int stepR = 0;

void stepMotorL(int dir) {
  stepL = (stepL + (dir==DIR_CW ? 1 : -1) + 8) % 8;
  digitalWrite(L1, SEQ[stepL][0]);
  digitalWrite(L2, SEQ[stepL][1]);
  digitalWrite(L3, SEQ[stepL][2]);
  digitalWrite(L4, SEQ[stepL][3]);
}

void stepMotorR(int dir) {
  stepR = (stepR + (dir==DIR_CW ? 1 : -1) + 8) % 8;
  digitalWrite(R1, SEQ[stepR][0]);
  digitalWrite(R2, SEQ[stepR][1]);
  digitalWrite(R3, SEQ[stepR][2]);
  digitalWrite(R4, SEQ[stepR][3]);
}

void setup() {
  Serial.begin(115200);
  espSerial.begin(115200);
  pinMode(L1,OUTPUT); pinMode(L2,OUTPUT);
  pinMode(L3,OUTPUT); pinMode(L4,OUTPUT);
  pinMode(R1,OUTPUT); pinMode(R2,OUTPUT);
  pinMode(R3,OUTPUT); pinMode(R4,OUTPUT);
  Serial.println(" Arduino listo ");
}

void loop() {
  // Leer serial
  while (espSerial.available()) {
    uint8_t b = espSerial.read();
    if (!synced) {
      if (b == MAGIC_BYTE) { synced=true; rxBuf[0]=b; rxCount=1; }
      continue;
    }
    rxBuf[rxCount++] = b;
    if (rxCount == PACKET_SIZE) {
      leftSteps  = rxBuf[1];
      leftDir    = rxBuf[2];
      rightSteps = rxBuf[3];
      rightDir   = rxBuf[4];
      stopped    = (leftSteps==0 && rightSteps==0);
      rxCount=0; synced=false;
    }
  }

  // Mover ambos motores al mismo tiempo
  if (!stopped) {
    stepMotorL(leftDir);
    stepMotorR(rightDir);
    delayMicroseconds(1200);
  }
}