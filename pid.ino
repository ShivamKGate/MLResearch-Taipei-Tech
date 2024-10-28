#include <Pixy2.h>
#include <PID_v1.h>

// Pixy2
Pixy2 pixy;

// Piny dla sterownika L298
#define ENA 6
#define ENB 7
#define IN1 2
#define IN2 3
#define IN3 4
#define IN4 5

// PID constants
double Kp = 2.2, Ki = 0.10, Kd = 0;
double Setpoint, Input, Output;
PID myPID(&Input, &Output, &Setpoint, Kp, Ki, Kd, DIRECT);

// Prędkości silników
int motorSpeedA = 0;
int motorSpeedB = 0;

// Funkcje dla silników
void setMotorA(int speed) {
  if (speed >= 0) {
    digitalWrite(IN1, HIGH);
    digitalWrite(IN2, LOW);
  } else {
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, HIGH);
    speed = -speed;
  }
  analogWrite(ENA, speed);
}

void setMotorB(int speed) {
  if (speed >= 0) {
    digitalWrite(IN3, HIGH);
    digitalWrite(IN4, LOW);
  } else {
    digitalWrite(IN3, LOW);
    digitalWrite(IN4, HIGH);
    speed = -speed;
  }
  analogWrite(ENB, speed);
}

void setup() {
  // Inicjalizacja pinów
  pinMode(ENA, OUTPUT);
  pinMode(ENB, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  // Inicjalizacja Pixy
  pixy.init();
  //pixy.setLamp(1, 1);  // Włącz diody LED kamery

  // Ustawienie początkowego Setpoint dla PID (środek obrazu)
  Setpoint = 160;  // Rozdzielczość kamery to 320x200, więc środek X to 160

  // Ustawienia PID
  myPID.SetMode(AUTOMATIC);
  myPID.SetOutputLimits(-255, 255);  // Zakres wyjścia PID dostosowany do PWM
}

void loop() {
  // Odczyt danych z Pixy
  pixy.ccc.getBlocks();

  if (pixy.ccc.numBlocks) {
    // Przechodzenie przez wszystkie wykryte obiekty
    for (int i = 0; i < pixy.ccc.numBlocks; i++) {
      int blockX = pixy.ccc.blocks[i].m_x;
      int blockSignature = pixy.ccc.blocks[i].m_signature;

      // Reaguj tylko na kostki czerwone (Signature 1) i zielone (Signature 2)
      if (blockSignature == 1 || blockSignature == 2) {
        // Wartość wejściowa dla PID to pozycja X wykrytego obiektu
        Input = blockX;

        // Obliczanie wartości wyjściowej PID
        myPID.Compute();

        if (abs(Output) > 15) {  // Jeśli kostka nie jest na wprost robota, obracaj się
          motorSpeedB = -Output;  // Przeciwnie ustawiona prędkość silników, aby obracać robota
          motorSpeedA = Output;
        } else {  // Jeśli kostka jest na wprost, jedź prosto
          motorSpeedA = 175;
          motorSpeedB = 175;
        }

        setMotorA(motorSpeedA);
        setMotorB(motorSpeedB);
        return;  // Wyjście z pętli po znalezieniu i zareagowaniu na odpowiedni blok
      }
    }
  }

  // Jeśli brak wykrycia odpowiednich kostek, zatrzymaj robota
  setMotorA(90);
  setMotorB(90);
  delay(100);
}