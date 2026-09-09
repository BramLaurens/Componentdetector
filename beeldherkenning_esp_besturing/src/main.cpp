#include <Arduino.h>
#include <Adafruit_NeoPixel.h>
#include <cmath>

constexpr uint8_t LED_PIN = 5;
constexpr uint16_t LED_COUNT = 24;
constexpr uint8_t STEPPER_DIR_PIN = 19;
constexpr uint8_t STEPPER_STEP_PIN = 18;

constexpr float MOTOR_STEP_ANGLE_DEGREES = 0.9f;
constexpr uint8_t MICROSTEPS = 8;
constexpr float STEPS_PER_DEGREE = MICROSTEPS / MOTOR_STEP_ANGLE_DEGREES;
constexpr uint16_t STEP_PULSE_DELAY_US = 500;

Adafruit_NeoPixel leds(LED_COUNT, LED_PIN, NEO_GRB + NEO_KHZ800);

void setLed(bool on) {
  if (on) {
    leds.fill(leds.Color(255, 255, 100));
  } else {
    leds.clear();
  }
  leds.show();
}

void rotateStepper(float degrees) {
  if (degrees == 0.0f) {
    return;
  }

  digitalWrite(STEPPER_DIR_PIN, degrees > 0.0f ? HIGH : LOW);
  const uint32_t stepCount = static_cast<uint32_t>(roundf(fabsf(degrees) * STEPS_PER_DEGREE));

  for (uint32_t step = 0; step < stepCount; ++step) {
    digitalWrite(STEPPER_STEP_PIN, HIGH);
    delayMicroseconds(STEP_PULSE_DELAY_US);
    digitalWrite(STEPPER_STEP_PIN, LOW);
    delayMicroseconds(STEP_PULSE_DELAY_US);
  }
}

void handleCommand(String command) {
  command.trim();
  command.toUpperCase();

  if (command == "LED ON") {
    setLed(true);
    Serial.println("OK LED ON");
    return;
  }

  if (command == "LED OFF") {
    setLed(false);
    Serial.println("OK LED OFF");
    return;
  }

  if (command.startsWith("ROTATE ")) {
    const String angleText = command.substring(7);
    char *end;
    const float degrees = strtof(angleText.c_str(), &end);

    if (end != angleText.c_str() && *end == '\0' && isfinite(degrees)) {
      rotateStepper(degrees);
      Serial.printf("OK ROTATE %.2f\n", degrees);
      return;
    }
  }

  Serial.println("ERROR Use: LED ON, LED OFF, or ROTATE <degrees>");
}

void setup() {
  pinMode(STEPPER_DIR_PIN, OUTPUT);
  pinMode(STEPPER_STEP_PIN, OUTPUT);
  digitalWrite(STEPPER_STEP_PIN, LOW);

  Serial.begin(115200);

  leds.begin();
  setLed(false);
  Serial.println("Ready. Use LED ON, LED OFF, or ROTATE <degrees>");
}

void loop() {
  if (Serial.available() > 0) {
    handleCommand(Serial.readStringUntil('\n'));
  }
}