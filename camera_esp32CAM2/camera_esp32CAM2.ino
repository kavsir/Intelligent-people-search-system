// Test LED Flash ESP32-CAM AI Thinker



void setup() {
  pinMode(FLASH_LED_PIN, OUTPUT);
  digitalWrite(FLASH_LED_PIN, LOW);

  Serial.begin(115200);
  Serial.println("ESP32-CAM Flash LED Test");
}

void loop() {
  Serial.println("LED ON");
  digitalWrite(FLASH_LED_PIN, HIGH);   // Bật LED
  delay(500);

  Serial.println("LED OFF");
  digitalWrite(FLASH_LED_PIN, LOW);    // Tắt LED
  delay(500);
}