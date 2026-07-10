/*
  esp32_servo.ino
  -----------------
  MỘT ESP32 (Dev Module) + MỘT PCA9685 (I2C) điều khiển CẢ 4 servo của hệ
  thống, tất cả giao tiếp với server qua WebSocket (đã GỘP
  esp32_pca9685_controller.ino vào đây -- không còn Serial-USB-to-PC cho
  pan/tilt nữa):

    PCA9685 channel 0 -> Pan   (theo dõi người đăng ký, servo_controller.py)
    PCA9685 channel 1 -> Tilt
    PCA9685 channel 2 -> Cửa Phòng 1 (SG90) -- id "cam1"
    PCA9685 channel 3 -> Cửa Phòng 2 (SG90) -- id "cam2"

  Giao thức WebSocket multiplex trên MỘT kết nối (giữ nguyên định dạng cửa
  cũ như esp32_servo.ino bản trước, thêm loại message mới cho pan/tilt):

    ESP32 -> Server (khi vừa kết nối, và sau mỗi lần đổi trạng thái cửa):
        {"door": "cam1", "state": "closed"}
        {"door": "cam2", "state": "closed"}

    Server -> ESP32 (khi bấm nút cửa trên dashboard):
        {"door": "cam1", "cmd": "OPEN"}
        {"door": "cam2", "cmd": "CLOSE"}

    Server -> ESP32 (MỚI -- operation/servo_controller.py gửi liên tục
    theo vòng lặp PID, ~20 lần/giây, KHÔNG cần ack):
        {"servo": "pantilt", "pan": 95, "tilt": 88}

  Thư viện cần cài (Arduino IDE > Library Manager):
    - WebSocketsClient   (Markus Sattler)
    - ArduinoJson
    - Adafruit PWM Servo Driver Library
  (KHÔNG cần ESP32Servo nữa -- mọi servo giờ ra PWM qua PCA9685/I2C, không
  còn servo nào gắn thẳng vào GPIO của ESP32.)

  Đấu nối:
    ESP32 GPIO21 (SDA) -> PCA9685 SDA
    ESP32 GPIO22 (SCL) -> PCA9685 SCL
    ESP32 GND -- PCA9685 GND -- GND nguồn ngoài PHẢI NỐI CHUNG (mass chung).
    Nguồn ngoài 5-6V, đủ dòng cho cả 4 servo cộng dồn -> PCA9685 V+
    (KHÔNG lấy nguồn servo từ chân 5V/3V3 của ESP32).
    Servo Pan          -> PCA9685 channel 0
    Servo Tilt         -> PCA9685 channel 1
    Servo cửa Phòng 1  -> PCA9685 channel 2
    Servo cửa Phòng 2  -> PCA9685 channel 3

  AN TOÀN:
    - Pan/tilt: nếu quá SAFETY_TIMEOUT_MS không nhận lệnh mới, chỉ CẢNH BÁO
      qua Serial (log), KHÔNG tự ý quay -- việc đưa servo về vị trí an toàn
      khi mất mục tiêu do phía Python (servo_controller.py) chủ động quyết
      định và gửi lệnh, giống hệt hành vi cũ khi còn dùng Serial USB.
    - Cửa: mất kết nối WebSocket không tự mở/đóng cửa -- giữ nguyên trạng
      thái hiện tại; WebSocketsClient tự động thử kết nối lại (3s/lần).
*/

#include <Wire.h>
#include <WiFi.h>
#include <WebSocketsClient.h>
#include <ArduinoJson.h>
#include <Adafruit_PWMServoDriver.h>

// ---- WiFi ----
const char* ssid     = "seele";
const char* password = "0123456789.";

// ---- Server running app_dashboard.py / door_ws_server.py ----
const char* ws_host = "10.208.229.207";   // <-- SET THIS to your server's LAN IP
const uint16_t ws_port = 8765;           // must match config.DOOR_WS_PORT
const char* ws_path = "/";

// ---- PCA9685 ----
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);

#define PAN_CHANNEL        0
#define TILT_CHANNEL       1
#define DOOR_CAM1_CHANNEL  2   // Phòng 1 door servo
#define DOOR_CAM2_CHANNEL  3   // Phòng 2 door servo

// Hiệu chỉnh theo servo thật của bạn (đo bằng cách quét thử và quan sát góc thực tế)
#define SERVO_MIN_PULSE  150   // tick tương ứng góc 0 độ (~500us ở 50Hz, 4096 tick/chu kỳ)
#define SERVO_MAX_PULSE  600   // tick tương ứng góc 180 độ (~2500us)

#define SAFETY_TIMEOUT_MS 3000  // pan/tilt: quá lâu không có lệnh mới -> chỉ log cảnh báo

const char* DOOR_ID_CAM1 = "cam1";
const char* DOOR_ID_CAM2 = "cam2";

int stateCam1 = 0;  // 0 = CLOSED, 1 = OPEN
int stateCam2 = 0;

int currentPan = 90;
int currentTilt = 90;
unsigned long lastPanTiltCommandTime = 0;

WebSocketsClient webSocket;

// ---------------------------------------------------------------------
// PCA9685 helpers
// ---------------------------------------------------------------------
int angleToPulse(int angle) {
  angle = constrain(angle, 0, 180);
  return map(angle, 0, 180, SERVO_MIN_PULSE, SERVO_MAX_PULSE);
}

void setServoAngle(uint8_t channel, int angle) {
  pwm.setPWM(channel, 0, angleToPulse(angle));
}

// ---------------------------------------------------------------------
// Door helpers (kênh PCA9685 thay cho Servo::attach() trực tiếp GPIO)
// ---------------------------------------------------------------------
int doorChannelFor(const String& doorId) {
  if (doorId == DOOR_ID_CAM1) return DOOR_CAM1_CHANNEL;
  if (doorId == DOOR_ID_CAM2) return DOOR_CAM2_CHANNEL;
  return -1;
}

int* stateForDoor(const String& doorId) {
  if (doorId == DOOR_ID_CAM1) return &stateCam1;
  if (doorId == DOOR_ID_CAM2) return &stateCam2;
  return nullptr;
}

void sendState(const char* doorId, int state) {
  StaticJsonDocument<96> doc;
  doc["door"] = doorId;
  doc["state"] = (state == 1) ? "open" : "closed";
  String out;
  serializeJson(doc, out);
  webSocket.sendTXT(out);
}

void sendAllStates() {
  sendState(DOOR_ID_CAM1, stateCam1);
  sendState(DOOR_ID_CAM2, stateCam2);
}

void setDoor(const String& doorId, int newState) {
  int channel = doorChannelFor(doorId);
  int* state = stateForDoor(doorId);
  if (channel < 0 || state == nullptr) {
    Serial.println("[Door] Unknown door id: " + doorId);
    return;
  }
  *state = newState;
  setServoAngle(channel, newState == 1 ? 90 : 0);
  sendState(doorId.c_str(), newState);
}

// ---------------------------------------------------------------------
// Pan/tilt handler (thay cho loop() đọc Serial ở esp32_pca9685_controller.ino cũ)
// ---------------------------------------------------------------------
void setPanTilt(int pan, int tilt) {
  pan = constrain(pan, 0, 180);
  tilt = constrain(tilt, 0, 180);
  currentPan = pan;
  currentTilt = tilt;
  setServoAngle(PAN_CHANNEL, currentPan);
  setServoAngle(TILT_CHANNEL, currentTilt);
  lastPanTiltCommandTime = millis();
}

// ---------------------------------------------------------------------
// Message dispatch: 1 kết nối WebSocket, 2 loại message ("servo" hoặc "door")
// ---------------------------------------------------------------------
void handleCommand(const String& msg) {
  StaticJsonDocument<128> doc;
  DeserializationError err = deserializeJson(doc, msg);
  if (err) {
    Serial.println("[WS] Bad JSON, ignoring: " + msg);
    return;
  }

  // ----- Pan/tilt (servo_controller.py, không cần ack) -----
  if (doc.containsKey("servo")) {
    const char* servoType = doc["servo"];
    if (strcmp(servoType, "pantilt") == 0 && doc.containsKey("pan") && doc.containsKey("tilt")) {
      setPanTilt(doc["pan"].as<int>(), doc["tilt"].as<int>());
    } else {
      Serial.println("[WS] Unknown/incomplete servo message: " + msg);
    }
    return;
  }

  // ----- Cửa (giữ nguyên logic cũ) -----
  const char* doorId = doc["door"];
  const char* cmd = doc["cmd"];
  if (doorId == nullptr || cmd == nullptr) {
    Serial.println("[WS] Message missing 'door'/'cmd' (and not a servo message): " + msg);
    return;
  }

  String door = String(doorId);
  if (strcmp(cmd, "OPEN") == 0) {
    setDoor(door, 1);
    Serial.println("[Door " + door + "] -> OPEN");
  } else if (strcmp(cmd, "CLOSE") == 0) {
    setDoor(door, 0);
    Serial.println("[Door " + door + "] -> CLOSE");
  } else {
    Serial.println("[WS] Unknown cmd: " + String(cmd));
  }
}

// ---------------------------------------------------------------------
// WebSocket event callback
// ---------------------------------------------------------------------
void onWsEvent(WStype_t type, uint8_t* payload, size_t length) {
  switch (type) {
    case WStype_DISCONNECTED:
      Serial.println("[WS] Disconnected from server");
      break;

    case WStype_CONNECTED:
      Serial.println("[WS] Connected to server");
      sendAllStates();  // announce BOTH doors' current state immediately
      break;

    case WStype_TEXT: {
      String msg = String((char*)payload, length);
      handleCommand(msg);
      break;
    }

    case WStype_ERROR:
      Serial.println("[WS] Error event");
      break;

    default:
      break;
  }
}

// ---------------------------------------------------------------------
// Setup / loop
// ---------------------------------------------------------------------
void setup() {
  Serial.begin(115200);

  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected. IP: " + WiFi.localIP().toString());

  Wire.begin(21, 22);  // SDA=21, SCL=22 (chân I2C mặc định của ESP32 Dev Module)
  pwm.begin();
  pwm.setPWMFreq(50);  // servo analog chuẩn hoạt động ở 50Hz
  delay(200);

  setServoAngle(PAN_CHANNEL, currentPan);
  setServoAngle(TILT_CHANNEL, currentTilt);
  lastPanTiltCommandTime = millis();

  setServoAngle(DOOR_CAM1_CHANNEL, 0);  // start CLOSED
  setServoAngle(DOOR_CAM2_CHANNEL, 0);
  stateCam1 = 0;
  stateCam2 = 0;

  webSocket.begin(ws_host, ws_port, ws_path);
  webSocket.onEvent(onWsEvent);
  webSocket.setReconnectInterval(3000);  // auto-retry every 3s if the link drops
}

void loop() {
  webSocket.loop();

  // Cảnh báo (chỉ log, không tự hành động) nếu pan/tilt mất giao tiếp quá lâu
  static unsigned long lastWarn = 0;
  if (millis() - lastPanTiltCommandTime > SAFETY_TIMEOUT_MS
      && millis() - lastWarn > SAFETY_TIMEOUT_MS) {
    Serial.println("WARN: Pan/tilt khong nhan duoc lenh moi.");
    lastWarn = millis();
  }
}
