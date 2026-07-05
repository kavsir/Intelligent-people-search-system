#include <ESP32Servo.h>
#include <WiFi.h>
#include <WebSocketsClient.h>
#include <ArduinoJson.h>

// ---- WiFi ----
const char* ssid     = "seele";
const char* password = "0123456789.";

// ---- Server running app_dashboard.py / door_ws_server.py ----
const char* ws_host = "10.153.15.207";   // <-- SET THIS to your server's LAN IP
const uint16_t ws_port = 8765;         // must match config.DOOR_WS_PORT
const char* ws_path = "/";

// ---- Servos: one pin per door, must match config.CAMERAS ids below ----
#define DOOR_CAM1_PIN 14   // Phòng 2 door servo
#define DOOR_CAM2_PIN 13   // Phòng 1 door servo

const char* DOOR_ID_CAM1 = "cam1";
const char* DOOR_ID_CAM2 = "cam2";

Servo servoCam1;
Servo servoCam2;
int stateCam1 = 0;  // 0 = CLOSED, 1 = OPEN
int stateCam2 = 0;

WebSocketsClient webSocket;

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------
Servo* servoForDoor(const String& doorId) {
  if (doorId == DOOR_ID_CAM1) return &servoCam1;
  if (doorId == DOOR_ID_CAM2) return &servoCam2;
  return nullptr;
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
  Servo* servo = servoForDoor(doorId);
  int* state = stateForDoor(doorId);
  if (servo == nullptr || state == nullptr) {
    Serial.println("[Door] Unknown door id: " + doorId);
    return;
  }
  *state = newState;
  servo->write(newState == 1 ? 90 : 0);
  sendState(doorId.c_str(), newState);
}

void handleCommand(const String& msg) {
  StaticJsonDocument<128> doc;
  DeserializationError err = deserializeJson(doc, msg);
  if (err) {
    Serial.println("[WS] Bad JSON, ignoring: " + msg);
    return;
  }

  const char* doorId = doc["door"];
  const char* cmd = doc["cmd"];
  if (doorId == nullptr || cmd == nullptr) {
    Serial.println("[WS] Message missing 'door' or 'cmd': " + msg);
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
      Serial.println("[WS] Received: " + msg);
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

  servoCam1.attach(DOOR_CAM1_PIN, 500, 2400);
  servoCam2.attach(DOOR_CAM2_PIN, 500, 2400);
  servoCam1.write(0);  // start CLOSED
  servoCam2.write(0);
  stateCam1 = 0;
  stateCam2 = 0;

  webSocket.begin(ws_host, ws_port, ws_path);
  webSocket.onEvent(onWsEvent);
  webSocket.setReconnectInterval(3000);  // auto-retry every 3s if the link drops
}

void loop() {
  webSocket.loop();
}
