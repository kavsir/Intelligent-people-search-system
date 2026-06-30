#include "esp_camera.h"
#include <WiFi.h>
#include <ESP32Servo.h>

// Thông tin WiFi của bạn
const char* ssid = "WIFI SINH VIEN"; 
const char* password = "";

// Chân điều khiển servo pan/tilt
#define SERVO_PAN_PIN   2
#define SERVO_TILT_PIN  41

Servo servoPan;
Servo servoTilt;

int currentPan = 90;
int currentTilt = 90;

// Cấu hình chân Camera cho hầu hết board ESP32-S3 CAM phổ biến (Định dạng CAMERA_MODEL_ESP32S3_EYE)
#define PWDN_GPIO_NUM    -1
#define RESET_GPIO_NUM   -1
#define XCLK_GPIO_NUM    15
#define SIOD_GPIO_NUM    4
#define SIOC_GPIO_NUM    5
#define Y9_GPIO_NUM      16
#define Y8_GPIO_NUM      17
#define Y7_GPIO_NUM      18
#define Y6_GPIO_NUM      12
#define Y5_GPIO_NUM      10
#define Y4_GPIO_NUM      8
#define Y3_GPIO_NUM      9
#define Y2_GPIO_NUM      11
#define VSYNC_GPIO_NUM   6
#define HREF_GPIO_NUM    7
#define PCLK_GPIO_NUM    13

WiFiServer server(80);

// Đọc lệnh servo đang chờ trên Serial (nếu có) và áp dụng ngay.
// Định dạng lệnh từ Python gửi sang: "P:<pan>,T:<tilt>\n"
// Sau khi áp dụng, ESP32 echo lại góc thực tế đã ghi vào servo theo dạng
// "A:<pan>,<tilt>\n" để Python đo được latency thật (command -> actuation),
// thay vì chỉ đo thời gian gửi lệnh.
void handleServoSerial() {
  if (Serial.available() <= 0) {
    return;
  }

  String data = Serial.readStringUntil('\n');

  int pIdx = data.indexOf("P:");
  int tIdx = data.indexOf("T:");

  if (pIdx < 0 || tIdx < 0) {
    return;
  }

  int pVal = data.substring(pIdx + 2, data.indexOf(",", pIdx)).toInt();
  int tVal = data.substring(tIdx + 2).toInt();

  if (pVal >= 10 && pVal <= 170) {
    currentPan = pVal;
    servoPan.write(currentPan);
  }
  if (tVal >= 10 && tVal <= 170) {
    currentTilt = tVal;
    servoTilt.write(currentTilt);
  }

  // Echo lại góc vừa áp dụng để Python tính round-trip latency.
  Serial.print("A:");
  Serial.print(currentPan);
  Serial.print(",");
  Serial.println(currentTilt);
}

// Hàm xử lý truyền luồng video (MJPEG Stream), đồng thời tranh thủ đọc
// lệnh servo trên Serial mỗi vòng lặp gửi frame.
void handleStream(WiFiClient& client) {
  client.println("HTTP/1.1 200 OK");
  client.println("Content-Type: multipart/x-mixed-replace; boundary=frame");
  client.println();

  while (client.connected()) {

    // --- ĐỌC LỆNH SERVO TỪ CỔNG SERIAL NGAY TRONG VÒNG LẶP STREAM ---
    handleServoSerial();

    // --- TIẾP TỤC TRUYỀN HÌNH ẢNH CAMERA ---
    camera_fb_t * fb = esp_camera_fb_get();
    if (!fb) {
      delay(10);
      continue;
    }

    client.print("--frame\r\n");
    client.print("Content-Type: image/jpeg\r\n");
    client.print("Content-Length: " + String(fb->len) + "\r\n\r\n");

    client.write(fb->buf, fb->len);
    client.print("\r\n");

    esp_camera_fb_return(fb);
    delay(1);
  }
}

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println();

  // Khởi tạo servo pan/tilt
  servoPan.setPeriodHertz(50);
  servoTilt.setPeriodHertz(50);

  servoPan.attach(SERVO_PAN_PIN, 500, 2400);
  servoTilt.attach(SERVO_TILT_PIN, 500, 2400);

  servoPan.write(currentPan);
  servoTilt.write(currentTilt);

  // Khởi tạo cấu hình các chân phần cứng Camera
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  if (psramFound()) {
    config.frame_size = FRAMESIZE_VGA;
    config.jpeg_quality = 10;
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_QVGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }

  // Khởi tạo Camera
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Khởi tạo Camera thất bại: 0x%x\n", err);
    return;
  }

  // Bắt đầu kết nối WiFi
  WiFi.begin(ssid, password);

  Serial.print("Đang kết nối WiFi");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nWiFi đã kết nối thành công!");

  // Khởi động HTTP Server
  server.begin();

  Serial.print("Truy cập link stream tại: http://");
  Serial.print(WiFi.localIP());
  Serial.println("/");
}

void loop() {

  // Vẫn đọc lệnh servo qua Serial ngay cả khi chưa có client nào kết nối
  // stream (ví dụ lúc mới mở app Python, trước khi cv2.VideoCapture() kết
  // nối tới /stream) -- nếu không, lệnh "ép về 90,90 khi khởi động" trong
  // servo_controller.py sẽ không bao giờ được ESP32 xử lý.
  handleServoSerial();

  WiFiClient client = server.available();

  if (client) {

    String currentLine = "";
    String requestPath = "";

    while (client.connected()) {

      if (client.available()) {

        char c = client.read();

        if (c == '\n') {

          if (currentLine.length() == 0) {

            if (requestPath.indexOf("GET /stream") >= 0) {

              handleStream(client);

            }
            else {

              client.println("HTTP/1.1 200 OK");
              client.println("Content-Type: text/html");
              client.println();

              client.println("<html><head><title>ESP32-S3 Camera</title></head>");
              client.println("<body style='text-align:center; background:#222; color:white; font-family:sans-serif;'>");
              client.println("<h1>ESP32-S3 Live Video Stream</h1>");
              client.println("<img src='/stream' style='border:5px solid #555; border-radius:8px;'/>");
              client.println("</body></html>");
            }

            break;

          } else {

            if (currentLine.startsWith("GET ")) {
              requestPath = currentLine;
            }

            currentLine = "";
          }

        } else if (c != '\r') {

          currentLine += c;
        }
      }
    }

    client.stop();
  }
}
