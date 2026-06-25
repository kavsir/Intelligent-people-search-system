#include "esp_camera.h"
#include <WiFi.h>

// Thông tin WiFi của bạn
const char* ssid = "TOTO";
const char* password = "O123456789";

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

// Hàm xử lý truyền luồng video (MJPEG Stream)
void handleStream(WiFiClient& client) {
  client.println("HTTP/1.1 200 OK");
  client.println("Content-Type: multipart/x-mixed-replace; boundary=frame");
  client.println();

  while (client.connected()) {
    camera_fb_t * fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Camera capture failed");
      delay(100);
      continue;
    }

    // Gửi boundary và header của khung hình
    client.print("--frame\r\n");
    client.print("Content-Type: image/jpeg\r\n");
    client.print("Content-Length: " + String(fb->len) + "\r\n\r\n");
    
    // Gửi dữ liệu ảnh nhị phân
    client.write(fb->buf, fb->len);
    client.print("\r\n");

    // Trả lại bộ đệm cho camera giải phóng RAM
    esp_camera_fb_return(fb);
    
    // Tạo độ trễ nhỏ để tránh nghẽn luồng và duy trì độ ổn định
    delay(1); 
  }
}

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println();

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

  // Tự động tối ưu độ phân giải dựa trên việc bật PSRAM
  if (psramFound()) {
    config.frame_size = FRAMESIZE_VGA;  // Độ phân giải 640x480 mượt mà
    config.jpeg_quality = 10;           // Chất lượng ảnh (10-63, số càng nhỏ ảnh càng nét)
    config.fb_count = 2;                // Sử dụng cơ chế double-buffer
  } else {
    config.frame_size = FRAMESIZE_QVGA; // Hạ xuống nếu không có hoặc không bật PSRAM
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
  WiFiClient client = server.available();
  if (client) {
    String currentLine = "";
    String requestPath = "";
    while (client.connected()) {
      if (client.available()) {
        char c = client.read();
        if (c == '\n') {
          if (currentLine.length() == 0) {
            // Khi nhận hết HTTP Header, kiểm tra đường dẫn request
            if (requestPath.indexOf("GET /stream") >= 0) {
              handleStream(client);
            } else {
              // Giao diện trang chủ hiển thị khung Stream
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
    client.stop(); // Ngắt kết nối client sau khi xử lý xong (trừ luồng stream sẽ chạy vô hạn)
  }
}