#include "esp_camera.h"
#include <WiFi.h>

//===================== WIFI =====================
const char* ssid = "TOTO";
const char* password = "O123456789";

//================= AI Thinker ESP32-CAM =================
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0

#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27

#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5

#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// Flash LED
#define LED_FLASH 4

WiFiServer server(80);

//=========================================================
// MJPEG STREAM
//=========================================================
void handleStream(WiFiClient &client)
{
  client.println("HTTP/1.1 200 OK");
  client.println("Content-Type: multipart/x-mixed-replace; boundary=frame");
  client.println();

  while (client.connected())
  {
    camera_fb_t *fb = esp_camera_fb_get();

    if (!fb)
    {
      Serial.println("Camera capture failed");
      delay(30);
      continue;
    }

    client.print("--frame\r\n");
    client.print("Content-Type: image/jpeg\r\n");
    client.print("Content-Length: ");
    client.print(fb->len);
    client.print("\r\n\r\n");

    client.write(fb->buf, fb->len);
    client.print("\r\n");

    esp_camera_fb_return(fb);

    delay(1);
  }
}

//=========================================================
// SETUP
//=========================================================
void setup()
{
  Serial.begin(115200);
  Serial.setDebugOutput(true);

  pinMode(LED_FLASH, OUTPUT);
  digitalWrite(LED_FLASH, LOW);

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

#if ESP_ARDUINO_VERSION_MAJOR >= 3
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
#else
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
#endif

  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;

  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  if (psramFound())
  {
    config.frame_size = FRAMESIZE_VGA;   // 640x480
    config.jpeg_quality = 12;
    config.fb_count = 2;
    config.grab_mode = CAMERA_GRAB_LATEST;
  }
  else
  {
    config.frame_size = FRAMESIZE_QVGA;  // 320x240
    config.jpeg_quality = 15;
    config.fb_count = 1;
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  }

  esp_err_t err = esp_camera_init(&config);

  if (err != ESP_OK)
  {
    Serial.printf("Camera Init Failed 0x%x\n", err);
    while (true)
      delay(1000);
  }

  sensor_t *s = esp_camera_sensor_get();

  s->set_brightness(s, 0);
  s->set_contrast(s, 0);
  s->set_saturation(s, 0);
  s->set_framesize(s, psramFound() ? FRAMESIZE_VGA : FRAMESIZE_QVGA);

  Serial.println();
  Serial.println("Connecting WiFi...");

  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED)
  {
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.println("WiFi Connected.");

  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());

  server.begin();

  Serial.println();
  Serial.println("================================");
  Serial.print("Open Browser: http://");
  Serial.println(WiFi.localIP());
  Serial.println("================================");
}

//=========================================================
// LOOP
//=========================================================
void loop()
{
  WiFiClient client = server.available();

  if (client)
  {
    String currentLine = "";
    String requestPath = "";

    while (client.connected())
    {
      if (client.available())
      {
        char c = client.read();

        if (c == '\n')
        {
          if (currentLine.length() == 0)
          {
            if (requestPath.indexOf("GET /stream") >= 0)
            {
              handleStream(client);
            }
            else
            {
              client.println("HTTP/1.1 200 OK");
              client.println("Content-Type: text/html");
              client.println();

              client.println("<!DOCTYPE html>");
              client.println("<html>");
              client.println("<head>");
              client.println("<meta charset='UTF-8'>");
              client.println("<title>ESP32-CAM Live Stream</title>");
              client.println("</head>");

              client.println("<body style='background:#111;color:white;text-align:center;font-family:Arial;'>");

              client.println("<h1>ESP32-CAM Live Stream</h1>");

              client.println("<img src='/stream' style='width:80%;max-width:720px;border-radius:10px;border:5px solid #666;'>");

              client.println("</body>");
              client.println("</html>");
            }

            break;
          }
          else
          {
            if (currentLine.startsWith("GET "))
            {
              requestPath = currentLine;
            }

            currentLine = "";
          }
        }
        else if (c != '\r')
        {
          currentLine += c;
        }
      }
    }

    client.stop();
  }
}