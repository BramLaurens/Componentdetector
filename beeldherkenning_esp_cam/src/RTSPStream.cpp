#include <WiFi.h>
#include "RTSPConfig.h" // Include a RTSPConfig.h file if want to change defined options
#include <ESP32-RTSPServer.h>

// Pin map for the AI Thinker ESP32-CAM.
// Camera pins are GPIO 0, 5, 18, 19, 21, 22, 23, 25, 26, 27, 32, 34, 35, 36 and 39.
// Do not connect pull-downs or pull-ups that prevent normal boot.

const char *ssid = "ESP-CAM";
const char *wifiPassword = "esp32cam123";

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


// RTSPServer instance
RTSPServer rtspServer;

// Can set a username and password for RTSP authentication or leave blank for no authentication
const char *rtspUser = "";
const char *rtspPassword = "";

// Set camera quality parameter
int quality = 10;

bool setupCamera() {
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
  config.frame_size = FRAMESIZE_HD;
  config.jpeg_quality = quality;
  config.fb_count = psramFound() ? 2 : 1;
  config.grab_mode = CAMERA_GRAB_LATEST;
  config.fb_location = psramFound() ? CAMERA_FB_IN_PSRAM : CAMERA_FB_IN_DRAM;

  return esp_camera_init(&config) == ESP_OK;
}

void getFrameQuality() { 
  sensor_t * s = esp_camera_sensor_get(); 
  quality = s->status.quality; 
  Serial.printf("Camera Quality is: %d\n", quality);
}

void sendVideo() {
  if (rtspServer.readyToSendFrame()) {
    camera_fb_t* fb = esp_camera_fb_get();
    if (fb != nullptr) {
      rtspServer.sendRTSPFrame(fb->buf, fb->len, quality, fb->width, fb->height);
      esp_camera_fb_return(fb);
    }
  }
}

void setup() {
  Serial.begin(115200);

  WiFi.mode(WIFI_AP);
  if (!WiFi.softAP(ssid, wifiPassword)) {
    Serial.println("Failed to start ESP-CAM access point");
    return;
  }
  Serial.printf("ESP-CAM network started: %s\n", ssid);
  Serial.printf("ESP-CAM IP address: %s\n", WiFi.softAPIP().toString().c_str());

  if (!setupCamera()) {
    Serial.println("Camera setup failed");
    return;
  }

  getFrameQuality(); //Retrieve frame quality

  rtspServer.setCredentials(rtspUser, rtspPassword); // Set RTSP authentication
  rtspServer.maxRTSPClients = 5;

  if (rtspServer.init()) {
   Serial.println("RTSP server started successfully"); 
  } else {
   Serial.println("Failed to start RTSP server"); 
  }

}

void loop() {
  sendVideo();
  delay(1);
}
   