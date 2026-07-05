Work in progress..., welp will try to finish by tuesday ;-;
This branch will not be merged since its just code to flash into ESP32.
This is not final code, since i need to integrate this code that communicates with the webapp with Zhen Ye's one that he flashed on his ESP32.


Circuit diagram from Neo Zhen Ye:
<img width="782" height="597" alt="Screenshot 2026-07-04 215818" src="https://github.com/user-attachments/assets/d3e4f76f-cc14-43b9-a8d7-ff14aaf81368" />


ESP32 Firmware code will be based on this circuit diagram

## Two firmware files — which to use when
### navassist_test_firmware — flash this on your bare ESP32-CAM (the one without camera, IMU or haptic motors set up):

- No camera, IMU, or motors needed
- Sends fake heartbeats, fake IMU, fake frames
- Confirms WiFi → AWS server → phone browser pipeline works end to end
- You'll see "Glasses connected successfully" on your phone when it connects

### navassist_full_firmware — flash this when you meet your team:

- Requires camera (OV5640), MPU-6050, and motors wired up
- Sends real JPEG frames to Roboflow via ai.js
- Reads real gyroscope data for turn detection
- Fires actual left/right motors based on haptic commands from server
- Startup confirmation buzz on both motors when WebSocket connects

## Notes on circuit diagram:
### For the ULN2003A

- pins connected: Pin 1, 2, 8, 9, 16.
- Pin 1 is connected to GPIO 12 pin of ESP32 CAM MODULE OV5640
- Pin 2 is connected to GPIO 13 pin of ESP32 CAM MODULE OV5640
- Pin 8 is connected to GND pin of ESP32 CAM MODULE OV5640
- pin 16 is connected to haptic motor 1 to negative side
- pin 15 is connected to haptic motor 2 on negative side
- pin 9 is connected to the 3.3V pin on ESP32 CAM MODULE 0V5640 which I think is also connected to the flipswitch that can control powersupply from the lipo battery.
- haptic motor 1 and 2 positive sides are connected as one connection end to end.

### For MPU 6050
- SDA pin connected to GPIO 14 of ESP32 CAM module
- SCL pin connected to GPIO 15 of ESP32 CAM module
