Work in progress..., welp will try to finish by tuesday ;-;
This branch will not be merged since its just code to flash into ESP32
This is not final code, since i need to integrate this code with Zhen Ye's one that he flashed on his ESP32


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

