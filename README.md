Work in progress..., welp will try to finish by tuesday ;-;
This branch will not be merged since its just code to flash into ESP32.
This is not final code, since i need to integrate this code that communicates with the webapp with Zhen Ye's one that he flashed on his ESP32.


Circuit diagram from Neo Zhen Ye:


<img width="578" height="551" alt="image" src="https://github.com/user-attachments/assets/76a95467-8b8a-42d4-94d2-8d7de51ca659" />



ESP32 Firmware code will be based on this circuit diagram


## Notes on circuit diagram:
### For the ULN2003A

- pins connected: Pin 1, 2, 8, 16.
- Pin 1 is connected to GPIO 1 pin of ESP32 S3 OV5640
- Pin 2 is connected to GPIO 2 pin of ESP32 S3 OV5640
- Pin 8 is connected to the negative terminal of lipo battery
- pin 16 is connected to haptic motor 1 to negative side
- pin 15 is connected to haptic motor 2 on negative side

### For MPU 6050
- SDA pin connected to GPIO 5 of ESP32 S3 OV5640
- SCL pin connected to GPIO 4 of ESP32 S3 OV5640
