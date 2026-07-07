# Model directory

Place the exported detection model here as:

    pedestrian.onnx

It is a YOLO11 model (classes: red, green, traffic-light) exported to ONNX
(input 1x3x640x640, opset 12) and loaded in the browser by `../js/ai.js`
via onnxruntime-web.

Export it from the training notebook with:
    YOLO('ped_lights_yolo11s_esp_best.pt').export(format='onnx', imgsz=640, opset=12, simplify=True)
