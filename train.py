from ultralytics import YOLO

model = YOLO("yolov8n-cls.pt")

results = model.train(data = "Data", epochs = 100, imgsz = 640)