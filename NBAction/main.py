from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO('Yolo-Weights/yolov8n.pt')
    results = model.train(data=r'basketball.v1i.yolov8\data.yaml', epochs=100, imgsz=640)
