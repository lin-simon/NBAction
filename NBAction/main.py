from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO('Yolo-Weights/yolov8n.pt')
    results = model.train(data=r'NBAction/nbaction-cps843.v2i.yolov8/data.yaml', epochs=100, imgsz=640, workers=8)
