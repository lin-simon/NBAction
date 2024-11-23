from ultralytics import YOLO

#Boilerplate model training code from ultralytics documentation.
#CURRENT ITERATION OF OUR MODEL: v5.
if __name__ == "main":
    model = YOLO('Yolo-Weights/yolov8n.pt')
    """
    Each iteration of NBAction was trained using following hardware:
    GPU: NVIDIA RTX 3080Ti
    CPU: AMD Ryzen 5 5600x Processor (6 cores)
    RAM: 32gb ddr4
    I offloaded the training to my GPU for much, much faster results. When training on CPU alone, training took an upwards of 2-3 hours, versus 10-15 minutes on my 3080Ti, YMMV.
    I followed Ultralytics documentation and stuck with the recommended epoch size of 100 to avoid overfitting, with a scaled image resolution of 640px. -- this was recommended for medium-large size object detection models
    8 workers were allocated for stable training.
    The setup for CUDA and CuDNN compatibility for PyTorch was a complicated process, and I go over specifics further in detail in my IEEE paper. 
    Read the section on experimental setup for my experimentation for accelerating training time on RTX GPUs.
    """
    results = model.train(data=r'NBAction/NBActionV5/data.yaml', epochs=100, imgsz=640, workers=8)