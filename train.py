from ultralytics import YOLO

# Load your previously trained model weights instead of starting from scratch
# Note: Verify that 'train8' is your actual latest/best training run folder!
# model = YOLO("runs/detect/train13/weights/best.pt") 


# Load the model configuration (.yaml) to initialize with random weights
# This builds the YOLOv8 Nano architecture from scratch
model = YOLO("yolov8n.yaml") 

# Continue training. 
results = model.train(data="cans.yaml", epochs=20, imgsz=1024)
