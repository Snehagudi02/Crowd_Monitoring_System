import os
from ultralytics import YOLO


class PersonDetector:
    def __init__(self, model_path='models/cv_weights/yolov8m.pt'):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model = YOLO(model_path if os.path.exists(model_path) else 'yolov8m.pt')
        if not os.path.exists(model_path) and os.path.exists('yolov8m.pt'):
            import shutil
            shutil.move('yolov8m.pt', model_path)
            self.model = YOLO(model_path)

    def detect(self, image):
        results = self.model(
            image,
            classes=[0],       # person class only
            imgsz=1024,
            conf=0.15,
            verbose=False
        )
        boxes = []
        if len(results) > 0:
            for box in results[0].boxes.xyxy.cpu().numpy().tolist():
                boxes.append([float(box[0]), float(box[1]), float(box[2]), float(box[3])])
        return boxes
