from ultralytics import YOLO
import cv2
import cvzone
import math
#just for testing the models on still frames/images
class ImageReaderWithDetection:
    def __init__(self):
        self.model = YOLO("runs/detect/train14/weights/best.pt")
        self.class_names = ['Basketball', 'Basketball Hoop', 'Defence', 'Player', 'shooting']

        self.image_path = "testset/shoot.jpg"
        self.image = cv2.imread(self.image_path)

        self.detect_objects()

    def detect_objects(self):
        results = self.model(self.image, stream=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                current_class = self.class_names[cls]

                if current_class == "shooting" and conf > 0:
                    cvzone.cornerRect(self.image, (x1, y1, w, h), l=10)
                    cv2.putText(
                        self.image,
                        f"Shooting ({conf:.2f})",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2
                    )
                if current_class == "Basketball" and conf > 0.2:
                    cvzone.cornerRect(self.image, (x1, y1, w, h), l=10)
                    cv2.putText(
                        self.image,
                        f"Basketball ({conf:.2f})",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2
                    )
                if current_class == "Defence" and conf > 0.1:
                    cvzone.cornerRect(self.image, (x1, y1, w, h), l=10)
                    cv2.putText(
                        self.image,
                        f"Defence ({conf:.2f})",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2
                    )
                if current_class == "Defence" and conf > 0.1:
                    cvzone.cornerRect(self.image, (x1, y1, w, h), l=10)
                    cv2.putText(
                        self.image,
                        f"Shooting ({conf:.2f})",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2
                    )



        cv2.imshow("Detection Viewer", self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    ImageReaderWithDetection()
