from ultralytics import YOLO
import cv2
import cvzone
import math

class ImageReaderWithDetection:
    def __init__(self):
        # Load the YOLO model
        self.model = YOLO("runs/detect/train6/weights/best.pt")
        self.class_names = ['Basketball', 'Basketball Hoop', 'shooting']

        # Load the image
        self.image_path = "testset/shoot.jpg"
        self.image = cv2.imread(self.image_path)

        if self.image is None:
            print("Error: Image not found or could not be loaded.")
            return

        # Run detection on the image
        self.detect_objects()

    def detect_objects(self):
        # Perform object detection
        results = self.model(self.image, stream=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                # Get confidence and class label
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                current_class = self.class_names[cls]

                # Draw bounding box and label for "shooting" class
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
                        f"Shooting ({conf:.2f})",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2
                    )

        # Display the image with detections
        cv2.imshow("Detection Viewer", self.image)
        cv2.waitKey(0)  # Wait for a key press
        cv2.destroyAllWindows()  # Close the window

if __name__ == "__main__":
    ImageReaderWithDetection()