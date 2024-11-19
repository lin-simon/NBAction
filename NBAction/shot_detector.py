from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
from utils import score, detect_down, detect_up, in_hoop_region, clean_hoop_pos, clean_ball_pos

class ShotDetector:
    def __init__(self):
        self.model = YOLO("runs/detect/train14/weights/best.pt")
        self.class_names = ['Basketball', 'Basketball Hoop', 'Defence', 'Player', 'shooting']

        #self.cap = cv2.VideoCapture(0) -- for live capture of games
        self.cap = cv2.VideoCapture("testset/test.mov") # 2 & 3 5
        #self.cap = cv2.VideoCapture("testset/glaze.mp4") 
        self.ball_pos = []
        self.hoop_pos = []  
        self.frame_count = 0
        self.frame = None
        self.frame2 = None
        self.makes = 0
        self.attempts = 0

        self.up = False
        self.down = False   
        self.up_frame = 0
        self.down_frame = 0

        self.fade_frames = 0
        self.fade_counter = 0
        self.overlay_color = (0, 0, 0)

        self.show_score_text = False
        self.score_text_frame_count = 0
        self.score_text_duration = 200
        self.font = cv2.FONT_HERSHEY_DUPLEX
        cv2.namedWindow('NBAction', cv2.NORM_MINMAX)
        self.run()

    def run(self):
        while True:
            ret, self.frame = self.cap.read()

            if not ret:
                break
            #if self.frame_count % 2 == 0: potential performance boost maybe 
            results = self.model(self.frame, stream=True)
            self.display_scoring_radius()
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1

                    conf = math.ceil((box.conf[0] * 100)) / 100 #adj as needed
                    cls = int(box.cls[0])
                    current_class = self.class_names[cls]
                    print(f"Detected Class: {current_class}, Confidence: {conf:.2f}")
                    center = (int(x1 + w / 2), int(y1 + h / 2))

                    if (conf > 0.4 or (in_hoop_region(center, self.hoop_pos) and conf > 0.15)) and current_class == "Basketball":
                        self.ball_pos.append((center, self.frame_count, w, h, conf))
                        x2, y2 = x1 + w, y1 + h
                        cv2.rectangle(self.frame, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
                        cv2.putText(self.frame, f"Basketball ({conf:.2f})", (x1, y1 - 10), self.font, 0.6, (0, 0, 255), 1, lineType=cv2.LINE_AA)

                    if conf > 0.5 and current_class == "Basketball Hoop":
                        self.hoop_pos.append((center, self.frame_count, w, h, conf))
                        x2, y2 = x1 + w, y1 + h
                        cv2.rectangle(self.frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
                        cv2.putText(self.frame, f"Basketball Hoop ({conf:.2f})", (x1, y1 - 10), self.font, 0.6, (0, 255, 0), 1, lineType=cv2.LINE_AA)
                        
                    if conf > 0.2 and current_class == "Defence":
                        x2, y2 = x1 + w, y1 + h
                        cv2.rectangle(self.frame, (x1, y1), (x2, y2), (0, 0, 0), thickness=2)
                        cv2.putText(self.frame, f"Defence ({conf:.2f})", (x1, y1 - 10), self.font, 0.6, (0, 0, 0), 1, lineType=cv2.LINE_AA)
                        
                    if conf > 0.1 and current_class == "Player":
                        x2, y2 = x1 + w, y1 + h
                        cv2.rectangle(self.frame, (x1, y1), (x2, y2), (0, 152, 248), thickness=2)
                        cv2.putText(self.frame, f"Player ({conf:.2f})", (x1, y1 - 10), self.font, 0.6, (0, 152, 248), 1, lineType=cv2.LINE_AA)
                        
                    if conf > 0 and current_class == "shooting":
                        self.hoop_pos.append((center, self.frame_count, w, h, conf))
                        cvzone.cornerRect(self.frame, (x1, y1, w, h), colorR=(255,0,0))
                        cv2.putText(self.frame, f"Shooting ({conf:.2f})", (x1, y1 - 10), self.font, 0.6, (255,0,0), 1, lineType=cv2.LINE_AA)
                        
            self.clean_motion()
            self.shot_detection()
            self.display_score()    
            self.frame_count += 1

            cv2.imshow('NBAction', self.frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
    def display_scoring_radius(self):
        if len(self.hoop_pos) > 0:
            # Get the latest hoop position
            hoop_center = self.hoop_pos[-1][0]
            hoop_radius = int(0.4 * self.hoop_pos[-1][2])  # Adjust as per your scoring logic
            # Draw the scoring radius
            cv2.circle(self.frame, hoop_center, hoop_radius, (255, 255, 0), thickness=3)  # Yellow circle
    def clean_motion(self):
        try:
            self.ball_pos = clean_ball_pos(self.ball_pos, self.frame_count)
            for i in range(len(self.ball_pos)):
                cv2.circle(self.frame, self.ball_pos[i][0], 2, (0, 0, 255), 2)
            print(f"Ball positions: {self.ball_pos}")  # Debugging line

            if len(self.hoop_pos) > 1:
                self.hoop_pos = clean_hoop_pos(self.hoop_pos)
                cv2.circle(self.frame, self.hoop_pos[-1][0], 2, (128, 128, 0), 2)
            print(f"Hoop positions: {self.hoop_pos}")  # Debugging line
                
        except IndexError as e:
            print(f"IndexError encountered in clean_motion: {e}")
        except Exception as e:
            print(f"Unexpected error in clean_motion: {e}")
            
    def shot_detection(self):
        if len(self.hoop_pos) > 0 and len(self.ball_pos) > 0:
            # Detect upward motion
            if not self.up:
                self.up = detect_up(self.ball_pos, self.hoop_pos)
                if self.up:
                    self.up_frame = self.ball_pos[-1][1]  # Record frame when motion starts upward

            # Detect downward motion
            if self.up and not self.down:
                self.down = detect_down(self.ball_pos, self.hoop_pos)
                if self.down:
                    self.down_frame = self.ball_pos[-1][1]  # Record frame when motion starts downward

            # Check if a shot attempt was made
            if self.up and self.down and self.up_frame < self.down_frame:
                self.attempts += 1
                self.up = False
                self.down = False

                if score(self):
                    self.makes += 1
                    self.overlay_color = (0, 255, 0)  # Green for success
                    self.fade_counter = self.fade_frames
                    self.show_score_text = True
                    self.score_text_frame_count = self.score_text_duration
                else:
                    self.display_text("Miss!")
                    self.overlay_color = (0, 0, 255)  
                    self.fade_counter = self.fade_frames
                    
    def display_score(self):
        text = str(self.makes) + " / " + str(self.attempts)
        frame_height, frame_width, _ = self.frame.shape

        text_size = cv2.getTextSize(text, self.font, 2, 5)[0] 
        text_x = (frame_width - text_size[0]) // 2 
        text_y = frame_height - 30 

        cv2.putText(self.frame, text, (text_x, text_y), self.font, 2, (0, 200, 252), 9, lineType=cv2.LINE_AA)  
        cv2.putText(self.frame, text, (text_x, text_y), self.font, 2, (255, 0, 0), 4, lineType=cv2.LINE_AA)  

        if self.show_score_text:
            self.display_text("Score!")
            self.score_text_frame_count -= 1
            if self.score_text_frame_count <= 0:
                self.show_score_text = False

        if self.fade_counter > 0:
            alpha = 0.2 * (self.fade_counter / self.fade_frames)
            self.frame = cv2.addWeighted(self.frame, 1 - alpha, np.full_like(self.frame, self.overlay_color), alpha, 0)
            self.fade_counter -= 1

    def display_text(self, text):
        cv2.putText(self.frame, text, (320, 100), self.font, 2, (0,255,0), 6)

if __name__ == "__main__":
    ShotDetector()
