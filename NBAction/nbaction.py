from ultralytics import YOLO
import cv2
import math
import numpy as np
from processing import score, in_hoop_region, clean_hoop_pos, clean_ball_pos


class NBAction:
    def __init__(self):

        self.model = YOLO("runs/detect/train14/weights/best.pt")
        self.class_names = ['Basketball', 'Basketball Hoop', 'Defence', 'Player', 'shooting']
        self.shots_made = 0  
        self.ball_in_hoop = False 
        self.cap = cv2.VideoCapture("testset/TMU.mp4")
        self.cooldown_frames = 500 
        self.last_attempt_frame = -self.cooldown_frames 
        self.ball_pos = []  
        self.hoop_pos = []  

        self.frame_count = 0
        self.frame = None
        
        self.show_score_text = False
        self.up = False
        self.down = False
        self.up_frame = 0
        self.down_frame = 0
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.fade_frames = 20
        self.fade_counter = 0
        self.overlay_color = (0, 0, 0)
        cv2.namedWindow('NBAction', cv2.NORM_MINMAX)
        self.run()

    def run(self):
        while True:
            ret, self.frame = self.cap.read()
            if not ret:
                break

            results = self.model(self.frame, stream=True)

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1

                    conf = math.ceil((box.conf[0] * 100)) / 100

                    cls = int(box.cls[0])
                    current_class = self.class_names[cls]

                    center = (int(x1 + w / 2), int(y1 + h / 2))


                    if (conf > 0.2 or (in_hoop_region(center, self.hoop_pos) and conf > 0.1)) and current_class == "Basketball":
                        self.ball_pos.append((center, self.frame_count, w, h, conf))
                        x2, y2 = x1 + w, y1 + h
                        cv2.rectangle(self.frame, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
                        cv2.putText(self.frame, f"Basketball ({conf:.2f})", (x1, y1 - 10), self.font, 0.6, (0, 0, 255), 1, lineType=cv2.LINE_AA)

                    if conf > 0.5 and current_class == "Basketball Hoop":
                        self.hoop_pos.append((center, self.frame_count, w, h, conf))
                        x2, y2 = x1 + w, y1 + h
                        cv2.rectangle(self.frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
                        cv2.putText(self.frame, f"Basketball Hoop ({conf:.2f})", (x1, y1 - 10), self.font, 0.6, (0, 255, 0), 1, lineType=cv2.LINE_AA)
                        
                    if conf > 0.4 and current_class == "Defence":
                        x2, y2 = x1 + w, y1 + h
                        cv2.rectangle(self.frame, (x1, y1), (x2, y2), (0, 0, 0), thickness=2)
                        cv2.putText(self.frame, f"Defence ({conf:.2f})", (x1, y1 - 10), self.font, 0.6, (0, 0, 0), 1, lineType=cv2.LINE_AA)
                        
                    if conf > 0.2 and current_class == "Player":
                        x2, y2 = x1 + w, y1 + h
                        cv2.rectangle(self.frame, (x1, y1), (x2, y2), (0, 152, 248), thickness=2)
                        cv2.putText(self.frame, f"Player ({conf:.2f})", (x1, y1 - 10), self.font, 0.6, (0, 152, 248), 1, lineType=cv2.LINE_AA)
                        
                    if conf > 0 and current_class == "shooting":
                        self.hoop_pos.append((center, self.frame_count, w, h, conf))
                        cv2.rectangle(self.frame, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)
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

    def clean_motion(self):
        self.ball_pos = clean_ball_pos(self.ball_pos, self.frame_count)
        for i in range(len(self.ball_pos)):
            cv2.circle(self.frame, self.ball_pos[i][0], 2, (0, 0, 255), 2) 

        if len(self.hoop_pos) > 0:
            self.hoop_pos = clean_hoop_pos(self.hoop_pos)
            hoop_center = self.hoop_pos[-1][0]
            
            hoop_width = self.hoop_pos[-1][2]
            hoop_height = self.hoop_pos[-1][3]
            
            hoop_radius = int((hoop_width + hoop_height) / 4)  

            max_hoop_radius = 50
            
            hoop_radius = min(hoop_radius, max_hoop_radius)

            cv2.circle(self.frame, hoop_center, hoop_radius, (128, 128, 0), 2) 

    def shot_detection(self):
        if len(self.hoop_pos) > 0 and len(self.ball_pos) > 0:
            if self.frame_count - self.last_attempt_frame >= self.cooldown_frames:
                if score(self) and not self.ball_in_hoop:
                    self.shots_made += 1  
                    self.overlay_color = (0, 255, 0)
                    self.fade_counter = self.fade_frames

                    self.ball_in_hoop = True

                    self.last_attempt_frame = self.frame_count 
                elif not score(self):
                    self.ball_in_hoop = False
                        
    def display_score(self):
        text = "Shots Scored: " + str(self.shots_made)
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
        frame_height, frame_width, _ = self.frame.shape

        text_size = cv2.getTextSize(text, self.font, 2, 6)[0]
        text_width, text_height = text_size

        text_x = (frame_width - text_width) // 2
        text_y = (frame_height + text_height) // 2
        cv2.putText(self.frame, text, (text_x, text_y-100), self.font, 2, (0, 255, 0), 6, lineType=cv2.LINE_AA)


if __name__ == "__main__":
    NBAction()