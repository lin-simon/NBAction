from ultralytics import YOLO
import numpy as np
import cv2
import math

from processing import score, within_shot_radius, stabilize_hoop, stabilize_ball

class NBAction:
    def __init__(self):

        self.model = YOLO("runs/detect/train14/weights/best.pt")
        self.classes = ['Basketball', 'Basketball Hoop', 'Defence', 'Player', 'shooting']
        self.video = cv2.VideoCapture("testset/trim5.mp4")
       
       
        self.current_frame = None
        self.cooldown_current_frames = 500
        self.last_attempt_current_frame = -self.cooldown_current_frames
         
        #(X,Y) Coordinates for current ball and hoop centers
        self.ball = []  
        self.hoop = []  

        #Current arc of ball
        self.up = False
        self.down = False
        self.ball_in_hoop = False 
        
        self.show_score_text = False
        self.frame_count = 0
        self.total = 0
        self.shots_made = 0  
        self.up_current_frame = 0
        self.down_current_frame = 0
        self.revert_frames = 20

        self.overlay_color = (0, 0, 0)
        self.font = cv2.FONT_HERSHEY_DUPLEX
        #Declare window and allow to be resized
        cv2.namedWindow('NBAction', cv2.NORM_MINMAX)
        self.run()

    def run(self):
        while True:
            ret, self.current_frame = self.video.read()
            if not ret:
                break
            results = self.model(self.current_frame, stream=True)

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    center = (int(x1 + w / 2), int(y1 + h / 2))
                    confidence = math.ceil((box.conf[0] * 100)) / 100
                    cclass = self.classes[int(box.cls[0])]

                
                

                    if (confidence > 0.2 or (within_shot_radius(center, self.hoop) and confidence > 0.1)) and cclass == "Basketball":
                        self.ball.append((center, self.total, w, h, confidence))
                        x2, y2 = x1 + w, y1 + h
                        cv2.rectangle(self.current_frame, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
                        cv2.putText(self.current_frame, f"Basketball ({confidence:.2f})", (x1, y1 - 10), self.font, 0.6, (0, 0, 255), 1, lineType=cv2.LINE_AA)

                    if confidence > 0.5 and cclass == "Basketball Hoop":
                        self.hoop.append((center, self.total, w, h, confidence))
                        x2, y2 = x1 + w, y1 + h
                        cv2.rectangle(self.current_frame, (x1, y1), (x2, y2), (255, 55, 174), thickness=2)
                        cv2.putText(self.current_frame, f"Basketball Hoop ({confidence:.2f})", (x1, y1 - 10), self.font, 0.6, (0, 255, 0), 1, lineType=cv2.LINE_AA)
                        
                    if confidence > 0.5 and cclass == "Defence":
                        x2, y2 = x1 + w, y1 + h
                        cv2.rectangle(self.current_frame, (x1, y1), (x2, y2), (0, 0, 0), thickness=2)
                        cv2.putText(self.current_frame, f"Defence ({confidence:.2f})", (x1, y1 - 10), self.font, 0.6, (0, 0, 0), 1, lineType=cv2.LINE_AA)
                        
                    if confidence > 0.2 and cclass == "Player":
                        x2, y2 = x1 + w, y1 + h
                        cv2.rectangle(self.current_frame, (x1, y1), (x2, y2), (0, 152, 248), thickness=2)
                        cv2.putText(self.current_frame, f"Player ({confidence:.2f})", (x1, y1 - 10), self.font, 0.6, (0, 152, 248), 1, lineType=cv2.LINE_AA)
                        
                    if confidence > 0 and cclass == "shooting":
                        self.hoop.append((center, self.total, w, h, confidence))
                        cv2.rectangle(self.current_frame, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)
                        cv2.putText(self.current_frame, f"Shooting ({confidence:.2f})", (x1, y1 - 10), self.font, 0.6, (255,0,0), 1, lineType=cv2.LINE_AA)
            
            #Run state functions every current_frame
            self.update_state()
            #Increment frame
            self.total += 1

            cv2.imshow('NBAction', self.current_frame)
            #Adjust how many current_frames program halts for, Q to exit window.
            if cv2.waitKey(1) & 0xFF == ord('q'):  
                break

        self.video.release()
        cv2.destroyAllWindows()


    #Keep track of most recent ball and hoop locations,
    def stabilize(self):
        self.ball = stabilize_ball(self.ball, self.total)
        for i in range(len(self.ball)):
            cv2.circle(self.current_frame, self.ball[i][0], 2, (0, 0, 255), 2) 

        if len(self.hoop) > 0:
            self.hoop = stabilize_hoop(self.hoop)
            hoop_center = self.hoop[-1][0]
            
            hoop_width = self.hoop[-1][2]
            hoop_height = self.hoop[-1][3]
            
            hoop_radius = int((hoop_width + hoop_height) / 4)  

            max_hoop_radius = 50
            
            hoop_radius = min(hoop_radius, max_hoop_radius)

            cv2.circle(self.current_frame, hoop_center, hoop_radius, (0, 255, 0), 2) 

    def check_score(self):
        if len(self.hoop) > 0 and len(self.ball) > 0:
            if self.total - self.last_attempt_current_frame >= self.cooldown_current_frames:
                if score(self) and not self.ball_in_hoop:
                    self.shots_made += 1  
                    self.overlay_color = (0, 255, 0)
                    self.frame_count = self.revert_frames

                    self.ball_in_hoop = True
                    self.last_attempt_current_frame = self.total  

                    self.show_score_text = True
                    self.score_text_total = 250 

                elif not score(self):
                    self.ball_in_hoop = False
                        
    def display_score(self):
        text = "Shots Scored: " + str(self.shots_made)
        current_frame_height, current_frame_width, _ = self.current_frame.shape
        
        text_size = cv2.getTextSize(text, self.font, 2, 5)[0] 
        text_x = (current_frame_width - text_size[0]) // 2 
        text_y = current_frame_height - 30 

        cv2.putText(self.current_frame, text, (text_x, text_y), self.font, 2, (0, 200, 252), 9, lineType=cv2.LINE_AA)  
        cv2.putText(self.current_frame, text, (text_x, text_y), self.font, 2, (255, 0, 0), 4, lineType=cv2.LINE_AA)  

        if self.show_score_text:
            self.display_text("Score!")
            self.score_text_total -= 1
            if self.score_text_total <= 0:
                self.show_score_text = False  

        if self.frame_count > 0:
            alpha = 0.2 * (self.frame_count / self.revert_frames)
            self.current_frame = cv2.addWeighted(self.current_frame, 1 - alpha, np.full_like(self.current_frame, self.overlay_color), alpha, 0)
            self.frame_count -= 1

    def update_state(self):
        self.stabilize()
        self.check_score()
        self.display_score()
        
    def display_text(self, text):
        current_frame_height, current_frame_width, _ = self.current_frame.shape

        text_size = cv2.getTextSize(text, self.font, 2, 6)[0]
        text_width, text_height = text_size

        text_x = (current_frame_width - text_width) // 2
        text_y = (current_frame_height + text_height) // 2

        cv2.putText(self.current_frame, text, (text_x, text_y-300), self.font, 2, (0, 0, 0), 9, lineType=cv2.LINE_AA)
        cv2.putText(self.current_frame, text, (text_x, text_y-300), self.font, 2, (0, 255, 0), 4, lineType=cv2.LINE_AA)

if __name__ == "__main__":
    NBAction()