from ultralytics import YOLO
import numpy as np
import cv2

from math import ceil
from processing import in_hoop, within_shot_radius, stabilize_hoop, stabilize_ball

class NBAction:
    def __init__(self):
        #Load our best iteration of NBAction object detection model.
        self.model = YOLO("runs/detect/train9/weights/best.pt")
        #self.model = YOLO("NBAction-main/NBAction/best.pt")
        #Image Classes we trained our model on to detect
        self.classes = ['Basketball', 'Basketball Hoop', 'Defence', 'Player', 'shooting']
        #Load a video to be analyzed. (/testset contains all our test videos, but feel free to upload your own basketball footage and change the path to the video.)
        #Higher resolution is preferred, majority of our test videos are recored in 1080p 60fps for better accuracy, but can get away with 720p 30fps,
        self.video = cv2.VideoCapture("testset/best.mp4") # file path here..
        #Initialize the current frame and total frames (Variables C and T, as defined in our IEEE paper.)
        self.current_frame = None
        self.frame_count = 0
        self.total = 0

        #We set a cooldown between each shot tracking to avoid recording multiple scores from a single successful shot
        self.cooldown_current_frames = 100
        self.last_attempt_current_frame = -self.cooldown_current_frames
        
        #(X,Y) Coordinate locations for current ball and hoop centers
        self.ball = []  
        self.hoop = []  

        #Boolean to check if the ball is inside the scoring circle radius.
        self.ball_in_hoop = False 
        self.show_score_text = False
        self.revert_frames = 20 #Frame time between visual effects
        self.overlay_color = (0, 0, 0) #Visual effect on scores

        #Current arc/trajectory of ball across each frame 
        # (if the ball has positive y value displacements == up, negative y displacement == ball is falling in a downward arc)
        # probably a better mathematical way to detect this without booleans, but we kept it simple in favour of better performance and less complex calculations during runtime.
        self.up = False
        self.up_current_frame = 0
        self.falling = False 
        self.falling_current_frame = 0

        #Total shots made in current video.
        self.shots_made = 0   

        #Text font to be used
        self.font = cv2.FONT_HERSHEY_DUPLEX

        #Declare window and allow to be resized, run the system with declared attributes.
        cv2.namedWindow('NBAction', cv2.NORM_MINMAX)
        self.run()

    def run(self):
        target_width = 1920
        target_height = 1080
        while True:
            ret, self.current_frame = self.video.read()
            if not ret:
                break
            #Resize too small or too large videos to 1080p (lot of data was filmed in 4k)
            current_height, current_width, _ = self.current_frame.shape
            scale_width = target_width / current_width
            scale_height = target_height / current_height
            scale = min(scale_width, scale_height)  
            new_width = int(current_width * scale)
            new_height = int(current_height * scale)
            self.current_frame = cv2.resize(self.current_frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

            #Black background for phone portrait videos
            canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            x_offset = (target_width - new_width) // 2
            y_offset = (target_height - new_height) // 2
            canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = self.current_frame
            self.current_frame = canvas

            detections = self.model(self.current_frame, stream=True)

            #Analyze all potential classes detected by the model
            for detection in detections:
                boxes = detection.boxes
                most_confident_ball = None  #Placeholder for the basketball with the highest confidence

                for box in boxes:
                    #Extract bounding box coordinates
                    x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                    width, height = x_max - x_min, y_max - y_min
                    center = (x_min + width // 2, y_min + height // 2)

                    #Round class confidence to 2 sig figs
                    confidence = ceil((box.conf[0]) * 100) / 100 
                    #Grab the current class detected by the model.
                    cclass = self.classes[int(box.cls[0])]
                    #Check each class sequentially for a match and compare confidence thresholds -- current values are our most consistent across tests. 
                    #Approx. 92% Detection succession rate of successful shots with current values.
                    if cclass == "Basketball":
                        if confidence > 0.5 or (within_shot_radius(center, self.hoop) and confidence > 0.1):
                            """
                            Keep track of the single most confident basketball -- helps avoid any anomalies, 
                            i.e more than one basketball in frame, basketball shaped objects.
                            """
                            if most_confident_ball is None or confidence > most_confident_ball["confidence"]:
                                #Store attributes of most confident current ball in frame
                                most_confident_ball = {
                                    "center": center, "confidence": confidence, "box": (x_min, y_min, x_max, y_max), "width": width, "height": height
                                }

                        # Process the most confident basketball-- if any
                        if most_confident_ball:
                            center = most_confident_ball["center"]
                            confidence = most_confident_ball["confidence"]
                            x_min, y_min, x_max, y_max = most_confident_ball["box"]
                            width, height = most_confident_ball["width"], most_confident_ball["height"]

                            self.ball.append((center, self.total, width, height, confidence))
                            #We draw bounding boxes and label the detected ball in the current frame.
                            cv2.rectangle(self.current_frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), thickness=2)
                            cv2.putText(self.current_frame, f"Basketball ({confidence:.2f})", (x_min, y_min - 10), self.font, 0.6, (0, 0, 255), 1, lineType=cv2.LINE_AA)
                    
                    #Identify hoops in current frame, we add its current position in frame to the hoop list to later process shots.
                    if confidence > 0.5 and cclass == "Basketball Hoop":
                        self.hoop.append((center, self.total, width, height, confidence))
                        x_max, y_max = x_min + width, y_min + height
                        cv2.rectangle(self.current_frame, (x_min, y_min), (x_max, y_max), (255, 55, 174), thickness=2)
                        cv2.putText(self.current_frame, f"Basketball Hoop ({confidence:.2f})", (x_min, y_min - 10), self.font, 0.6, (0, 255, 0), 1, lineType=cv2.LINE_AA)
                    
                    #Identify any likely defending players (arms wide, arms high, no ball in possession)
                    if confidence > 0.5 and cclass == "Defence":
                        x_max, y_max = x_min + width, y_min + height
                        cv2.rectangle(self.current_frame, (x_min, y_min), (x_max, y_max), (0, 0, 0), thickness=2)
                        cv2.putText(self.current_frame, f"Defence ({confidence:.2f})", (x_min, y_min - 10), self.font, 0.6, (0, 0, 0), 1, lineType=cv2.LINE_AA)
                    
                    #Identify everyone else
                    if confidence > 0.4 and cclass == "Player":
                        x_max, y_max = x_min + width, y_min + height
                        cv2.rectangle(self.current_frame, (x_min, y_min), (x_max, y_max), (0, 152, 248), thickness=2)
                        cv2.putText(self.current_frame, f"Player ({confidence:.2f})", (x_min, y_min - 10), self.font, 0.6, (0, 152, 248), 1, lineType=cv2.LINE_AA)
                    
                    #Identify any people currently appearing to perform a shooting motion, (3 pointer, layup, etc.)
                    if confidence > 0 and cclass == "shooting":
                        self.hoop.append((center, self.total, width, height, confidence))
                        cv2.rectangle(self.current_frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), thickness=2)
                        cv2.putText(self.current_frame, f"Shooting ({confidence:.2f})", (x_min, y_min - 10), self.font, 0.6, (255,0,0), 1, lineType=cv2.LINE_AA)
            
            #Run state functions every current_frame
            #Stabilize the motion of the hoop and ball tracking, prevent any sudden motion jumps, check for any valid scores in current frame, display visuals.
            self.update_state()
            #Increment total frames
            self.total += 1

            #Display the frame.
            cv2.imshow('NBAction', self.current_frame)
            #Adjust how many current_frames program halts for, Q to exit window.
            if cv2.waitKey(1) & 0xFF == ord('q'):  
                break

        self.video.release()
        cv2.destroyAllWindows()


    #Keep track of most recent ball and hoop locations
    def stabilize(self):
        #Stabilize our tracked ball, discard any anomalies in X,Y coordinates. (Ball should not jump from multiple locations and each location should be within a certain displacement)
        self.ball = stabilize_ball(self.ball, self.total) 
        #Leave a trail most recent ball locations (we track the center of the ball at all times--crucial for our scoring detection)
        for i in range(len(self.ball)):
            cv2.circle(self.current_frame, self.ball[i][0], 2, (0, 0, 255), 2) 

        
        if len(self.hoop) > 0:
            #Stabilize hoops in frame,
            self.hoop = stabilize_hoop(self.hoop)

            #Fetch current hoop key points
            hoop_center = self.hoop[-1][0]
            hoop_width = self.hoop[-1][2]
            hoop_height = self.hoop[-1][3]
            
            #Declare the radius of the net from the center -- we use this for determining whether a shot has gone through the net or not.
            hoop_radius = int((hoop_width + hoop_height) / 4)  
            #Avoid large scoring radiuses caused by anomalies, we set a limit of 50 pixels to the radius of any given hoop.
            max_hoop_radius = 50 
            hoop_radius = min(hoop_radius, max_hoop_radius)
            #Draw the radius, helpful for visualizing the scoring zone for basketballs. Could comment out to cut down on processing time, we keep it here for the purpose of demoing.
            cv2.circle(self.current_frame, hoop_center, hoop_radius, (0, 255, 0), 2) 

    def check_score(self):
        # Only attempt to check for scores each frame when both a ball and hoop are in the current frame.
        if len(self.hoop) > 0 and len(self.ball) > 0:
            # Make sure enough time has passed before checking another shot -- helps avoids the same shot being counted multiple times.
            if self.total - self.last_attempt_current_frame >= self.cooldown_current_frames:
                #On successful shots, increment the score and display visuals
                if in_hoop(self) and not self.ball_in_hoop:
                    self.shots_made += 1  
                    self.overlay_color = (0, 255, 0)
                    self.frame_count = self.revert_frames

                    self.ball_in_hoop = True
                    self.last_attempt_current_frame = self.total  

                    self.show_score_text = True
                    self.score_text_total = 50

                elif not in_hoop(self):
                    self.ball_in_hoop = False
                        
    def display_score(self):
        # Function to show the current score each frame.
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
        # Show the visual effects for a certain amount of frames.
        if self.frame_count > 0:
            effect = (self.frame_count / self.revert_frames)*0.2
            self.current_frame = cv2.addWeighted(self.current_frame, 1 - effect, np.full_like(self.current_frame, self.overlay_color), effect, 0)
            self.frame_count -= 5

    #We put the necessary functions that need to run every frame in a function to be ran together.
    def update_state(self):
        self.stabilize()
        self.check_score()
        self.display_score()
    
    def display_text(self, text):
        # A function just for displaying the score text.
        current_frame_height, current_frame_width, _ = self.current_frame.shape

        text_size = cv2.getTextSize(text, self.font, 2, 6)[0]
        text_width, text_height = text_size

        text_x = (current_frame_width - text_width) // 2
        text_y = (current_frame_height + text_height) // 2

        #Attempt to get a top-middle location for our score text, probably has odd behaviour on lower resolution videos
        cv2.putText(self.current_frame, text, (text_x, text_y-300), self.font, 2, (0, 0, 0), 9, lineType=cv2.LINE_AA)
        cv2.putText(self.current_frame, text, (text_x, text_y-300), self.font, 2, (0, 255, 0), 4, lineType=cv2.LINE_AA)

if __name__ == "__main__":
    NBAction()
