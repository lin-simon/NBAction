import math
import numpy as np

def in_hoop(self):
    #Check if the ball is inside the net. -- True if a successful shot
    #We make sure at least one hoop and ball is in the current frame
    if len(self.hoop) == 0 or len(self.ball) == 0:
        return False
    #Grab the center of the hoop, grab the most recent center of the ball
    hoop_center = self.hoop[-1][0] 
    ball_center = self.ball[-1][0] 
    #We draw a radius from the center of the hoop with a diameter same size as the full length of the net of the hoop.
    hoop_radius = int(max(self.hoop[-1][2], self.hoop[-1][3]) * 0.5)  
    
    #We use a simple distance formula to calculate the difference between the center of the ball, to the center of the hoop.
    distance = math.sqrt((ball_center[0] - hoop_center[0]) ** 2 + (ball_center[1] - hoop_center[1]) ** 2)
    #Check if the ball is within the hoop radius and below the hoop's center. (a score)
    if distance <= hoop_radius and ball_center[1] > hoop_center[1]:
        return True

    return False

def within_shot_radius(center, hoop):
    #Checks if a point is within a shot radius near the hoop.
    if len(hoop) < 1:
        return False
    x = center[0]
    y = center[1]
    #Get the center of the ball and hoop
    x1 = hoop[-1][0][0] - 1 * hoop[-1][2]
    x2 = hoop[-1][0][0] + 1 * hoop[-1][2]
    y1 = hoop[-1][0][1] - 1 * hoop[-1][3]
    y2 = hoop[-1][0][1] + 0.5 * hoop[-1][3]
    #Check if the point is within the hoops bounds
    if x1 < x < x2 and y1 < y < y2:
        return True
    return False


def stabilize_ball(ball, frame_count):
    #Upper limit on recent ball positions (the red dots)
    #This is here to avoid some glitches where dots appear all over the screen when the ball travels too quickly and disappears offscreen
    max_ball_count = 30
    if len(ball) > max_ball_count:
        return []  #Reset all ball positions if limit exceeded

    if len(ball) > 1:
        #Grab positions of the last two ball detections
        x1 = ball[-2][0][0]
        y1 = ball[-2][0][1]
        x2 = ball[-1][0][0]
        y2 = ball[-1][0][1]
        w1 = ball[-2][2]
        h1 = ball[-2][3]
        w2 = ball[-1][2]
        h2 = ball[-1][3]

        f1 = ball[-2][1]
        f2 = ball[-1][1]
        f_dif = f2 - f1
        #Calculate the Euclidean distance between the last two ball positions
        #More indepth explanation on IEEE report methodology section.
        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        max_dist = 4 * math.sqrt((w1) ** 2 + (h1) ** 2)
        #Anomaly checker -- look for large distance jumps and remove
        if (dist > max_dist and f_dif < 5) or (w2 * 1.4 < h2) or (h2 * 1.4 < w2):
            ball.pop()
    #Remove old ball positions for performance
    if len(ball) > 0 and frame_count - ball[0][1] > 5:
        ball.pop(0)

    return ball

def stabilize_hoop(hoop):
    #Same method but for the hoop,
    if len(hoop) > 1:
        x1 = hoop[-2][0][0]
        y1 = hoop[-2][0][1]
        x2 = hoop[-1][0][0]
        y2 = hoop[-1][0][1]
        w1 = hoop[-2][2]
        h1 = hoop[-2][3]
        w2 = hoop[-1][2]
        h2 = hoop[-1][3]

        f1 = hoop[-2][1]
        f2 = hoop[-1][1]

        f_dif = f2-f1

        dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        #Calculate the Euclidean distance between the last two hoop positions
        max_dist = 0.5 * math.sqrt(w1 ** 2 + h1 ** 2)
        #Factor here is a lot smaller since hoops dont typically move (I hope)
        if dist > max_dist and f_dif < 5:
            hoop.pop()
        #Remove if distance is too large -- an anomaly.
        if (w2*1.3 < h2) or (h2*1.3 < w2):
            hoop.pop()
    #Remove old hoop positions for performance.
    if len(hoop) > 25:
        hoop.pop(0)

    return hoop
