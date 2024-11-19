import math
import numpy as np


def score(self):

    if len(self.hoop_pos) == 0 or len(self.ball_pos) == 0:
        return False

    hoop_center = self.hoop_pos[-1][0] 
    hoop_radius = int(max(self.hoop_pos[-1][2], self.hoop_pos[-1][3]) * 0.5)  
    ball_center = self.ball_pos[-1][0] 

    distance = math.sqrt((ball_center[0] - hoop_center[0]) ** 2 + (ball_center[1] - hoop_center[1]) ** 2)

    if distance <= hoop_radius and ball_center[1] > hoop_center[1]:
        return True

    return False

def detect_down(ball_pos, hoop_pos):
    y = hoop_pos[-1][0][1] + 0.5 * hoop_pos[-1][3]
    if ball_pos[-1][0][1] > y:
        return True
    return False

def detect_up(ball_pos, hoop_pos):
    x1 = hoop_pos[-1][0][0] - 4 * hoop_pos[-1][2]
    x2 = hoop_pos[-1][0][0] + 4 * hoop_pos[-1][2]
    y1 = hoop_pos[-1][0][1] - 2 * hoop_pos[-1][3]
    y2 = hoop_pos[-1][0][1]

    if x1 < ball_pos[-1][0][0] < x2 and y1 < ball_pos[-1][0][1] < y2 - 0.5 * hoop_pos[-1][3]:
        return True
    return False

def in_hoop_region(center, hoop_pos):
    if len(hoop_pos) < 1:
        return False
    x = center[0]
    y = center[1]

    x1 = hoop_pos[-1][0][0] - 1 * hoop_pos[-1][2]
    x2 = hoop_pos[-1][0][0] + 1 * hoop_pos[-1][2]
    y1 = hoop_pos[-1][0][1] - 1 * hoop_pos[-1][3]
    y2 = hoop_pos[-1][0][1] + 0.5 * hoop_pos[-1][3]

    if x1 < x < x2 and y1 < y < y2:
        return True
    return False


def clean_ball_pos(ball_pos, frame_count):
    if len(ball_pos) > 1:
        w1 = ball_pos[-2][2]
        h1 = ball_pos[-2][3]
        w2 = ball_pos[-1][2]
        h2 = ball_pos[-1][3]

        x1 = ball_pos[-2][0][0]
        y1 = ball_pos[-2][0][1]
        x2 = ball_pos[-1][0][0]
        y2 = ball_pos[-1][0][1]

        f1 = ball_pos[-2][1]
        f2 = ball_pos[-1][1]
        f_dif = f2 - f1

        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        max_dist = 4 * math.sqrt((w1) ** 2 + (h1) ** 2)

        if (dist > max_dist) and (f_dif < 5):
            ball_pos.pop()

        elif (w2*1.4 < h2) or (h2*1.4 < w2):
            ball_pos.pop()

    if len(ball_pos) > 0:
        if frame_count - ball_pos[0][1] > 15:
            ball_pos.pop(0)

    return ball_pos


def clean_hoop_pos(hoop_pos):
    if len(hoop_pos) > 1:
        x1 = hoop_pos[-2][0][0]
        y1 = hoop_pos[-2][0][1]
        x2 = hoop_pos[-1][0][0]
        y2 = hoop_pos[-1][0][1]

        w1 = hoop_pos[-2][2]
        h1 = hoop_pos[-2][3]
        w2 = hoop_pos[-1][2]
        h2 = hoop_pos[-1][3]

        f1 = hoop_pos[-2][1]
        f2 = hoop_pos[-1][1]

        f_dif = f2-f1

        dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)

        max_dist = 0.5 * math.sqrt(w1 ** 2 + h1 ** 2)

        if dist > max_dist and f_dif < 5:
            hoop_pos.pop()

        if (w2*1.3 < h2) or (h2*1.3 < w2):
            hoop_pos.pop()

    if len(hoop_pos) > 10:
        hoop_pos.pop(0)

    return hoop_pos
