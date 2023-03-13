import cv2
import math
import random

class MovingCircle:
    def __init__(self, centerX, centerY, radiusOfMask):
        self.x = random.randint(0, centerX - radiusOfMask + 80) + radiusOfMask
        self.y = random.randint(0, centerY - radiusOfMask + 80) + radiusOfMask
        self.step = random.randint(2, 6)
        self.angle = random.uniform(0, 359)
        self.radiusOfMask = radiusOfMask
        self.color = (10, 10, 20)
        self.size = 10
        self.center = (centerX, centerY)
        self.toRadians =  math.pi / 180

    def move(self):
        angle_dist = random.uniform(-11, 6)
        step_dist = random.randint(5, 9)

        self.angle += angle_dist
        
        new_x = self.x + self.step * math.cos(self.angle * self.toRadians)
        new_y = self.y + self.step * math.sin(self.angle * self.toRadians)

        distance = math.sqrt((new_x - self.center[0]) ** 2 + (new_y - self.center[1]) ** 2)

        if distance >= self.radiusOfMask:
            self.angle += 180
            new_x = self.x + self.step * math.cos(self.angle * self.toRadians)
            new_y = self.y + self.step * math.sin(self.angle * self.toRadians)

        self.x = new_x
        self.y = new_y
        self.size = int(4 * (self.radiusOfMask - distance) / self.radiusOfMask) + 1

        if random.randint(0, 121) == 0:
            self.angle = angle_dist
            self.step = step_dist

    def draw(self, frame, color):
        cv2.circle(frame, (int(self.x), int(self.y)), self.size, color, -1)

