from maestro import Controller
import time

MIDDLE = 6000
HEAD_UP_DOWN_PORT = 2
HEAD_LEFT_RIGHT_PORT = 4
LEFT_WHEEL_PORT = 0
RIGHT_WHEEL_PORT = 1

class RobotControl:
    _instance = None

    @staticmethod
    def getInst():
        if RobotControl._instance is None:
            RobotControl._instance = RobotControl()
        return RobotControl._instance

    def __init__(self):
        self.m = Controller()

    def pan_left(self):
        self.m.setTarget(HEAD_LEFT_RIGHT_PORT, 6000)
        self.m.setTarget(HEAD_LEFT_RIGHT_PORT, 5900)
        time.sleep(.1)
        
    def body_reset(self):
        self.m.setTarget(2, 6000)
        self.m.setTarget(3, 6000)
        self.m.setTarget(4, 6000)
        self.m.setTarget(5, 6000)
        self.m.setTarget(6, 6000)
        self.m.setTarget(7, 6000)
        self.m.setTarget(8, 6000)
        self.m.setTarget(9, 6000)
        self.m.setTarget(10, 6000)
        self.m.setTarget(11, 6000)
        self.m.setTarget(12, 6000)
        self.m.setTarget(13, 6000)
        self.m.setTarget(14, 6000)
        self.m.setTarget(15, 6000)
        self.m.setTarget(16, 6000)




    def pan_right(self):
        self.m.setTarget(HEAD_LEFT_RIGHT_PORT, 6000)
        self.m.setTarget(HEAD_LEFT_RIGHT_PORT, 6100)
        time.sleep(.1)

    def turn_left(self, duration):
        print("left")
        self.m.setTarget(RIGHT_WHEEL_PORT, 6000)
        self.m.setTarget(RIGHT_WHEEL_PORT, 7000)
        time.sleep(duration)
        self.m.setTarget(RIGHT_WHEEL_PORT, 6000)

    def turn_right(self, duration):
        print("right")
        self.m.setTarget(RIGHT_WHEEL_PORT, 6000)
        time.sleep(0.5)
        self.m.setTarget(RIGHT_WHEEL_PORT, 5000)
        time.sleep(duration)
        self.m.setTarget(RIGHT_WHEEL_PORT, 6000)

    def move_forward(self):
        self.m.setTarget(LEFT_WHEEL_PORT, 6000)
        self.m.setTarget(0, 5000)
        time.sleep(1.1)
        self.m.setTarget(LEFT_WHEEL_PORT, 6000)

robot = RobotControl()


def main():
    #robot.body_reset()
    robot.turn_right(0.8)
    time.sleep(0.75)
    robot.turn_left(0.89)
    
main()
