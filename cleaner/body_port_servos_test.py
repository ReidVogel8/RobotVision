from maestro import Controller
import time

MIDDLE = 6000

LEFT_WHEEL_PORT = 0
RIGHT_WHEEL_PORT = 1
WAIST_ROTATION = 2 #lower than 6000 turns to its right
HEAD_UP_DOWN_PORT = 3
HEAD_LEFT_RIGHT_PORT = 4
LEFT_SHOULDER_UP_DOWN = 5
LEFT_SHOULDER_LEFT_RIGHT = 6
LEFT_ELBOW = 7
LEFT_WRIST = 8

TEST_PORT = 9

class RobotControl:
    _instance = None

    @staticmethod
    def getInst():
        if RobotControl._instance is None:
            RobotControl._instance = RobotControl()
        return RobotControl._instance

    def __init__(self):
        self.m = Controller()

    def body_reset(self):
        self.m.setTarget(2, 6600)
        time.sleep(0.5)
        self.m.setTarget(3, 5200)
        time.sleep(0.5)
        self.m.setTarget(4, 6000)
        time.sleep(0.5)
        self.m.setTarget(5, 6000)
        time.sleep(0.5)
        self.m.setTarget(6, 6000)
        time.sleep(0.5)
        self.m.setTarget(7, 6000)
        time.sleep(0.5)
        self.m.setTarget(8, 6000)
        time.sleep(0.5)
        self.m.setTarget(9, 6000)
        time.sleep(0.5)
        self.m.setTarget(10, 6000)
        time.sleep(0.5)
        self.m.setTarget(11, 6000)
        time.sleep(0.5)
        self.m.setTarget(12, 6000)
        time.sleep(0.5)
        self.m.setTarget(13, 6000)
        time.sleep(0.5)
        self.m.setTarget(14, 6000)
        time.sleep(0.5)
        self.m.setTarget(15, 6000)
        time.sleep(0.5)
        self.m.setTarget(16, 6000)
        time.sleep(0.5)
        self.m.setTarget(17, 6000)
        time.sleep(0.5)
        self.m.setTarget(18, 6000)
        time.sleep(0.5)
        self.m.setTarget(19, 6000)
        time.sleep(0.5)
        self.m.setTarget(20, 6000)
        time.sleep(0.5)

    def servo_test(self):
        self.m.setTarget(TEST_PORT, 7500)
        time.sleep(1)
        self.m.setTarget(TEST_PORT, 6000)

robot = RobotControl()

def main():
    #robot.body_reset()
    robot.servo_test()
    
main()
