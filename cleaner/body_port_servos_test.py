from maestro import Controller
import time

MIDDLE = 6000

LEFT_WHEEL_PORT = 0
RIGHT_WHEEL_PORT = 1
WAIST_ROTATION = 2
HEAD_UP_DOWN_PORT = 3
HEAD_LEFT_RIGHT_PORT = 4

TEST_PORT = 4

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
        self.m.setTarget(TEST_PORT, 7500)
        time.sleep(1)
        self.m.setTarget(TEST_PORT, 600)

robot = RobotControl()

def main():
    robot.body_reset()
    
main()
