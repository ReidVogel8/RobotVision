import time
from maestro import Controller

MIDDLE = 5800
HEAD_UP_DOWN_PORT = 4
HEAD_LEFT_RIGHT_PORT = 2
LEFT_WHEEL_PORT = 0
RIGHT_WHEEL_PORT = 1

class HeadControl:
    _instance = None
    @staticmethod
    def getInst():
        if HeadControl._instance == None:
            HeadControl._instance = HeadControl()
        return HeadControl._instance
        
    def __init__(self):
        self.m = Controller()
        pass
    
    def head_tilt(self, num):
        print("turn Head")
        self.m.setTarget(HEAD_UP_DOWN_PORT, num)
        time.sleep(1)
        pass

    def head_rotate(self, num):
        print("rotate Head")
        self.m.setTarget(HEAD_LEFT_RIGHT_PORT, num)
        time.sleep(1)
        pass
    
    def wheel_rotate_left(self):
        print("Left Wheel Forward")
        self.m.setTarget(LEFT_WHEEL_PORT, 7300)
        self.m.setTarget(RIGHT_WHEEL_PORT, 7300)
        time.sleep(1)
        self.m.setTarget(LEFT_WHEEL_PORT, 6000)
        self.m.setTarget(RIGHT_WHEEL_PORT, 6000)
        pass


x = HeadControl().getInst()
x.head_tilt(9000)
x.head_tilt(3000)
x.head_tilt(6000)

x.head_rotate(9000)
x.head_rotate(3000)
x.head_rotate(6000)

x.wheel_rotate_left()
