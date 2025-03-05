import time
from maestro import Controller

MIDDLE = 5800
UPDOWNPORT = 4
LEFTRIGHTPORT = 3

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
    
    def tilt(self, num):
        print("turn Head")
        self.m.setTarget(UPDOWNPORT, num)
        time.sleep(1)
        pass

    def rotate(self, num):
        print("rotate Head")
        self.m.setTarget(LEFTRIGHTPORT, num)
        time.sleep(1)
        pass


x = HeadControl().getInst()
x.tilt(9000)
x.tilt(3000)
x.tilt(6000)

x.rotate(9000)
x.rotate(3000)
x.rotate(6000)
