
from maestro import Controller

MIDDLE = 5800
PORT = 4

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
        self.m.setTarget(PORT, num)
        pass

    def rotate(self):
        print("rotate Head")
        self.m.setTarget(PORT, 7000)
        self.m.setTarget(PORT, 3000)
        pass
    
    
x = HeadControl().getInst()
x.tilt(9000)
x.tilt(3000)
x.tilt(MIDDLE)
