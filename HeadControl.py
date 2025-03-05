
from maestro import Controller

MIDDLE = 5800
PORT = 3

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
    
    def tilt(self):
        print("turn Head")
        self.m.setTarget(PORT, MIDDLE)
        pass
