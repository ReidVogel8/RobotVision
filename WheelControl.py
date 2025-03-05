from maestro import Controller
import time

MIDDLE = 5800
LEFT_WHEEL_PORT = 8
RIGHT_WHEEL_PORT = 12


class WheelControl:
    _instance = None

    @staticmethod
    def getInst():
        if WheelControl._instance == None:
            WheelControl._instance = WheelControl()
        return WheelControl._instance

    def __init__(self):
        self.m = Controller()
        pass

    def rotate_left(self):
        print("rotate Head")
        self.m.setTarget(LEFT_WHEEL_PORT, 9000)
        time.sleep(1)
        self.m.setTarget(LEFT_WHEEL_PORT, 6000)
        pass


x = WheelControl().getInst()
x.rotate_left()

