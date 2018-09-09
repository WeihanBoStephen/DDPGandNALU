import time
from uarm.wrapper import SwiftAPI
from uarm.utils.log import logger

class uarm:
    def __init__(self):
        logger.setLevel(logger.VERBOSE)
        self.swift = SwiftAPI(filters={'hwid': 'USB VID:PID=2341:0042'}, callback_thread_pool_size=1)
        self.swift.waiting_ready()
        self.swift.set_speed_factor(factor=20)
        self.swift.set_position(x=239.43, y=0, z=170)
    def jump(self,press_time):
        self.swift.set_position(x=239.43, y=0, z=-30)
        print("robot:%f"%press_time)
        time.sleep(press_time)
        self.swift.set_position(x=239.43, y=0, z=170)

# if __name__ == '__main__':
#     arm = uarm()
#     arm.jump(0.75)
#     time.sleep(5)
#     arm.jump(0.75)