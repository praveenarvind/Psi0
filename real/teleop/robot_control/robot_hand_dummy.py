import numpy as np

class Dex3_1_Controller:
    def __init__(self, *args, **kwargs):
        print("[DummyHand] No hand controller - running without hands")
    
    def get_current_dual_hand_q(self):
        return np.zeros(14)
    
    def get_current_dual_hand_pressure(self):
        return np.zeros(216)
    
    def ctrl_dual_hand(self, *args, **kwargs):
        pass
    
    def shutdown(self):
        pass
    
    def reset(self):
        pass