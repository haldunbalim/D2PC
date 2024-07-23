import numpy as np

class OLController:
    def __init__(self):
        pass

    def control(self, state):
        raise NotImplementedError("Base Class")

    def __call__(self):
        return self.control()

class RandomNormalController(OLController):
    def __init__(self, nu: int, sigma_u: float = 1):
        self.nu = nu
        self.sigma_u = sigma_u

    def control(self):
        return np.random.normal(scale=self.sigma_u, size=self.nu).astype(np.float64)
    
class StaticController(OLController):
    def __init__(self, us: np.ndarray):
        self.us = us
        self.t = 0

    def control(self):
        u = self.us[self.t]
        self.t += 1
        return u
    
class RepController(OLController):
    def __init__(self, nu:int, choices: np.ndarray, rep: int):
        self.nu = nu
        self.choices = choices
        self.state = np.random.choice(choices, nu)
        self.ct = 0
        self.rep = rep

    def control(self):
        if self.ct % self.rep == 0:
            self.state = np.random.choice(self.choices, self.nu)
        self.ct += 1
        return self.state
