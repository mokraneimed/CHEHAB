class EncParams:
    def __init__(self, n=1024, q=27, t=20):
        self.n = n
        self.t = t
        self.q = q

    def set_params(self, n, q, t):
        self.n = n
        self.t = t
        self.q = q
        
    def increase_q(self, value):
        self.q = self.q + value