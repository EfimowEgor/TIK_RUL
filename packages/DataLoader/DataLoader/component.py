import numpy as np

class Component:
    def __init__(self,
                 data: np.ndarray,
                 direction: str,
                 idx: 'str' ) -> None:
        self.direction = direction
        self.data = data
        self.idx = idx

    def __repr__(self):
        return f'data: {self.data}, direction: {self.direction}, idx: {self.idx}'
