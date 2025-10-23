import numpy as np
from typing import List


class Sensor:
    def __init__(self, padding: int = 0) -> None:
        self.id: str = None

        # Intrinsic Parameters
        self.fx: float = None    # Focal length X-axis
        self.fy: float = None    # Focal length Y-axis

        self.cx: float = None    # Principal point X-axis
        self.cy: float = None    # Principal point Y-axis

        self.fovx: float = None  # Field of View X-axis (radians)
        self.fovy: float = None  # Field of View Y-axis (radians)

        self.width: int = None   # Resolution width
        self.height: int = None  # Resolution height

        # Distortion Parameters (Brown-Conrady)
        self.k1: float = None  # 1st Radial coefficient
        self.k2: float = None  # 2nd Radial coefficient
        self.k3: float = None  # 3rd Radial coefficient

        self.p1: float = None  # 1st Tangential coefficient
        self.p2: float = None  # 2nd Tangential coefficient
