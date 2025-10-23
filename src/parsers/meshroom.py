import numpy as np
import json
from typing import List

from .parser import Parser
from ..cameras import Sensor


class Meshroom(Parser):
    def parse(self, file_path: str) -> List[Sensor]:
        extension = file_path.split('.')[-1]
        if extension.lower() != 'sfm':
            raise NameError(f"Invalid filename extension '{extension}', expected an SFM file.")

        data = None
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Parse calibration parameters
        sensors: List[Sensor] = []
        for sensor in data['intrinsics']:
            s = Sensor()

            s.width = int(sensor['width'])
            s.height = int(sensor['height'])

            focal_length = float(sensor['focalLength'])  # mm

            px = float(sensor['sensorWidth']) / s.width  # mm / pixels
            py = float(sensor['sensorHeight']) / s.height # mm / pixels

            s.fx = focal_length / px  # Focal length in pixels
            s.fy = focal_length / py  # Focal length in pixels

            s.fovx = 2 * np.arctan(s.width / (2 * s.fx))
            s.fovy = 2 * np.arctan(s.height / (2 * s.fy))

            s.cx = s.width / 2.0 + float(sensor['principalPoint'][0])
            s.cy = s.height / 2.0 + float(sensor['principalPoint'][1])

            if len(sensor['distortionParams']) < 5:
                raise ValueError("Distortion provided doesn't follow Brown-Conrady model.")

            s.k1 = float(sensor['distortionParams'][0])
            s.k2 = float(sensor['distortionParams'][1])
            s.k3 = float(sensor['distortionParams'][2])

            s.p1 = float(sensor['distortionParams'][3])
            s.p2 = float(sensor['distortionParams'][4])

            s.id = sensor['intrinsicId']

            sensors.append(s)

        return sensors

