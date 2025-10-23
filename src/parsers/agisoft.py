import numpy as np
import xml.etree.ElementTree as ET
from typing import List

from .parser import Parser
from ..cameras import Sensor


class Agisoft(Parser):
    def parse(self, file_path: str) -> List[Sensor]:
        extension = file_path.split('.')[-1]
        if extension.lower() != 'xml':
            raise NameError(f"Invalid filename extension '{extension}', expected an XML file.")

        # Parse XML file
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Parse calibration parameters
        chunk = root.find('chunk')
        sensors: List[Sensor] = []
        for sensor in chunk.find('sensors').findall('sensor'):
            s = Sensor()

            calibration = sensor.find('calibration')
            if calibration is None:
                continue

            s.width = int(calibration.find('resolution').attrib['width'])
            s.height = int(calibration.find('resolution').attrib['height'])

            s.fx = float(calibration.find('f').text)
            s.fy = s.fx

            s.fovx = 2 * np.arctan(s.width / (2 * s.fx))
            s.fovy = 2 * np.arctan(s.height / (2 * s.fy))

            s.cx = s.width / 2.0 + float(calibration.find('cx').text)
            s.cy = s.height / 2.0 + float(calibration.find('cy').text)

            s.p1 = float(calibration.find('p1').text)
            s.p2 = float(calibration.find('p2').text)

            s.k1 = float(calibration.find('k1').text)
            s.k2 = float(calibration.find('k2').text)
            s.k3 = float(calibration.find('k3').text)

            s.id = sensor.attrib['id']

            sensors.append(s)

        return sensors

