import numpy as np
import cv2, csv, gc
from typing import List, Tuple, Dict

from .colormap import Colormap
from .cameras import Sensor
from .metrics import calculate_ground_resolution, calculate_slant, calculate_UCIQE, calculate_UIQM


class Pipeline:
    fields: Tuple[str] = [
        "label",
        "image",
        "camera",
        "res. width",
        "res. height",
        "visibility",
        "centroid u",
        "centroid v",
        "med. depth (mm)",
        "avg. depth (mm)",
        "area (pixels)",
        "ground res. (mm / pixel)",
        "camera slant (degrees)",
        "UIQM",
        "UCIQUE",
    ]

    def __init__(self, sensor: Sensor, colormap: Colormap, target_classes: Tuple[str], output_path: str):
        self.data: List[Dict[str, str]]  = []

        self.sensor = sensor
        self.colormap = colormap
        self.target_classes = target_classes
        self.output_path = output_path

    def toFile(self):
        if self.output_path.split('.')[-1].lower() != 'csv':
            self.output_path += '.csv'

        with open(self.output_path, 'w') as f:
            writer = csv.DictWriter(f, Pipeline.fields)
            writer.writeheader()
            writer.writerows(self.data)

    def processImages(self,
        imgs_path: List[str],
        masks_path: List[str],
        depths_path: List[str],
        camera_type: str,
        visibility: float
    ):
        for img_path, mask_path, depth_path in zip(imgs_path, masks_path, depths_path):
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

            slant = calculate_slant(self.sensor, depth)
            uiqm = calculate_UIQM(img)
            ucique = calculate_UCIQE(img)

            for color in np.unique(mask.reshape(-1, mask.shape[-1]), axis=0):
                label = self.colormap.get_label(tuple(int(c) for c in color))

                if label is None or label.split('_')[0] not in self.target_classes:
                    continue

                indices = np.logical_and(np.all(mask == color, axis=2), depth > 0)
                u, v = np.where(indices)

                self.data.append({
                    "label": label,
                    "image": ''.join(img_path.split('/')[-1].split('.')[:-1]),
                    "camera": camera_type,
                    "res. width": depth.shape[1],
                    "res. height": depth.shape[0],
                    "visibility": visibility,
                    "centroid u": np.median(u),
                    "centroid v": np.median(v),
                    "med. depth (mm)": np.median(depth[u, v]),
                    "avg. depth (mm)": np.mean(depth[u, v]),
                    "area (pixels)": u.size,
                    "ground res. (mm / pixel)": calculate_ground_resolution(self.sensor, u, v, depth[u, v]),
                    "camera slant (degrees)": slant,
                    "UIQM": uiqm,
                    "UCIQUE": ucique
                })

                self.toFile()

            del img, depth, mask
            gc.collect()


