import csv
from typing import Dict, Tuple


class Colormap:
    hex_to_bgr = lambda h: tuple(int(h[i:i + 2], 16) for i in (5, 3, 1))

    def __init__(self, csv_path):
        self.file_path = csv_path
        self.colormap: Dict[Tuple[int, int, int], str] = {}

        with open(csv_path, 'r') as f:
            reader = csv.reader(f)

            for row in reader:
                label, color = row
                self.colormap[Colormap.hex_to_bgr(color)] = label

    def get_label(self, color: Tuple[int, int, int]) -> None | str:
        if self.colormap.get(color) is None:
            return None

        return self.colormap[color]

