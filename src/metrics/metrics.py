import numpy as np

from ..cameras import Sensor


def calculate_ground_resolution(sensor: Sensor, u: np.ndarray, v:np.ndarray, z: np.ndarray) -> float:
    x = z * (u - sensor.cx) / sensor.fx
    y = z * (v - sensor.cy) / sensor.fy

    min_x, max_x = np.argmin(x), np.argmax(x)
    min_y, max_y = np.argmin(y), np.argmax(y)

    u_res = np.linalg.norm((u[max_x] - u[min_x], v[max_x] - v[min_x]))  # px
    v_res = np.linalg.norm((u[max_y] - u[min_y], v[max_y] - v[min_y]))  # px

    x_res = np.linalg.norm((x[max_x] - x[min_x], y[max_x] - y[min_x]))  # mm
    y_res = np.linalg.norm((x[max_y] - x[min_y], y[max_y] - y[min_y]))  # mm

    return (x_res/u_res + y_res/v_res) / 2  # mm / px