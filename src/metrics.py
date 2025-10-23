import numpy as np
import cv2
from skimage.filters import sobel
from typing import List

from .cameras import Sensor


def calculate_ground_resolution(sensor: Sensor, u: np.ndarray, v:np.ndarray, z: np.ndarray) -> float:
    x = z * (u - sensor.cx) / sensor.fx
    y = z * (v - sensor.cy) / sensor.fy

    return np.linalg.norm((x[1] - x[0], y[1] - y[0], z[1] - z[0])) / np.linalg.norm((u, v))  # mm / px


def calculate_slant(sensor: Sensor, depth: np.ndarray, n_ref: np.ndarray = np.array((0, -1, 0))) -> float:
    # v is row index (y-coordinate), u is col index (x-coordinate)
    v, u = np.indices(depth.shape) 

    # Calculate 3D coordinates (X, Y, Z) for desk pixels
    Z = depth.flatten()
    X = Z * (u.flatten() - sensor.cx) / sensor.fx
    Y = Z * (v.flatten() - sensor.cy) / sensor.fy

    # Center the points by subtracting the centroid
    points = np.stack((X, Y, Z), axis=1)
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid

    # Perform SVD on the centered points
    # The normal vector is the eigenvector corresponding to the smallest singular value
    _, _, Vt = np.linalg.svd(points_centered)

    # The normal vector is the last row of Vt (corresponding to the smallest singular value in S)
    normal_vector = Vt[-1, :]
    if normal_vector[2] < 0:  # Ensure the normal vector points towards the camera
        normal_vector = -normal_vector

    # Ensure both normals are unit vectors
    n_unit = normal_vector / np.linalg.norm(normal_vector)
    n_ref_unit = n_ref / np.linalg.norm(n_ref)

    # Clamp the dot product to [-1, 1] to avoid floating-point errors
    dot_product = np.clip(np.dot(n_unit, n_ref_unit), -1.0, 1.0)
    angle = np.arccos(np.abs(dot_product))

    return np.degrees(angle)


def calculate_UCIQE(img: np.ndarray) -> float:
    # Source: https://github.com/xueleichen/PSNR-SSIM-UCIQE-UIQM-Python/blob/main/nevaluate.py

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    #1st term
    chroma = np.linalg.norm((lab[:, :, 1], lab[:, :, 2]))
    sc = np.sqrt(np.mean(np.power(chroma - np.mean(chroma), 2)))

    #2nd term
    top = int(0.01 * lab.shape[0] * lab.shape[1])
    sl = np.sort(lab[:, :, 0], axis=None)
    conl = np.mean(sl[top:]) - np.mean(sl[:top])

    #3rd term
    satur = 0
    for c1, l1 in zip(chroma.flatten(), lab[:, :, 0].flatten()):
        if l1 != 0:
            satur += c1 / l1

    return 0.4680 * sc + 0.2745 * conl + 0.2576 * satur / chroma.size  # c1 * sc + c2 * conl + c3 * us


def calculate_eme_logamee(img: np.ndarray, blocksize=8, gamma=1026, k=1026) -> List[float]:
    # Source: https://github.com/xueleichen/PSNR-SSIM-UCIQE-UIQM-Python/blob/main/nevaluate.py

    emes = []
    for c in range(4):
        if c == 3:
            ch = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            ch = np.round(img[:, :, i] * sobel(img[:, :, i])).astype(np.uint8)

        num_x = np.ceil(ch.shape[0] / blocksize)
        num_y = np.ceil(ch.shape[1] / blocksize)

        eme = 0
        w = 1 / (num_x * num_y)
        for i in range(num_x):
            xlb = i * blocksize
            if i < num_x - 1:
                xrb = (i + 1) * blocksize
            else:
                xrb = ch.shape[0]

            for j in range(num_y):
                ylb = j * blocksize
                if j < num_y - 1:
                    yrb = (j + 1) * blocksize
                else:
                    yrb = ch.shape[1]

                block = ch[xlb:xrb, ylb:yrb]
                blockmin = np.float(np.min(block))
                blockmax = np.float(np.max(block))

                if c == 3:  # Logamee
                    top = k * (blockmax - blockmin) / (k - blockmin)
                    bottom = blockmax + blockmin - blockmax * blockmin / gamma

                    m = top/bottom
                    if m != 0:
                        eme += m * np.log(m)
                else:  # Regular eme
                    if blockmin == 0:
                        blockmin = 1

                    if blockmax == 0:
                        blockmax = 1

                    eme += 2 * w * np.log(blockmax / blockmin)

        if c == 3:
            emes.append(gamma - gamma * np.power(1 - eme / gamma, w))
        else:
            emes.append(eme)

    return emes


def calculate_UIQM(img: np.ndarray) -> float:
    # Source: https://github.com/xueleichen/PSNR-SSIM-UCIQE-UIQM-Python/blob/main/nevaluate.py

    rgl = np.sort(img[:, :, 2] - img[:, :, 1], axis=None)
    ybl = np.sort((img[:, :, 2] + img[:, :, 1]) / 2 - img[:, :, 0], axis=None)

    T = int(0.1 * len(rgl))

    urg = np.mean(rgl[T:-T])
    uyb = np.mean(ybl[T:-T])

    s2rg = np.mean(np.power(rgl[T:-T] - urg, 2))
    s2yb = np.mean(np.power(ybl[T:-T] - uyb, 2))

    uicm = -0.0268 * np.linalg.norm((urg, uyb)) + 0.1586 * np.sqrt(s2rg + s2yb)

    Beme, Geme, Reme, uiconm = calculate_eme_logamee(img)
    uism = 0.299 * Reme + 0.587 * Geme + 0.114 * Beme

    return 0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm
