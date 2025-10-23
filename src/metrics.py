import numpy as np
import cv2
from skimage.filters import sobel
from typing import List

from .cameras import Sensor


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


def calculate_slant(sensor: Sensor, depth: np.ndarray) -> float:
    # v is row index (y-coordinate), u is col index (x-coordinate)
    v, u = np.indices(depth.shape)

    # Calculate 3D coordinates (X, Y, Z) for desk pixels
    indices = depth.flatten() > 0

    Z = depth.flatten()[indices]
    X = Z * (u.flatten()[indices] - sensor.cx) / sensor.fx
    Y = Z * (v.flatten()[indices] - sensor.cy) / sensor.fy

    # Create the design matrix for the polynomial terms
    A = np.vstack([
        np.ones_like(Z),  # p1
        X,                # p2 * x
        Y,                # p3 * y
        X**2,             # p4 * x^2
        X * Y,            # p5 * x * y
        Y**2,             # p6 * y^2
        X**2 * Y,         # p7 * x^2 * y
        X * Y**2,         # p8 * x * y^2
        Y**3              # p9 * y^3
    ]).T

    # Perform least squares fitting
    # coeffs are p1, p2, p3, p4, p5, p6, p7, p8, and p9
    coeffs, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)

    # Calculate the fitted z values using the polynomial equation
    z_fitted = (
        coeffs[0] +
        coeffs[1] * X +
        coeffs[2] * Y +
        coeffs[3] * X**2 +
        coeffs[4] * X * Y +
        coeffs[5] * Y**2 +
        coeffs[6] * X**2 * Y +
        coeffs[7] * X * Y**2 +
        coeffs[8] * Y**3
    )

    n_z = -1
    n_x = coeffs[1]  # Coefficient for x
    n_y = coeffs[2]  # Coefficient for y

    # Calculate angle with respect to the vertical using the magnitude of the linear normal vector
    normal_magnitude = np.sqrt(n_x ** 2 + n_y ** 2 + n_z ** 2)
    return np.degrees(np.arccos(n_z / (normal_magnitude + 1e-6)))


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
            ch = np.round(img[:, :, c] * sobel(img[:, :, c])).astype(np.uint8)

        num_x = int(np.ceil(ch.shape[0] / blocksize))
        num_y = int(np.ceil(ch.shape[1] / blocksize))

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
                blockmin = np.astype(np.min(block), float)
                blockmax = np.astype(np.max(block), float)

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
