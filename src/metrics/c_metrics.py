import numpy as np
import cv2
from skimage.filters import sobel
from typing import List
import cv2, os, ctypes, platform

from ..cameras import Sensor


# Find the library file based on OS
lib_name = "libfastmetrics.so"
if platform.system() == "Darwin":
    lib_name = "libfastmetrics.dylib"
elif platform.system() == "Windows":
    lib_name = "libfastmetrics.dll" # Note: Compilation on Windows is different

try:
    # Get the absolute path to the library, assuming it's in the same directory
    lib_path = os.path.join(os.path.dirname('.'), lib_name)
    lib = ctypes.CDLL(lib_path)
except OSError as e:
    print(f"Error loading library: {e}")
    print(f"Please make sure '{lib_name}' is compiled and in the same directory as this script.")
    exit()

# --- Define ctypes argument types for type safety ---

# Define a pointer to a double
POINTER_DOUBLE = ctypes.POINTER(ctypes.c_double)

# Define NumPy-compatible pointer types
# We enforce C-contiguous arrays for safe C++ processing
NP_FLOAT64_C = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
NP_UINT8_C = np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS')

# --- Set up function signatures for calculate_slant_c ---
lib.calculate_slant_c.restype = None
lib.calculate_slant_c.argtypes = [
    NP_FLOAT64_C,      # double* depth_data
    ctypes.c_int,      # int rows
    ctypes.c_int,      # int cols
    ctypes.c_double,   # double fx
    ctypes.c_double,   # double fy
    ctypes.c_double,   # double cx
    ctypes.c_double,   # double cy
    POINTER_DOUBLE     # double* out_slant_angle
]

# --- Set up function signatures for calculate_uciqe_c ---
lib.calculate_uciqe_c.restype = None
lib.calculate_uciqe_c.argtypes = [
    NP_UINT8_C,        # unsigned char* lab_data
    ctypes.c_int,      # int rows
    ctypes.c_int,      # int cols
    POINTER_DOUBLE     # double* out_uciqe
]

# --- Set up function signatures for calculate_channel_eme_c ---
lib.calculate_channel_eme_c.restype = None
lib.calculate_channel_eme_c.argtypes = [
    NP_UINT8_C,        # unsigned char* ch_data
    ctypes.c_int,      # int rows
    ctypes.c_int,      # int cols
    ctypes.c_int,      # int blocksize
    ctypes.c_bool,     # bool is_logamee
    ctypes.c_double,   # double gamma
    ctypes.c_double,   # double k
    POINTER_DOUBLE     # double* out_eme
]

# --- Set up function signatures for calculate_uicm_c ---
lib.calculate_uicm_c.restype = None
lib.calculate_uicm_c.argtypes = [
    NP_FLOAT64_C,      # double* rgl_trimmed
    NP_FLOAT64_C,      # double* ybl_trimmed
    ctypes.c_int,      # int n
    POINTER_DOUBLE     # double* out_uicm
]


# =================================================================
#  PYTHON FUNCTIONS (Wrapper API)
# =================================================================

def calculate_slant(sensor: Sensor, depth: np.ndarray) -> float:
    """
    Calculates the slant angle using the fast C++/Eigen backend.
    """
    # Ensure data is C-contiguous and in the correct format (float64)
    depth_c = np.ascontiguousarray(depth, dtype=np.float64)
    rows, cols = depth_c.shape
    
    # Create a C-style double to store the result
    result = ctypes.c_double(0.0)
    
    # Call the C++ function
    lib.calculate_slant_c(
        depth_c, rows, cols,
        sensor.fx, sensor.fy, sensor.cx, sensor.cy,
        ctypes.byref(result)
    )
    
    return result.value


def calculate_UCIQE(img: np.ndarray) -> float:
    """
    Calculates the UCIQE metric using the fast C++ backend.
    The BGR2LAB conversion is still done in Python.
    """
    # 1. Pre-processing (still in Python)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Ensure data is C-contiguous and in the correct format (uint8)
    lab_c = np.ascontiguousarray(lab, dtype=np.uint8)
    rows, cols, _ = lab_c.shape
    
    # Create a C-style double to store the result
    result = ctypes.c_double(0.0)
    
    # 2. Call the C++ function
    lib.calculate_uciqe_c(lab_c, rows, cols, ctypes.byref(result))
    
    return result.value


def calculate_eme_logamee(img: np.ndarray, blocksize=8, gamma=1026, k=1026) -> List[float]:
    """
    Calculates EME/LogAMEE using the fast C++ backend for the block-wise calculation.
    Sobel and Grayscale conversions are still done in Python.
    """
    emes = []
    result = ctypes.c_double(0.0) # We can re-use this result variable
    
    for c in range(4):
        # 1. Pre-processing (still in Python)
        if c == 3:
            ch = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            is_logamee = True
        else:
            # Apply sobel filter to the channel
            sobel_ch = sobel(img[:, :, c])
            # Normalize and scale sobel output similar to original
            ch = np.round(img[:, :, c] * sobel_ch).astype(np.uint8)
            is_logamee = False

        # Ensure data is C-contiguous and in the correct format
        ch_c = np.ascontiguousarray(ch, dtype=np.uint8)
        rows, cols = ch_c.shape
        
        # 2. Call the C++ helper function
        lib.calculate_channel_eme_c(
            ch_c, rows, cols, 
            blocksize, is_logamee, 
            gamma, k,
            ctypes.byref(result)
        )
        emes.append(result.value)

    return emes


def calculate_UIQM(img: np.ndarray) -> float:
    """
    Calculates the UIQM metric, using C++ helpers for UICM and EME.
    """
    # --- 1. UICM Calculation ---
    
    # Pre-processing (Python)
    # **FIX:** Cast to float64 *before* subtraction to avoid uint8 overflow
    rgl = np.sort(img[:, :, 2].astype(np.float64) - img[:, :, 1].astype(np.float64), axis=None)
    ybl = np.sort((img[:, :, 2].astype(np.float64) + img[:, :, 1].astype(np.float64)) / 2 - img[:, :, 0].astype(np.float64), axis=None)

    T = int(0.1 * len(rgl))
    
    rgl_trimmed = np.ascontiguousarray(rgl[T:-T], dtype=np.float64)
    ybl_trimmed = np.ascontiguousarray(ybl[T:-T], dtype=np.float64)
    n_trimmed = len(rgl_trimmed)

    uicm_result = ctypes.c_double(0.0)
    
    # Call C++ helper for UICM
    if n_trimmed > 0:
        lib.calculate_uicm_c(rgl_trimmed, ybl_trimmed, n_trimmed, ctypes.byref(uicm_result))
    
    uicm = uicm_result.value

    # --- 2. UISM & UIConM Calculation ---
    
    # This now calls our *new*, fast calculate_eme_logamee function
    Beme, Geme, Reme, uiconm = calculate_eme_logamee(img)
    uism = 0.299 * Reme + 0.587 * Geme + 0.114 * Beme

    # --- 3. Final Result ---
    return 0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm
