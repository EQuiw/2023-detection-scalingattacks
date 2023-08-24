import numpy as np
import typing


def fourier(img: np.ndarray) -> np.ndarray:
    """Transform greyscale image (shape (x, y, 1) or shape (x, y)) to Fourier spectrogram"""
    if len(np.shape(img)) == 2:
        img = np.expand_dims(img, axis=2)
    img = np.mean(img, axis=2)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    spec = 20 * np.log(np.abs(fshift).clip(min=0.00000001))
    return spec


def fourier_3_channels(img: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get 3 Fourier shifts (no spectrogram! we do not do the log-step) of image with 3 channels"""
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    # r
    f_r = np.fft.fft2(r)
    fshift_r = np.fft.fftshift(f_r)

    # g
    f_g = np.fft.fft2(g)
    fshift_g = np.fft.fftshift(f_g)

    # b
    f_b = np.fft.fft2(b)
    fshift_b = np.fft.fftshift(f_b)

    return fshift_r, fshift_g, fshift_b


def get_main_peaks(f_img_shape: typing.Tuple[int, int, int],
                   target_shape: typing.Tuple[int, int, int]):
    """
    Get indices of locations at which peaks are to be expected in an attack instance.
     .   .   .   .   .   .   .   .   .   .   .
    .   .   .   .   .   .   .   .   .   .   .
    .   x   .   x   .   x   .   x   .   x   .
    .   .   .   .   .   .   .   .   .   .   .
    .   .   .   .   .   .   .   .   .   .   .
    .   x   .   x   .   .   .   x   .   x   .
    .   .   .   .   .   .   .   .   .   .   .
    .   .   .   .   .   .   .   .   .   .   .
    .   x   .   x   .   x   .   x   .   x   .
    .   .   .   .   .   .   .   .   .   .   .
    .   .   .   .   .   .   .   .   .   .   .
    """
    peaks = []
    row_ratio = int(f_img_shape[0] / target_shape[0] / 2 - 0.001)
    col_ratio = int(f_img_shape[1] / target_shape[1] / 2 - 0.001)
    center = (int(f_img_shape[0] / 2), int(f_img_shape[1] / 2))

    # center axis
    for r_r in range(1, row_ratio + 1):
        peaks.append((center[0] + r_r * target_shape[0], center[1]))
        peaks.append((center[0] - r_r * target_shape[0], center[1]))
    for c_r in range(1, col_ratio + 1):
        peaks.append((center[0], center[1] + c_r * target_shape[1]))
        peaks.append((center[0], center[1] - c_r * target_shape[1]))

    # quadrants
    for r_r in range(1, row_ratio + 1):
        for c_r in range(1, col_ratio + 1):
            # quadrant 4
            peaks.append((center[0] + r_r * target_shape[0], center[1] + c_r * target_shape[1]))
            # quadrant 3
            peaks.append((center[0] + r_r * target_shape[0], center[1] - c_r * target_shape[1]))
            # quadrant 2
            peaks.append((center[0] - r_r * target_shape[0], center[1] - c_r * target_shape[1]))
            # quadrant 1
            peaks.append((center[0] - r_r * target_shape[0], center[1] + c_r * target_shape[1]))
    return np.array(peaks)


def get_main_and_secondary_peaks(f_img_shape: typing.Tuple[int, int, int],
                                 target_shape: typing.Tuple[int, int, int]) -> np.ndarray:
    """
    Get indices of locations at which peaks are to be expected in an attack instance,
    plus one additional, secondary peak north, south, east, west of each main peak.
    """
    peaks = []
    row_ratio = int(f_img_shape[0] / target_shape[0] / 2)
    row_rest = f_img_shape[0] / target_shape[0] % 1
    col_ratio = int(f_img_shape[1] / target_shape[1] / 2)
    col_rest = f_img_shape[1] / target_shape[1] % 1
    center = (int(f_img_shape[0] / 2), int(f_img_shape[1] / 2))

    # quadrants
    for r_r in range(0, row_ratio + 2):
        for c_r in range(0, col_ratio + 2):
            # quadrant 4
            peaks.append((center[0] + r_r * target_shape[0], center[1] + c_r * target_shape[1]))
            # quadrant 3
            peaks.append((center[0] + r_r * target_shape[0], center[1] - c_r * target_shape[1]))
            # quadrant 2
            peaks.append((center[0] - r_r * target_shape[0], center[1] - c_r * target_shape[1]))
            # quadrant 1
            peaks.append((center[0] - r_r * target_shape[0], center[1] + c_r * target_shape[1]))

    additional_peaks = []
    for p in peaks:
        # north
        additional_peaks.append((p[0] - row_rest * target_shape[0], p[1]))
        # south
        additional_peaks.append((p[0] + row_rest * target_shape[0], p[1]))
        # west
        additional_peaks.append((p[0], p[1] - col_rest * target_shape[1]))
        # east
        additional_peaks.append((p[0], p[1] + col_rest * target_shape[1]))
    peaks = peaks + additional_peaks

    # filter peaks that are out of bound
    peaks = [p for p in peaks if not (p[0] < 0 or p[0] > f_img_shape[0] - 1) and not (p[1] < 0 or p[1] > f_img_shape[1] - 1)]

    # filter center peak
    peaks = [p for p in peaks if not (p[0] == center[0] and p[1] == center[1])]

    # remove identical peaks
    return np.unique(np.array(peaks), axis=0).astype("int")


def get_peaks_mask(f_img_shape: typing.Tuple[int, int, int],
                   target_shape: typing.Tuple[int, int, int],
                   peak_finding_method: typing.Callable = get_main_peaks,
                   sampling_radius: int = 0) -> np.ndarray:
    """
    Gets mask that defines areas around peaks based on method (main or main+secondary peaks) and sampling radius
    Cuts out area around center because there, we always have high intensities.
    """
    peaks = peak_finding_method(f_img_shape, target_shape)

    rows = f_img_shape[0]
    cols = f_img_shape[1]

    mask = np.zeros(f_img_shape, dtype=bool)

    if sampling_radius == 0:
        for p in peaks:
            mask[p[0], p[1]] = True
    else:
        for p in peaks:
            mask[max(p[0] - sampling_radius, 0):
                 min(p[0] + sampling_radius, rows), max(p[1] - sampling_radius, 0):
                                           min(p[1] + sampling_radius, cols)] = True

    # filter center
    center = (int(f_img_shape[0] / 2), int(f_img_shape[1] / 2))
    mask[center[0] - target_shape[0] // 2: center[0] + target_shape[0] // 2,
    center[1] - target_shape[1] // 2: center[1] + target_shape[1] // 2] = False

    return mask
