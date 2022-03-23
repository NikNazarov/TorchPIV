import numpy as np
import torch
import os
import re
from typing import Generator, Tuple
from torch.utils.data import Dataset
from scipy import interpolate
import cv2
import fastSubpixel
from time import time


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def load_pair(name_a: str, name_b: str, transforms) -> Tuple[torch.Tensor]:
    try:
        frame_b = cv2.imread(name_b, cv2.IMREAD_GRAYSCALE)
        frame_a = cv2.imread(name_a, cv2.IMREAD_GRAYSCALE)
    except FileNotFoundError:
        print("Invalid File Path!")
        return None
    if transforms:
        frame_a = transforms(frame_a)
        frame_b = transforms(frame_b)
    return frame_a, frame_b

class ToTensor:
    def __init__(self, dtype:  type) -> None:
        self.dtype = dtype
    def __call__(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=self.dtype)[None, ...]

class PIVDataset(Dataset):
    def __init__(self, folder, file_fmt, transform=None):
        self.transform = transform
        filenames = [os.path.join(folder, name) for name 
            in os.listdir(folder) if name.endswith(file_fmt)]
        filenames.sort(key=natural_keys)
        self.img_pairs = list(zip(filenames[::2], filenames[1::2]))
    def __len__(self):
        return len(self.img_pairs)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor]:        

        if torch.is_tensor(index):
            index = index.tolist()

        pair = self.img_pairs[index]
        img_b = cv2.imread(pair[1], cv2.IMREAD_GRAYSCALE)
        img_a = cv2.imread(pair[0], cv2.IMREAD_GRAYSCALE)
        if self.transform:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)
        
        return img_a, img_b

def moving_window_array(array: torch.Tensor, window_size, overlap) -> torch.Tensor:
    """
    This is a nice numpy trick. The concept of numpy strides should be
    clear to understand this code.

    Basically, we have a 2d array and we want to perform cross-correlation
    over the interrogation windows. An approach could be to loop over the array
    but loops are expensive in python. So we create from the array a new array
    with three dimension, of size (n_windows, window_size, window_size), in
    which each slice, (along the first axis) is an interrogation window.

    """
    shape = array.shape
    strides = (
        shape[-2]*shape[-1],
        shape[-1] * (window_size - overlap),
        (window_size - overlap),
        shape[-1],
        1
    )
    shape = (
        shape[0],
        int((shape[-2] - window_size) / (window_size - overlap)) + 1,
        int((shape[-1] - window_size) / (window_size - overlap)) + 1,
        window_size,
        window_size,
    )
    return torch.as_strided(
        array, size=shape, stride=strides 
    ).reshape((shape[0], shape[1]*shape[2], *shape[-2:]))

def correalte_fft(images_a, images_b) -> torch.Tensor:
    """Compute cross correlation based on fft method"""
    corr = torch.fft.fftshift(torch.fft.irfft2(torch.fft.rfft2(images_a).conj() *
                               torch.fft.rfft2(images_b)), dim=(-2, -1))
    return corr


def find_first_peak_position(corr: torch.Tensor) -> torch.Tensor:
    """Return Tensor (n, c, 2) of peak coordinates"""
    n, c, d, k = corr.shape
    m = corr.view(n,c, -1).argmax(-1, keepdim=True)
    return torch.cat((m // d, m % k), -1)

def interpolate_nan(
        vec: np.ndarray,
        method: str = 'linear',
        fill_value: int = 0
    ) -> np.ndarray:
    """
    :param vec (:, :): 2D field
    :param method: interpolation method, one of
        'nearest', 'linear', 'cubic'.
    :param fill_value: which value to use for filling up data outside the
        convex hull of known pixel values.
        Default is linear.
    :return: the image with missing values interpolated
    
    """
    if not np.isnan(vec).any():
        return vec
    mask = np.ma.masked_invalid(vec).mask
    h, w = vec.shape
    xx, yy = np.meshgrid(np.arange(0, w), np.arange(0, h))


    known_x = xx[~mask]
    known_y = yy[~mask]
    known_v = vec[~mask]
    missing_x = xx[mask]
    missing_y = yy[mask]


    interp_values = interpolate.griddata(
        (known_x, known_y), 
        known_v, 
        (missing_x, missing_y),
        method=method, fill_value=fill_value
    )
    interp_image = vec.copy()
    interp_image[missing_y, missing_x] = interp_values
    return interp_image

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

def interpolate_boarders(vec: np.ndarray) -> np.ndarray:
    if not np.isnan(vec).any():
        return vec
    nans, x = nan_helper(vec[0,:])
    vec[0,nans]   = np.interp(x(nans), x(~nans), vec[0,:][~nans])    
    nans, x = nan_helper(vec[-1,:])
    vec[-1,nans] = np.interp(x(nans), x(~nans), vec[-1,:][~nans])
    nans, x = nan_helper(vec[:,0])
    vec[nans,0]   = np.interp(x(nans), x(~nans), vec[:,0][~nans])
    nans, x = nan_helper(vec[:,-1])
    vec[nans,-1] = np.interp(x(nans), x(~nans), vec[:,-1][~nans])
    
    return vec

def c_correlation_to_displacement(
    corr: torch.Tensor, 
    n_rows, n_cols, interp_nan=True) -> Tuple[np.ndarray]:
    """
    Correlation maps are converted to displacement for each interrogation
    window using the convention that the size of the correlation map
    is 2N -1 where N is the size of the largest interrogation window
    (in frame B) that is called search_area_size
    Inputs:
        corr : 4D torch.Tesnsor [batch, channels, :, :]
            contains output of the fft_correlate_images
        n_rows, n_cols : number of interrogation windows, output of the
            get_field_shape
    """
    # iterate through interrogation widows and search areas
    eps = 1e-7
    n = corr.shape[0]
    first_peak = find_first_peak_position(corr)
    # center point of the correlation map
    default_peak_position = np.floor(np.array(corr[0, 0, :, :].shape)/2)
    corr += eps
    corr = corr.cpu().numpy()
    first_peak = first_peak.cpu().numpy()

    print(f"corr size {((corr.size * corr.itemsize) / 1024 / 1024):.2f} Mb", end=" ")
    

    temp = fastSubpixel.find_subpixel_position(corr, first_peak, n_rows, n_cols)
    peak = (np.array(temp).T - default_peak_position.T).T
    u, v = peak[0], peak[1]
    u, v = u.squeeze(), v.squeeze()
    u = interpolate_boarders(u)
    v = interpolate_boarders(v)
    if interp_nan:
        u = interpolate_nan(u)
        v = interpolate_nan(v)
    return u, v


def get_field_shape(image_size, search_area_size, overlap) -> Tuple:
    """Compute the shape of the resulting flow field.

    Given the image size, the interrogation window size and
    the overlap size, it is possible to calculate the number
    of rows and columns of the resulting flow field.

    Parameters
    ----------
    image_size: two elements tuple
        a two dimensional tuple for the pixel size of the image
        first element is number of rows, second element is
        the number of columns, easy to obtain using .shape

    search_area_size: tuple
        the size of the interrogation windows (if equal in frames A,B)
        or the search area (in frame B), the largest  of the two

    overlap: tuple
        the number of pixel by which two adjacent interrogation
        windows overlap.


    Returns
    -------
    field_shape : three elements tuple
        the shape of the resulting flow field
    """
    field_shape = (np.array(image_size) - search_area_size) // (
        search_area_size - overlap
    ) + 1
    return field_shape

def resize_iteration(arr: np.ndarray, iter: int):
    arr = cv2.resize(arr, None, fx=iter, fy=iter, interpolation=cv2.INTER_LINEAR)
    # Depricated method, may use later
    # arr = tinterpolate(torch.from_numpy(arr[None, None, ...]), scale_factor=2, mode='bilinear', align_corners=True).numpy()
    return arr.squeeze()

def extended_search_area_piv(
    frame_a,
    frame_b,
    window_size,
    overlap=0,
    dt=1,
    search_area_size=None,
) -> Tuple[np.ndarray]:
    """Standard PIV cross-correlation algorithm, with an option for
    extended area search that increased dynamic range. The search region
    in the second frame is larger than the interrogation window size in the
    first frame. ZERO ORDER!!!

    This is a pure python implementation of the standard PIV cross-correlation
    algorithm. It is a zero order displacement predictor, and no iterative
    process is performed.

    Parameters
    ----------
    frame_a : 2d torch.tensor
        an two dimensions array of integers containing grey levels of
        the first frame.

    frame_b : 2d torch.tensor
        an two dimensions array of integers containing grey levels of
        the second frame.

    window_size : int
        the size of the (square) interrogation window, [default: 32 pix].

    overlap : int
        the number of pixels by which two adjacent windows overlap
        [default: 16 pix].

    dt : float
        the time delay separating the two frames [default: 1.0].

    """

    # check the inputs for validity
    if search_area_size is None:
        search_area_size = window_size

    if overlap >= window_size:
        raise ValueError("Overlap has to be smaller than the window_size")

    if search_area_size < window_size:
        raise ValueError("Search size cannot be smaller than the window_size")
    if (window_size > frame_a.shape[-2]) or (window_size > frame_a.shape[-1]):
        raise ValueError("window size cannot be larger than the image")
    torch.cuda.synchronize()
    start = time()
    _, _, n_rows, n_cols = get_field_shape(frame_a.shape, search_area_size=search_area_size, overlap=overlap)
    aa = moving_window_array(frame_a, search_area_size, overlap)
    bb = moving_window_array(frame_b, search_area_size, overlap)
    torch.cuda.synchronize()
    print(f"moving window time {(time() - start):.3f} sec", end=' ')
    torch.cuda.synchronize()
    corr_time = time()
    corr = correalte_fft(aa, bb)
    torch.cuda.synchronize()
    print(f"correlation time {(time() - corr_time):.3f} sec")
    u, v = c_correlation_to_displacement(corr, n_rows, n_cols)
    return u, v

def get_coordinates(image_size, search_area_size, overlap):
    """Compute the x, y coordinates of the centers of the interrogation windows.
    the origin (0,0) is like in the image, top left corner
    positive x is an increasing column index from left to right
    positive y is increasing row index, from top to bottom


    Parameters
    ----------
    image_size: two elements tuple
        a two dimensional tuple for the pixel size of the image
        first element is number of rows, second element is
        the number of columns.

    search_area_size: int
        the size of the search area windows, sometimes it's equal to
        the interrogation window size in both frames A and B

    overlap: int = 0 (default is no overlap)
        the number of pixel by which two adjacent interrogation
        windows overlap.


    Returns
    -------
    x : 2d torch.tensor
        a two dimensional array containing the x coordinates of the
        interrogation window centers, in pixels.

    y : 2d torch.tensor
        a two dimensional array containing the y coordinates of the
        interrogation window centers, in pixels.

        Coordinate system 0,0 is at the top left corner, positive
        x to the right, positive y from top downwards, i.e.
        image coordinate system

    """

    # get shape of the resulting flow field
    field_shape = get_field_shape(image_size,
                                  search_area_size,
                                  overlap)

    # compute grid coordinates of the search area window centers
    # note the field_shape[1] (columns) for x
    x = (
        np.arange(field_shape[-1]) * (search_area_size - overlap)
        + (search_area_size) / 2.0
    )
    # note the rows in field_shape[0]
    y = (
        np.arange(field_shape[-2]) * (search_area_size - overlap)
        + (search_area_size) / 2.0
    )

    # moving coordinates further to the center, so that the points at the
    # extreme left/right or top/bottom
    # have the same distance to the window edges. For simplicity only integer
    # movements are allowed.
    x += (
        image_size[-1]
        - 1
        - ((field_shape[-1] - 1) * (search_area_size - overlap) +
            (search_area_size - 1))
    ) // 2
    y += (
        image_size[-2] - 1
        - ((field_shape[-2] - 1) * (search_area_size - overlap) +
           (search_area_size - 1))
    ) // 2

    # the origin 0,0 is at top left
    # the units are pixels

    return np.meshgrid(x, y)

def piv_iteration(
    frame_a: np.ndarray, 
    frame_b: np.ndarray, 
    x:       np.ndarray, 
    y:       np.ndarray, 
    u0:      np.ndarray, 
    v0:      np.ndarray, 
    wind_size: int, 
    device: torch.device):
    iter_proc = time()
    uin = np.rint(u0/2).astype(np.int64)
    vin = np.rint(v0/2).astype(np.int64)

    bb2 = torch.from_numpy(fastSubpixel.iter_displacement(frame_b, x, y, uin, vin, wind_size))
    aa2 = torch.from_numpy(fastSubpixel.iter_displacement(frame_a, x, y, -uin, -vin, wind_size))
    aa2, bb2 = aa2[None,...].to(device), bb2[None,...].to(device)
    corr = correalte_fft(aa2, bb2)
    du, dv = c_correlation_to_displacement(corr, x.shape[-2], x.shape[-1], interp_nan=True)
    du = du.squeeze()
    dv = dv.squeeze()
    v = 2*vin + dv
    u = 2*uin + du

    mask_u = (du > u0) * (np.rint(u0) > 0)
    mask_v = (dv > v0) * (np.rint(v0) > 0)
    v[mask_v] = v0[mask_v]
    u[mask_u] = u0[mask_u]
    torch.cuda.synchronize()
    print(f"Iteration finished in {(time() - iter_proc):.3f} sec", end=" ")
    return u, v

def calc_mean(v_list: list):
    np_list = np.stack(v_list, axis=0)
    return np.mean(np_list, axis=0).squeeze()

class OfflinePIV:
    def __init__(
        self, folder: str, 
        device: str,
        file_fmt: str, 
        wind_size: int, 
        overlap: int,
        iterations: int = 1,
        dt: int = 1,
        scale:float = 1.,
        resize: int = 2,
        iter_scale:float = 2.
                ) -> None:
        self._wind_size = wind_size
        self._overlap = overlap
        self._dt = dt
        self._iter = iterations
        self._iter_scale = iter_scale
        self._resize = resize
        self._scale = scale
        self._device = torch.device(device)
        self._batch_size = 1
        self._dataset = PIVDataset(folder, file_fmt, 
                       transform=ToTensor(dtype=torch.uint8)
                      )
        self.loader = torch.utils.data.DataLoader(self._dataset, 
            batch_size=self._batch_size, num_workers=0, pin_memory=True)
    def __len__(self) -> int:
        return len(self._dataset)

    def __call__(self) -> Generator:
        end_time = time() 
        for a, b in self.loader:
            print(f"Load time {(time() - end_time):.3f} sec", end=' ')
            start = time()
            a_gpu, b_gpu = a.to(self._device), b.to(self._device)
            print(f"Convert to {self._device} time {(time() - start):.3f} sec", end =' ')
            u, v = extended_search_area_piv(a_gpu, b_gpu, window_size=self._wind_size, 
                                            overlap=self._overlap, dt=1)
            x, y = get_coordinates(a.shape, self._wind_size, self._overlap)
            u = resize_iteration(u, iter=self._resize)
            v = resize_iteration(v, iter=self._resize)
            x = resize_iteration(x, iter=self._resize)
            y = resize_iteration(y, iter=self._resize)
            bn = b.numpy().squeeze()
            an = a.numpy().squeeze()
            wind_size = self._wind_size
            for _ in range(self._iter-1):
                wind_size = int(wind_size//self._iter_scale)
                u, v = piv_iteration(an, bn, x.astype(np.int64), y.astype(np.int64), u, v, wind_size, self._device)
            u =  np.flip(u, axis=0)
            v = -np.flip(v, axis=0)
            yield x, y, u, v
            end_time = time()
            print(f"Batch finished in {(end_time - start):.3f} sec")


 
