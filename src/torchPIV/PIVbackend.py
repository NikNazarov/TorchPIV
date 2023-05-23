import numpy as np
import torch
import os
import torch.nn.functional as F
from typing import Generator, Tuple
from torch.utils.data import Dataset
from scipy import interpolate
import cv2
from time import time
from torchPIV.PlotterFunctions import natural_keys

class DeviceMap:
    devicies = {
        torch.cuda.get_device_name(i) : torch.device(i)  
        for i in range(torch.cuda.device_count())
    }
    devicies["cpu"] = torch.device("cpu")


def free_cuda_memory():
    # torch.cuda.synchronize()
    if torch.cuda.is_available(): torch.cuda.empty_cache() 

def load_pair(name_a: str, name_b: str, transforms) -> Tuple[torch.Tensor]:
    """
    Helper method, Can be used later in right on fly version of PIV
    Reads image pair from disk as numpy array and performs transforms on it
    """
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
    """
    Basic transform class. Converts numpy array to torch.Tensor with dtype
    """
    def __init__(self, dtype:  type) -> None:
        self.dtype = dtype
    def __call__(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=self.dtype)

class PIVDataset(Dataset):
    def __init__(self, folder, file_fmt, transform=None):
        self.transform = transform
        filenames = [os.path.join(folder, name) for name 
            in os.listdir(folder) if name.endswith(file_fmt)]
        filenames.sort(key=natural_keys)
        self.img_pairs = list(zip(filenames[::2], filenames[1::2]))
        # self.img_pairs = list(zip(filenames[:-1], filenames[1:]))
    def __len__(self):
        return len(self.img_pairs)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor]:        

        if torch.is_tensor(index):
            index = index.tolist()

        pair = self.img_pairs[index]
        #imread function works only with latin file path
        img_b = cv2.imread(pair[1], cv2.IMREAD_GRAYSCALE)
        img_a = cv2.imread(pair[0], cv2.IMREAD_GRAYSCALE)
        if self.transform:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)
        
        return img_a, img_b


def biliniar_interpolation_CWS(array: torch.Tensor, grid:torch.Tensor, 
                                vel_x: torch.Tensor, vel_y: torch.Tensor) -> torch.Tensor:
    """
    PyTorch implementation of bilinear interpolation for CWS
    Key idea is to shift corresponding IW indices then perform torch.gather operation
    array: 2d torch.Tensor 
        an two dimensions array of integers containing grey levels of
        the frame.
    grid: 3d torch.Tensor
        [c, w, h] Tensor of image elments flatten indices to be shifted
    vel_x: 3d torch.Tensor
        [c, 1, 1] three dimensional tensor of torch.float32 x velocities in pixel units 
    vel_y: 3d torch.Tensor
        [c, 1, 1] three dimensional tensor of torch.float32 y velocities in pixel units 
    """
    frame_shape = array.shape
    grid_y, grid_x = grid // frame_shape[-1], grid % frame_shape[-1] 
    new_y, new_x = grid_y + vel_y, grid_x + vel_x

    up_x = torch.ceil(new_x).type(torch.int64)
    up_y = torch.ceil(new_y).type(torch.int64)
    down_x = torch.floor(new_x).type(torch.int64)
    down_y = torch.floor(new_y).type(torch.int64)
    mask = (up_x - down_x)*(up_y - down_y) == 0

    Q12 = up_y   * frame_shape[-1] + down_x
    Q11 = down_y * frame_shape[-1] + down_x
    Q22 = up_y   * frame_shape[-1] + up_x
    Q21 = down_y * frame_shape[-1] + up_x

    Q11.clamp_(0, array.numel()-1)
    Q21.clamp_(0, array.numel()-1)
    Q12.clamp_(0, array.numel()-1)
    Q22.clamp_(0, array.numel()-1)


    f_Q11 = torch.gather(array.view(-1), -1, Q11.view(-1)).reshape(grid.shape)
    f_Q12 = torch.gather(array.view(-1), -1, Q12.view(-1)).reshape(grid.shape)
    f_Q21 = torch.gather(array.view(-1), -1, Q21.view(-1)).reshape(grid.shape)
    f_Q22 = torch.gather(array.view(-1), -1, Q22.view(-1)).reshape(grid.shape)
    f_new = (
        f_Q11 * (up_x - new_x)   * (up_y - new_y)   + 
        f_Q21 * (new_x - down_x) * (up_y - new_y)   +
        f_Q12 * (up_x - new_x)   * (new_y - down_y) + 
        f_Q22 * (new_x - down_x) * (new_y - down_y)
    )
    f_new[mask] = f_Q11[mask].type(torch.float32)
    return f_new


def interpolation_DWS(array: torch.Tensor, grid:torch.Tensor, 
                    vel_x: torch.Tensor, vel_y: torch.Tensor) -> torch.Tensor:
    """
    PyTorch implementation of nearest interpolation for DWS
    Key idea is to shift corresponding IW indices then perform torch.gather operation
    array: 2d torch.Tensor 
        an two dimensions array of integers containing grey levels of
        the frame.
    grid: 3d torch.Tensor
        [c, w, h] Tensor of image elments flatten indices to be shifted
    vel_x: 3d torch.Tensor
        [c, 1, 1] three dimensional tensor of torch.int64 x velocities in pixel units 
    vel_y: 3d torch.Tensor
        [c, 1, 1] three dimensional tensor of torch.int64 y velocities in pixel units 
    """
    frame_shape = array.shape
    new_grid = grid + vel_y*frame_shape[-1] + vel_x
    new_grid.clamp_(0, array.numel()-1)
    f_new = torch.gather(array.view(-1), -1, new_grid.view(-1)).reshape(grid.shape)
    return f_new
    


def moving_window_array(array: torch.Tensor, window_size, overlap) -> torch.Tensor:
    """
    This is a nice numpy and torch trick. The concept of numpy strides should be
    clear to understand this code.

    Basically, we have a 2d array and we want to perform cross-correlation
    over the interrogation windows. An approach could be to loop over the array
    but loops are expensive in python. So we create from the array a new array
    with three dimension, of size (n_windows, window_size, window_size), in
    which each slice, (along the first axis) is an interrogation window.

    """
    shape = array.shape
    strides = (
        shape[-1] * (window_size - overlap),
        (window_size - overlap),
        shape[-1],
        1
    )
    shape = (
        int((shape[-2] - window_size) / (window_size - overlap)) + 1,
        int((shape[-1] - window_size) / (window_size - overlap)) + 1,
        window_size,
        window_size,
    )
    return torch.as_strided(
        array, size=shape, stride=strides 
    ).reshape(-1, window_size, window_size)

def correalte_fft(images_a: torch.Tensor, images_b: torch.Tensor) -> torch.Tensor:
    """
    Compute cross correlation based on fft method
    Between two torch.Tensors of shape [c, width, height]
    fft performed over last two dimensions of tensors
    """
    corr = torch.fft.fftshift(torch.fft.irfft2(torch.fft.rfft2(images_a).conj() *
                               torch.fft.rfft2(images_b)), dim=(-2, -1))
    return corr


def find_first_peak_position(corr: torch.Tensor) -> torch.Tensor:
    """Return Tensor (c, 2) of peak coordinates"""
    c, d, k = corr.shape
    m = corr.view(c, -1).argmax(-1, keepdim=True)
    return torch.cat((m // d, m % k), -1)

def getPixelsForInterp(img): 
    """
    Calculates a mask of pixels neighboring invalid values - 
        to use for interpolation. 
    """
    # mask invalid pixels
    invalid_mask = np.isnan(img)
    # plt.imshow(invalid_mask)
    # plt.show()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    #dilate to mark borders around invalid regions
    dilated_mask = cv2.dilate(invalid_mask.astype('uint8'), kernel, 
                        borderType=cv2.BORDER_CONSTANT, borderValue=int(0))
    # pixelwise "and" with valid pixel mask (~invalid_mask)
    masked_for_interp = dilated_mask *  ~invalid_mask
    return masked_for_interp.astype('bool'), invalid_mask

def fillMissingValues(target_for_interp,
                      interpolator=interpolate.LinearNDInterpolator):
    '''
    Interpolates missing values in matrix using bilinear interpolation
    as deafoult
    target_for_interp: np.ndarray
    interpolator: scipy interpolator class, default LinearNDInterpolator
    ''' 
    

    # Mask pixels for interpolation
    mask_for_interp, invalid_mask = getPixelsForInterp(target_for_interp)
    # Interpolate only holes, only using these pixels
    points = np.argwhere(mask_for_interp)
    values = target_for_interp[mask_for_interp]
    if points.size:
        interp = interpolator(points, values)
        target_for_interp[invalid_mask] = interp(np.argwhere(invalid_mask))
    else:
        print("Warning! to many false vectors")
    return target_for_interp


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
    if not nans.all():
        vec[0,nans]   = np.interp(x(nans), x(~nans), vec[0,:][~nans])    
    nans, x = nan_helper(vec[-1,:])
    if not nans.all():    
        vec[-1,nans] = np.interp(x(nans), x(~nans), vec[-1,:][~nans])
    nans, x = nan_helper(vec[:,0])
    if not nans.all():
        vec[nans,0]   = np.interp(x(nans), x(~nans), vec[:,0][~nans])
    nans, x = nan_helper(vec[:,-1])
    if not nans.all():
        vec[nans,-1] = np.interp(x(nans), x(~nans), vec[:,-1][~nans])
    
    return vec

def peak2peak_secondpeak(
    corr: torch.Tensor, imax: torch.Tensor, 
    wind: int=2) -> torch.Tensor:

    c, d, k = corr.shape
    cor = corr.view(c, -1)
    for i in range(-wind, wind+1):
        for j in range(-wind, wind+1):
            ids = imax + i + k * j
            torch.clamp_(ids, 0, k*d-1)
            cor.scatter_(-1, ids, 0.0)
    second_max = cor.argmax(-1, keepdim=True)
    return second_max

def correlation_to_displacement(
    corr: torch.Tensor,
    n_rows, n_cols,
    validate: bool=True,
    val_ratio=1.2, 
    validation_window=3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Correlation maps are converted to displacement for 
    each interrogation window
    Inputs:
        corr : 3D torch.Tesnsor [channels, :, :]
            contains output of the correlate_fft
        n_rows, n_cols : number of interrogation windows, output of the
            get_field_shape
        validate: bool Flag for validation 
        val_ratio: int = 1.2 peak2peak validation coefficient
        validation_window: int = 3 half of peak2peak validation window
    """
    validation_mask = None
    c, d, k = corr.shape
    eps = 1e-7
    corr += eps
    cor = corr.view(c, -1).type(torch.float64)
    m = corr.view(c, -1).argmax(-1, keepdim=True)

    left = m + 1
    right = m - 1
    top = m + k 
    bot = m - k
    left[left >= k*d - 1] = m[left >= k*d - 1]
    right[right <= 0] = m[right <= 0]
    top[top >= k*d - 1] = m[top >= k*d - 1]
    bot[bot <= 0] = m[bot <= 0]

    cm = torch.gather(cor, -1, m)
    cl = torch.gather(cor, -1, left)
    cr = torch.gather(cor, -1, right)
    ct = torch.gather(cor, -1, top)
    cb = torch.gather(cor, -1, bot)
    nom1 = torch.log(cr) - torch.log(cl) 
    den1 = 2 * (torch.log(cl) + torch.log(cr)) - 4 * torch.log(cm) 
    nom2 = torch.log(cb) - torch.log(ct) 
    den2 = 2 * (torch.log(cb) + torch.log(ct)) - 4 * torch.log(cm) 

    m2d = torch.cat((m // d, m % k), -1)
    
    v = m2d[:, 0][:, None] + nom2/den2
    u = m2d[:, 1][:, None] + nom1/den1

    if validate:
        m2 = peak2peak_secondpeak(corr, m, validation_window)
        validation_mask = (cm / torch.gather(cor, -1, m2)) < val_ratio
        validation_mask[(left >= k*d - 1) * (right <= 0) * (top >= k*d - 1) * (bot <= 0)] = True
        validation_mask = validation_mask.reshape(n_rows, n_cols).cpu().numpy()

    default_peak_position = corr.shape[-2:]
    v = v - int(default_peak_position[0] / 2)
    u = u - int(default_peak_position[1] / 2)
    torch.nan_to_num_(v)
    torch.nan_to_num_(u)
    u = u.reshape(n_rows, n_cols).cpu().numpy()
    v = v.reshape(n_rows, n_cols).cpu().numpy()
    return u, v, validation_mask


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


def extended_search_area_piv(
    frame_a,
    frame_b,
    window_size=32,
    overlap=0,
    validate: bool = False,
    validation_ratio:float = 1.2
) -> Tuple[np.ndarray, ...]:
    """Standard PIV cross-correlation algorithm, with an option for
    extended area search that increased dynamic range. The search region
    in the second frame is larger than the interrogation window size in the
    first frame. ZERO ORDER!!!

    This is a pure python implementation of the standard PIV cross-correlation
    algorithm. It is a zero order displacement predictor, and no iterative
    process is performed.

    Parameters
    ----------
    frame_a : 2d torch.Tensor
        an two dimensions array of integers containing grey levels of
        the first frame.

    frame_b : 2d torch.Tensor
        an two dimensions array of integers containing grey levels of
        the second frame.

    window_size : int
        the size of the (square) interrogation window, [default: 32 pix].

    overlap : int
        the number of pixels by which two adjacent windows overlap
        [default: 0 pix].

    validate: bool=False
        peak2peak validation flag

    validation_ratio: float=1.2
        peak2peak validation ratio

    """

    # frame_a, frame_b = frame_a.to(device), frame_b.to(device)

    if overlap >= window_size:
        raise ValueError("Overlap has to be smaller than the window_size")

    if (window_size > frame_a.shape[-2]) or (window_size > frame_a.shape[-1]):
        raise ValueError("window size cannot be larger than the image")
    n_rows, n_cols = get_field_shape(frame_a.shape, search_area_size=window_size, overlap=overlap)
    x, y = get_coordinates(frame_a.shape, window_size, overlap)
    aa = moving_window_array(frame_a, window_size, overlap)
    bb = moving_window_array(frame_b, window_size, overlap)
    # Normalize Intesity
    # aa = aa / torch.mean(aa, (-2,-1), dtype=torch.float32, keepdim=True)
    # bb = bb / torch.mean(bb, (-2,-1), dtype=torch.float32, keepdim=True)
    
    corr = correalte_fft(aa, bb)
    # Normalize correlation
    corr = corr - torch.amin(corr, (-2, -1), keepdim=True)
    u, v, validation_mask = correlation_to_displacement(corr, n_rows, n_cols, validate, val_ratio=validation_ratio)
    return u, v, x, y, validation_mask

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
        np.arange(field_shape[-1], dtype=np.int32) * (search_area_size - overlap)
        + (search_area_size) / 2.0
    )
    # note the rows in field_shape[0]
    y = (
        np.arange(field_shape[-2], dtype=np.int32) * (search_area_size - overlap)
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

class piv_iteration_CWS_Fast:
    def __init__(self, frame_shape, wind_size, overlap, device) -> None:
        self.n_rows, self.n_cols = get_field_shape(frame_shape, search_area_size=wind_size, overlap=overlap)
        
        self.x, self.y = get_coordinates(frame_shape, wind_size, overlap)
        self.slice_x, self.slice_y = self.x[0,:], self.y[:,0]
        
        self.idx = moving_window_array(
            torch.arange(0, frame_shape[-2]*frame_shape[-1], dtype=torch.int64, device=device).reshape(frame_shape), 
            wind_size, overlap
            )
        
        affine_transform = torch.tensor([[1., 0., 0.],
                                        [0., 1., 0.]]).to(device)
        self.affine_transform = affine_transform.repeat(self.n_rows * self.n_cols, 1, 1)

    def __call__(self,
    frame_a: torch.Tensor,
    frame_b: torch.Tensor, 
    x0: np.ndarray,  
    y0: np.ndarray,  
    u0: np.ndarray,  
    v0: np.ndarray,
    validation_mask: np.ndarray,
    wind_size: int, 
    overlap: int,
    device: torch.device) -> tuple[np.ndarray, ...]:
        iter_proc = time()
        spline_u = interpolate.RectBivariateSpline(y0[:,0], x0[0,:], u0)
        spline_v = interpolate.RectBivariateSpline(y0[:,0], x0[0,:], v0)
        u0 = spline_u(self.slice_y, self.slice_x)
        v0 = spline_v(self.slice_y, self.slice_x)
        validate = False
        if validation_mask is not None:
            validate = True
            spline_val = interpolate.RectBivariateSpline(y0[:,0], x0[0,:], validation_mask)
            val = spline_val(self.slice_y, self.slice_x) >= .5
            u0[val] = 0.0
            v0[val] = 0.0
        uflat = torch.from_numpy(u0.flatten()).to(device)
        vflat = torch.from_numpy(v0.flatten()).to(device)
        frame_a, frame_b = frame_a.to(device), frame_b.to(device)
        aa = moving_window_array(frame_a, wind_size, overlap)[:,None,...].float()
        bb = moving_window_array(frame_b, wind_size, overlap)[:,None,...].float()
        
    
        self.affine_transform[:, 1, 2] = -vflat/wind_size
        self.affine_transform[:, 0, 2] = -uflat/wind_size
        grid = F.affine_grid(self.affine_transform, aa.size())
        aa = F.grid_sample(aa, grid, mode='bicubic',padding_mode="border")

        self.affine_transform[:, 1, 2] = vflat/wind_size
        self.affine_transform[:, 0, 2] = uflat/wind_size
        grid = F.affine_grid(self.affine_transform, bb.size())
        bb = F.grid_sample(bb, grid, mode='bicubic',padding_mode="border")

        # Normalize Intesity
        aa = aa / torch.mean(aa, (-2,-1), dtype=torch.float32, keepdim=True)
        bb = bb / torch.mean(bb, (-2,-1), dtype=torch.float32, keepdim=True)
        
        corr = correalte_fft(aa, bb)
        corr = corr - torch.amin(corr, (-2, -1), keepdim=True)
        du, dv, val = correlation_to_displacement(corr.squeeze(), self.n_rows, self.n_cols, validate)

        v = v0 + dv
        u = u0 + du

        mask_u = (du > u0) * (np.rint(u0) > 0)
        mask_v = (dv > v0) * (np.rint(v0) > 0)
        if val is not None:
            mask_u[val] = True
            mask_v[val] = True

        v[mask_v] = v0[mask_v]
        u[mask_u] = u0[mask_u]
        print(f"Iteration finished in {(time() - iter_proc):.3f} sec", end=" ")
        return u, v, self.x, self.y, val

class piv_iteration_CWS:
    def __init__(self, frame_shape, wind_size, overlap, device) -> None:
        self.n_rows, self.n_cols = get_field_shape(frame_shape, search_area_size=wind_size, overlap=overlap)
        self.device = device
        self.x, self.y = get_coordinates(frame_shape, wind_size, overlap)
        self.slice_x, self.slice_y = self.x[0,:], self.y[:,0]
        
        self.idx = moving_window_array(
            torch.arange(0, frame_shape[-2]*frame_shape[-1], dtype=torch.int64, device=device).reshape(frame_shape), 
            wind_size, overlap
            )


    def __call__(self,
    frame_a: torch.Tensor,
    frame_b: torch.Tensor, 
    x0: np.ndarray,  
    y0: np.ndarray,  
    u0: np.ndarray,  
    v0: np.ndarray,
    validation_mask: np.ndarray) -> tuple[np.ndarray, ...]:
        
        iter_proc = time()
        spline_u = interpolate.RectBivariateSpline(y0[:,0], x0[0,:], u0)
        spline_v = interpolate.RectBivariateSpline(y0[:,0], x0[0,:], v0)

        u0 = spline_u(self.slice_y, self.slice_x)
        v0 = spline_v(self.slice_y, self.slice_x)
        u2 = u0 / 2 
        v2 = v0 / 2 
        validate = False
        if validation_mask is not None:
            validate = True
            spline_val = interpolate.RectBivariateSpline(y0[:,0], x0[0,:], validation_mask)
            val = spline_val(self.slice_y, self.slice_x) >= .5
            u0[val] = 0.0
            v0[val] = 0.0
        v2t = torch.from_numpy(v2).type(torch.float32).to(self.device)
        u2t = torch.from_numpy(u2).type(torch.float32).to(self.device)
        v2t = v2t.view(-1)[..., None, None]
        u2t = u2t.view(-1)[..., None, None]

        frame_a, frame_b = frame_a.to(self.device), frame_b.to(self.device)
        aa = biliniar_interpolation_CWS(frame_a, self.idx, -u2t, -v2t)
        bb = biliniar_interpolation_CWS(frame_b, self.idx, u2t, v2t)

        corr = correalte_fft(aa, bb)
        corr = corr - torch.amin(corr, (-2, -1), keepdim=True)

        du, dv, val = correlation_to_displacement(corr, self.n_rows, self.n_cols, validate)

        v = 2*v2 + dv
        u = 2*u2 + du

        mask_u = (du > u0) * (np.rint(u0) > 0)
        mask_v = (dv > v0) * (np.rint(v0) > 0)
        if val is not None:
            mask_u[val] = True
            mask_v[val] = True

        v[mask_v] = v0[mask_v]
        u[mask_u] = u0[mask_u]
        print(f"Iteration finished in {(time() - iter_proc):.3f} sec", end=" ")
        return u, v, self.x, self.y, val



class piv_iteration_DWS:
    def __init__(self, frame_shape, wind_size, overlap, device) -> None:
        self.n_rows, self.n_cols = get_field_shape(frame_shape, search_area_size=wind_size, overlap=overlap)
        self.device = device
        self.x, self.y = get_coordinates(frame_shape, wind_size, overlap)
        self.slice_x, self.slice_y = self.x[0,:], self.y[:,0]
        
        self.idx = moving_window_array(
            torch.arange(0, frame_shape[-2]*frame_shape[-1], dtype=torch.int64, device=device).reshape(frame_shape), 
            wind_size, overlap
            )
        
    
    def __call__(self,
    frame_a: torch.Tensor, 
    frame_b: torch.Tensor, 
    x0:       np.ndarray, 
    y0:       np.ndarray, 
    u0:      np.ndarray, 
    v0:      np.ndarray,
    validation_mask: np.ndarray)->tuple[np.ndarray, ...]:

        iter_proc = time()


        spline_u = interpolate.RectBivariateSpline(y0[:,0], x0[0,:], u0)
        spline_v = interpolate.RectBivariateSpline(y0[:,0], x0[0,:], v0)

        u0 = spline_u(self.slice_y, self.slice_x)
        v0 = spline_v(self.slice_y, self.slice_x)
        validate = False
        if validation_mask is not None:
            validate = True
            spline_val = interpolate.RectBivariateSpline(y0[:,0], x0[0,:], validation_mask)
            val = spline_val(self.slice_y, self.slice_x) >= .5
            u0[val] = 0.0
            v0[val] = 0.0

        vin = v0/2
        uin = u0/2
        v2  = np.rint(vin)
        u2  = np.rint(uin)

        v2t = torch.from_numpy(v2).to(self.device).type(torch.int64)
        u2t = torch.from_numpy(u2).to(self.device).type(torch.int64)
        v2t = v2t.view(-1)[..., None, None]
        u2t = u2t.view(-1)[..., None, None]
        frame_a, frame_b = frame_a.to(self.device), frame_b.to(self.device)
        aa = interpolation_DWS(frame_a, self.idx, -u2t, -v2t)
        bb = interpolation_DWS(frame_b, self.idx, u2t, v2t)

        corr = correalte_fft(aa, bb)
        corr = corr - torch.amin(corr, (-2, -1), keepdim=True)

        du, dv, val = correlation_to_displacement(corr, self.n_rows, self.n_cols, validate)

        v = 2*np.rint(v2) + dv
        u = 2*np.rint(u2) + du

        mask_u = (du > u0) * (np.rint(u0) > 0)
        mask_v = (dv > v0) * (np.rint(v0) > 0)
        if val is not None:
            mask_u[val] = True
            mask_v[val] = True

        v[mask_v] = v0[mask_v]
        u[mask_u] = u0[mask_u]
        print(f"Iteration finished in {(time() - iter_proc):.3f} sec", end=" ")
        return u, v, self.x, self.y, val

class IterModMap:
    functions = {
        "DWS": piv_iteration_DWS,
        "CWS": piv_iteration_CWS
    }

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
        iter_mod: str="DWS",
        dt: int = 1,
        scale:float = 1.,
        iter_scale:float = 2.
                ) -> None:
        self._wind_size = wind_size
        self._overlap = overlap
        self._dt = dt
        self._iter = iterations
        self._iter_scale = iter_scale
        self._scale = scale
        
        self._device = DeviceMap.devicies[device]
        self._dataset = PIVDataset(folder, file_fmt, 
                       transform=ToTensor(dtype=torch.uint8)
                      )
        self._iter_function = IterModMap.functions[iter_mod]
        if not self:
            return
        frame_a, frame_b = self._dataset[0]
        self._iter_functions = []
        for _ in range(self._iter-1):
            wind_size = int(wind_size//self._iter_scale)
            overlap = int(overlap//self._iter_scale)  
            self._iter_functions.append(self._iter_function(frame_a.shape, wind_size, overlap, self._device))
    def __len__(self) -> int:
        return len(self._dataset)

    def __call__(self) -> Generator:
        loader = torch.utils.data.DataLoader(self._dataset, 
            batch_size=None, num_workers=0, pin_memory=True)

        end_time = time()

        for a, b in loader:

            print(f"Load time {(time() - end_time):.3f} sec", end=' ')
            start = time()
            a, b = a.to(self._device), b.to(self._device)
            u, v, x, y, val = extended_search_area_piv(a, b, window_size=self._wind_size, 
                                            overlap=self._overlap, validate=True)

            wind_size = self._wind_size
            overlap = self._overlap
            for iter in range(self._iter-1):
                wind_size = int(wind_size//self._iter_scale)
                overlap = int(overlap//self._iter_scale)                    
                u, v, x, y, val = self._iter_functions[iter](a, b, x, y, u, v, val)

            if val is not None:
                u[val] = np.nan
                v[val] = np.nan
                u = interpolate_boarders(u)
                v = interpolate_boarders(v)
                u = fillMissingValues(u)
                v = fillMissingValues(v)
            
            u =  np.flip(u, axis=0)
            v = -np.flip(v, axis=0)

            yield x, y, u, v
            end_time = time()
            print(f"Batch finished in {(end_time - start):.3f} sec")


class OnlinePIV:
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
        
        self._device = DeviceMap.devicies[device]