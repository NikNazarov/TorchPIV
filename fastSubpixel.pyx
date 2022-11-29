import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport log, isnan, floor, round
from cython.parallel import prange
from cython.view cimport array as cvarray

#Test function
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef float sum(float[:] arr):
    cdef int i
    cdef float res = 0
    for i in range(arr.shape[0]):
        res += arr[i]
    return res


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
def find_subpixel_position(
    float[:,:,:] corr, 
    long long[:,:] first_peak, 
    int n_rows, int n_cols):
    """ 
        Find subpixel position gaussian method
        Input params:
        corr[chan, :, :] float32 
        first_peak[chan, 2] long long - peak positions
        n_rows int - velocity field shape[0]
        n_cols int - velocity field shape[1]
        Returns:
        tuple[np.ndarray] u, v [: , :] double
    """
    cdef int k, m, n_channels, peak_i, peak_j, chan_n, i
    cdef float c, cl, cr, cd, cu, nom1, nom2, den1, den2

    n_channels = corr.shape[0] 
    bounds_h = corr.shape[-2]
    bounds_w = corr.shape[-1]

    cdef double[:,:] u = np.empty((n_rows, n_cols), dtype=np.double)
    cdef double[:,:] v = np.empty((n_rows, n_cols), dtype=np.double)

    
    # Find subpixel fields, gaussian method
    for k in range(n_rows):
        for m in range(n_cols):
            chan_n = k*n_cols+m
            peak_i = first_peak[chan_n, 0]
            peak_j = first_peak[chan_n, 1]  
            if ((peak_i == 0) | (peak_i == corr.shape[-2]-1) |
                    (peak_j == 0) | (peak_j == corr.shape[-1]-1)):
                v[k, m] = float("nan")
                u[k, m] = float("nan")
                continue
            
            c = corr[chan_n, peak_i, peak_j]
            cl = corr[chan_n, peak_i - 1, peak_j]
            cr = corr[chan_n, peak_i + 1, peak_j]
            cd = corr[chan_n, peak_i, peak_j - 1]
            cu = corr[chan_n, peak_i, peak_j + 1]
            nom1 = log(cl) - log(cr)
            den1 = 2 * log(cl) - 4 * log(c) + 2 * log(cr)
            nom2 = log(cd) - log(cu)
            den2 = 2 * log(cd) - 4 * log(c) + 2 * log(cu)

            v[k, m] = peak_i + nom1 / den1
            u[k, m] = peak_j + nom2 / den2

    for i in range(3):
        _replace_nans(v)
        _replace_nans(u)
     
    return np.asarray(u), np.asarray(v)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef void _replace_nans(double[:,:] vec):

    """ 
        Raplace NaNs by averaging neighborhood
        Input params:
        vec[:,:] double 
        Returns:
        vec[:,:] double
    """

    cdef double neighbours[8]
    cdef int k, m, n_rows, n_cols, i, nan_count
    n_rows    = vec.shape[0]
    n_cols    = vec.shape[1]
    for k in range(1, n_rows - 1):
        for m in range(1, n_cols - 1):
            if not isnan(vec[k, m]):
                continue
            
            neighbours[0] = vec[k, m-1]
            neighbours[1] = vec[k+1, m-1]
            neighbours[2] = vec[k-1, m-1]
            neighbours[3] = vec[k-1, m]
            neighbours[4] = vec[k+1, m]
            neighbours[5] = vec[k, m+1]
            neighbours[6] = vec[k+1, m+1]
            neighbours[7] = vec[k-1, m+1]
            nan_count = 0
            for i in range(8):
                if isnan(neighbours[i]):
                    nan_count += 1
            if nan_count < 3:
                vec[k, m] = 0
                for i in range(8):
                    if not isnan(neighbours[i]): 
                        vec[k, m] += neighbours[i]
                vec[k, m] = vec[k, m] / (8 - nan_count)



@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
def replace_nans(double[:,:] vec):

    """ 
        Raplace NaNs by averaging neighborhood
        Input params:
        vec[:,:] double 
        Returns:
        vec[:,:] double
    """

    cdef double neighbours[8]
    cdef double* weights = [1., 0.7071, 0.7071, 1., 1., 1., 0.7071, 0.7071]
    cdef double weight
    cdef int k, m, n_rows, n_cols, i, nan_count, whole_nans, iterations
    n_rows    = vec.shape[0]
    n_cols    = vec.shape[1]
    iterations = 0
    while True:
        whole_nans = 0
        for k in range(1, n_rows - 1):
            for m in range(1, n_cols - 1):
                if not isnan(vec[k, m]):
                    continue
                whole_nans += 1
                neighbours[0] = vec[k, m-1]
                neighbours[1] = vec[k+1, m-1]
                neighbours[2] = vec[k-1, m-1]
                neighbours[3] = vec[k-1, m]
                neighbours[4] = vec[k+1, m]
                neighbours[5] = vec[k, m+1]
                neighbours[6] = vec[k+1, m+1]
                neighbours[7] = vec[k-1, m+1]
                nan_count = 0
                for i in range(8):
                    if isnan(neighbours[i]):
                        nan_count += 1
                if nan_count < 3:
                    vec[k, m] = 0.
                    weight = 0.
                    for i in range(8):
                        if not isnan(neighbours[i]): 
                            vec[k, m] += neighbours[i] * weights[i]
                            weight += weights[i]
                    if (weight + 1e-3) > 0:  
                        vec[k, m] = vec[k, m] / weight
        iterations += 1
        if whole_nans == 0 or iterations > 100:
            break
    return np.asarray(vec)



@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function 
cdef void _mem_view_sum(
    unsigned char[:, :] image_a,
    unsigned char[:, :] image_b,
    double a, double b
    ) nogil:
    cdef:
        int i, j
        double value
    for i in range(image_a.shape[0]):
        for j in range(image_b.shape[1]):
            value = a * image_a[i, j] + b * image_b[i, j]
            if value > 255:
                value = 255
            value = round(value)
            image_a[i, j] = <unsigned char>value


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function 
@cython.cdivision(True)
def iter_displacement_CWS( 
    unsigned char[:, :] image_b,
    int[:, :] x,
    int[:, :] y,
    double[:, :] dx,
    double[:, :] dy,
    int wind_size
    ):
    cdef:
        int i, j, wind_id, old_y, old_x, bound_y_right, bound_y_left, bound_x_right, bound_x_left, new_x, new_y
        double dxi, dyi, rdx, rdy
        int wind_half = wind_size//2
        unsigned char[:,:,:] moving_window_array = np.empty((x.shape[0]*x.shape[1], wind_size, wind_size), dtype=np.uint8)
        bint xvalid, yvalid
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            wind_id = i*x.shape[1]+j
            old_x = x[i, j]
            old_y = y[i, j]
            # Check boundaries before displacement 
            bound_x_left = old_x - wind_half
            bound_x_right = old_x + wind_half
            bound_y_left = old_y - wind_half
            bound_y_right = old_y + wind_half
            if bound_x_left < 0:
                old_x = old_x - bound_x_left
            if bound_x_right > image_b.shape[1]:
                old_x = old_x - (bound_x_right - x.shape[1])
            if bound_y_left < 0:
                old_y = old_y - bound_y_left
            if bound_y_right > image_b.shape[0]:
                old_y = old_y - (bound_y_right - x.shape[0])

            dxi = dx[i, j]
            dyi = dy[i, j]
            new_x = <int>round(old_x + dxi)
            new_y = <int>round(old_y + dyi)
            rdx = (old_x + dxi) % 1
            rdy = (old_y + dyi) % 1
            # Check boundaries after displacement 
            xvalid = (
                (0 <= (new_x - wind_half)) *
                (image_b.shape[1] >= (new_x + wind_half + 1))
            )

            yvalid = (
                (0 <= (new_y - wind_half)) *
                (image_b.shape[0] >= (new_y + wind_half + 1))
            )
            if not xvalid:
                new_x = old_x
            if not yvalid:
                new_y = old_y
            _mem_view_sum(
                image_b[(new_y-wind_half):(new_y+wind_half),(new_x-wind_half):(new_x+wind_half)],
                image_b[(new_y+1-wind_half):(new_y+wind_half+1), (new_x-wind_half):(new_x+wind_half)],
                rdy, 1-rdy
                )
            _mem_view_sum(
                image_b[(new_y-wind_half):(new_y+wind_half),(new_x-wind_half):(new_x+wind_half)],
                image_b[(new_y-wind_half):(new_y+wind_half), (new_x+1-wind_half):(new_x+1+wind_half)],
                rdx, 1-rdx
                )
            # print(np.asarray(image_b[(new_y-wind_half):(new_y+wind_half),(new_x-wind_half):(new_x+wind_half)]))
            moving_window_array[wind_id, :, :] = image_b[(new_y-wind_half):(new_y+wind_half), (new_x-wind_half):(new_x+wind_half)]
    return np.asarray(moving_window_array)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function 
def iter_displacement_DWS( 
    const unsigned char[:, :] image_b,
    int[:, :] x,
    int[:, :] y,
    double[:, :] dx,
    double[:, :] dy,
    int wind_size
    ):
    cdef:
        int i, j, new_x, new_y, wind_id, old_y, old_x, bound_y_right, bound_y_left, bound_x_right, bound_x_left
        double dxi, dyi
        int wind_half = wind_size//2
        unsigned char[:,:,:] moving_window_array = np.empty((x.shape[0]*x.shape[1], wind_size, wind_size), dtype=np.uint8)
        bint xvalid, yvalid
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            wind_id = i*x.shape[1]+j
            old_x = x[i, j]
            old_y = y[i, j]
            # Check boundaries before displacement 
            bound_x_left = old_x - wind_half
            bound_x_right = old_x + wind_half
            bound_y_left = old_y - wind_half
            bound_y_right = old_y + wind_half
            if bound_x_left < 0:
                old_x = old_x - bound_x_left
            if bound_x_right > image_b.shape[1]:
                old_x = old_x - (bound_x_right - x.shape[1])
            if bound_y_left < 0:
                old_y = old_y - bound_y_left
            if bound_y_right > image_b.shape[0]:
                old_y = old_y - (bound_y_right - x.shape[0])

            dxi = dx[i, j]
            dyi = dy[i, j]
            if isnan(dxi) or isnan(dyi):
                new_x = old_x
                new_y = old_y
            else:
                new_x = <int>round(old_x + dxi)
                new_y = <int>round(old_y + dyi)
            # Check boundaries after displacement 
            xvalid = (
                (0 <= (new_x - wind_half)) *
                (image_b.shape[1] >= (new_x + wind_half))
            )

            yvalid = (
                (0 <= (new_y - wind_half)) *
                (image_b.shape[0] >= (new_y + wind_half))
            )
            if not xvalid:
                new_x = old_x
            if not yvalid:
                new_y = old_y
            moving_window_array[wind_id, :, :] = image_b[(new_y-wind_half):(new_y+wind_half), (new_x-wind_half):(new_x+wind_half)]
    return np.asarray(moving_window_array)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function 
def moving_window( 
    float[:, :] image_a,
    long long[:, :] x,
    long long[:, :] y,
    int wind_size
    ):
    cdef int i, j, new_x, new_y, wind_id
    cdef int wind_half = wind_size//2
    cdef float[:,:,:] moving_window_array = np.empty((x.shape[0]*x.shape[1], wind_size, wind_size), dtype=np.float32)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            wind_id = i*x.shape[1]+j
            new_x = x[i, j]
            new_y = y[i, j]
            moving_window_array[wind_id, :, :] = image_a[(new_y-wind_half):(new_y+wind_half), (new_x-wind_half):(new_x+wind_half)]
    return np.asarray(moving_window_array)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function 
cdef float[:,:,:] peak2peak_valid_mask(
    float[:,:,:] corr, 
    long long[:,:] first_peak, 
    int width):
    '''
        Create a masked view of the corr
        Input params:
        corr[chan, :, :] float32 
        first_peak[chan, 2] long long - peak positions
        width int - ignore window size
    -------
        Returns:
        masked corr
    '''

    cdef int window_h = corr.shape[-2]
    cdef int window_w = corr.shape[-1]
    cdef int c, i, j, i_ini, i_fin, j_ini, j_fin
    # create a masked view of the corr

        # for c in prange(corr.shape[1], nogil=True, num_threads=4):
    for c in range(corr.shape[0]):
        i = first_peak[c, 0]
        j = first_peak[c, 1]
        i_ini = i - width
        i_fin = i + width + 1
        j_ini = j - width
        j_fin = j + width + 1
        if i_ini < 0:
            i_ini = 0
        if i_fin > window_h:
            i_fin = window_h
        if j_ini < 0:
            j_ini = 0
        if j_fin > window_w:
            i_fin = window_w 
        corr[c, i_ini:i_fin, j_ini:j_fin] = 0
    return corr


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function 
cdef int[:,:] find_max_id(float[:,:,:] corr):
    cdef int[:,:] arr = np.empty((corr.shape[0], 2), dtype=np.int32)
    cdef int c, i, j, max_i, max_j
    cdef float max_val, curr_val
    for c in range(corr.shape[0]):
        max_val = 0.0
        for i in range(corr.shape[1]):
            for j in range(corr.shape[2]):
                curr_val = corr[c, i, j]
                if curr_val > max_val:
                    max_val = curr_val
                    max_i = i
                    max_j = j
        arr[c, 0] = max_i
        arr[c, 1] = max_j
    return arr

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function 
@cython.cdivision(True)
cpdef np.ndarray validate_peak2peak(float[:,:,:] corr, 
    long long[:,:] first_peak, 
    int width,
    float ratio):
    cdef float[:,:,:] masked_corr = peak2peak_valid_mask(corr, first_peak, width) 
    cdef int[:,:] second_peak = find_max_id(masked_corr)
    cdef bint[:] arr = np.empty((corr.shape[0]), dtype=bool)
    cdef int c
    cdef float first_max, second_max
    for c in range(corr.shape[0]):
        first_max = corr[
            c, 
            first_peak[c, 0], 
            first_peak[c, 1]
            ]
        second_max = corr[
            c, 
            second_peak[c, 0], 
            second_peak[c, 1]
            ]
        if first_max / second_max < ratio:
            arr[c] = True
        else:
            arr[c] = False
    return np.asarray(arr)

# def adaptive_median_filter(float[:,:] u, float[:,:] v):
