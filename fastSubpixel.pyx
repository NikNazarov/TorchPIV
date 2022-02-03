import numpy
cimport numpy
cimport cython
from libc.math cimport log, isnan

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
    float[:,:,:,:] corr, 
    long long[:,:,:] first_peak, 
    int n_rows, int n_cols):
    """ 
        Find subpixel position gaussian method
        Input params:
        corr[batch, chan, :, :] float32 
        first_peak[batch, chan, 2] long long - peak positions
        n_rows int - velocity field shape[0]
        n_cols int - velocity field shape[1]
        Returns:
        tuple[np.ndarray] u, v [batch, : , :] double
    """
    cdef int k, m, n_batches, n_channels, peak_i, peak_j, chan_n, i
    cdef float c, cl, cr, cd, cu, nom1, nom2, den1, den2

    n_batches = corr.shape[0]
    n_channels = corr.shape[1] 
    bounds_h = corr.shape[-2]
    bounds_w = corr.shape[-1]

    cdef double[:,:,:] u = numpy.empty((n_batches, n_rows, n_cols), dtype=numpy.double)
    cdef double[:,:,:] v = numpy.empty((n_batches, n_rows, n_cols), dtype=numpy.double)

    
    # Find subpixel fields, gaussian method
    for batch in range(n_batches):
        for k in range(n_rows):
            for m in range(n_cols):
                chan_n = k*n_cols+m
                peak_i = first_peak[batch, chan_n, 0]
                peak_j = first_peak[batch, chan_n, 1]  
                if ((peak_i == 0) | (peak_i == corr.shape[-2]-1) |
                        (peak_j == 0) | (peak_j == corr.shape[-1]-1)):
                    v[batch, k, m] = float("nan")
                    u[batch, k, m] = float("nan")
                    continue
                
                c = corr[batch, chan_n, peak_i, peak_j]
                cl = corr[batch, chan_n, peak_i - 1, peak_j]
                cr = corr[batch, chan_n, peak_i + 1, peak_j]
                cd = corr[batch, chan_n, peak_i, peak_j - 1]
                cu = corr[batch, chan_n, peak_i, peak_j + 1]
                nom1 = log(cl) - log(cr)
                den1 = 2 * log(cl) - 4 * log(c) + 2 * log(cr)
                nom2 = log(cd) - log(cu)
                den2 = 2 * log(cd) - 4 * log(c) + 2 * log(cu)

                v[batch, k, m] = peak_i + nom1 / den1
                u[batch, k, m] = peak_j + nom2 / den2

    for i in range(3):
        v = _replace_nans(v)
        u = _replace_nans(u)
     
    return numpy.asarray(u), numpy.asarray(v)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef double[:,:,:] _replace_nans(double[:,:,:] vec):

    """ 
        Raplace NaNs by averaging neighborhood
        Input params:
        vec[:,:,:] double 
        Returns:
        vec[:,:,:] double
    """

    cdef double neighbours[8]
    cdef int k, m, n_batches, n_rows, n_cols, i, nan_count
    n_batches = vec.shape[0]
    n_rows    = vec.shape[1]
    n_cols    = vec.shape[2]
    for batch in range(n_batches):
        for k in range(1, n_rows - 1):
            for m in range(1, n_cols - 1):
                if not isnan(vec[batch, k, m]):
                    continue
                
                neighbours[0] = vec[batch, k, m-1]
                neighbours[1] = vec[batch, k+1, m-1]
                neighbours[2] = vec[batch, k-1, m-1]
                neighbours[3] = vec[batch, k-1, m]
                neighbours[4] = vec[batch, k+1, m]
                neighbours[5] = vec[batch, k, m+1]
                neighbours[6] = vec[batch, k+1, m+1]
                neighbours[7] = vec[batch, k-1, m+1]
                nan_count = 0
                for i in range(8):
                    if isnan(neighbours[i]):
                        nan_count += 1
                if nan_count < 3:
                    vec[batch, k, m] = 0
                    for i in range(8):
                        if not isnan(neighbours[i]): 
                            vec[batch, k, m] += neighbours[i]
                    vec[batch, k, m] = vec[batch, k, m] / (8 - nan_count)
    return vec



@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function 
def iter_displacement( 
    float[:, :] image_b,
    long long[:, :] x,
    long long[:, :] y,
    long long[:, :] dx,
    long long[:, :] dy,
    int wind_size
    ):
    cdef int i, j, dxi, dyi, new_x, new_y, wind_id, old_y, old_x, bound_y_right, bound_y_left, bound_x_right, bound_x_left
    cdef int wind_half = wind_size//2
    cdef float[:,:,:] moving_window_array = numpy.empty((x.shape[0]*x.shape[1], wind_size, wind_size), dtype=numpy.float32)
    cdef bint valid
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
            new_x = old_x + dxi
            new_y = old_y + dyi
            # Check boundaries after displacement 
            valid = (
                (0 <= (new_x - wind_half)) *
                (image_b.shape[1] >= (new_x + wind_half)) *
                (0 <= (new_y - wind_half)) *
                (image_b.shape[0] >= (new_y + wind_half))
            )
            if not valid:
                new_x = old_x
                new_y = old_y
            moving_window_array[wind_id, :, :] = image_b[(new_y-wind_half):(new_y+wind_half), (new_x-wind_half):(new_x+wind_half)]
    return numpy.asarray(moving_window_array)

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
    cdef float[:,:,:] moving_window_array = numpy.empty((x.shape[0]*x.shape[1], wind_size, wind_size), dtype=numpy.float32)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            wind_id = i*x.shape[1]+j
            new_x = x[i, j]
            new_y = y[i, j]
            moving_window_array[wind_id, :, :] = image_a[(new_y-wind_half):(new_y+wind_half), (new_x-wind_half):(new_x+wind_half)]
    return numpy.asarray(moving_window_array)