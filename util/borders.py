import numpy as np

def zeros(mat: np.ndarray, size: tuple = None) -> np.ndarray:
    '''
    Returns a matrix with zero boundary conditions.

    Parameters:
        mat (ndarray): Matrix to add boundary conditions to.
        size (tuple): Number of pixels to include in the boundary.
                      Use None to specify all pixels.
    
    Returns:
        New matrix with boundary conditions.
    '''

    mat_zeros = np.zeros(mat.shape)
    zeros_stack = np.vstack((mat_zeros, mat_zeros, mat_zeros))
    mid_stack   = np.vstack((mat_zeros, mat,       mat_zeros))
    result      = np.hstack((zeros_stack, mid_stack, zeros_stack))

    if size:
        return resize(result, size)
    else:
        return result

def periodic(mat: np.ndarray, size: tuple = None) -> np.ndarray:
    '''
    Returns a matrix with periodic boundary conditions.

    Parameters:
        mat (ndarray): Matrix to add boundary conditions to.
        size (tuple): Number of pixels to include in the boundary.
                      Use None to specify all pixels.
    
    Returns:
        New matrix with boundary conditions.
    '''

    stack  = np.vstack((mat, mat, mat))
    result = np.hstack((stack, stack, stack))

    if size:
        return resize(result, size)
    else:
        return result

def reflexive(mat: np.ndarray, size: tuple = None) -> np.ndarray:
    '''
    Returns a matrix with reflexive boundary conditions.

    Parameters:
        mat (ndarray): Matrix to add boundary conditions to.
        size (tuple): Number of pixels to include in the boundary.
                      Use None to specify all pixels.
    
    Returns:
        New matrix with boundary conditions.
    '''

    mat_lr = np.fliplr(mat)
    mat_ud = np.flipud(mat)
    mat_x  = np.fliplr(mat_ud)

    left_stack  = np.vstack((mat_x,  mat_lr, mat_x))
    mid_stack   = np.vstack((mat_ud, mat,    mat_ud))
    right_stack = np.vstack((mat_x,  mat_lr, mat_x))
    result      = np.hstack((left_stack, mid_stack, right_stack))

    if size:
        return resize(result, size)
    else:
        return result

def resize(mat, padding):
    '''
        Returns a matrix with only the padding needed.  
        Returned matrix has the following structure:
        (x+y) (y) (x+y)  
        (x)   mat   (x)  
        (x+y) (y) (x+y)  

        Parameters:
            mat (mat): original matrix
            padding (tuple): how many pixels to pad, ex (5, 5)
        
        Returns:
            matrix with padding applied
    '''

    padding_x, padding_y = padding

    original_pos = mat.shape[0]/3
    left_bound   = int(original_pos - 2*padding_x)
    right_bound  = int(2*original_pos + 2*padding_x)
    bottom_bound = int(original_pos - 2*padding_y)
    top_bound    = int(2*original_pos + 2*padding_y)
    return mat[bottom_bound:top_bound, left_bound:right_bound]

def undo(mat: np.ndarray, size: tuple = None) -> np.ndarray:
    '''
        Extracts the center part of a matrix by "undoing" the boundary condition operations. If a kernel size
        is given, only the given pixels will be removed; otherwise, it will unpack a matrix into a 3x3 proportion
        and return the middle matrix.

        Parameters:
            mat (ndarray): Matrix containing the original image with boundary conditions.
            size (tuple): Dimensions of padding applied, ex. (3, 3)

        Returns:
            Center of matrix.
    '''

    # kernel size given, so slice it
    if size:
        padding_x = size[0]
        padding_y = size[1]
        return mat[padding_x:-padding_x, padding_y:-padding_y]
    # no kernel size, so assume it's a 3x3 construction
    else:
        _, mid_stack, _ = np.hsplit(mat, 3)
        _, mid_mat,   _ = np.vsplit(mid_stack, 3)
        return mid_mat