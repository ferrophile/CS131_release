import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    for m in range(Hi):
        for n in range(Wi):
            for hk in range(Hk):
                for wk in range(Wk):
                    k = hk - 1
                    l = wk - 1
                    if m-k<0 or n-l<0 or m-k>=Hi or n-l>=Wi:
                        continue
                    out[m, n] += kernel[hk, wk] * image[m - k, n - l]
    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W)
        pad_width: width of the zero padding (left and right padding)
        pad_height: height of the zero padding (bottom and top padding)

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width)
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE
    rows = H + pad_height * 2
    cols = W + pad_width * 2

    out = np.zeros((rows, cols))
    for h in range(H):
        for w in range(W):
            out[h + pad_height, w + pad_width] = image[h, w]
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pad_img = zero_pad(image, Hk//2, Wk//2)
    flip_ker = np.flip(np.flip(kernel, 1), 0)

    for m in range(Hi):
        for n in range(Wi):
            out[m, n] = np.sum(pad_img[m:m+Hk, n:n+Wk] * flip_ker)

    ### END YOUR CODE

    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pad_img = zero_pad(image, Hk//2, Wk//2)
    flip_ker = np.flip(np.flip(kernel, 1), 0)

    # Each row is the frame image corresponding to each pixel.
    trans_img = np.zeros((Hi*Wi, Hk*Wk))

    for m in range(Hi):
        for n in range(Wi):
            trans_img[m*Wi+n, :] = pad_img[m:m+Hk, n:n+Wk].reshape((1, -1))

    # Calculate dot product for each row (each pixel) with the kernel.
    # [HiWi, HkWk] * [HkWk, 1] => [HiWi, 1]
    # Result is a column of pixels of convolved image.
    trans_img = trans_img.dot(kernel.reshape(-1, 1))

    # Remap this column into image shape
    out = trans_img.reshape((Hi, Wi))

    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    flip_g = np.flip(np.flip(g, 0), 1)
    out = conv_fast(f, flip_g)
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g

    Subtract the mean of g from g so that its mean becomes zero

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    nomean_g = g - np.mean(g)
    out = cross_correlation(f, nomean_g)
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    Hf, Wf = f.shape
    Hg, Wg = g.shape
    out = np.zeros((Hf, Wf))
    norm_g = (g - np.mean(g)) / np.std(g)
    pad_img = zero_pad(f, Hg//2, Wg//2)

    for m in range(Hf):
        for n in range(Wf):
            f = pad_img[m:m+Hg, n:n+Wg]
            norm_f = (f - np.mean(f)) / np.std(f)
            out[m, n] = np.sum(norm_f * norm_g)
    ### END YOUR CODE

    return out
