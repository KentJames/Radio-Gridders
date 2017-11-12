# Radio Gridders

This is a repository that holds several implementations of imagers for radio telescopes.

It is primarily focused on finding more efficient methods of implementation of classic algorithms, such as Tim Cornwell's W-Projection, and the Direct Discrete Fourier Transform.

The end result is the "Dirty Map": the true sky brightness distribution convolved with the point spread function (p.s.f) of the instrument. Removing the p.s.f is done through a deconvolution step using either CLEAN or MaxEnt.


## Discrete Fourier Transform

This is the most mathematically accurate method of reconstruction the sky brightness distribution function, by summing up the contribution at each discretised part of the domain, being limited by floating point error.

However it is by *far* the most computationally intensive, being of O(m*n^2), and heavily bandwidth limited. On a CPU the time taken feels like waiting for the heat death of the universe. Thus a basic GPU version is presented that enables a quicker DFT, taking advantage of the GPU's parallel architecture. Contributions to each grid point are summed in a local register than added to the grid non-atomically(a single thread looks at one grid point and only that grid-point), thanks to the trivially parallelisable nature of the algorithm.

## W-Projection

Originally presented by Tim Cornwell, this corrects for the W kernel in a computationally more efficient method at the expense of more error. The numerical error, compared to the DFT, still creates perfectly usable images. This method uses a convolution kernel to convolve the 3-D(u,v,w) visibility onto the 2-D (u,v) grid. Then an inverse FFT is executed to take us back to the image domain.



## W-Towers

Similar to WSClean, but further divides the grid to take advantage of the reduced complexity of small 2-D FFT's. 

