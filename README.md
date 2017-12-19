# Radio Gridders

This is a repository that holds several reference GPU implementations of imagers for radio telescopes.

The end result of these imagers is the "Dirty Map": the true sky brightness distribution convolved with the point spread function (p.s.f) of the instrument. Removing the p.s.f is done through a deconvolution step using either CLEAN or MaxEnt.

## Imagers

The imagers that this repository describes are CUDA implementations of:

* Direct Discrete Fourier Transform
* W-Projection
* W-Towers

### Discrete Fourier Transform

This is the most mathematically accurate method of reconstruction the sky brightness distribution function, by summing up the contribution at each discretised part of the domain, being limited by floating point error.

However it is by *far* the most computationally intensive, being of O(m*n^2), and heavily bandwidth limited. On a CPU the time taken feels like waiting for the heat death of the universe. Thus a basic GPU version is presented that enables a quicker DFT, taking advantage of the GPU's parallel architecture. Contributions to each grid point are summed in a local register than added to the grid non-atomically(a single thread looks at one grid point and only that grid-point), thanks to the trivially parallelisable nature of the algorithm.

### W-Projection

Originally presented by Tim Cornwell, this corrects for the W kernel in a computationally more efficient method at the expense of more error. The numerical error, compared to the DFT, still creates perfectly usable images. This method uses a convolution kernel to convolve the 3-D(u,v,w) visibility onto the 2-D (u,v) grid. Then an inverse FFT is executed to take us back to the image domain.

### W-Towers

Similar to W-Stacking, but further divides the grid to take advantage of the reduced complexity of small 2-D FFT's. Uses the concept
of Image Domain Gridding, and Romein's W-Projection gridder.

## Build

To build the gridders, you need:

* Fully functioning CUDA installation.
* HDF5 headers. 

Past this everything should be standard on most UNIX systems. In top level directory run `make`, or for enabled in-kernel visibility counting `make CXXFLAGS+=-D__COUNT_VIS__`.


## License

The MIT License (MIT)

Copyright (c) 2016-2017 James Kent

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.



