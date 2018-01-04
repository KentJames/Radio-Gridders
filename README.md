# Radio Gridders

This is a repository that holds several reference GPU implementations of imagers for radio telescopes.

The end result of these imagers is the "Dirty Map": the true sky brightness distribution convolved with the point spread function (p.s.f) of the instrument. Removing the p.s.f is done through a deconvolution step using either CLEAN or MaxEnt.

## Imagers

The imagers that this repository describes are CUDA implementations of:

* Direct Discrete Fourier Transform
* W-Projection
* W-Towers

There is also a CPU reference in src/reference, which was written mostly by Peter Wortmann. I added some OpenMP 
parallelisation to W-Towers. 

The CPU reference has been copied from the crocodile repo, available at: https://github.com/SKA-ScienceDataProcessor/crocodile

Lots of functionality is shared between this repo and crocodile so make sure to check it out. The W-Kernels
and visibilities are generated from the scripts and notebooks in that repository.

### Discrete Fourier Transform

This is the most mathematically accurate method of reconstruction the sky brightness distribution function, by summing up the contribution at each discretised part of the domain, being limited by floating point error.

However it is by *far* the most computationally intensive, being of O(m*n^2), and heavily bandwidth limited. A basic GPU version is presented that enables a quicker DFT, taking advantage of the GPU's parallel architecture. Contributions to each grid point are summed in a local register than added to the grid non-atomically(a single thread looks at one grid point and only that grid-point), thanks to the trivially parallelisable nature of the algorithm.

### W-Projection

Originally presented by Tim Cornwell, this corrects for the W kernel in a computationally more efficient method at the expense of more error. The numerical error, compared to the DFT, still creates perfectly usable images. This method uses a convolution kernel to convolve the 3-D(u,v,w) visibility onto the 2-D (u,v) grid. Then an inverse FFT is executed to take us back to the image domain.

The W-Projection kernel implemented is a reduced bandwidth scatter gridder designed originally by John Romein.

### W-Towers

Similar to W-Stacking, but further divides the grid to take advantage of the reduced complexity of small 2-D FFT's. Uses the concept
of Image Domain Gridding, and Romein's W-Projection gridder.

## Build

To build the gridders, you need:

* Fully functioning CUDA installation.
* HDF5 headers. 

Past this everything should be standard on most UNIX systems. In top level directory run `make`, or for enabled in-kernel visibility counting `make CXXFLAGS+=-D__COUNT_VIS__`.

## Usage

There are two datasets included in this repository under `data/crocodile_data`:

* A VLA dataset with phase centre significantly away from zenith. This dataset will stress test any w-correction gridder.
* A SKA dataset with phase centre at the zenith, simulated with OSKAR. Baselines are significantly larger than VLA.

Builds are by default in bin/

### DFT

The DFT needs to be configured with a visibility file, a field of view size and other parameters. To get more information,
in /bin, run:

``` shell
./dft.out --help
```

An example DFT using the repo's datasets:

``` shell
./dft.out -theta=0.1 -lambda=20480 -image=SKA_DFT.out -vis=../data/crocodile_data/vis/SKA1_Low_quick.h5 -blocks=4096 -threadblock=1024 -device=0
```

This will run a DFT on our SKA dataset, on device 0 (can just leave this out if you have one GPU)
with a grid size of (theta \times lambda). The numbers of blocks and the numbers of threads per
block need to be configured to allow full coverage of the grid. This can be calculated as such:

No. of blocks = \frac{Grid_Size^2}{Threads Per Block}

### W-Projection / W-Towers

Similar to the DFT, we need a visibility file, a field of view and other parameters. To get more information, in /bin run:

``` shell
./grid.out --help
```

An example W-Projection Gridder:

``` shell
./grid.out -theta=0.1 -lambda=20480 -image=VLA_image.out -wkernel=../data/crocodile_data/kernels/vla_w10_static_size16.h5 \
-vis=../data/crocodile_data/vis/vlaa_theta0.1.h5 -bl_max=10000 -wproj -flat
```

This creates an image from our VLA dataset, where we limit the baseline to a maximum of 10km, and we also flatten our hierarchical
dataset (can significantly increase speed).

An example W-Towers Gridder:

``` shell
./grid.out -theta=0.1 -lambda=20480 -image=VLA_image.out -wkernel=../data/crocodile_data/kernels/vla_w10_static_size16.h5 \
-vis=../data/crocodile_data/vis/vlaa_theta0.1.h5 -bl_max=10000 -subgrid=512 -margin=16 -winc=10 -flat
```

This sets our subgrids at 512x512 with a margin o 16. The distances between floors of the towers in w is 10. 

### Reference

The reference gridder(make to sure to compile with ```CFLAGS+=-DOMP_TOWERS``` to use multithreading) can be used as follows:

An example W-Towers gridder:

``` shell
./grid_CPU.out --theta=0.1 --lambda=20480 --image=image.out --wkernel=../data/crocodile_data/kernels/vla_w10_static_size16.h5 \
--max-bl=10000 --subgrid=64 --margin=20 --winc=10 ../data/crocodile_data/vis/vlaa_theta0.1.h5
```

An example W-Projection gridder:

``` shell
./grid_CPU.out --theta=0.1 --lambda=20480 --image=image.out --wkernel=../data/crocodile_data/kernels/vla_w10_static_size16.h5 \
--max-bl=10000 ../data/crocodile_data/vis/vlaa_theta0.1.h5
```

There is also a CPU DFT implementation (VERY SLOW) which is built in /bin too. Run `./dft_CPU.out --help` for usage details.

## Caveats

Hopefully in future I will get to implement optimsations such as streaming to make this a production ready gridder, however for 
now it serves (alongside the SKA/crocodile repo) as a way to compare W-Towers to other gridders.

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



