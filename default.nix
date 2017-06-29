with import <nixpkgs> {};

stdenv.mkDerivation rec {
    name = "env";
    env = buildEnv { name = name;
                    paths = buildInputs; };
    buildInputs = [
        gcc
        cudatoolkit
        fftw
        fftwFloat
        python36
        python36Packages.numpy
        python36Packages.pycuda
        python36Packages.pandas
        python36Packages.astropy
    ];
}
