with import <nixpkgs> {};

stdenv.mkDerivation rec {
  name = "env";
  env = buildEnv { name = name;
                  paths = buildInputs; };
  buildInputs = [
    gcc-snapshot
    fftw
    hdf5
    astropy
    graphviz
    kdeApplications.kcachegrind
    python36
    python36Packages.pylru
    python36Packages.h5py
    python36Packages.scipy
    python36Packages.numpy
    valgrind
    fftwFloat
  ];
}


