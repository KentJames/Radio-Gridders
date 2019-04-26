include ../user.mk


override CXXFLAGS += -fopenmp -std=c++14 -I../src/common
override LDFLAGS += -L../src/common -lhdf5_r -L../src/wstack -lpredict -lhdf5 -lm -lomp -lfftw3_threads -lfftw3

all: test_wstack.out test_deconvolve.out

clean:
	rm -f *.o 
	rm -f *.out

test_wstack.out: test_wstack.o 
	$(CXX) $^ -o test_wstack.out $(LDFLAGS) -L../src/wstack/ -lpredict

test_deconvolve.out: test_deconvolve.o 
	$(CXX) $^ -o test_deconvolve.out $(LDFLAGS) -L../src/wstack/ -lpredict


test_wstack.o: test_wstack.cpp
	$(CXX) -c $< -o $@  $(CXXFLAGS) -I../src/wstack/

test_deconvolve.o: test_deconvolve.cpp
	$(CXX) -c $< -o $@  $(CXXFLAGS) -I../src/wstack/

