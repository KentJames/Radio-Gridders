include user.mk


GRID_DIR = ./src/grid
DFT_DIR = ./src/dft
COMMON_DIR = ./src/common
REF_DIR = ./src/reference
EPIC_DIR = ./src/epic

.PHONY: project_code

project_code: binmk
	$(MAKE) -C $(COMMON_DIR) CXXFLAGS=$(CXXFLAGS) LDFLAGS=$(LDFLAGS)
	$(MAKE) -C $(GRID_DIR) CXXFLAGS=$(CXXFLAGS) LDFLAGS=$(LDFLAGS) CUDA_FLAGS=$(CUDA_FLAGS)
	$(MAKE) -C $(DFT_DIR) CXXFLAGS=$(CXXFLAGS) LDFLAGS=$(LDFLAGS) CUDA_FLAGS=$(CUDA_FLAGS)
	$(MAKE) -C $(EPIC_DIR) CXXFLAGS=$(CXXFLAGS) LDFLAGS=$(LDFLAGS) CUDA_FLAGS=$(CUDA_FLAGS)
	$(MAKE) -C $(REF_DIR) CFLAGS=$(CFLAGS) LDFLAGS=$(LDFLAGS)

binmk:
	-mkdir bin 2> /dev/null


distclean:
	-$(MAKE) -C $(COMMON_DIR) clean
	-$(MAKE) -C $(GRID_DIR) clean
	-$(MAKE) -C $(DFT_DIR) clean  
	-$(MAKE) -C $(EPIC_DIR) clean
	-$(MAKE) -C $(REF_DIR) clean
clean:
	rm -rf ./bin 2> /dev/null
