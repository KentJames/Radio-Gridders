GRID_DIR = ./src/grid
DFT_DIR = ./src/dft
COMMON_DIR = ./src/common
REF_DIR = ./src/reference

.PHONY: project_code

project_code:
	$(MAKE) -C $(COMMON_DIR)
	$(MAKE) -C $(GRID_DIR)
	$(MAKE) -C $(DFT_DIR)
	$(MAKE) -C $(REF_DIR)
clean:
	-$(MAKE) -C $(GRID_DIR) clean
	-$(MAKE) -C $(DFT_DIR) clean
	-$(MAKE) -C $(COMMON_DIR) clean
	-$(MAKE) -C $(REF_DIR) clean
