include user.mk


GRID_DIR = ./src/grid
DFT_DIR = ./src/dft
COMMON_DIR = ./src/common
REF_DIR = ./src/reference
EPIC_DIR = ./src/epic

.PHONY: project_code

project_code: binmk make_grid make_dft make_epic make_ref



make_common: 
	make -C $(COMMON_DIR)

make_grid: make_common
	make -C $(GRID_DIR)

make_dft: make_common
	make -C $(DFT_DIR)

make_epic: make_common
	make -C $(EPIC_DIR)

make_ref: make_common
	make -C $(REF_DIR)

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
