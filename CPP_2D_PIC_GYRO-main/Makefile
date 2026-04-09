PROJECT := cpp_2d_pic_gyro
BUILD_DIR ?= build
OBJ_DIR := $(BUILD_DIR)/obj
BIN_DIR := $(BUILD_DIR)/bin

CXX ?= c++
BUILD_TYPE ?= Release
OPENMP ?= 0

SRC := $(shell find . -name '*.cpp' -not -path './$(BUILD_DIR)/*' | sort)
OBJ := $(patsubst ./%.cpp,$(OBJ_DIR)/%.o,$(SRC))
DEP := $(OBJ:.o=.d)

CPPFLAGS := -I.
CXXFLAGS_DEBUG := -std=c++17 -O0 -g -Wall -Wextra -Wpedantic
CXXFLAGS_RELEASE := -std=c++17 -O3 -Wall -Wextra -Wpedantic
LDFLAGS :=
LDLIBS :=

ifeq ($(BUILD_TYPE),Debug)
  CXXFLAGS := $(CXXFLAGS_DEBUG)
else
  CXXFLAGS := $(CXXFLAGS_RELEASE)
endif

ifeq ($(OPENMP),1)
  CXXFLAGS += -Xpreprocessor -fopenmp -DCPP_2D_PIC_GYRO_USE_OPENMP=1
  LDFLAGS += -lomp
endif

.PHONY: all run clean rebuild debug release help

all: $(BIN_DIR)/$(PROJECT)

$(BIN_DIR)/$(PROJECT): $(OBJ)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(OBJ) $(LDFLAGS) $(LDLIBS) -o $@

$(OBJ_DIR)/%.o: ./%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

run: all
	./$(BIN_DIR)/$(PROJECT)

clean:
	rm -rf $(BUILD_DIR)

rebuild: clean all

debug:
	$(MAKE) BUILD_TYPE=Debug

release:
	$(MAKE) BUILD_TYPE=Release

help:
	@echo "Targets:"
	@echo "  make            - build Release version"
	@echo "  make run        - build and run ./$(BIN_DIR)/$(PROJECT)"
	@echo "  make debug      - build Debug version"
	@echo "  make release    - build Release version"
	@echo "  make clean      - remove $(BUILD_DIR)"
	@echo "Variables:"
	@echo "  CXX=<compiler>       default: c++"
	@echo "  BUILD_TYPE=<type>    Release or Debug (default: Release)"
	@echo "  OPENMP=1             enable OpenMP flags if your compiler supports them"

-include $(DEP)
