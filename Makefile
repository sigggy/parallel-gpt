PYTHON ?= python3
CXX ?= c++
CXXFLAGS ?= -std=c++17 -O3 -Wall -Wextra -pedantic

# Profiling flags
PROFILE_CXXFLAGS := -std=c++17 -O2 -g -fno-omit-frame-pointer -fno-inline-functions -Wall -Wextra -pedantic

# macOS Instruments
#INSTRUMENTS_CXXFLAGS := -std=c++17 -O2 -g -fno-omit-frame-pointer -Wall -Wextra -pedantic


NVCC ?= nvcc
NVCCFLAGS ?= -std=c++17 -O3 -Xcompiler -Wall,-Wextra,-pedantic

BUILD_DIR := build
DATASET := training_data/datasets/names.txt
FIXTURE_DIR := training_data/fixtures/small_case
SAMPLE_NAME := anna

SERIAL_CPP_SRCS := methods/serial_cpp/main.cpp methods/serial_cpp/kernel.cpp methods/serial_cpp/utils.cpp
PARALLEL_CPP_SRCS := methods/parallel_cpp/main.cpp methods/parallel_cpp/kernel.cu methods/parallel_cpp/utils.cpp

.PHONY: all fixtures clean profile

all: $(BUILD_DIR)/serial_cpp $(BUILD_DIR)/parallel_cpp

profile: $(BUILD_DIR)/serial_cpp_profile

fixtures:
	$(PYTHON) methods/serial_python/serial.py --mode dump-fixtures --dataset $(DATASET) --fixture-dir $(FIXTURE_DIR) --sample-name $(SAMPLE_NAME)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Normal optimized build
$(BUILD_DIR)/serial_cpp: $(SERIAL_CPP_SRCS) methods/serial_cpp/kernel.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(SERIAL_CPP_SRCS) -o $@

# Profiling build (clean symbols, minimal inlining)
$(BUILD_DIR)/serial_cpp_profile: $(SERIAL_CPP_SRCS) methods/serial_cpp/kernel.hpp | $(BUILD_DIR)
	$(CXX) $(PROFILE_CXXFLAGS) $(SERIAL_CPP_SRCS) -o $@

# CUDA build (unchanged)
$(BUILD_DIR)/parallel_cpp: $(PARALLEL_CPP_SRCS) methods/parallel_cpp/kernel.hpp | $(BUILD_DIR)
	@command -v $(NVCC) >/dev/null 2>&1 || { echo "parallel_cpp requires nvcc on PATH"; exit 1; }
	$(NVCC) $(NVCCFLAGS) $(PARALLEL_CPP_SRCS) -o $@

clean:
	rm -rf $(BUILD_DIR)
