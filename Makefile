PYTHON ?= python3
CXX ?= c++
CXXFLAGS ?= -std=c++17 -O3 -Wall -Wextra -pedantic

BUILD_DIR := build
DATASET := training_data/datasets/names.txt
FIXTURE_DIR := training_data/fixtures/small_case
SAMPLE_NAME := anna

SERIAL_CPP_SRCS := methods/serial_cpp/main.cpp methods/serial_cpp/kernel.cpp
PARALLEL_CPP_SRCS := methods/parallel_cpp/main.cpp methods/parallel_cpp/kernel.cpp

.PHONY: all fixtures clean

all: $(BUILD_DIR)/serial_cpp $(BUILD_DIR)/parallel_cpp

fixtures:
	$(PYTHON) methods/serial_python/serial.py --mode dump-fixtures --dataset $(DATASET) --fixture-dir $(FIXTURE_DIR) --sample-name $(SAMPLE_NAME)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/serial_cpp: $(SERIAL_CPP_SRCS) methods/serial_cpp/kernel.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(SERIAL_CPP_SRCS) -o $@

$(BUILD_DIR)/parallel_cpp: $(PARALLEL_CPP_SRCS) methods/parallel_cpp/kernel.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(PARALLEL_CPP_SRCS) -o $@

clean:
	rm -rf $(BUILD_DIR)
