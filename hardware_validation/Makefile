NVCC ?= nvcc
CXXFLAGS ?= -O3 -std=c++17
CUDAFLAGS ?= --expt-relaxed-constexpr

BUILD_DIR := build
SOURCES := $(wildcard src/tcgen05_*_probe.cu)
RUNNERS := $(patsubst src/%.cu,$(BUILD_DIR)/%,$(SOURCES))

.PHONY: all clean

all: $(RUNNERS)

$(BUILD_DIR)/%: src/%.cu | $(BUILD_DIR)
	$(NVCC) $(CXXFLAGS) $(CUDAFLAGS) -arch=sm_100a -lineinfo -o $@ $<

$(BUILD_DIR):
	mkdir -p $@

clean:
	rm -rf $(BUILD_DIR)
