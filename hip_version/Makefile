HIP_PATH ?= $(wildcard /opt/rocm/hip)
HIP_PLATFORM = $(shell $(HIP_PATH)/bin/hipconfig --platform)
HIP_INCLUDE  = -I$(HIP_PATH)/include
BUILD_DIR ?= build 

HIPCC=$(HIP_PATH)/bin/hipcc

ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif

CC =gcc
CXX=g++ #${HIPCC}
CXXFLAGS = -I.
LDFLAGS = -lhdf5 -lhdf5_cpp -shared
LD_LIBRARY_PATH := 
LIBS=-L$(LD_LIBRARY_PATH) $(LDFLAGS)
PRECISION = 

ifeq (, ${LD_LIBRARY_PATH})
	LIBS=$(LDFLAGS)
endif


SOURCES=$(filter-out src/gpu_state.cpp, $(wildcard src/*.cpp))
HEADERS=$(wildcard src/*.hpp)
# OBJECTS=$(SOURCES:.cpp=.o)

all: pySimbi

pySimbi: setup.py $(OBJECTS)
	python setup.py build_ext --inplace $(PRECISION) && rm -f src/gpu_state.cpp && rm -Rf build

clean:
	rm *.so 

