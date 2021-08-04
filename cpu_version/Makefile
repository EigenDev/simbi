CC=gcc
CXX=g++
OMP_NUM_THREADS=2

SOURCES = $(filter-out src/state.cpp, $(wildcard src/*.cpp))
HEADERS = $(wildcard src/*.hpp)

default: pySimbi

pySimbi: setup.py $(SOURCES) $(HEADERS)
	python setup.py build_ext --inplace && rm -f src/state.cpp && rm -Rf build

clean: 
	rm *.so