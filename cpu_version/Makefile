CC=gcc
CXX=g++
PYTHON=python3

SOURCES = $(filter-out src/state.cpp, $(wildcard src/*.cpp))
HEADERS = $(wildcard src/*.hpp)

default: pySimbi

pySimbi: setup.py $(SOURCES) $(HEADERS)
	$(PYTHON) setup.py build_ext --inplace && rm -f src/state.cpp && rm -Rf build

clean: 
	rm *.so