CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -pedantic -pthread

.PHONY: all test run benchmark-only clean

all: correctness_test benchmark

benchmark: benchmark.cpp stack.hpp
	$(CXX) $(CXXFLAGS) benchmark.cpp -o benchmark

correctness_test: correctness_test.cpp stack.hpp
	$(CXX) $(CXXFLAGS) correctness_test.cpp -o correctness_test

test: correctness_test
	./correctness_test

run: test benchmark
	./benchmark

benchmark-only: benchmark
	./benchmark

clean:
	rm -f benchmark correctness_test
