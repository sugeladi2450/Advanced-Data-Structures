CXX := g++
CXXFLAGS := -O2 -std=c++17 -Wall -Wextra -pedantic
BIN_DIR := bin

.PHONY: all clean run-benchmark

all: $(BIN_DIR)/rbtree_demo $(BIN_DIR)/splay_demo $(BIN_DIR)/benchmark

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/rbtree_demo: rbtree/main.cpp rbtree/rbtree.cpp rbtree/rbtree.h | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -I./rbtree rbtree/main.cpp rbtree/rbtree.cpp -o $@

$(BIN_DIR)/splay_demo: splaytree/main.cpp splaytree/splaytree.cpp splaytree/splaytree.h | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -I./splaytree splaytree/main.cpp splaytree/splaytree.cpp -o $@

$(BIN_DIR)/benchmark: benchmark.cpp rbtree/rbtree.cpp rbtree/rbtree.h splaytree/splaytree.cpp splaytree/splaytree.h | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -I./rbtree -I./splaytree benchmark.cpp rbtree/rbtree.cpp splaytree/splaytree.cpp -o $@

run-benchmark: $(BIN_DIR)/benchmark
	./$(BIN_DIR)/benchmark

clean:
	rm -rf $(BIN_DIR)
