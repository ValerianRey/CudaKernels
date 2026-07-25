NVCC := nvcc
CXXSTD := -std=c++17
SRC_DIR := src
INCLUDES := -I$(SRC_DIR)
BUILD_DIR := build

GTEST_VERSION := v1.15.2
GTEST_REPO_DIR := third_party/googletest
GTEST_ROOT := $(GTEST_REPO_DIR)/googletest
GTEST_INCLUDE := $(GTEST_ROOT)/include
GTEST_LIB := $(BUILD_DIR)/libgtest.a
GTEST_MAIN_LIB := $(BUILD_DIR)/libgtest_main.a

.PHONY: all demo bench test clean distclean

all: demo bench test

demo: $(BUILD_DIR)/dot_product_demo $(BUILD_DIR)/add_vector_demo
bench: $(BUILD_DIR)/dot_product_bench

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# --- dot product library, shared by the demo, the benchmark and the tests ---

$(BUILD_DIR)/dot_product.o: $(SRC_DIR)/dot_product.cu $(SRC_DIR)/dot_product.cuh $(SRC_DIR)/cuda_check.cuh | $(BUILD_DIR)
	$(NVCC) $(CXXSTD) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/dot_product_main.o: $(SRC_DIR)/dot_product_main.cu $(SRC_DIR)/dot_product.cuh $(SRC_DIR)/cuda_check.cuh | $(BUILD_DIR)
	$(NVCC) $(CXXSTD) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/dot_product_demo: $(BUILD_DIR)/dot_product.o $(BUILD_DIR)/dot_product_main.o
	$(NVCC) $(CXXSTD) $^ -o $@

# --- add vector library, shared by the demo and the tests ---

$(BUILD_DIR)/add_vector.o: $(SRC_DIR)/add_vector.cu $(SRC_DIR)/add_vector.cuh $(SRC_DIR)/cuda_check.cuh | $(BUILD_DIR)
	$(NVCC) $(CXXSTD) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/add_vector_main.o: $(SRC_DIR)/add_vector_main.cu $(SRC_DIR)/add_vector.cuh $(SRC_DIR)/cuda_check.cuh | $(BUILD_DIR)
	$(NVCC) $(CXXSTD) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/add_vector_demo: $(BUILD_DIR)/add_vector.o $(BUILD_DIR)/add_vector_main.o
	$(NVCC) $(CXXSTD) $^ -o $@

# --- benchmark ---

$(BUILD_DIR)/dot_product_bench.o: benchmarks/dot_product_bench.cu $(SRC_DIR)/dot_product.cuh $(SRC_DIR)/cuda_check.cuh | $(BUILD_DIR)
	$(NVCC) $(CXXSTD) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/dot_product_bench: $(BUILD_DIR)/dot_product.o $(BUILD_DIR)/dot_product_bench.o
	$(NVCC) $(CXXSTD) $^ -o $@

# --- GoogleTest, vendored and built from source (no system package needed) ---

$(GTEST_REPO_DIR):
	git clone --depth 1 --branch $(GTEST_VERSION) https://github.com/google/googletest.git $(GTEST_REPO_DIR)

$(BUILD_DIR)/gtest-all.o: | $(BUILD_DIR) $(GTEST_REPO_DIR)
	g++ $(CXXSTD) -pthread -isystem $(GTEST_INCLUDE) -I $(GTEST_ROOT) -c $(GTEST_ROOT)/src/gtest-all.cc -o $@

$(BUILD_DIR)/gtest_main.o: | $(BUILD_DIR) $(GTEST_REPO_DIR)
	g++ $(CXXSTD) -pthread -isystem $(GTEST_INCLUDE) -I $(GTEST_ROOT) -c $(GTEST_ROOT)/src/gtest_main.cc -o $@

$(GTEST_LIB): $(BUILD_DIR)/gtest-all.o
	ar rcs $@ $^

$(GTEST_MAIN_LIB): $(BUILD_DIR)/gtest_main.o
	ar rcs $@ $^

# --- correctness tests ---

$(BUILD_DIR)/dot_product_test.o: tests/dot_product_test.cu $(SRC_DIR)/dot_product.cuh $(SRC_DIR)/cuda_check.cuh | $(BUILD_DIR) $(GTEST_REPO_DIR)
	$(NVCC) $(CXXSTD) $(INCLUDES) -I $(GTEST_INCLUDE) -c $< -o $@

$(BUILD_DIR)/dot_product_test: $(BUILD_DIR)/dot_product.o $(BUILD_DIR)/dot_product_test.o $(GTEST_LIB) $(GTEST_MAIN_LIB)
	$(NVCC) $(CXXSTD) $^ -lpthread -o $@

$(BUILD_DIR)/add_vector_test.o: tests/add_vector_test.cu $(SRC_DIR)/add_vector.cuh $(SRC_DIR)/cuda_check.cuh | $(BUILD_DIR) $(GTEST_REPO_DIR)
	$(NVCC) $(CXXSTD) $(INCLUDES) -I $(GTEST_INCLUDE) -c $< -o $@

$(BUILD_DIR)/add_vector_test: $(BUILD_DIR)/add_vector.o $(BUILD_DIR)/add_vector_test.o $(GTEST_LIB) $(GTEST_MAIN_LIB)
	$(NVCC) $(CXXSTD) $^ -lpthread -o $@

test: $(BUILD_DIR)/dot_product_test $(BUILD_DIR)/add_vector_test
	./$(BUILD_DIR)/dot_product_test
	./$(BUILD_DIR)/add_vector_test

clean:
	rm -rf $(BUILD_DIR)

distclean: clean
	rm -rf $(GTEST_REPO_DIR)
