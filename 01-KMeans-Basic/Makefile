# ============================================================================
# Makefile for Lloyd's K-Means Clustering - Basic Implementation
# ============================================================================

CC = gcc
CFLAGS = -ansi -Wall -Wextra -Werror -pedantic-errors
LIBS = -lm
TARGET = lloyd
SOURCE = lloyd_clustering.c
PYTHON = python3

# Default target
.PHONY: all
all: $(TARGET)

# Compile C program
$(TARGET): $(SOURCE)
	@echo "Compiling $(SOURCE) with strict ANSI C compliance..."
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCE) $(LIBS)
	@echo "✓ Successfully compiled: $(TARGET)"

# Generate test data
.PHONY: test_data
test_data: sample_data.csv

sample_data.csv:
	@echo "Generating sample test data..."
	@$(PYTHON) << 'EOF'
import random
random.seed(42)

# Cluster 1: around (0, 0)
for _ in range(50):
    print(f"{random.gauss(0, 0.5):.2f},{random.gauss(0, 0.5):.2f}")

# Cluster 2: around (10, 10)
for _ in range(50):
    print(f"{random.gauss(10, 0.5):.2f},{random.gauss(10, 0.5):.2f}")

# Cluster 3: around (20, 0)
for _ in range(50):
    print(f"{random.gauss(20, 0.5):.2f},{random.gauss(0, 0.5):.2f}")
EOF
	@echo "✓ Sample data created: sample_data.csv"

# Run tests
.PHONY: test
test: $(TARGET) sample_data.csv
	@echo ""
	@echo "=========================================="
	@echo "Testing Lloyd's Clustering"
	@echo "=========================================="
	@echo ""
	@echo "🔍 Running C version..."
	@./$(TARGET) 3 300 < sample_data.csv > c_output.txt
	@echo "✓ C version output saved to c_output.txt"
	@echo ""
	@echo "🐍 Running Python version..."
	@$(PYTHON) lloyd_clustering.py 3 300 < sample_data.csv > python_output.txt
	@echo "✓ Python version output saved to python_output.txt"
	@echo ""
	@echo "Comparing outputs (should be identical):"
	@echo "--- C Output ---"
	@cat c_output.txt
	@echo ""
	@echo "--- Python Output ---"
	@cat python_output.txt
	@echo ""

# Clean up
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	@rm -f $(TARGET) $(TARGET).o
	@rm -f c_output.txt python_output.txt
	@echo "✓ Cleanup complete"

# Clean everything
.PHONY: clean_all
clean_all: clean
	@rm -f sample_data.csv
	@echo "✓ Full cleanup complete"

# Help
.PHONY: help
help:
	@echo ""
	@echo "Lloyd's K-Means Clustering - Build System"
	@echo "=========================================="
	@echo ""
	@echo "Available targets:"
	@echo "  make all        - Compile C implementation"
	@echo "  make test       - Run full test suite"
	@echo "  make test_data  - Generate sample test data"
	@echo "  make clean      - Remove compiled files"
	@echo "  make clean_all  - Remove all generated files"
	@echo "  make help       - Display this help"
	@echo ""
	@echo "Quick start:"
	@echo "  make              # Compile"
	@echo "  make test_data    # Generate data"
	@echo "  make test         # Run tests"
	@echo ""
