.PHONY: all build clean help

all: build

build:
	@echo "Building C extension..."
	python3 setup.py build_ext --inplace
	@echo "✓ Build complete"

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -f clustering_engine*.so
	rm -f *.pyc
	rm -rf __pycache__
	@echo "✓ Clean complete"

help:
	@echo "Makefile targets:"
	@echo "  make build  - Build C extension"
	@echo "  make clean  - Clean build artifacts"
	@echo "  make help   - Show this message"
