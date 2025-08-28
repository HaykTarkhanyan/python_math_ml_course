# Makefile for Homework 3: Calculus Solutions
# Usage:
#   make html           - Render to HTML with solutions
#   make html-no-sol    - Render to HTML without solutions
#   make pdf            - Render to PDF with solutions
#   make pdf-no-sol     - Render to PDF without solutions
#   make clean          - Clean up auxiliary files
#   make help           - Show this help message

# Variables
QUARTO_FILE = 03_calculus.qmd
OUTPUT_DIR = .
HTML_OUTPUT = 03_calculus.html
PDF_OUTPUT = 03_calculus.pdf

# Default target
.DEFAULT_GOAL := help

# Help target
help:
	@echo "Available targets:"
	@echo "  html           - Render to HTML with solutions"
	@echo "  html-no-sol    - Render to HTML without solutions"
	@echo "  pdf            - Render to PDF with solutions"
	@echo "  pdf-no-sol     - Render to PDF without solutions"
	@echo "  clean          - Clean up auxiliary files"
	@echo "  help           - Show this help message"

# HTML rendering with solutions
html:
	@echo "Rendering to HTML with solutions..."
	quarto render $(QUARTO_FILE) --to html --profile solution
	@echo "✓ HTML with solutions rendered to $(HTML_OUTPUT)"

# HTML rendering without solutions
html-no-sol:
	@echo "Rendering to HTML without solutions..."
	quarto render $(QUARTO_FILE) --to html
	@echo "✓ HTML without solutions rendered to $(HTML_OUTPUT)"

# PDF rendering with solutions
pdf:
	@echo "Rendering to PDF with solutions..."
	quarto render $(QUARTO_FILE) --to pdf --profile solution
	@echo "✓ PDF with solutions rendered to $(PDF_OUTPUT)"

# PDF rendering without solutions
pdf-no-sol:
	@echo "Rendering to PDF without solutions..."
	quarto render $(QUARTO_FILE) --to pdf
	@echo "✓ PDF without solutions rendered to $(PDF_OUTPUT)"

# Clean up auxiliary files
clean:
	@echo "Cleaning up auxiliary files..."
	-del /Q $(HTML_OUTPUT) 2>nul || true
	-del /Q $(PDF_OUTPUT) 2>nul || true
	-del /Q *.synctex.gz 2>nul || true
	-del /Q *.tex 2>nul || true
	-del /Q *.log 2>nul || true
	-del /Q *.aux 2>nul || true
	-del /Q *.fdb_latexmk 2>nul || true
	-del /Q *.fls 2>nul || true
	-del /Q *.out 2>nul || true
	-del /Q *.toc 2>nul || true
	-del /Q *.bak 2>nul || true
	-del /Q *~ 2>nul || true
	-rmdir /S /Q 03_calculus_files 2>nul || true
	-rmdir /S /Q _freeze 2>nul || true
	@echo "✓ Auxiliary files cleaned up"

# Clean and rebuild HTML with solutions
rebuild-html: clean html

# Clean and rebuild PDF with solutions
rebuild-pdf: clean pdf

# Build both HTML and PDF with solutions
all: html pdf

# Build both HTML and PDF without solutions
all-no-sol: html-no-sol pdf-no-sol

# Phony targets
.PHONY: help html html-no-sol pdf pdf-no-sol clean rebuild-html rebuild-pdf all all-no-sol
