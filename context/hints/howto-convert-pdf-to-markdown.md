# How to Convert PDF to Markdown Using MarkItDown

## Overview

MarkItDown is a powerful Python library developed by Microsoft that converts various file formats into Markdown. It's particularly useful for extracting content from PDFs for use with LLMs and text analysis pipelines.

## Installation

In our system, we use `uvx` to run MarkItDown without permanent installation:

```powershell
# Basic usage - no installation required
uvx markitdown path-to-file.pdf > output.md
```

For permanent installation (if needed):

```powershell
pip install markitdown[pdf]
```

## Basic PDF Conversion

### Command Line Usage

```powershell
# Convert PDF to markdown and output to stdout
uvx markitdown document.pdf

# Convert PDF to markdown and save to file
uvx markitdown document.pdf > document.md

# Convert PDF with output file specified
uvx markitdown document.pdf -o document.md

# Use with piping
Get-Content document.pdf | uvx markitdown
```

### Python API Usage

```python
from markitdown import MarkItDown

# Basic conversion
md = MarkItDown()
result = md.convert("document.pdf")
print(result.text_content)

# Save to file
with open("output.md", "w", encoding="utf-8") as f:
    f.write(result.text_content)
```

## Image Handling in PDFs

### Important Limitations

⚠️ **Key Limitations for PDF Image Extraction:**

1. **Standard MarkItDown**: Does NOT extract images from PDFs by default
2. **Text-only PDFs**: Only extracts text content, images are ignored
3. **Image-based PDFs**: Requires OCR preprocessing for text extraction
4. **Formatting Loss**: PDF text extraction loses most formatting (headings become plain text)

### Current Image Capabilities

MarkItDown can handle images in these scenarios:

- **Standalone Image Files**: Can process `.jpg`, `.png`, etc. with LLM descriptions
- **Office Documents**: Can extract images from Word, PowerPoint files
- **NOT PDF Images**: Cannot extract embedded images from PDF files

### Working with Image Descriptions (for standalone images)

```python
from markitdown import MarkItDown
from openai import OpenAI

# Setup LLM client for image descriptions
client = OpenAI(api_key="your-api-key")
md = MarkItDown(llm_client=client, llm_model="gpt-4o")

# This works for standalone images, NOT PDF images
result = md.convert("image.jpg")
print(result.text_content)
```

## Advanced Options

### Using Azure Document Intelligence

For better PDF processing with potential image handling:

```powershell
# Command line with Azure Document Intelligence
uvx markitdown document.pdf -o output.md -d -e "your-document-intelligence-endpoint"
```

```python
from markitdown import MarkItDown

# Python API with Document Intelligence
md = MarkItDown(docintel_endpoint="your-document-intelligence-endpoint")
result = md.convert("document.pdf")
print(result.text_content)
```

### Using Plugins

```powershell
# List available plugins
uvx markitdown --list-plugins

# Use with plugins enabled
uvx markitdown --use-plugins document.pdf
```

## Alternative Solutions for PDF Images

Since MarkItDown doesn't extract PDF images directly, consider these alternatives:

### 1. Extract Images First (Manual Process)

```powershell
# Using poppler-utils (if available)
pdfimages document.pdf output-prefix

# Using PDFBox
java -jar pdfbox-app.jar ExtractImages document.pdf
```

### 2. Third-party Tools

- **markitdown-go**: A Go implementation that extracts images to `assets` folder
- **Custom solutions**: Write scripts using `PyMuPDF` or `pdfplumber`

### 3. OCR Solutions

For image-heavy PDFs, consider OCR preprocessing:

```python
# Example with pytesseract (conceptual)
import pytesseract
from pdf2image import convert_from_path

pages = convert_from_path('document.pdf')
for page in pages:
    text = pytesseract.image_to_string(page)
    # Process extracted text
```

## Best Practices

### 1. Check PDF Type First

```python
import fitz  # PyMuPDF

doc = fitz.open("document.pdf")
page = doc[0]
text_blocks = page.get_text("dict")["blocks"]

# Check if PDF has text or is image-based
has_text = any(block.get("type") == 0 for block in text_blocks)
print(f"PDF has extractable text: {has_text}")
```

### 2. Preprocessing for Better Results

- Ensure PDFs have OCR text layer for image-based documents
- Use high-quality source PDFs when possible
- Consider PDF optimization before conversion

### 3. Post-processing

```python
# Clean up extracted markdown
import re

def clean_markdown(text):
    # Remove excessive whitespace
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    # Add proper heading formatting (manual detection needed)
    # ... additional cleanup
    return text
```

## Common Use Cases

### 1. Research Papers

```powershell
# Convert academic paper to markdown
uvx markitdown research-paper.pdf -o paper.md
```

### 2. Batch Processing

```powershell
# PowerShell batch processing
Get-ChildItem "*.pdf" | ForEach-Object {
    $outputName = $_.BaseName + ".md"
    uvx markitdown $_.FullName -o $outputName
}
```

### 3. Integration with LLM Workflows

```python
from markitdown import MarkItDown

def pdf_to_llm_context(pdf_path):
    md = MarkItDown()
    result = md.convert(pdf_path)
    
    # Clean and prepare for LLM
    context = result.text_content
    return context

# Use in LLM pipeline
context = pdf_to_llm_context("document.pdf")
# Feed to your LLM...
```

## Troubleshooting

### Common Issues

1. **Empty Output**: PDF might be image-based without text layer
2. **Poor Formatting**: Expected - PDF formatting is not preserved
3. **Missing Images**: Not supported for PDF extraction
4. **Encoding Errors**: Use UTF-8 encoding when saving files

### Solutions

```python
# Handle encoding issues
try:
    result = md.convert("document.pdf")
    with open("output.md", "w", encoding="utf-8", errors="ignore") as f:
        f.write(result.text_content)
except Exception as e:
    print(f"Conversion error: {e}")
```

## Summary

- ✅ **Good for**: Text-based PDFs, quick conversion, LLM preprocessing
- ❌ **Not good for**: PDF image extraction, preserving complex formatting
- 🔧 **Best practice**: Use `uvx markitdown` for simple text extraction
- 🔍 **For images**: Consider separate image extraction tools before conversion

## Related Tools

- **Pandoc**: Alternative conversion tool
- **PyMuPDF**: Python library for PDF processing with image extraction
- **pdfplumber**: PDF text and table extraction
- **Azure Document Intelligence**: Enterprise OCR solution
