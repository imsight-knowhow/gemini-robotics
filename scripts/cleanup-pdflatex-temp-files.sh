#!/bin/bash

# LaTeX Cleanup Script
# Cleans up LaTeX intermediate files from the tex directory.

# Script directory and tex directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEX_DIR="$(dirname "$SCRIPT_DIR")/tex"

# Default options
FORCE=false
WHATIF=false

# LaTeX intermediate file extensions to clean up
LATEX_TEMP_EXTENSIONS=(
    "aux"        # Auxiliary files
    "log"        # Compilation logs
    "toc"        # Table of contents
    "lof"        # List of figures
    "lot"        # List of tables
    "out"        # Hyperref bookmarks
    "bbl"        # Bibliography
    "blg"        # Bibliography log
    "idx"        # Index
    "ilg"        # Index log
    "ind"        # Index
    "fls"        # File list
    "fdb_latexmk" # Latexmk database
    "synctex.gz" # SyncTeX files
    "nav"        # Beamer navigation
    "snm"        # Beamer
    "vrb"        # Beamer verbatim
    "figlist"    # Figure list
    "makefile"   # Makefile
    "run.xml"    # Biblatex
    "bcf"        # Biblatex control file
)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
GRAY='\033[0;37m'
NC='\033[0m' # No Color

show_help() {
    echo "LaTeX Cleanup Script"
    echo "==================="
    echo ""
    echo "Cleans up LaTeX intermediate files from the tex directory."
    echo ""
    echo "USAGE:"
    echo "    $0 [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "    -f, --force     Skip confirmation and delete files immediately"
    echo "    -w, --whatif    Show what files would be deleted without actually deleting them"
    echo "    -h, --help      Show this help message"
    echo ""
    echo "EXAMPLES:"
    echo "    $0              Shows files to be deleted and prompts for confirmation"
    echo "    $0 -f           Deletes files without confirmation"
    echo "    $0 -w           Shows what files would be deleted without prompting or deleting"
}

# Parse command line options
while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--force)
            FORCE=true
            shift
            ;;
        -w|--whatif)
            WHATIF=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            show_help
            exit 1
            ;;
    esac
done

echo -e "${CYAN}LaTeX Cleanup Script${NC}"
echo -e "${CYAN}===================${NC}"

# Check if tex directory exists
if [[ ! -d "$TEX_DIR" ]]; then
    echo -e "${RED}Error: tex directory not found at: $TEX_DIR${NC}" >&2
    exit 1
fi

echo -e "${YELLOW}Searching for LaTeX intermediate files in: $TEX_DIR${NC}"

# Create temporary file to store list of files to delete
TEMP_FILE=$(mktemp)
trap "rm -f $TEMP_FILE" EXIT

# Find all matching files
for ext in "${LATEX_TEMP_EXTENSIONS[@]}"; do
    find "$TEX_DIR" -type f -name "*.$ext" >> "$TEMP_FILE"
done

# Remove duplicates and sort
sort -u "$TEMP_FILE" -o "$TEMP_FILE"

FILE_COUNT=$(wc -l < "$TEMP_FILE")

if [[ $FILE_COUNT -eq 0 ]]; then
    echo -e "${GREEN}No LaTeX intermediate files found to clean up.${NC}"
    exit 0
fi

# Display files found
echo -e "\n${YELLOW}Found $FILE_COUNT intermediate files:${NC}"
while IFS= read -r file; do
    relative_path="${file#$TEX_DIR/}"
    echo -e "  ${GRAY}tex/$relative_path${NC}"
done < "$TEMP_FILE"

# Calculate total size
if command -v du >/dev/null 2>&1; then
    TOTAL_SIZE_KB=0
    while IFS= read -r file; do
        if [[ -f "$file" ]]; then
            SIZE=$(du -k "$file" 2>/dev/null | cut -f1)
            TOTAL_SIZE_KB=$((TOTAL_SIZE_KB + SIZE))
        fi
    done < "$TEMP_FILE"
    
    TOTAL_SIZE_MB=$(echo "scale=2; $TOTAL_SIZE_KB / 1024" | bc 2>/dev/null || echo "0")
    echo -e "\n${YELLOW}Total size: ${TOTAL_SIZE_MB} MB${NC}"
fi

if [[ "$WHATIF" == true ]]; then
    echo -e "\n${MAGENTA}[WhatIf] Would delete $FILE_COUNT files.${NC}"
    exit 0
fi

# Confirm deletion unless Force is specified
if [[ "$FORCE" != true ]]; then
    echo ""
    read -p "Delete these files? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Cleanup cancelled.${NC}"
        exit 0
    fi
fi

# Delete files
echo -e "\n${GREEN}Deleting files...${NC}"
DELETED_COUNT=0
ERROR_COUNT=0

while IFS= read -r file; do
    if [[ -f "$file" ]]; then
        if rm -f "$file" 2>/dev/null; then
            DELETED_COUNT=$((DELETED_COUNT + 1))
        else
            ERROR_COUNT=$((ERROR_COUNT + 1))
            relative_path="${file#$TEX_DIR/}"
            echo -e "${RED}Warning: Failed to delete: tex/$relative_path${NC}" >&2
        fi
    fi
done < "$TEMP_FILE"

# Summary
echo -e "\n${GREEN}Cleanup complete!${NC}"
echo -e "${GREEN}Deleted: $DELETED_COUNT files${NC}"
if [[ $ERROR_COUNT -gt 0 ]]; then
    echo -e "${RED}Errors: $ERROR_COUNT files could not be deleted${NC}"
fi
if [[ -n "$TOTAL_SIZE_MB" ]] && [[ "$TOTAL_SIZE_MB" != "0" ]]; then
    echo -e "${GREEN}Freed: ${TOTAL_SIZE_MB} MB of disk space${NC}"
fi