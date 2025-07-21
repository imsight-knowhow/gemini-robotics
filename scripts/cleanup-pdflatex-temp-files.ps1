#!/usr/bin/env powershell

<#
.SYNOPSIS
    Cleans up LaTeX intermediate files from the tex directory.

.DESCRIPTION
    This script removes common LaTeX intermediate files (.aux, .log, .toc, etc.)
    from the tex directory and all subdirectories, while preserving source .tex 
    files and output .pdf files.

.PARAMETER Force
    Skip confirmation and delete files immediately.

.PARAMETER WhatIf  
    Show what files would be deleted without actually deleting them.

.EXAMPLE
    .\cleanup-pdflatex-temp-files.ps1
    Shows files to be deleted and prompts for confirmation.

.EXAMPLE
    .\cleanup-pdflatex-temp-files.ps1 -Force
    Deletes files without confirmation.

.EXAMPLE
    .\cleanup-pdflatex-temp-files.ps1 -WhatIf
    Shows what files would be deleted without prompting or deleting.
#>

param(
    [switch]$Force,
    [switch]$WhatIf
)

# Define the tex directory relative to script location
$TexDir = Join-Path (Split-Path $PSScriptRoot -Parent) "tex"

# LaTeX intermediate file extensions to clean up
$LatexTempExtensions = @(
    "*.aux",    # Auxiliary files
    "*.log",    # Compilation logs
    "*.toc",    # Table of contents
    "*.lof",    # List of figures
    "*.lot",    # List of tables
    "*.out",    # Hyperref bookmarks
    "*.bbl",    # Bibliography
    "*.blg",    # Bibliography log
    "*.idx",    # Index
    "*.ilg",    # Index log
    "*.ind",    # Index
    "*.fls",    # File list
    "*.fdb_latexmk", # Latexmk database
    "*.synctex.gz",  # SyncTeX files
    "*.nav",    # Beamer navigation
    "*.snm",    # Beamer
    "*.vrb",    # Beamer verbatim
    "*.figlist", # Figure list
    "*.makefile", # Makefile
    "*.run.xml", # Biblatex
    "*.bcf"     # Biblatex control file
)

Write-Host "LaTeX Cleanup Script" -ForegroundColor Cyan
Write-Host "===================" -ForegroundColor Cyan

# Check if tex directory exists
if (-not (Test-Path $TexDir)) {
    Write-Warning "tex directory not found at: $TexDir"
    exit 1
}

Write-Host "Searching for LaTeX intermediate files in: $TexDir" -ForegroundColor Yellow

# Find all matching files
$FilesToDelete = @()
foreach ($Extension in $LatexTempExtensions) {
    $Files = Get-ChildItem -Path $TexDir -Filter $Extension -Recurse -File
    $FilesToDelete += $Files
}

if ($FilesToDelete.Count -eq 0) {
    Write-Host "No LaTeX intermediate files found to clean up." -ForegroundColor Green
    exit 0
}

# Display files found
Write-Host "`nFound $($FilesToDelete.Count) intermediate files:" -ForegroundColor Yellow
foreach ($File in $FilesToDelete) {
    $RelativePath = $File.FullName.Replace($TexDir, "tex")
    Write-Host "  $RelativePath" -ForegroundColor Gray
}

# Calculate total size
$TotalSizeBytes = ($FilesToDelete | Measure-Object -Property Length -Sum).Sum
$TotalSizeMB = [Math]::Round($TotalSizeBytes / 1MB, 2)
Write-Host "`nTotal size: $TotalSizeMB MB" -ForegroundColor Yellow

if ($WhatIf) {
    Write-Host "`n[WhatIf] Would delete $($FilesToDelete.Count) files." -ForegroundColor Magenta
    exit 0
}

# Confirm deletion unless Force is specified
if (-not $Force) {
    $Confirm = Read-Host "`nDelete these files? [y/N]"
    if ($Confirm -notmatch "^[Yy]") {
        Write-Host "Cleanup cancelled." -ForegroundColor Yellow
        exit 0
    }
}

# Delete files
Write-Host "`nDeleting files..." -ForegroundColor Green
$DeletedCount = 0
$ErrorCount = 0

foreach ($File in $FilesToDelete) {
    try {
        Remove-Item $File.FullName -Force
        $DeletedCount++
    }
    catch {
        $ErrorCount++
        $RelativePath = $File.FullName.Replace($TexDir, "tex")
        Write-Warning "Failed to delete: $RelativePath - $($_.Exception.Message)"
    }
}

# Summary
Write-Host "`nCleanup complete!" -ForegroundColor Green
Write-Host "Deleted: $DeletedCount files" -ForegroundColor Green
if ($ErrorCount -gt 0) {
    Write-Warning "Errors: $ErrorCount files could not be deleted"
}
Write-Host "Freed: $TotalSizeMB MB of disk space" -ForegroundColor Green