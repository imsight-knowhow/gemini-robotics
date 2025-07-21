# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a learning and research project focused on surveying Google's Gemini Robotics technologies. The repository is currently in its initial stages and serves as a knowledge base for understanding:

- Embodied reasoning models (Gemini Robotics-ER)
- On-device AI capabilities (Gemini Robotics On-Device)
- Developer tools and SDKs for robotics applications
- Complex task learning for robotic systems
- Safety considerations in AI-powered robotics

## Repository Structure

This is a documentation-focused repository organized around research and documentation of Gemini Robotics concepts. The architecture follows a knowledge-first approach optimized for systematic research exploration.

### Core Architecture

**Primary Technologies:**
- **Markdown** - Primary documentation format (28+ files)
- **LaTeX/TeX** - Academic document preparation and presentations
- **PowerShell/Shell Scripts** - Utility scripts for workflow automation

### Key Directories

**`context/` - Centralized Knowledge Base**
The heart of the repository following standardized organization:
- `context/design/` - Technical specifications and architecture diagrams
- `context/hints/` - Research methodologies and development guides  
- `context/logs/` - Research session records and findings
- `context/plans/` - Implementation roadmaps and research strategies
- `context/refcode/` - Reference implementations and code examples
- `context/summaries/` - Knowledge base documents and concept explanations
- `context/tasks/` - Current and planned research work items
- `context/tools/` - Development utilities and analysis scripts

**`analysis/` - Research Analysis**
- Comprehensive analysis of research papers and findings
- Source materials and converted markdown documents
- Technical reports on Gemini Robotics capabilities

**`tex/` - LaTeX Documents**
- Academic document preparation
- Presentation slides and formatted reports
- Research notes and publications

**`scripts/` - Utility Scripts**
- LaTeX cleanup automation
- Development workflow helpers

Always check the context directory for existing knowledge before starting new research tasks.

## Development Workflow

### Research Workflow
1. **Check existing knowledge** in `context/` before starting new tasks
2. **Track progress** using markdown files in `context/tasks/`
3. **Document findings** in appropriate `context/` subdirectories
4. **Update analysis** in `analysis/` directory for comprehensive outputs

### LaTeX Workflow
- LaTeX documents stored in `tex/` directory
- **Cleanup command:** Use `scripts/cleanup-pdflatex-temp-files.ps1` (PowerShell) or `scripts/cleanup-pdflatex-temp-files.sh` (Shell) to remove intermediate files
- When using `pdflatex`, enable tracing back from PDF to LaTeX source code for debugging
- **Supported file types:** .tex, .pdf, presentation slides, academic papers

### Common Commands
- **LaTeX cleanup:** `./scripts/cleanup-pdflatex-temp-files.ps1` (removes .aux, .log, .toc, .out, .synctex.gz, etc.)
- **LaTeX cleanup (Linux/Mac):** `./scripts/cleanup-pdflatex-temp-files.sh`
- **No build/test commands** - This is a documentation-only repository

## Development Notes

- This is primarily a research and documentation repository
- No build tools, test frameworks, or development dependencies are configured
- Content focuses on educational exploration of Gemini Robotics technologies
- The project is not affiliated with Google or DeepMind
- **Architecture Pattern:** Knowledge-first organization optimized for AI assistant continuity

## Research Focus Areas

The project specifically explores:
- **Gemini Robotics-ER** - Embodied reasoning models for physical world understanding
- **Gemini Robotics On-Device** - Optimized models for local robotic hardware execution  
- **Gemini Robotics SDK** - Developer tools and integration frameworks
- **Complex Task Learning** - Multi-step robotic task execution capabilities
- **Safety Considerations** - AI safety in robotics applications

## Key References

- Google DeepMind Blog: https://deepmind.google/discover/blog/gemini-robotics-brings-ai-into-the-physical-world/
- Research Paper: https://arxiv.org/abs/2503.20020
- Research Paper (HTML): https://arxiv.org/html/2503.20020v1