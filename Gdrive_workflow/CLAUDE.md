# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Google Drive LLM Analyzer designed for the 100x Engineers cohort. It's a single-file Python CLI application that analyzes Google Drive documents using OpenAI GPT models with an integrated "Second-Brain" system prompt for comprehensive educational and business evaluation.

## Setup & Configuration

### Environment Setup
```bash
pip install -r requirements_simple.txt
python gdrive_analyzer.py setup  # Creates .env template
```

### Required Environment Variables (.env)
```bash
GOOGLE_CREDENTIALS_PATH=path/to/credentials.json
GOOGLE_DRIVE_TOKEN=path/to/token.json
GOOGLE_DOCS_TOKEN=path/to/docstoken.json
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-4o-mini
SYSTEM_PROMPT_PATH=sytem_prompt.md
```

### Google API Setup
1. Enable Google Drive API and Google Docs API in Google Cloud Console
2. Create OAuth 2.0 credentials (Desktop Application)
3. Download `credentials.json` file
4. First run will trigger OAuth flow to generate tokens

## Core Commands

### Main Analysis Commands
```bash
# Comprehensive analysis with web research
python gdrive_analyzer.py analyze -t "Multi agentic systems" --query "name contains 'agent'"

# Q&A against Drive documents
python gdrive_analyzer.py ask "How do I build production AI agents?"

# Web research only (no Drive documents)
python gdrive_analyzer.py research "Latest AI trends 2025"

# Test all connections
python gdrive_analyzer.py test

# Setup wizard
python gdrive_analyzer.py setup
```

### Common Options
- `-t, --topic`: Focus topic for analysis
- `-q, --query`: Google Drive search query
- `-l, --limit`: Max documents to analyze (default: 10)
- `--web-research/--no-web-research`: Enable/disable web search (default: enabled)
- `--docs-only`: Analyze only documents, no web research

## Architecture Overview

### Core Classes (all in gdrive_analyzer.py)
- **Config**: Environment configuration and validation
- **GoogleDriveClient**: Google Drive/Docs API integration with authentication
- **DocumentReader**: Multi-format document parsing (PDF, DOCX, Google Docs, TXT)
- **OpenAIAnalyzer**: OpenAI integration with system prompt and web search
- **ReportGenerator**: Report formatting and dual output (local + Drive upload)

### Data Flow
1. **Config** loads environment and system prompt
2. **GoogleDriveClient** authenticates and searches/downloads documents
3. **DocumentReader** extracts text and URLs from various formats
4. **OpenAIAnalyzer** processes content with system prompt + web research
5. **ReportGenerator** creates formatted reports with active hyperlinks

## System Prompt Integration (100x Engineers Framework)

The tool uses a comprehensive system prompt (`sytem_prompt.md`) that implements:

### Evaluation Framework
- **Gate A**: Curriculum Relevance scoring (must score ≥4.0 to proceed to Teach Lane)
- **Teach Lane**: Build-Now readiness, Student Impact assessment
- **Biz-Scout Lane**: Commercial opportunity evaluation with 72-hour spike recommendations
- **11-Section Output Format**: TL;DR, First-Principles History, Cohort Relevance, etc.

### Key Behaviors
- Prioritizes Google Drive documents over web sources
- Uses Asia/Kolkata timestamps
- Applies audience priors (Engineers 41%, Founders 11%, etc.)
- Enforces structured decision gates with numerical thresholds

## Google Drive Query Syntax

### Basic Queries
```bash
# Search by filename
--query "name contains 'agent'"

# Search by content
--query "fullText contains 'artificial intelligence'"

# Multiple conditions
--query "name contains 'RAG' or fullText contains 'retrieval'"

# File type filtering (handled automatically)
# Supports: PDF, DOCX, DOC, TXT, Google Docs
```

### Important Notes
- Use `contains` not `contain`
- Use single quotes inside double quotes for string values
- Combine with `and`, `or` operators

## Report Generation

### Automatic Behavior
- **Local Save**: Always saves to `C:\Users\visha\Downloads` (markdown format)
- **Google Drive Upload**: Always uploads with active hyperlinks (Google Docs format)
- **Hyperlink Processing**: URLs in reports become clickable links in Google Drive version

### Report Types
- **Analysis Reports**: Comprehensive topic evaluation with 100x framework
- **Q&A Reports**: Question-answer format with source documents
- **Research Reports**: Web research summaries

## Authentication & Tokens

### OAuth Flow
- First run triggers browser-based OAuth
- Creates separate tokens for Drive (`token.json`) and Docs (`docstoken.json`)
- Tokens auto-refresh when expired

### Scopes Required
- Google Drive: `drive.readonly`, `drive.file`
- Google Docs: `documents` (full access for hyperlink creation)

### Token Issues
If authentication fails:
```bash
# Delete tokens to force re-auth
rm "path/to/token.json"
rm "path/to/docstoken.json"
python gdrive_analyzer.py test  # Triggers re-auth
```

## Development & Debugging

### Testing Individual Components
```bash
# Test syntax
python -m py_compile gdrive_analyzer.py

# Test connections
python gdrive_analyzer.py test

# Test with minimal query
python gdrive_analyzer.py analyze -t "test" --limit 1
```

### Common Issues
- **Invalid Query Error**: Check Google Drive query syntax (use `contains` not `contain`)
- **403 Permission Error**: Delete and regenerate tokens after scope changes
- **Unicode Errors**: The tool includes unicode sanitization functions
- **Timeout Errors**: Large document sets may timeout; use `--limit` to reduce scope

### Logging
- Logs to `gdrive_analyzer.log` 
- Shows authentication flow, API calls, and document processing status
- Set logging levels in the Config class

## Key Implementation Notes

### Document Processing
- **Concurrent Processing**: Uses ThreadPoolExecutor for batch document downloads
- **Retry Logic**: Exponential backoff for Google API calls
- **Error Handling**: Graceful degradation when documents fail to process
- **Format Support**: PDF (PyPDF2), DOCX (python-docx), Google Docs (native API)

### LLM Integration
- **Model Flexibility**: Configurable via OPENAI_MODEL environment variable
- **Web Search**: Uses OpenAI Responses API with web search preview
- **Token Management**: Includes tiktoken for token counting and limits
- **System Prompt**: Always sent with user queries for consistent evaluation framework

### Report Formatting
- **Dual Output**: Local markdown + Google Drive with hyperlinks
- **URL Processing**: Automatically converts URLs to active hyperlinks in Google Drive version
- **Rich CLI**: Uses Rich library for formatted console output and progress tracking