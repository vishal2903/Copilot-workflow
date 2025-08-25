# Complete Dependencies and Installation Guide

## Core Requirements

Install all required dependencies with pip:

```bash
pip install click PyPDF2 pytz rich google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client openai python-dotenv tiktoken python-docx pathlib
```

## LiveKit Voice Functionality (Optional)

For voice functionality, install LiveKit agents:

```bash
pip install livekit-agents livekit-plugins-openai livekit-plugins-silero
```

## Detailed Dependencies

### Core Python Libraries
- `click` - CLI framework
- `PyPDF2` - PDF reading
- `pytz` - Timezone handling
- `rich` - Beautiful CLI output
- `pathlib` - Path handling (usually built-in)

### Google APIs
- `google-auth` - Google authentication
- `google-auth-oauthlib` - OAuth flow
- `google-auth-httplib2` - HTTP library for Google APIs
- `google-api-python-client` - Google Drive/Docs API client

### AI/LLM Libraries
- `openai` - OpenAI GPT API
- `python-dotenv` - Environment variable loading
- `tiktoken` - Token counting for OpenAI models

### Document Processing
- `python-docx` - Word document creation/editing

### LiveKit Voice (Optional)
- `livekit-agents` - LiveKit agent framework
- `livekit-plugins-openai` - OpenAI integration for LiveKit
- `livekit-plugins-silero` - Silero VAD for voice detection

## Installation Commands

### Minimal Installation (Text Mode Only)
```bash
pip install click PyPDF2 pytz rich google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client openai python-dotenv tiktoken python-docx
```

### Full Installation (Including Voice)
```bash
pip install click PyPDF2 pytz rich google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client openai python-dotenv tiktoken python-docx livekit-agents livekit-plugins-openai livekit-plugins-silero
```

## Usage Modes

### Text Mode (Default)
```bash
python gdrive_analyzer.py
```

### Voice Mode (Text interface with voice indicators)
```bash
python gdrive_analyzer.py --voice
```

### LiveKit Server Mode (Full voice functionality)
```bash
python gdrive_analyzer.py --livekit
```

## Key Features Implemented

### 1. Voice Functionality (LiveKit Integration)
- ✅ Proper Agent class with function_tool decorators
- ✅ AgentSession with STT, LLM, TTS, and VAD
- ✅ Voice optimized responses (shorter, concise)
- ✅ Based on working api.py implementation

### 2. Analyze Trigger with Pattern Learning
- ✅ 'analyze' keyword triggers deep analysis
- ✅ Pattern learning from previous analyses
- ✅ Hypothesis generation using learned patterns
- ✅ Memory storage in learned_patterns.json

### 3. Chat Continuity
- ✅ Conversation context maintained across interactions
- ✅ Follow-up questions on previous topics
- ✅ Context-aware responses
- ✅ Last 10 interactions remembered

### 4. Automatic Tool Triggering
- ✅ 'analyze' → triggers analyze_with_patterns()
- ✅ 'upload', 'gdrive' → triggers upload_to_drive()
- ✅ 'save', 'export' → triggers save_report()
- ✅ Intelligent keyword detection in prompts

### 5. Enhanced Agent Functions
- `analyze_with_patterns()` - Deep analysis with pattern learning
- `ask_with_continuity()` - Q&A with conversation context
- `web_research_only()` - Web research without Drive docs
- `follow_up()` - Continue previous topic discussions
- `save_report()` - Save analyses locally
- `upload_to_drive()` - Upload to Google Drive

## Testing

Test syntax:
```bash
python -m py_compile gdrive_analyzer.py
```

Test core functionality:
```bash
python -c "from gdrive_analyzer import PatternLearningEngine; print('Working!')"
```

Test LiveKit availability:
```bash
python -c "from gdrive_analyzer import LIVEKIT_AVAILABLE; print(f'LiveKit: {LIVEKIT_AVAILABLE}')"
```

## Environment Setup

Create `.env` file with:
```
GOOGLE_CREDENTIALS_PATH=path/to/credentials.json
GOOGLE_DRIVE_TOKEN=path/to/token.json
GOOGLE_DOCS_TOKEN=path/to/docstoken.json
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-4o-mini
SYSTEM_PROMPT_PATH=sytem_prompt.md
```

## Common Issues

1. **LiveKit not available**: Install with `pip install livekit-agents`
2. **Google API errors**: Regenerate tokens by deleting token files
3. **Unicode errors**: Handled automatically in the code
4. **Memory issues**: Pattern learning keeps only last 20 patterns per category