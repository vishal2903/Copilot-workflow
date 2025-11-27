# ZENO Voice Copilot 🎙️

> An AI-powered teaching assistant with multi-agent architecture for curriculum development

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LiveKit](https://img.shields.io/badge/LiveKit-Voice%20AI-green.svg)](https://livekit.io/)

ZENO is a Jarvis-style voice copilot designed to help instructors create detailed, contextual lesson plans. It coordinates a team of specialized AI agents to gather requirements, retrieve course materials, research community sentiment, compose structured content, and validate quality—all through natural voice conversation.

---

## 🎯 Features

### Multi-Agent Architecture
- **Orchestrator Agent**: Coordinates all specialists, manages state, handles voice I/O
- **HITL Agent**: Human-in-the-loop discovery with strict question-by-question flow
- **Retrieval Agent**: Index-based lookup with automatic cross-referencing
- **Web Agent**: Community sentiment research with retry/backoff
- **Composer Agent**: Writes 8 priority sections grounded in retrieved content
- **QA Agent**: Validates completeness, grounding, and detects generic content

### Voice-First Experience
- Natural conversation flow like Jarvis from Iron Man
- Transparent agent handoffs ("Let me check with my retrieval specialist...")
- Progress updates with fun facts during long operations
- Real-time speech-to-text and text-to-speech via LiveKit

### Smart Retrieval
- Parses index documents for exact content line ranges
- Extracts relevant sections from course material databases
- Automatic cross-referencing of related topics
- Google Drive integration for supplementary materials

### Quality Assurance
- Validates all 8 sections are present and complete
- Ensures "Connecting Points" are grounded in actual course materials
- Detects and flags generic/filler content
- Marks genuinely missing data as `<TBD>`

---

## 📋 Table of Contents

- [Architecture](#-architecture)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [The 8 Priority Sections](#-the-8-priority-sections)
- [Agent Details](#-agent-details)
- [API Reference](#-api-reference)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🏗 Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                        USER VOICE INPUT                         │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR (ZENO)                          │
│         Coordinates agents, manages state, voice output         │
└─────────────────────────────┬───────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   HITL AGENT    │ │ RETRIEVAL AGENT │ │   WEB AGENT     │
│                 │ │                 │ │                 │
│ • 7 discovery   │ │ • Index parsing │ │ • Community     │
│   questions     │ │ • Data Doc      │ │   sentiment     │
│ • One-by-one    │ │   extraction    │ │ • Recent        │
│   validation    │ │ • Cross-refs    │ │   developments  │
│ • Skip option   │ │ • Drive search  │ │ • Retry/backoff │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                             ▼
              ┌─────────────────────────────┐
              │      COMPOSER AGENT         │
              │                             │
              │ • 8 priority sections       │
              │ • Grounded in context       │
              │ • HITL inputs prioritized   │
              └──────────────┬──────────────┘
                             │
                             ▼
              ┌─────────────────────────────┐
              │        QA AGENT             │
              │                             │
              │ • Completeness check        │
              │ • Grounding validation      │
              │ • Generic content detection │
              └──────────────┬──────────────┘
                             │
                             ▼
              ┌─────────────────────────────┐
              │    GOOGLE DRIVE UPLOAD      │
              │    + VOICE CONFIRMATION     │
              └─────────────────────────────┘
```

---

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- Google Cloud Project with Drive API enabled
- OpenAI API key
- LiveKit account (for voice functionality)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/zeno-voice-copilot.git
cd zeno-voice-copilot
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
# Core dependencies
pip install click PyPDF2 pytz rich google-auth google-auth-oauthlib \
    google-auth-httplib2 google-api-python-client openai python-dotenv \
    tiktoken python-docx

# Voice functionality (LiveKit)
pip install livekit-agents livekit-plugins-openai livekit-plugins-silero
```

Or install from requirements file:
```bash
pip install -r requirements.txt
```

### Step 4: Set Up Google Cloud Credentials

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable **Google Drive API** and **Google Docs API**
4. Create OAuth 2.0 credentials (Desktop application)
5. Download the credentials JSON file

### Step 5: Configure Environment

Create a `.env` file in the project root:
```env
# Google APIs
GOOGLE_CREDENTIALS_PATH=path/to/credentials.json
GOOGLE_DRIVE_TOKEN=path/to/token.json
GOOGLE_DRIVE_FOLDER_ID=your_folder_id  # Optional: specific folder for uploads

# OpenAI
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o-mini

# System Configuration
SYSTEM_PROMPT_PATH=system_prompt.txt
DATA_DOC_PATH=Gdrive_workflow/Data_Doc.txt
INDEX_DOC_PATH=Gdrive_workflow/index_with_tags_lines_only.txt

# LiveKit (for voice)
LIVEKIT_URL=wss://your-livekit-server.livekit.cloud
LIVEKIT_API_KEY=your_api_key
LIVEKIT_API_SECRET=your_api_secret
```

---

## ⚙️ Configuration

### Data Documents

ZENO requires two key documents for retrieval:

#### Index Document (`index_with_tags_lines_only.txt`)

Maps topics to line ranges in the Data Doc:
```
topic = Lecture 5: ControlNets (line no = 239-319) — tags: Subject, Medium, Key ControlNet Types
topic = Lecture 6: IP-Adapters, InstantID, and Inpainting (line no = 320-415) — tags: Inpainting, IP-Adapter
```

#### Data Document (`Data_Doc.txt`)

Contains all course materials, lecture notes, and reference content. The Retrieval Agent extracts specific line ranges based on Index lookups.

### System Prompt

The `system_prompt.txt` file defines agent behavior, section requirements, and quality gates. Customize it for your specific curriculum needs.

---

## 🎮 Usage

### Starting the Voice Agent
```bash
python w2.py
```

This launches the LiveKit voice agent. Connect via a LiveKit-compatible client.

### Example Conversation
```
You: "Make a lesson plan for Lecture 5: ControlNets"

Zeno: "I'll coordinate with my specialist team to build this lesson plan. 
       Let me start with my curriculum discovery specialist..."

Zeno: "Question 1 of 7: Who is the target audience and what's their 
       experience level with AI/ML concepts?"

You: "Cohort 5, mix of technical and non-technical backgrounds"

Zeno: "Got it - mixed technical/non-technical Cohort 5. 
       Question 2 of 7: What specific learning outcomes should students 
       achieve by the end of this lecture?"

You: "They should understand how to use Canny and Depth ControlNets 
      and build a basic workflow in ComfyUI"

... [continues through all 7 questions] ...

Zeno: "Discovery complete. Let me check with my retrieval specialist..."

Zeno: "Found 3 relevant sections in course materials, plus 2 related 
       topics for cross-reference..."

Zeno: "My web research specialist is checking community discussions..."

Zeno: "Fun fact: ControlNets were introduced in February 2023 and have 
       become the most popular method for guided image generation..."

Zeno: "Composing your lesson plan with 8 priority sections..."

Zeno: "Running quality checks... All sections validated, content is 
       grounded in your course materials..."

Zeno: "Done! Your lesson plan has been uploaded to Google Drive. 
       Feel free to ask follow-up questions about any section!"
```

### Voice Commands

| Command | Action |
|---------|--------|
| "Start lesson plan for [topic]" | Begins HITL discovery |
| "Skip" or "Use defaults" | Skips remaining questions |
| "Quick analysis of [topic]" | Fast retrieval without full generation |
| "Save transcript" | Saves conversation to Drive |
| Follow-up questions | Answered from last lesson plan context |

---

## 📁 Project Structure
```
zeno-voice-copilot/
│
├── w2.py                      # Main application (multi-agent system)
├── system_prompt.txt          # Agent behavior configuration
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables (not in repo)
│
├── Gdrive_workflow/
│   ├── Data_Doc.txt           # Course materials database
│   ├── index_with_tags_lines_only.txt  # Topic index with line ranges
│   ├── Cohort notes.txt       # Additional reference materials
│   └── requirements_simple.txt # Minimal dependencies list
│
├── credentials/               # Google API credentials (not in repo)
│   ├── credentials.json
│   └── token.json
│
└── README.md                  # This file
```

---

## 📚 The 8 Priority Sections

Every lesson plan generated by ZENO includes these sections:

| # | Section | Description |
|---|---------|-------------|
| 1 | **First-Principles Thinking** | 4-6 Socratic questions guiding core understanding |
| 2 | **Assignments and Practice Sets** | 2-3 detailed assignments with rubrics |
| 3 | **Connecting Points** | Past dependencies + future unlocks (MUST be grounded) |
| 4 | **HITL Conversation Data** | Full discovery Q&A documentation |
| 5 | **Drive Data** | Referenced documents and extracted examples |
| 6 | **User Journeys and Scenarios** | 2-3 realistic use cases with edge cases |
| 7 | **Requirements** | Functional, model/tool, and data requirements |
| 8 | **Risks and Mitigations** | Technical, pedagogical, and operational risks |

---

## 🤖 Agent Details

### HITL Agent

Manages the discovery conversation with strict validation:

**7 Discovery Questions:**
1. Target audience and experience level
2. Specific learning outcomes
3. Prior lectures to connect (past dependencies)
4. Future lectures this enables (future unlocks)
5. Time constraints
6. Tools/platform requirements
7. Common misconceptions to address

**Features:**
- One question at a time with answer validation
- LLM-based answer quality checking
- Paraphrasing for key points extraction
- "Skip" command for using defaults

### Retrieval Agent

Handles all content retrieval:

**Pipeline:**
1. Parse Index Doc for topic matches (title + tags)
2. Extract exact line ranges from Data Doc
3. Find cross-references for related topics
4. Search Google Drive for supplementary materials
5. Combine with clear source provenance

**Cross-Reference Logic:**
- Shares tags with primary topic
- Mentioned in HITL connection answers
- Topically similar based on index proximity

### Web Agent

Gathers external context:

- Community sentiment (Reddit, Twitter, LinkedIn, blogs)
- Recent developments (last 6 months)
- Retry with exponential backoff (3 attempts)
- Graceful degradation on failure

### Composer Agent

Writes all 8 sections:

- Section-specific prompts for focused generation
- Full context injection (HITL + retrieval + web)
- HITL inputs prioritized over defaults
- Grounding verification for each section

### QA Agent

Validates output quality:

**Checks:**
- [ ] All 8 sections present and non-empty
- [ ] Connecting Points references actual course materials
- [ ] No generic filler content detected
- [ ] HITL answers incorporated
- [ ] `<TBD>` markers only where genuinely missing

---

## 🔌 API Reference

### Function Tools
```python
@function_tool
async def start_lesson_plan(context: RunContext, topic: str) -> str:
    """Start generating a lesson plan for a topic"""

@function_tool
async def process_discovery_answer(context: RunContext, answer: str) -> str:
    """Process an answer during HITL discovery"""

@function_tool
async def skip_discovery(context: RunContext) -> str:
    """Skip discovery questions and use defaults"""

@function_tool
async def followup_question(context: RunContext, question: str) -> str:
    """Answer follow-up questions about the last lesson plan"""

@function_tool
async def quick_analysis(context: RunContext, topic: str) -> str:
    """Quick topic analysis without full generation"""

@function_tool
async def save_transcript(context: RunContext) -> str:
    """Save conversation transcript to Google Drive"""
```

### Data Classes
```python
@dataclass
class HITLState:
    stage: HITLStage
    topic: Optional[str]
    current_question_idx: int
    questions: List[HITLQuestion]
    assumptions: List[str]
    notes: List[str]

@dataclass
class RetrievalResult:
    topic: str
    index_matches: List[Dict]
    data_doc_content: str
    cross_references: List[Dict]
    drive_docs: List[Dict]
    drive_content: str

@dataclass
class LessonPlan:
    topic: str
    sections: List[ComposedSection]
    hitl_summary: str
    retrieval_sources: List[str]
    qa_passed: bool
    qa_issues: List[str]
```

---

## 🛠 Development

### Running Tests
```bash
# Syntax check
python -m py_compile w2.py

# Test imports
python -c "from w2 import ZenoOrchestrator; print('✓ Imports OK')"

# Test retrieval agent
python -c "
from w2 import Config, RetrievalAgent, GoogleDriveService, OpenAI
config = Config()
# ... test retrieval
"
```

### Adding New Agents

1. Create a new class following the agent pattern:
```python
class MyNewAgent:
    def __init__(self, openai_client: OpenAI, model: str):
        self.client = openai_client
        self.model = model
    
    async def process(self, input_data: Any) -> Result:
        # Agent logic here
        pass
```

2. Register in `ZenoOrchestrator.__init__`:
```python
self.my_new_agent = MyNewAgent(self.openai_client, config.openai_model)
```

3. Integrate into the generation pipeline in `_generate_lesson_plan`

### Customizing Sections

Edit `ComposerAgent.PRIORITY_SECTIONS` and corresponding prompts in `_compose_section` to add/modify sections.

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Use type hints for all function signatures
- Document classes and methods with docstrings
- Keep agent responsibilities focused and single-purpose

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [LiveKit](https://livekit.io/) for real-time voice infrastructure
- [OpenAI](https://openai.com/) for language model APIs
- [100xEngineers](https://100xengineers.com/) for the curriculum framework

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/zeno-voice-copilot/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/zeno-voice-copilot/discussions)
- **Email**: your.email@example.com

---

<p align="center">
  <strong>Built with ❤️ for educators by educators</strong>
</p>
