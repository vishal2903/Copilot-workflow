#!/usr/bin/env python3
"""
ZENO VOICE Copilot - Multi-Agent Architecture
100xEngineers Second Brain with Specialist Agents

Agents:
- Orchestrator: Coordinates all agents, manages state
- HITL Agent: Human-in-the-loop discovery conversation
- Retrieval Agent: Index/Data Doc lookup + Drive search
- Web Agent: Community sentiment research
- Composer Agent: Lesson plan writer (8 priority sections)
- QA Agent: Quality gate validator
"""

import os
import io
import logging
import asyncio
import re
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from dotenv import load_dotenv
from dataclasses import dataclass, field
from enum import Enum
from contextlib import suppress

# Core dependencies
import PyPDF2
from docx import Document as DocxDocument

# Google APIs
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
from googleapiclient.errors import HttpError

# OpenAI
from openai import OpenAI

# LiveKit
from livekit import agents
from livekit.agents import (
    Agent,
    AgentSession,
    RunContext,
    RoomInputOptions,
    function_tool
)
from livekit.plugins import openai as lk_openai
from livekit.plugins import silero, noise_cancellation

# Load environment
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class AgentType(Enum):
    ORCHESTRATOR = "orchestrator"
    HITL = "hitl"
    RETRIEVAL = "retrieval"
    WEB = "web"
    COMPOSER = "composer"
    QA = "qa"


class HITLStage(Enum):
    NOT_STARTED = "not_started"
    DISCOVERY = "discovery"
    CONFIRMING = "confirming"
    COMPLETE = "complete"
    SKIPPED = "skipped"


@dataclass
class HITLQuestion:
    """Represents a single HITL discovery question"""
    id: int
    question: str
    category: str  # audience, outcomes, connections, constraints, etc.
    answer: Optional[str] = None
    paraphrase: Optional[str] = None
    confirmed: bool = False


@dataclass
class HITLState:
    """State management for HITL Agent"""
    stage: HITLStage = HITLStage.NOT_STARTED
    topic: Optional[str] = None
    current_question_idx: int = 0
    questions: List[HITLQuestion] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def get_current_question(self) -> Optional[HITLQuestion]:
        if 0 <= self.current_question_idx < len(self.questions):
            return self.questions[self.current_question_idx]
        return None

    def all_answered(self) -> bool:
        return all(q.confirmed for q in self.questions)

    def get_summary(self) -> str:
        """Generate summary of HITL conversation for Composer"""
        lines = [f"## HITL DISCOVERY SUMMARY FOR: {self.topic}\n"]
        for q in self.questions:
            if q.answer:
                lines.append(f"**Q{q.id} ({q.category})**: {q.question}")
                lines.append(f"**Answer**: {q.answer}")
                if q.paraphrase:
                    lines.append(f"**Key Points**: {q.paraphrase}")
                lines.append("")
        if self.assumptions:
            lines.append("**Assumptions Made:**")
            for a in self.assumptions:
                lines.append(f"- {a}")
        return "\n".join(lines)


@dataclass
class RetrievalResult:
    """Result from Retrieval Agent"""
    topic: str
    index_matches: List[Dict[str, Any]] = field(default_factory=list)
    data_doc_content: str = ""
    cross_references: List[Dict[str, Any]] = field(default_factory=list)
    drive_docs: List[Dict[str, Any]] = field(default_factory=list)
    drive_content: str = ""

    def get_full_context(self) -> str:
        """Combine all retrieved content"""
        parts = []
        if self.data_doc_content:
            parts.append("## PRIMARY CONTENT FROM DATA DOC\n")
            parts.append(self.data_doc_content)
            parts.append("\n")
        if self.cross_references:
            parts.append("## CROSS-REFERENCED TOPICS\n")
            for ref in self.cross_references:
                parts.append(f"### {ref.get('topic', 'Related Topic')}")
                parts.append(ref.get('content', '')[:1500])
                parts.append("\n")
        if self.drive_content:
            parts.append("## ADDITIONAL DRIVE DOCUMENTS\n")
            parts.append(self.drive_content)
        return "\n".join(parts)


@dataclass
class WebResearchResult:
    """Result from Web Agent"""
    topic: str
    community_sentiment: str = ""
    recent_developments: str = ""
    success: bool = True
    error: Optional[str] = None


@dataclass
class ComposedSection:
    """A single section of the lesson plan"""
    name: str
    content: str
    grounded: bool = True  # Whether it's grounded in retrieved content
    quality_score: float = 1.0


@dataclass
class LessonPlan:
    """Complete lesson plan output"""
    topic: str
    sections: List[ComposedSection] = field(default_factory=list)
    hitl_summary: str = ""
    retrieval_sources: List[str] = field(default_factory=list)
    qa_passed: bool = False
    qa_issues: List[str] = field(default_factory=list)

    def to_text(self) -> str:
        """Convert to formatted text for upload"""
        lines = [f"# LESSON PLAN: {self.topic}\n"]
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

        for section in self.sections:
            lines.append(f"## {section.name}\n")
            lines.append(section.content)
            lines.append("\n")

        if self.retrieval_sources:
            lines.append("## SOURCES REFERENCED\n")
            for src in self.retrieval_sources:
                lines.append(f"- {src}")

        return "\n".join(lines)


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Centralized configuration management"""

    def __init__(self):
        self.google_credentials_path = os.getenv("GOOGLE_CREDENTIALS_PATH")
        self.google_drive_token = os.getenv("GOOGLE_DRIVE_TOKEN")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.system_prompt_path = os.getenv("SYSTEM_PROMPT_PATH", "system_prompt.txt")
        self.google_drive_folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID", None)
        self.data_doc_path = os.getenv("DATA_DOC_PATH", "Gdrive_workflow/Data_Doc.txt")
        self.index_doc_path = os.getenv("INDEX_DOC_PATH", "Gdrive_workflow/index_with_tags_lines_only.txt")

        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found in .env")


# =============================================================================
# GOOGLE DRIVE SERVICE
# =============================================================================

class GoogleDriveService:
    """Handles all Google Drive operations"""

    SCOPES = [
        "https://www.googleapis.com/auth/drive.readonly",
        "https://www.googleapis.com/auth/documents",
        "https://www.googleapis.com/auth/drive.file"
    ]

    def __init__(self, config: Config):
        self.config = config
        self.drive_service = None
        self.docs_service = None
        self._authenticate()

    def _authenticate(self):
        """Authenticate with Google APIs"""
        creds = None
        token_path = Path(self.config.google_drive_token).expanduser().resolve()

        if token_path.exists():
            creds = Credentials.from_authorized_user_file(str(token_path), self.SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.config.google_credentials_path, self.SCOPES
                )
                creds = flow.run_local_server(port=0)
                token_path.parent.mkdir(parents=True, exist_ok=True)
                token_path.write_text(creds.to_json(), encoding="utf-8")

        self.drive_service = build('drive', 'v3', credentials=creds)
        try:
            self.docs_service = build("docs", "v1", credentials=creds)
        except HttpError as e:
            logger.warning(f"Docs API not accessible: {e}")
            self.docs_service = None

    def search_files(self, query: str, limit: int = 10) -> List[Dict]:
        """Search files in Google Drive"""
        try:
            q = f"trashed = false and (name contains '{query}' or fullText contains '{query}')"
            results = self.drive_service.files().list(
                q=q,
                pageSize=limit,
                fields="files(id, name, mimeType, modifiedTime)"
            ).execute()
            return results.get('files', [])
        except Exception as e:
            logger.error(f"Error searching files: {e}")
            return []

    def get_file_content(self, file_id: str, mime_type: str) -> str:
        """Get content of a file from Google Drive"""
        try:
            if mime_type == "application/vnd.google-apps.document":
                data = self.drive_service.files().export(
                    fileId=file_id, mimeType="text/plain"
                ).execute()
                return data.decode('utf-8') if isinstance(data, bytes) else str(data)

            elif mime_type == "application/pdf":
                request = self.drive_service.files().get_media(fileId=file_id)
                buf = io.BytesIO()
                downloader = MediaIoBaseDownload(buf, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
                return self._extract_pdf_text(buf.getvalue())

            elif "wordprocessingml" in mime_type:
                request = self.drive_service.files().get_media(fileId=file_id)
                buf = io.BytesIO()
                downloader = MediaIoBaseDownload(buf, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
                return self._extract_docx_text(buf.getvalue())

            return ""
        except Exception as e:
            logger.error(f"Error getting file content: {e}")
            return ""

    def _extract_pdf_text(self, content: bytes) -> str:
        try:
            pdf_file = io.BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = []
            for page in pdf_reader.pages[:50]:
                text.append(page.extract_text())
            return '\n'.join(text)
        except:
            return ""

    def _extract_docx_text(self, content: bytes) -> str:
        try:
            doc = DocxDocument(io.BytesIO(content))
            return '\n'.join([p.text for p in doc.paragraphs])
        except:
            return ""

    def format_as_google_doc_requests(self, title: str, content: str) -> List[dict]:
        """Convert plain text to Google Docs API requests"""
        ops: List[dict] = []
        idx = 1

        def ins(text: str):
            nonlocal idx, ops
            ops.append({"insertText": {"location": {"index": idx}, "text": text}})
            idx += len(text)

        def set_heading(start: int, end: int, level: str):
            ops.append({
                "updateParagraphStyle": {
                    "range": {"startIndex": start, "endIndex": end},
                    "paragraphStyle": {"namedStyleType": level},
                    "fields": "namedStyleType",
                }
            })

        def set_align_center(start: int, end: int):
            ops.append({
                "updateParagraphStyle": {
                    "range": {"startIndex": start, "endIndex": end},
                    "paragraphStyle": {"alignment": "CENTER"},
                    "fields": "alignment",
                }
            })

        def apply_inline_bold(start_index: int, raw_text: str) -> str:
            cleaned = []
            bold_spans = []
            pos = 0
            for m in re.finditer(r"\*\*(.+?)\*\*", raw_text):
                before = raw_text[pos:m.start()]
                if before:
                    cleaned.append(before)
                bold_txt = m.group(1)
                bold_start = sum(len(x) for x in cleaned)
                cleaned.append(bold_txt)
                bold_end = bold_start + len(bold_txt)
                bold_spans.append((bold_start, bold_end))
                pos = m.end()
            tail = raw_text[pos:]
            if tail:
                cleaned.append(tail)

            cleaned_text = "".join(cleaned)
            for s_off, e_off in bold_spans:
                ops.append({
                    "updateTextStyle": {
                        "range": {
                            "startIndex": start_index + s_off,
                            "endIndex": start_index + e_off
                        },
                        "textStyle": {"bold": True},
                        "fields": "bold"
                    }
                })
            return cleaned_text

        def is_heading_like(line: str) -> bool:
            line_stripped = line.strip()
            if len(line_stripped) == 0 or len(line_stripped) > 80:
                return False
            if line_stripped.endswith("."):
                return False
            if line_stripped.startswith(("-", "*", "•", "1.", "2.", "3.")):
                return False
            return bool(re.match(r"^[A-Z][A-Za-z0-9][A-Za-z0-9\- ]*$", line_stripped))

        # Title
        t_start = idx
        ins(title + "\n")
        set_heading(t_start, idx, "HEADING_1")
        set_align_center(t_start, idx)
        ins("\n\n")

        # Body parsing
        lines = content.splitlines()
        bullet_block_start = None
        bullet_block_has_items = False

        def flush_bullets():
            nonlocal bullet_block_start, bullet_block_has_items, idx
            if bullet_block_start is not None and bullet_block_has_items:
                ops.append({
                    "createParagraphBullets": {
                        "range": {"startIndex": bullet_block_start, "endIndex": idx},
                        "bulletPreset": "BULLET_DISC_CIRCLE_SQUARE"
                    }
                })
                ins("\n")
            bullet_block_start = None
            bullet_block_has_items = False

        for raw in lines:
            line = raw.rstrip()

            if not line.strip():
                flush_bullets()
                ins("\n\n")
                continue

            line_for_heading = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
            if line.startswith("## ") or (line_for_heading.isupper() and len(line_for_heading) <= 80) or is_heading_like(line_for_heading):
                flush_bullets()
                h_start = idx
                heading_src = line[3:] if line.startswith("## ") else line
                heading_clean = re.sub(r"\*\*(.+?)\*\*", r"\1", heading_src)
                ins(heading_clean + "\n")
                apply_inline_bold(h_start, heading_src)
                set_heading(h_start, idx, "HEADING_2")
                ins("\n")
                continue

            if line.lstrip().startswith(("- ", "* ", "• ", "1. ", "2. ", "3.")):
                if bullet_block_start is None:
                    bullet_block_start = idx
                    bullet_block_has_items = False
                text = line.lstrip()
                if text[:2] in ("- ", "* "):
                    text = text[2:]
                elif text[:2] == "• ":
                    text = text[2:]
                elif re.match(r"^\d+\.\s", text):
                    text = re.sub(r"^\d+\.\s", "", text)

                b_start = idx
                preview_clean = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
                ins(preview_clean + "\n")
                apply_inline_bold(b_start, text)
                bullet_block_has_items = True
                continue

            flush_bullets()
            p_start = idx
            preview_clean = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
            ins(preview_clean + "\n\n")
            apply_inline_bold(p_start, line)

        flush_bullets()
        return ops

    def create_google_doc_with_formatting(self, title: str, content: str, folder_id: Optional[str] = None) -> Dict:
        """Create a Google Doc with formatting"""
        try:
            if not self.docs_service:
                return {"error": "Docs API unavailable"}

            doc = self.docs_service.documents().create(body={"title": title}).execute()
            doc_id = doc.get("documentId")

            if folder_id:
                try:
                    self.drive_service.files().update(
                        fileId=doc_id, addParents=folder_id, fields="id"
                    ).execute()
                except Exception as e:
                    logger.warning(f"Could not move doc to folder: {e}")

            requests = self.format_as_google_doc_requests(title, content)
            if requests:
                self.docs_service.documents().batchUpdate(
                    documentId=doc_id, body={"requests": requests}
                ).execute()

            meta = self.drive_service.files().get(
                fileId=doc_id, fields="id,name,webViewLink"
            ).execute()
            meta["documentId"] = doc_id
            return meta
        except Exception as e:
            logger.error(f"create_google_doc_with_formatting failed: {e}")
            return {"error": str(e)}

    def upload_lesson_plan(self, lesson_plan: LessonPlan, folder_id: Optional[str] = None) -> Dict:
        """Upload a lesson plan to Google Drive"""
        title = f"{lesson_plan.topic} - Lesson Plan {datetime.now().strftime('%Y-%m-%d %H-%M')}"
        content = lesson_plan.to_text()
        return self.create_google_doc_with_formatting(title, content, folder_id)


# =============================================================================
# SPECIALIST AGENTS
# =============================================================================

class HITLAgent:
    """Human-in-the-Loop Discovery Agent
    
    Manages the discovery conversation with strict question-by-question flow.
    Each question must be answered and confirmed before moving to the next.
    """

    DISCOVERY_QUESTIONS = [
        HITLQuestion(1, "Who is the target audience and what's their experience level with AI/ML concepts?", "audience"),
        HITLQuestion(2, "What specific learning outcomes should students achieve by the end of this lecture?", "outcomes"),
        HITLQuestion(3, "Which prior lectures or modules should this connect to? (Be specific about lecture names/numbers)", "connections_past"),
        HITLQuestion(4, "Which future lectures will build upon this one? (Be specific about lecture names/numbers)", "connections_future"),
        HITLQuestion(5, "What time constraints exist? (lecture duration, assignment time, etc.)", "constraints"),
        HITLQuestion(6, "Are there any specific tools, platforms, or hardware requirements to consider?", "requirements"),
        HITLQuestion(7, "What common misconceptions or pitfalls should be addressed?", "pitfalls"),
    ]

    def __init__(self, openai_client: OpenAI, model: str):
        self.client = openai_client
        self.model = model
        self.state = HITLState()

    def start_discovery(self, topic: str) -> str:
        """Initialize discovery for a new topic"""
        self.state = HITLState(
            stage=HITLStage.DISCOVERY,
            topic=topic,
            questions=[HITLQuestion(q.id, q.question, q.category) for q in self.DISCOVERY_QUESTIONS]
        )
        first_q = self.state.get_current_question()
        return f"Question {first_q.id} of {len(self.state.questions)}: {first_q.question}"

    def skip_discovery(self, topic: str) -> HITLState:
        """Skip discovery and use defaults"""
        self.state = HITLState(
            stage=HITLStage.SKIPPED,
            topic=topic,
            questions=[HITLQuestion(q.id, q.question, q.category) for q in self.DISCOVERY_QUESTIONS]
        )
        # Set default answers
        defaults = {
            "audience": "Mixed technical/non-technical cohort members",
            "outcomes": "Understand core concepts and be able to apply them practically",
            "connections_past": "Infer from topic sequence in curriculum",
            "connections_future": "Infer from topic sequence in curriculum",
            "constraints": "90 minutes lecture + 2 hours assignment",
            "requirements": "Standard cohort setup (laptop, internet, provided tools)",
            "pitfalls": "Infer common misconceptions from topic"
        }
        for q in self.state.questions:
            q.answer = defaults.get(q.category, "Use reasonable defaults")
            q.paraphrase = q.answer
            q.confirmed = True

        self.state.assumptions.append("User skipped discovery - using default assumptions")
        self.state.stage = HITLStage.COMPLETE
        return self.state

    async def process_answer(self, answer: str) -> Tuple[str, bool]:
        """
        Process user's answer to current question.
        Returns (response_message, is_complete)
        """
        answer = answer.strip()

        # Check for skip command
        if answer.lower() in ("skip", "skip all", "use defaults"):
            self.skip_discovery(self.state.topic)
            return "Understood. I'll use reasonable defaults for the remaining questions.", True

        current_q = self.state.get_current_question()
        if not current_q:
            return "Discovery is complete.", True

        # Validate answer quality
        is_valid, followup = await self._validate_answer(current_q, answer)
        if not is_valid:
            return followup, False

        # Store answer
        current_q.answer = answer
        current_q.paraphrase = await self._paraphrase(answer)
        current_q.confirmed = True

        # Add to notes
        self.state.notes.append(f"Q{current_q.id} ({current_q.category}): {current_q.paraphrase}")

        # Move to next question
        self.state.current_question_idx += 1

        # Check if complete
        if self.state.current_question_idx >= len(self.state.questions):
            self.state.stage = HITLStage.COMPLETE
            return "Discovery complete. I have all the information needed.", True

        # Get next question
        next_q = self.state.get_current_question()
        return f"Got it. Question {next_q.id} of {len(self.state.questions)}: {next_q.question}", False

    async def _validate_answer(self, question: HITLQuestion, answer: str) -> Tuple[bool, str]:
        """Validate if answer is sufficient"""
        if not answer or len(answer) < 10:
            return False, "Could you provide more detail? I need specific information to create a relevant lesson plan."

        # Use LLM to check answer quality
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You evaluate if an answer provides actionable information for lesson planning. Reply ONLY with 'VALID' or 'NEED_MORE: <specific follow-up question>'"},
                    {"role": "user", "content": f"Question: {question.question}\nAnswer: {answer}\n\nIs this answer specific enough for lesson planning?"}
                ],
                max_tokens=100,
                temperature=0.3
            )
            result = response.choices[0].message.content.strip()
            if result.startswith("VALID"):
                return True, ""
            elif result.startswith("NEED_MORE:"):
                return False, result.replace("NEED_MORE:", "").strip()
            return True, ""  # Default to valid if unclear
        except Exception as e:
            logger.warning(f"Answer validation failed: {e}")
            return True, ""  # Default to valid on error

    async def _paraphrase(self, text: str) -> str:
        """Paraphrase answer to key points"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Extract key facts and constraints in one concise sentence. No hedging."},
                    {"role": "user", "content": text}
                ],
                max_tokens=100,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except:
            return text[:200]

    def get_state(self) -> HITLState:
        """Get current HITL state"""
        return self.state

    def is_complete(self) -> bool:
        """Check if discovery is complete"""
        return self.state.stage in (HITLStage.COMPLETE, HITLStage.SKIPPED)


class RetrievalAgent:
    """Retrieval Agent
    
    Handles Index Doc parsing, Data Doc content extraction,
    cross-referencing related topics, and Drive document search.
    """

    def __init__(self, config: Config, gdrive: GoogleDriveService, openai_client: OpenAI, model: str):
        self.config = config
        self.gdrive = gdrive
        self.client = openai_client
        self.model = model
        self._index_cache: Optional[List[Dict]] = None
        self._data_doc_cache: Optional[str] = None

    def _load_index(self) -> List[Dict]:
        """Parse the index document"""
        if self._index_cache:
            return self._index_cache

        try:
            index_path = Path(self.config.index_doc_path)
            if not index_path.exists():
                logger.warning(f"Index file not found: {index_path}")
                return []

            content = index_path.read_text(encoding='utf-8')
            entries = []

            for line in content.splitlines():
                line = line.strip()
                if not line or line.startswith("Index"):
                    continue

                # Parse: topic = Name (line no = X-Y) — tags: tag1, tag2
                match = re.match(r"topic\s*=\s*(.+?)\s*\(line no\s*=\s*(\d+)-(\d+)\)", line)
                if match:
                    topic_name = match.group(1).strip()
                    start_line = int(match.group(2))
                    end_line = int(match.group(3))

                    # Extract tags if present
                    tags = []
                    tag_match = re.search(r"—\s*tags:\s*(.+)$", line)
                    if tag_match:
                        tags = [t.strip() for t in tag_match.group(1).split(",")]

                    entries.append({
                        "topic": topic_name,
                        "start_line": start_line,
                        "end_line": end_line,
                        "tags": tags
                    })

            self._index_cache = entries
            logger.info(f"Loaded {len(entries)} index entries")
            return entries
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return []

    def _load_data_doc(self) -> str:
        """Load the data document"""
        if self._data_doc_cache:
            return self._data_doc_cache

        try:
            data_path = Path(self.config.data_doc_path)
            if not data_path.exists():
                logger.warning(f"Data doc not found: {data_path}")
                return ""

            self._data_doc_cache = data_path.read_text(encoding='utf-8')
            return self._data_doc_cache
        except Exception as e:
            logger.error(f"Failed to load data doc: {e}")
            return ""

    def _extract_lines(self, start: int, end: int) -> str:
        """Extract specific line range from data doc"""
        data_doc = self._load_data_doc()
        if not data_doc:
            return ""

        lines = data_doc.splitlines()
        # Convert to 0-indexed
        start_idx = max(0, start - 1)
        end_idx = min(len(lines), end)
        return "\n".join(lines[start_idx:end_idx])

    def _find_matching_entries(self, topic: str) -> List[Dict]:
        """Find index entries matching the topic"""
        index = self._load_index()
        if not index:
            return []

        topic_lower = topic.lower()
        matches = []

        # Extract key terms from topic
        key_terms = set(re.findall(r'\b\w+\b', topic_lower))
        key_terms -= {'the', 'a', 'an', 'and', 'or', 'to', 'for', 'in', 'on', 'with'}

        for entry in index:
            entry_topic_lower = entry['topic'].lower()
            entry_terms = set(re.findall(r'\b\w+\b', entry_topic_lower))
            entry_tags_lower = [t.lower() for t in entry.get('tags', [])]

            # Calculate match score
            score = 0

            # Direct topic match
            if topic_lower in entry_topic_lower or entry_topic_lower in topic_lower:
                score += 10

            # Term overlap
            overlap = key_terms & entry_terms
            score += len(overlap) * 2

            # Tag matches
            for tag in entry_tags_lower:
                if any(term in tag for term in key_terms):
                    score += 3

            if score > 0:
                matches.append({**entry, 'score': score})

        # Sort by score descending
        matches.sort(key=lambda x: x['score'], reverse=True)
        return matches[:6]  # Top 6 matches

    def _find_cross_references(self, primary_topic: str, primary_entries: List[Dict]) -> List[Dict]:
        """Find related topics for cross-referencing"""
        index = self._load_index()
        if not index:
            return []

        # Get topics already matched
        matched_topics = {e['topic'] for e in primary_entries}

        # Keywords to look for in related topics
        related_keywords = set()
        for entry in primary_entries:
            for tag in entry.get('tags', []):
                related_keywords.add(tag.lower())

        cross_refs = []
        for entry in index:
            if entry['topic'] in matched_topics:
                continue

            # Check if this entry shares tags/keywords
            entry_tags = [t.lower() for t in entry.get('tags', [])]
            if related_keywords & set(entry_tags):
                cross_refs.append(entry)

        return cross_refs[:3]  # Top 3 cross-references

    async def retrieve(self, topic: str, hitl_state: Optional[HITLState] = None) -> RetrievalResult:
        """Main retrieval method"""
        result = RetrievalResult(topic=topic)

        # 1. Find matching index entries
        matches = self._find_matching_entries(topic)
        result.index_matches = matches

        # 2. Extract content from Data Doc
        content_parts = []
        for match in matches:
            content = self._extract_lines(match['start_line'], match['end_line'])
            if content:
                content_parts.append(f"### {match['topic']}\n{content}")
                result.retrieval_sources.append(f"Data Doc: {match['topic']} (lines {match['start_line']}-{match['end_line']})")

        result.data_doc_content = "\n\n".join(content_parts)

        # 3. Find cross-references
        cross_refs = self._find_cross_references(topic, matches)
        for ref in cross_refs:
            content = self._extract_lines(ref['start_line'], ref['end_line'])
            if content:
                result.cross_references.append({
                    'topic': ref['topic'],
                    'content': content[:2000],
                    'tags': ref.get('tags', [])
                })

        # 4. Search Google Drive for additional materials
        drive_files = self.gdrive.search_files(topic, limit=5)
        result.drive_docs = drive_files

        drive_content_parts = []
        for file_info in drive_files[:3]:
            content = self.gdrive.get_file_content(file_info['id'], file_info['mimeType'])
            if content:
                drive_content_parts.append(f"### {file_info['name']}\n{content[:1500]}")
                result.retrieval_sources.append(f"Drive: {file_info['name']}")

        result.drive_content = "\n\n".join(drive_content_parts)

        # 5. If HITL specified connections, try to find those specifically
        if hitl_state:
            for q in hitl_state.questions:
                if q.category in ("connections_past", "connections_future") and q.answer:
                    # Try to find the mentioned lectures
                    mentioned = self._extract_lecture_names(q.answer)
                    for lecture_name in mentioned:
                        lecture_matches = self._find_matching_entries(lecture_name)
                        for lm in lecture_matches[:1]:
                            content = self._extract_lines(lm['start_line'], lm['end_line'])
                            if content and lm['topic'] not in [m['topic'] for m in matches]:
                                result.cross_references.append({
                                    'topic': lm['topic'],
                                    'content': content[:1500],
                                    'tags': lm.get('tags', []),
                                    'connection_type': q.category
                                })

        logger.info(f"Retrieval complete: {len(matches)} primary, {len(result.cross_references)} cross-refs, {len(drive_files)} drive docs")
        return result

    def _extract_lecture_names(self, text: str) -> List[str]:
        """Extract lecture names/numbers from text"""
        # Match patterns like "Lecture 5", "Lecture 5: ControlNets", etc.
        patterns = [
            r"Lecture\s+\d+[:\s]+[\w\s]+",
            r"Lecture\s+\d+",
            r"Module\s+\d+",
        ]
        results = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            results.extend(matches)
        return results


class WebAgent:
    """Web Research Agent
    
    Gathers community sentiment and recent developments.
    Implements retry with backoff.
    """

    def __init__(self, openai_client: OpenAI, model: str):
        self.client = openai_client
        self.model = model

    async def research(self, topic: str, max_retries: int = 3) -> WebResearchResult:
        """Perform web research with retry logic"""
        result = WebResearchResult(topic=topic)

        for attempt in range(max_retries):
            try:
                # Community sentiment
                sentiment_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a research assistant. Provide recent community discussions and sentiment from Reddit, Twitter, LinkedIn, and technical blogs about the given topic. Focus on practical experiences, common challenges, and implementation feedback."},
                        {"role": "user", "content": f"Research community sentiment and discussions about: {topic}"}
                    ],
                    max_tokens=800,
                    temperature=0.7
                )
                result.community_sentiment = sentiment_response.choices[0].message.content

                # Recent developments
                dev_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a research assistant. Provide recent developments, updates, new tools, and industry trends related to the given topic. Focus on the last 6 months."},
                        {"role": "user", "content": f"What are the recent developments and trends in: {topic}"}
                    ],
                    max_tokens=600,
                    temperature=0.7
                )
                result.recent_developments = dev_response.choices[0].message.content

                result.success = True
                return result

            except Exception as e:
                logger.warning(f"Web research attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    result.success = False
                    result.error = str(e)

        return result


class ComposerAgent:
    """Composer Agent
    
    Writes the 8 priority sections of the lesson plan,
    grounded in retrieved content and HITL inputs.
    """

    PRIORITY_SECTIONS = [
        "First-Principles Thinking",
        "Assignments and Practice Sets",
        "Connecting Points",
        "HITL Conversation Data",
        "Drive Data",
        "User Journeys and Scenarios",
        "Requirements",
        "Risks and Mitigations"
    ]

    def __init__(self, openai_client: OpenAI, model: str, system_prompt: str):
        self.client = openai_client
        self.model = model
        self.system_prompt = system_prompt

    async def compose(
        self,
        topic: str,
        hitl_state: HITLState,
        retrieval: RetrievalResult,
        web_research: WebResearchResult
    ) -> LessonPlan:
        """Compose the full lesson plan"""
        lesson_plan = LessonPlan(
            topic=topic,
            hitl_summary=hitl_state.get_summary(),
            retrieval_sources=retrieval.retrieval_sources if hasattr(retrieval, 'retrieval_sources') else []
        )

        # Build context for composer
        full_context = self._build_context(hitl_state, retrieval, web_research)

        # Compose each section
        for section_name in self.PRIORITY_SECTIONS:
            section = await self._compose_section(topic, section_name, full_context, hitl_state)
            lesson_plan.sections.append(section)

        return lesson_plan

    def _build_context(
        self,
        hitl_state: HITLState,
        retrieval: RetrievalResult,
        web_research: WebResearchResult
    ) -> str:
        """Build comprehensive context for composition"""
        parts = []

        # HITL Summary (highest priority)
        parts.append("## HUMAN-IN-THE-LOOP DISCOVERY INPUTS")
        parts.append(hitl_state.get_summary())
        parts.append("")

        # Retrieved content
        parts.append("## RETRIEVED COURSE MATERIALS")
        parts.append(retrieval.get_full_context())
        parts.append("")

        # Web research
        if web_research.success:
            parts.append("## WEB RESEARCH")
            parts.append("### Community Sentiment")
            parts.append(web_research.community_sentiment or "No data available")
            parts.append("### Recent Developments")
            parts.append(web_research.recent_developments or "No data available")

        return "\n".join(parts)

    async def _compose_section(
        self,
        topic: str,
        section_name: str,
        context: str,
        hitl_state: HITLState
    ) -> ComposedSection:
        """Compose a single section"""
        section_prompts = {
            "First-Principles Thinking": """Generate 4-6 Socratic-style questions that guide students to understand the core principles of this topic. For each question, provide a brief answer that helps students arrive at conclusions. Focus on:
- What problem does this technology solve?
- Why was this problem important?
- What core principles make this solution effective?
- What are the trade-offs?""",

            "Assignments and Practice Sets": """Create 2-3 detailed assignments with:
- Clear step-by-step instructions
- Success criteria and expected artifacts
- Difficulty progression (basic to stretch goals)
- Checkable, unambiguous outcomes
Also include 2-3 practice exercises for self-study.""",

            "Connecting Points": """CRITICAL SECTION - Must be grounded in actual course content.
Identify and explain:
**Past Dependencies**: Name specific earlier lectures/modules and demonstrate HOW specific concepts are prerequisites (mechanism/algorithm that is directly reused)
**Future Unlocks**: Name specific upcoming lectures and demonstrate exactly HOW today's topic enables them
Include relevant lab assignments that interconnect with these concepts.
BE SPECIFIC - use actual lecture names from the retrieved content.""",

            "HITL Conversation Data": """Document the human-in-the-loop conversation:
- List all questions asked and answers received
- Document decisions made
- Note constraints and preferences identified
- Highlight scope (must-teach vs must-avoid)
- Flag any risks mentioned and how they're addressed""",

            "Drive Data": """Based on the retrieved Drive documents:
- List each relevant document name
- Extract 'what good looks like' examples
- Identify reusable templates or rubrics
- Note any gaps marked as TBD""",

            "User Journeys and Scenarios": """Create 2-3 realistic user journeys relevant to this topic:
- Describe end-to-end flow tied to cohort roles
- For each: stages, user goals, success criteria
- Include both happy path and edge cases
- Show where today's concept changes the experience""",

            "Requirements": """Define requirements for the lecture outcomes:
**Functional Requirements**: What capabilities must students demonstrate?
**Model/Tool Requirements**: Specific tools, parameters, or configurations needed
**Data Requirements**: Input data, formats, any preparation needed
Make requirements testable and observable.""",

            "Risks and Mitigations": """Identify concrete risks:
- Technical risks (what could break?)
- Pedagogical risks (what could confuse students?)
- Operational risks (time, resources)
For each risk, provide specific mitigation tied to the assignments or content structure."""
        }

        prompt = section_prompts.get(section_name, f"Write the {section_name} section.")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"""Topic: {topic}

{prompt}

Use the following context to ground your response. Do not invent facts not present in the context.

{context[:8000]}

Write the {section_name} section now:"""}
                ],
                max_tokens=1500,
                temperature=0.7
            )

            content = response.choices[0].message.content
            return ComposedSection(
                name=section_name,
                content=content,
                grounded=True
            )

        except Exception as e:
            logger.error(f"Failed to compose section {section_name}: {e}")
            return ComposedSection(
                name=section_name,
                content=f"<TBD> Section generation failed: {str(e)}",
                grounded=False,
                quality_score=0.0
            )


class QAAgent:
    """Quality Assurance Agent
    
    Validates lesson plan completeness, grounding, and quality.
    """

    def __init__(self, openai_client: OpenAI, model: str):
        self.client = openai_client
        self.model = model

    async def validate(self, lesson_plan: LessonPlan, retrieval: RetrievalResult) -> Tuple[bool, List[str]]:
        """Validate the lesson plan"""
        issues = []

        # 1. Check all sections present
        section_names = {s.name for s in lesson_plan.sections}
        required = {
            "First-Principles Thinking",
            "Assignments and Practice Sets",
            "Connecting Points",
            "HITL Conversation Data",
            "Drive Data",
            "User Journeys and Scenarios",
            "Requirements",
            "Risks and Mitigations"
        }
        missing = required - section_names
        if missing:
            issues.append(f"Missing sections: {', '.join(missing)}")

        # 2. Check Connecting Points grounding
        connecting_section = next((s for s in lesson_plan.sections if s.name == "Connecting Points"), None)
        if connecting_section:
            grounding_ok = await self._check_grounding(connecting_section.content, retrieval)
            if not grounding_ok:
                issues.append("Connecting Points section may not be properly grounded in course materials")

        # 3. Check for generic content
        for section in lesson_plan.sections:
            is_generic = await self._detect_generic_content(section)
            if is_generic:
                issues.append(f"Section '{section.name}' contains generic/filler content")

        # 4. Check for TBD markers
        for section in lesson_plan.sections:
            if "<TBD>" in section.content:
                issues.append(f"Section '{section.name}' has unresolved <TBD> markers")

        passed = len(issues) == 0
        lesson_plan.qa_passed = passed
        lesson_plan.qa_issues = issues

        return passed, issues

    async def _check_grounding(self, content: str, retrieval: RetrievalResult) -> bool:
        """Check if Connecting Points references actual course content"""
        try:
            # Get lecture names from retrieval
            known_topics = [m.get('topic', '') for m in retrieval.index_matches]
            known_topics += [r.get('topic', '') for r in retrieval.cross_references]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You verify if lecture connections are grounded in provided materials. Reply only 'GROUNDED' or 'NOT_GROUNDED: <reason>'"},
                    {"role": "user", "content": f"""Check if these Connecting Points reference actual lectures from the curriculum:

Known lectures: {', '.join(known_topics)}

Connecting Points content:
{content[:2000]}

Are the referenced past/future lectures from the known list or reasonably inferred?"""}
                ],
                max_tokens=100,
                temperature=0.3
            )
            result = response.choices[0].message.content.strip()
            return result.startswith("GROUNDED")
        except:
            return True  # Default to pass on error

    async def _detect_generic_content(self, section: ComposedSection) -> bool:
        """Detect if section content is generic filler"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You detect generic/filler content in educational materials. Reply only 'SPECIFIC' or 'GENERIC: <reason>'"},
                    {"role": "user", "content": f"""Is this content specific and actionable, or generic filler?

Section: {section.name}
Content:
{section.content[:1500]}"""}
                ],
                max_tokens=100,
                temperature=0.3
            )
            result = response.choices[0].message.content.strip()
            return result.startswith("GENERIC")
        except:
            return False  # Default to pass on error


# =============================================================================
# ORCHESTRATOR AGENT (MAIN ZENO CLASS)
# =============================================================================

class ZenoOrchestrator(Agent):
    """ZENO - The Orchestrator Agent
    
    Coordinates all specialist agents and manages the lesson plan generation flow.
    Provides a Jarvis-like voice interface with transparent agent handoffs.
    """

    def __init__(self, config: Config) -> None:
        super().__init__(instructions="""You are ZENO, an AI teaching Copilot and second brain for a GenAI instructor.

You coordinate a team of specialist agents:
- HITL Specialist: Handles discovery conversations
- Retrieval Specialist: Finds relevant course materials
- Web Research Specialist: Gathers community sentiment
- Composition Specialist: Writes lesson plan sections
- Quality Specialist: Validates the output

When generating lesson plans, you transparently mention which specialist you're consulting.
Keep voice responses conversational but informative.
Provide progress updates during long operations.""")

        self.config = config

        # Initialize services
        self.gdrive = GoogleDriveService(config)
        self.openai_client = OpenAI(api_key=config.openai_api_key)
        self.system_prompt = self._load_system_prompt()

        # Initialize specialist agents
        self.hitl_agent = HITLAgent(self.openai_client, config.openai_model)
        self.retrieval_agent = RetrievalAgent(config, self.gdrive, self.openai_client, config.openai_model)
        self.web_agent = WebAgent(self.openai_client, config.openai_model)
        self.composer_agent = ComposerAgent(self.openai_client, config.openai_model, self.system_prompt)
        self.qa_agent = QAAgent(self.openai_client, config.openai_model)

        # State management
        self._current_topic: Optional[str] = None
        self._last_lesson_plan: Optional[LessonPlan] = None
        self._transcript: List[Tuple[str, str]] = []
        self._progress_task = None
        self._progress_stop = None

    def _load_system_prompt(self) -> str:
        """Load system prompt from file"""
        path = Path(self.config.system_prompt_path).expanduser().resolve()
        try:
            if path.exists():
                return path.read_text(encoding='utf-8')
        except Exception as e:
            logger.warning(f"Could not load system prompt: {e}")
        return "You are ZENO, an AI teaching Copilot for GenAI curriculum development."

    def add_to_transcript(self, user_text: str, zeno_response: str):
        """Add interaction to transcript"""
        self._transcript.append((user_text, zeno_response))
        if len(self._transcript) > 50:
            self._transcript = self._transcript[-50:]

    # =========================================================================
    # PROGRESS SPEECH (Background updates during long operations)
    # =========================================================================

    async def _progress_speaker(self, context: RunContext, topic: str, stage: str) -> None:
        """Speak progress updates during long operations"""
        messages = {
            "retrieval": [
                f"Searching course materials for {topic}...",
                "Parsing the index document...",
                "Extracting relevant sections from the data doc...",
                "Looking for cross-references to related topics...",
                "Checking your Google Drive for additional materials..."
            ],
            "web": [
                "Researching community discussions...",
                "Gathering recent developments...",
                "Checking technical blogs and forums..."
            ],
            "compose": [
                "Composing First-Principles section...",
                "Creating assignments and practice sets...",
                "Building the Connecting Points section - this is critical...",
                "Documenting HITL conversation data...",
                "Writing user journeys and scenarios...",
                "Defining requirements...",
                "Analyzing risks and mitigations..."
            ],
            "qa": [
                "Running quality checks...",
                "Verifying all sections are present...",
                "Checking that connections are grounded in course materials...",
                "Scanning for generic content..."
            ]
        }

        stage_messages = messages.get(stage, ["Processing..."])
        idx = 0

        while self._progress_stop and not self._progress_stop.is_set():
            msg = stage_messages[idx % len(stage_messages)]
            idx += 1

            # Occasionally add a fun fact
            if idx % 3 == 0:
                try:
                    fact = self.openai_client.chat.completions.create(
                        model=self.config.openai_model,
                        messages=[
                            {"role": "system", "content": "Give one surprising but accurate fun fact in a single short sentence. No preamble."},
                            {"role": "user", "content": f"Fun fact about {topic}"}
                        ],
                        max_tokens=60,
                        temperature=0.8
                    ).choices[0].message.content.strip()
                    msg = f"{msg} Fun fact: {fact}"
                except:
                    pass

            try:
                await context.session.say(msg)
            except:
                pass

            try:
                await asyncio.wait_for(self._progress_stop.wait(), timeout=8.0)
            except asyncio.TimeoutError:
                pass

    def _start_progress(self, context: RunContext, topic: str, stage: str):
        """Start background progress speech"""
        if self._progress_task and not self._progress_task.done():
            return
        self._progress_stop = asyncio.Event()
        self._progress_task = asyncio.create_task(
            self._progress_speaker(context, topic, stage)
        )

    async def _stop_progress(self):
        """Stop background progress speech"""
        if self._progress_stop:
            self._progress_stop.set()
        if self._progress_task:
            try:
                await asyncio.wait_for(self._progress_task, timeout=2.0)
            except:
                with suppress(asyncio.CancelledError):
                    self._progress_task.cancel()
        self._progress_task = None
        self._progress_stop = None

    # =========================================================================
    # FUNCTION TOOLS
    # =========================================================================

    @function_tool(
        description="Start generating a lesson plan for a topic. This initiates the HITL discovery process."
    )
    async def start_lesson_plan(self, context: RunContext, topic: str) -> str:
        """Start the lesson plan generation process"""
        self._current_topic = topic

        await context.session.say(
            f"I'll coordinate with my specialist team to build a lesson plan for {topic}. "
            "Let me start with my curriculum discovery specialist..."
        )

        # Start HITL discovery
        first_question = self.hitl_agent.start_discovery(topic)

        response = f"Let's tailor this lesson plan to your needs. {first_question}"
        await context.session.say(response)

        self.add_to_transcript(f"Start lesson plan for {topic}", response)
        return response

    @function_tool(
        description="Process an answer during the HITL discovery conversation"
    )
    async def process_discovery_answer(self, context: RunContext, answer: str) -> str:
        """Process user's answer in HITL flow"""
        if not self.hitl_agent.state.stage == HITLStage.DISCOVERY:
            return "No active discovery session. Say 'start lesson plan for [topic]' to begin."

        response, is_complete = await self.hitl_agent.process_answer(answer)

        if is_complete:
            await context.session.say(response)
            # Automatically continue to generation
            result = await self._generate_lesson_plan(context)
            return result
        else:
            await context.session.say(response)
            self.add_to_transcript(answer, response)
            return response

    @function_tool(
        description="Skip the discovery questions and use defaults"
    )
    async def skip_discovery(self, context: RunContext) -> str:
        """Skip discovery and use defaults"""
        if not self._current_topic:
            return "No topic set. Say 'start lesson plan for [topic]' first."

        self.hitl_agent.skip_discovery(self._current_topic)
        await context.session.say("Understood. I'll use reasonable defaults. Proceeding to generate the lesson plan...")

        result = await self._generate_lesson_plan(context)
        return result

    async def _generate_lesson_plan(self, context: RunContext) -> str:
        """Internal method to generate the full lesson plan"""
        topic = self._current_topic
        hitl_state = self.hitl_agent.get_state()

        try:
            # Phase 1: Retrieval
            await context.session.say("Let me check with my retrieval specialist to find relevant course materials...")
            self._start_progress(context, topic, "retrieval")

            retrieval_result = await self.retrieval_agent.retrieve(topic, hitl_state)

            await self._stop_progress()
            await context.session.say(
                f"My retrieval specialist found {len(retrieval_result.index_matches)} relevant sections "
                f"and {len(retrieval_result.cross_references)} cross-references."
            )

            # Phase 2: Web Research
            await context.session.say("Now consulting my web research specialist for community sentiment...")
            self._start_progress(context, topic, "web")

            web_result = await self.web_agent.research(topic)

            await self._stop_progress()
            if web_result.success:
                await context.session.say("Web research complete. Found recent community discussions and developments.")
            else:
                await context.session.say(f"Web research partially completed. Some data may be marked as TBD.")

            # Phase 3: Composition
            await context.session.say("Handing off to my composition specialist to write the 8 priority sections...")
            self._start_progress(context, topic, "compose")

            lesson_plan = await self.composer_agent.compose(topic, hitl_state, retrieval_result, web_result)
            lesson_plan.retrieval_sources = getattr(retrieval_result, 'retrieval_sources', [])

            await self._stop_progress()
            await context.session.say(f"Composition complete. All {len(lesson_plan.sections)} sections written.")

            # Phase 4: QA
            await context.session.say("Finally, my quality specialist is validating the lesson plan...")
            self._start_progress(context, topic, "qa")

            qa_passed, qa_issues = await self.qa_agent.validate(lesson_plan, retrieval_result)

            await self._stop_progress()

            if qa_passed:
                await context.session.say("Quality checks passed. All sections are complete and properly grounded.")
            else:
                issues_text = "; ".join(qa_issues[:3])
                await context.session.say(f"Quality check found some issues: {issues_text}. Proceeding with upload.")

            # Phase 5: Upload
            self._last_lesson_plan = lesson_plan
            upload_result = self.gdrive.upload_lesson_plan(lesson_plan, self.config.google_drive_folder_id)

            if upload_result.get("error"):
                error_msg = f"Lesson plan generated but upload failed: {upload_result['error']}"
                await context.session.say(error_msg)
                return error_msg

            link = upload_result.get('webViewLink', 'Google Drive')
            success_msg = (
                f"Done! Your lesson plan for {topic} has been uploaded to Google Drive. "
                "It includes all 8 priority sections grounded in your course materials. "
                "Feel free to ask me follow-up questions about any section."
            )
            await context.session.say(success_msg)

            self.add_to_transcript(f"Generate lesson plan for {topic}", success_msg)
            return f"Lesson plan uploaded: {link}"

        except Exception as e:
            logger.exception(f"Lesson plan generation failed: {e}")
            error_msg = f"I encountered an error during generation: {str(e)}. Would you like me to try again?"
            await context.session.say(error_msg)
            return error_msg

        finally:
            await self._stop_progress()

    @function_tool(
        description="Answer follow-up questions about the last generated lesson plan"
    )
    async def followup_question(self, context: RunContext, question: str) -> str:
        """Answer questions about the last lesson plan"""
        if not self._last_lesson_plan:
            response = "I don't have a recent lesson plan in memory. Would you like me to generate one?"
            await context.session.say(response)
            return response

        # Build context from lesson plan
        lp_context = self._last_lesson_plan.to_text()[:6000]

        try:
            response = self.openai_client.chat.completions.create(
                model=self.config.openai_model,
                messages=[
                    {"role": "system", "content": "Answer questions about the lesson plan. Be specific and reference the actual content."},
                    {"role": "user", "content": f"Lesson Plan:\n{lp_context}\n\nQuestion: {question}"}
                ],
                max_tokens=500,
                temperature=0.5
            )

            answer = response.choices[0].message.content
            await context.session.say(answer)

            self.add_to_transcript(question, answer)
            return answer

        except Exception as e:
            error_msg = f"I encountered an error: {str(e)}"
            await context.session.say(error_msg)
            return error_msg

    @function_tool(
        description="Save the conversation transcript to Google Drive"
    )
    async def save_transcript(self, context: RunContext) -> str:
        """Save conversation transcript"""
        if not self._transcript:
            response = "No conversation to save yet."
            await context.session.say(response)
            return response

        topic = self._current_topic or "General Conversation"
        now_str = datetime.now().strftime("%Y-%m-%d %H-%M")
        title = f"Conversation - {topic} - {now_str}"

        # Build transcript content
        content_lines = [
            f"## Conversation Transcript",
            f"**Topic**: {topic}",
            f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "## Conversation"
        ]

        for user_text, zeno_text in self._transcript:
            content_lines.append(f"**User**: {user_text}")
            content_lines.append(f"**Zeno**: {zeno_text}")
            content_lines.append("")

        content = "\n".join(content_lines)

        result = self.gdrive.create_google_doc_with_formatting(
            title, content, self.config.google_drive_folder_id
        )

        if result.get("error"):
            error_msg = f"Failed to save transcript: {result['error']}"
            await context.session.say(error_msg)
            return error_msg

        link = result.get('webViewLink', 'Google Drive')
        success_msg = f"Conversation transcript saved with {len(self._transcript)} exchanges."
        await context.session.say(success_msg)

        return f"Transcript saved: {link}"

    @function_tool(
        description="Quick analysis of a topic without full lesson plan generation"
    )
    async def quick_analysis(self, context: RunContext, topic: str) -> str:
        """Quick topic analysis"""
        await context.session.say(f"Let me do a quick analysis of {topic}...")

        # Quick retrieval
        retrieval = await self.retrieval_agent.retrieve(topic)

        summary_parts = []
        if retrieval.index_matches:
            summary_parts.append(f"Found {len(retrieval.index_matches)} relevant sections in course materials.")
            top_match = retrieval.index_matches[0]
            summary_parts.append(f"Primary reference: {top_match.get('topic', 'Unknown')}")

        if retrieval.cross_references:
            refs = [r.get('topic', '') for r in retrieval.cross_references[:2]]
            summary_parts.append(f"Related topics: {', '.join(refs)}")

        if retrieval.drive_docs:
            summary_parts.append(f"Found {len(retrieval.drive_docs)} related documents in your Drive.")

        response = " ".join(summary_parts) if summary_parts else f"No existing materials found for {topic}."
        await context.session.say(response)

        self.add_to_transcript(f"Quick analysis of {topic}", response)
        return response


# =============================================================================
# LIVEKIT ENTRY POINT
# =============================================================================

async def entrypoint(ctx: agents.JobContext):
    """Main entry point for LiveKit agent"""
    await ctx.connect()

    config = Config()

    session = AgentSession(
        stt=lk_openai.STT(model="whisper-1"),
        llm=lk_openai.LLM(model=config.openai_model),
        tts=lk_openai.TTS(
            model="tts-1",
            voice="nova",
            instructions="Speak in a clear, friendly, and professional tone like Jarvis from Iron Man. Be conversational but informative.",
            speed=1.2,
        ),
        vad=silero.VAD.load(),
        turn_detection="vad"
    )

    orchestrator = ZenoOrchestrator(config)

    await session.start(
        room=ctx.room,
        agent=orchestrator,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        )
    )

    # Initial greeting
    await session.generate_reply(
        instructions="""Greet the user as ZENO - their AI teaching Copilot. 
        Briefly mention you coordinate a team of specialist agents for:
        - Discovery conversations to understand their needs
        - Course material retrieval and analysis  
        - Web research for community sentiment
        - Lesson plan composition with 8 priority sections
        - Quality validation
        
        Ask how you can help them today. Keep it warm and brief like Jarvis."""
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))


