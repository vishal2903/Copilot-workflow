#!/usr/bin/env python3
"""
ZENO VOICE Copilot - 100xEngineers Second Brain
Enhanced with transcript saving and formatted reports
"""

import os
import io
import logging
import asyncio
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from dotenv import load_dotenv
import json
# Core dependencies
import PyPDF2
from docx import Document as DocxDocument
import random
from contextlib import suppress

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
        
        # Validate critical configs
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found in .env")

# =============================================================================
# SERVICE CLASSES
# =============================================================================

class GoogleDriveService:
    """Handles all Google Drive operations with enhanced formatting"""
    
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
                
                # Save token for future use
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
        """Extract text from PDF"""
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
        """Extract text from DOCX"""
        try:
            doc = DocxDocument(io.BytesIO(content))
            return '\n'.join([p.text for p in doc.paragraphs])
        except:
            return ""

    
    # ----- NEW: Docs API formatting (no HTML, no markdown) -----

    def format_as_google_doc_requests(self, title: str, content: str) -> List[dict]:
        """
        Plain text -> Google Docs API requests:
        - Title as HEADING_1 centered
        - Headings: lines starting with '## ', ALL-CAPS (<=80 chars), or Title/Hypen Case (<=80, no trailing '.')
        - Bullets: lines starting with -, *, •, or 1./2./3. -> bullet list
        - Inline **bold**: strip **...** and apply real bold with updateTextStyle
        - Spacing: blank line after bullet blocks and paragraphs
        """
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

        def transform_line_for_links(line: str):
            """
            Convert [label](url) -> label  and  (Label – url) -> (Label)
            Collect hyperlink spans (start,end,url) relative to the transformed line.
            Also auto-link bare URLs. Skips placeholders like example.com.
            """
            spans = []
            s = line

            # 1) Markdown [label](url)
            def md_repl(m):
                label, url = m.group(1), m.group(2)
                if "example.com" in url or "localhost" in url:
                    return label  # drop placeholder URL
                # we will hyperlink 'label'
                spans.append(("label", label, url))
                return label
            s = re.sub(r"\[([^\]]+)\]\((https?://[^\s)]+)\)", md_repl, s)

            # 2) Parenthetical (Label – url) or (Label - url)
            def par_repl(m):
                label, url = m.group(1).strip(), m.group(2)
                if "example.com" in url or "localhost" in url:
                    return f"({label})"
                spans.append(("par", label, url))  # hyperlink 'label'
                return f"({label})"
            s = re.sub(r"\(([^()]+?)\s[–-]\s(https?://[^\s)]+)\)", par_repl, s)

            # 3) Bare URLs: hyperlink the URL text itself
            for m in re.finditer(r"(https?://[^\s)]+)", s):
                url = m.group(1)
                if "example.com" in url or "localhost" in url:
                    continue
                spans.append(("bare", url, url))

            # Compute (start,end,url) spans in the transformed string
            link_spans = []
            cursor = 0
            for _kind, label, url in spans:
                i = s.find(label, cursor)
                if i != -1:
                    link_spans.append((i, i + len(label), url))
                    cursor = i + len(label)
            return s, link_spans

        def set_align_center(start: int, end: int):
            ops.append({
                "updateParagraphStyle": {
                    "range": {"startIndex": start, "endIndex": end},
                    "paragraphStyle": {"alignment": "CENTER"},
                    "fields": "alignment",
                }
            })

        def apply_inline_bold(start_index: int, raw_text: str) -> str:
            """
            Remove **...** and add updateTextStyle for each bold span.
            Returns cleaned_text; bold ranges are queued as ops.
            """
            cleaned = []
            bold_spans = []
            pos = 0
            for m in re.finditer(r"\*\*(.+?)\*\*", raw_text):
                # text before
                before = raw_text[pos:m.start()]
                if before:
                    cleaned.append(before)
                bold_txt = m.group(1)
                bold_start = sum(len(x) for x in cleaned)
                cleaned.append(bold_txt)
                bold_end = bold_start + len(bold_txt)
                bold_spans.append((bold_start, bold_end))
                pos = m.end()
            # tail
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
            # Mixed/Title case headings like "First-Principles History" or "Numbers-At-A-Glance"
            line_stripped = line.strip()
            if len(line_stripped) == 0 or len(line_stripped) > 80:
                return False
            if line_stripped.endswith("."):
                return False
            if line_stripped.startswith(("-", "*", "•", "1.", "2.", "3.")):
                return False
            # Starts with capital and composed of letters/digits/spaces/hyphens
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
                # Turn the block into bullets
                ops.append({
                    "createParagraphBullets": {
                        "range": {"startIndex": bullet_block_start, "endIndex": idx},
                        "bulletPreset": "BULLET_DISC_CIRCLE_SQUARE"
                    }
                })
                # Add a blank line after bullet block for spacing
                ins("\n")
            bullet_block_start = None
            bullet_block_has_items = False

        for raw in lines:
            line = raw.rstrip()

            # blank line => close bullet block, add spacing
            if not line.strip():
                flush_bullets()
                ins("\n\n")
                continue

            # Headings: '## ' or ALLCAPS short lines, or title/hyphen-case headings
            line_for_heading = re.sub(r"\*\*(.+?)\*\*", r"\1", line)  # remove ** for detection
            if line.startswith("## ") or (line_for_heading.isupper() and len(line_for_heading) <= 80) or is_heading_like(line_for_heading):
                flush_bullets()
                h_start = idx
                # Source with potential ** for bold spans, but cleaned for insertion
                heading_src = line[3:] if line.startswith("## ") else line
                heading_clean = re.sub(r"\*\*(.+?)\*\*", r"\1", heading_src)
                ins(heading_clean + "\n")      # the heading line itself
                apply_inline_bold(h_start, heading_src)
                set_heading(h_start, idx, "HEADING_2")
                ins("\n")                      # ADD: extra blank line after heading
                continue


            # Bullets
            if line.lstrip().startswith(("- ", "* ", "• ", "1. ", "2. ", "3. ")):
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


            # Headings or normal paragraphs:
            # For headings, you already do 'ins(...)' then set_heading(...)
            # For paragraphs:
            # Paragraph (normal text) with inline bold and extra blank line
            flush_bullets()
            p_start = idx
            preview_clean = re.sub(r"\*\*(.+?)\*\*", r"\1", line)  # strip ** for insertion
            ins(preview_clean + "\n\n")                            # ensure a full blank line after
            apply_inline_bold(p_start, line)                       # apply bold spans

        # Make sure the last bullet block (if any) is converted,
        # then return the batched requests so batchUpdate can run.
        flush_bullets()
        return ops


    def create_google_doc_with_formatting(self, title: str, content: str, folder_id: Optional[str] = None) -> Dict:
        """Create a Google Doc and format it using Docs API (no HTML)."""
        try:
            if not self.docs_service:
                return {"error": "Docs API unavailable"}
            # 1) create doc
            doc = self.docs_service.documents().create(body={"title": title}).execute()
            doc_id = doc.get("documentId")
            # 2) (optional) move to folder
            if folder_id:
                try:
                    self.drive_service.files().update(fileId=doc_id, addParents=folder_id, fields="id").execute()
                except Exception as e:
                    logger.warning(f"Could not move doc to folder {folder_id}: {e}")
            # 3) format content
            requests = self.format_as_google_doc_requests(title, content)
            logger.info("Docs formatter produced %d requests", len(requests) if requests else 0)

            if requests:
                self.docs_service.documents().batchUpdate(documentId=doc_id, body={"requests": requests}).execute()
            # 4) metadata
            meta = self.drive_service.files().get(fileId=doc_id, fields="id,name,webViewLink").execute()
            meta["documentId"] = doc_id
            return meta
        except Exception as e:
            logger.error(f"create_google_doc_with_formatting failed: {e}")
            return {"error": str(e)}

    def upload_report(self, content: str, file_name: str, folder_id: Optional[str] = None) -> Dict:
        """
        NEW: Upload using Google Docs API formatting (no markdown/HTML).
        Keeps the same signature, but delegates to create_google_doc_with_formatting().
        """
        return self.create_google_doc_with_formatting(file_name, content, folder_id)

    def create_transcript_doc(self, report_topic: str, transcript: List[Tuple[str, str]], folder_id: Optional[str] = None) -> Dict:
        """Create a formatted Google Doc with conversation transcript"""
        try:
            now_str = datetime.now().strftime("%Y-%m-%d %H-%M")
            title = f"Conversation - {report_topic} - {now_str}"
            
            # Build HTML content for transcript
            html_content = f"<b>Conversation Transcript</b><br>"
            html_content += f"<b>Topic:</b> {report_topic}<br>"
            html_content += f"<b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}<br><br>"
            html_content += "<b>Conversation:</b><br><br>"
            
            for user_text, zeno_text in transcript:
                html_content += f"<b>User:</b> {user_text}<br>"
                html_content += f"<b>Zeno:</b> {zeno_text}<br><br>"
            
            # Upload as formatted document
            metadata = {
                "name": title,
                "mimeType": "application/vnd.google-apps.document"
            }
            if folder_id:
                metadata["parents"] = [folder_id]
            
            media = MediaIoBaseUpload(
                io.BytesIO(html_content.encode("utf-8")),
                mimetype="text/html",
                resumable=True
            )
            
            file = self.drive_service.files().create(
                body=metadata,
                media_body=media,
                fields="id, name, webViewLink"
            ).execute()
            
            return file
        except Exception as e:
            logger.error(f"Failed to create transcript doc: {e}")
            return {"error": str(e)}

class OpenAIService:
    """Handles OpenAI interactions"""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(api_key=config.openai_api_key)
        self.system_prompt = self._load_system_prompt()
    
    def _load_system_prompt(self) -> str:
        """Load system prompt from file"""
        path = Path(self.config.system_prompt_path).expanduser().resolve()
        try:
            if path.exists():
                logger.info(f"Loading system prompt from: {path}")
                with open(path, "r", encoding="utf-8") as f:
                    return f.read()
            else:
                logger.warning(f"System prompt file not found at: {path}")
        except Exception as e:
            logger.warning(f"Could not load system prompt: {e}")
        return "You are ZENO, an AI teaching Copilot focused on analyzing GenAI topics for curriculum relevance."
    
    def generate_report(self, topic: str, analysis_text: str, web_research_text: str) -> str:
        """Generate a comprehensive report based on system prompt framework"""
        prompt = f"""
        Generate a comprehensive evaluation report on: {topic}

        Use the headings/decision rubric implied by the system prompt, but do NOT copy any literal sentences from it.
        Combine, in priority order:
        1) If present, “USER DISCOVERY NOTES” at the top of Document Analysis (these are the user's answers/assumptions),
        2) Document Analysis from Google Drive,
        3) Recent Web Research & Community Sentiment.

        Requirements:
        - Bold section headings; concise bullets under each.
        - Make explicit any assumptions carried from discovery (mark as "Assumptions").
        - Cite dates/sources ONLY if they appear in the inputs; do not invent facts.
        - Keep it under ~1,500–2,000 words.

        Document Analysis from Google Drive (may include 'USER DISCOVERY NOTES:' preface):
        {analysis_text[:4000]}

        Recent Web Research & Community Sentiment:
        {web_research_text[:2000]}
        """

        try:
            response = self.client.chat.completions.create(
                model=self.config.openai_model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=3000,
                temperature=0.8
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return f"Report generation error: {str(e)}"
    
    def fetch_web_results(self, query: str) -> str:
        """Fetch web results focusing on community sentiment"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Search for recent developments and community discussions from Reddit, Twitter, LinkedIn, Medium about the topic. Focus on practical experiences and implementation feedback."},
                    {"role": "user", "content": f"Research recent updates and community sentiment about: {query}"}
                ],
                max_tokens=1000,
                temperature=0.8
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Web research failed: {e}")
            return ""
    
    def get_brief_response(self, query: str, context: str = "") -> str:
        """Generate brief conversational responses"""
        try:
            response = self.client.chat.completions.create(
                model=self.config.openai_model,
                messages=[
                    {"role": "system", "content": "You are ZENO, a helpful voice assistant. Give brief, conversational responses (1-2 sentences max)."},
                    {"role": "user", "content": f"Context: {context}\nQuery: {query}"}
                ],
                max_tokens=100,
                temperature=0.8
            )
            return response.choices[0].message.content
        except Exception as e:
            return "I encountered an issue processing that request."

    def draft_topic_specific_questions(self, topic: str, system_prompt: str) -> List[str]:
        """
        Returns up to 7 topic-specific discovery questions, tailored to the user's topic.
        Questions must be short, specific, and follow the discovery style implied by the system prompt.
        """
        try:
            msg_sys = "You write only a numbered list of short, topic-specific discovery questions (max 7). No prose."
            msg_usr = (
                f"Topic: {topic}\n\n"
                "Use the system prompt's discovery style (scope, constraints, assets, decision drivers, risks, format, delivery), "
                "but customize each question to this topic. Return ONLY a numbered list (1..7)."
            )
            res = self.client.chat.completions.create(
                model=self.config.openai_model,
                messages=[
                    {"role": "system", "content": msg_sys},
                    {"role": "user", "content": msg_usr}
                ],
                max_tokens=400,
                temperature=0.3
            )
            text = (res.choices[0].message.content or "").strip()
            # Parse lines like "1. ..." / "- ..."
            qs = []
            for line in text.splitlines():
                line = line.strip()
                line = re.sub(r"^\s*(\d+\.|-|\*)\s*", "", line)
                if line:
                    qs.append(line)
            return qs[:7]
        except Exception as e:
            logger.warning(f"draft_topic_specific_questions failed: {e}")
            return []

    
    def analyze_transcript_for_insights(self, topic: str, transcript_text: str, date_range: str = "") -> dict:
        """
        Turn a User↔Zeno transcript into a teaching-prep report (STRICT JSON).
        Returns a dict with keys:
          title, topic, date_range, participants, overview, key_questions, answers,
          clarifications_and_misconceptions, decisions_and_agreements, open_issues,
          action_items, teaching_outline, quiz, followup_resources, transcript_coverage
        """
        system = (
            "You are a pedagogy-focused summarizer. Use ONLY transcript info. "
            "Return ONLY valid JSON matching the requested schema. No code fences."
        )
        user = f"""
Build a “Follow-up Conversation Report” from the transcript below.

Topic: {topic}
Date range: {date_range or datetime.now().strftime('%Y-%m-%d')}
Participants: User, Zeno

Transcript (verbatim lines):
{transcript_text}

Output format (STRICT JSON):
{{
  "title": "{topic} — Follow-up Conversation Report ({datetime.now().strftime('%Y-%m-%d')})",
  "topic": "{topic}",
  "date_range": "{date_range or datetime.now().strftime('%Y-%m-%d')}",
  "participants": ["User","Zeno"],
  "overview": "",
  "key_questions": [],
  "answers": [],
  "clarifications_and_misconceptions": [],
  "decisions_and_agreements": [],
  "open_issues": [],
  "action_items": [],
  "teaching_outline": [],
  "quiz": [],
  "followup_resources": [],
  "transcript_coverage": {{"turns_total": 0, "turns_used": 0, "coverage_pct": 0}}
}}
"""
        try:
            resp = self.client.chat.completions.create(
                model=self.config.openai_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                max_tokens=2000,
                temperature=0.6,
            )
            payload = (resp.choices[0].message.content or "").strip()
            return json.loads(payload)
        except Exception as e:
            logger.error(f"Transcript analysis JSON parse failed: {e}")
            # fallback minimal structure
            return {
                "title": f"{topic} — Follow-up Conversation Report ({datetime.now().strftime('%Y-%m-%d')})",
                "topic": topic,
                "date_range": date_range or datetime.now().strftime("%Y-%m-%d"),
                "participants": ["User", "Zeno"],
                "overview": transcript_text[:600],
                "key_questions": [],
                "answers": [],
                "clarifications_and_misconceptions": [],
                "decisions_and_agreements": [],
                "open_issues": [],
                "action_items": [],
                "teaching_outline": [],
                "quiz": [],
                "followup_resources": [],
                "transcript_coverage": {"turns_total": 0, "turns_used": 0, "coverage_pct": 0},
            }


# =============================================================================
# MAIN AGENT CLASS WITH FUNCTION TOOLS
# =============================================================================

class ZenoAssistant(Agent):
    """ZENO - The instructor's second brain for GenAI curriculum"""
    
    def __init__(self, config: Config) -> None:
        # Initialize with comprehensive instructions
        super().__init__(instructions="""You are ZENO, an AI teaching Copilot and second brain for a GenAI instructor.

Core Capabilities:
- Analyze documents from Google Drive for curriculum relevance
- Research recent developments and community sentiment from the web
- Generate comprehensive evaluation reports following the system prompt framework
- Upload formatted reports to Google Drive
- Save conversation transcripts when requested
- Maintain context for follow-up questions

Behavior:
- Keep voice responses brief and conversational
- Only generate full reports when explicitly asked
- Use the system prompt framework for evaluations
- Prioritize Drive documents over web sources
- Track conversation context for intelligent follow-ups

When asked to "make a report" or "generate report", you will:
1. Search relevant documents in Google Drive
2. Perform web research for recent updates
3. Generate a comprehensive evaluation following the system prompt
4. Upload formatted report to Google Drive automatically
5. Provide a brief confirmation, not the full report content

When asked to "save our conversation" or "save transcript", you will:
1. Format the conversation history
2. Upload it as a Google Doc
3. Provide the link to the saved transcript""")
        
        # Initialize services
        self.config = config
        self.gdrive = GoogleDriveService(config)
        self.openai = OpenAIService(config)
        
        # State management
        self._last_report_topic: Optional[str] = None
        self._last_report_content: Optional[str] = None
        self._last_analysis: Optional[str] = None
        self._last_web_research: Optional[str] = None
        self._conversation_context: List[Dict] = []
        self._progress_task = None
        self._progress_stop = None

       
        
        # Transcript tracking (FIX: Initialize these attributes)
        self._transcript: List[Tuple[str, str]] = []  # List of (user_text, zeno_text) tuples
        self._followup_active: bool = False  # Track if we're in follow-up mode
        self._transcript_topic: Optional[str] = None  # Current topic being discussed
        self._full_conversation: List[Dict] = []  # Full conversation history

         # --- HITL state (human-in-the-loop) ---
        self._hitl_active: bool = False
        self._hitl_topic: Optional[str] = None
        self._hitl_stage: Optional[str] = None   # "DISCOVERY" | "DEEP_DIVE" | "CONFIRM_DONE" | "CONFIRM_REPORT"
        self._hitl_q_index: int = 0
        self._hitl_log: List[Dict[str, Any]] = []    # [{q, user_raw, user_paraphrase, objective, assumptions[], next_step}]
        self._hitl_assumptions: List[str] = []
        self._hitl_q_list: List[str] = []  # topic-specific questions cache
        # HITL strictness / short-term memory
        self._hitl_min_questions: int = 6              # enforce minimum Qs (6–7)
        self._hitl_pending_q: Optional[str] = None     # last question asked; don't advance until satisfied
        self._hitl_notes: List[str] = []               # short notes for Discovery Notes preface



    
    def add_to_transcript(self, user_text: str, zeno_response: str):
        """Add an interaction to the transcript"""
        self._transcript.append((user_text, zeno_response))
        self._full_conversation.append({
            "timestamp": datetime.now(),
            "user": user_text,
            "zeno": zeno_response
        })
        # Keep only last 50 interactions in memory
        if len(self._transcript) > 50:
            self._transcript = self._transcript[-50:]


    def _hitl_reset(self, topic: Optional[str] = None):
        self._hitl_active = False
        self._hitl_topic = topic
        self._hitl_stage = None
        self._hitl_q_index = 0
        self._hitl_log = []
        self._hitl_assumptions = []
        self._hitl_q_list = []          # clear any cached topic-specific questions
        self._hitl_pending_q = None
        self._hitl_notes = []

    def _hitl_questions(self) -> List[str]:
        """Return cached topic-specific questions if present, else fallback to defaults."""
        if getattr(self, "_hitl_q_list", None):
            return self._hitl_q_list
        # fallback baseline (kept short)
        return [
            "To scope this properly, what EXACT outcome do you want from this report and who’s the audience?",
            "Any constraints I must respect (time, budget, compliance, platforms, data boundaries)?",
            "What prior assets should I reuse (Drive docs, datasets, links, brand voice)?",
            "Which decision drivers matter most (accuracy/latency/cost/human effort)? Give thresholds if possible.",
            "What are known risks/unknowns or areas that usually confuse readers?",
            "Preferred format & tone (bullets vs prose, level of technical depth, examples to include)?",
            "Delivery expectations (deadline, length, export type, sections you insist on)?"
        ]

    
    async def _paraphrase(self, text: str) -> str:
        """Return a one-line paraphrase focused on key facts and constraints."""
        try:
            q = (
                "Paraphrase in ONE short sentence, preserving key facts, constraints, and any numbers. "
                "Avoid hedging words. Do not add new details."
            )
            if hasattr(self.openai, "get_brief_response"):
                return (self.openai.get_brief_response(q, context=text) or "").strip()
            return text.strip()[:200]
        except Exception:
            return text.strip()[:200]

    def _reflect_block(self, topic: str, paraphrase: str, step_name: str) -> Dict[str, Any]:
        """Build Objective / Assumptions / Paraphrase / Next step dict."""
        # Objective: keep user anchored; short and explicit
        objective = f"Stay focused on '{topic}' — current step: {step_name}."
    # Numbered assumptions
        assumptions = [f"Assumption {i+1}: {a}" for i, a in enumerate(self._hitl_assumptions)]
        next_step = "I’ll ask the next discovery question."  # short; overwritten at confirm stages
        return {
            "objective": objective,
            "assumptions": assumptions,
            "user_input_paraphrase": paraphrase,
            "next_step": next_step
        }
    def _hitl_next_question(self) -> str:
        qs = self._hitl_questions()
        if self._hitl_q_index < len(qs):
            q = qs[self._hitl_q_index]
            self._hitl_q_index += 1
            return q
        # if we exhausted questions, move to confirm
        self._hitl_stage = "CONFIRM_DONE"
        return "Are you done with it, or is there more you want me to capture before I draft?"


    
    # =========================================================================
    # FUNCTION TOOLS
    # =========================================================================




    @function_tool(
        description="Search and analyze documents from Google Drive related to a topic. Returns key insights from found documents."
    )
    async def analyze_google_drive_documents(
        self, 
        context: RunContext, 
        topic: str,
        max_files: int = 5
    ) -> str:
        """Search and analyze Google Drive documents"""
        try:
            # Search for relevant files
            files = self.gdrive.search_files(topic, limit=max_files)
            
            if not files:
                response = f"No documents found in Google Drive for '{topic}'."
                self.add_to_transcript(f"Analyze documents about {topic}", response)
                return response
            
            # Analyze documents
            analysis_parts = []
            analysis_parts.append(f"Found {len(files)} relevant documents:\n")
            
            for file_info in files[:max_files]:
                content = self.gdrive.get_file_content(file_info['id'], file_info['mimeType'])
                if content:
                    analysis_parts.append(f"\nDocument: {file_info['name']}")
                    analysis_parts.append(f"Modified: {file_info.get('modifiedTime', 'Unknown')}")
                    analysis_parts.append(f"Content preview: {content[:500]}...\n")
            
            self._last_analysis = '\n'.join(analysis_parts)
            
            # Brief voice response
            voice_response = f"Analyzed {len(files)} documents from your Drive about {topic}."
            await context.session.say(voice_response)
            
            self.add_to_transcript(f"Analyze documents about {topic}", voice_response)
            
            return self._last_analysis
            
        except Exception as e:
            logger.error(f"Document analysis error: {e}")
            return f"Error analyzing documents: {str(e)}"
    
    @function_tool(
        description="Research recent web developments and community sentiment about a topic."
    )
    async def research_web_and_community(
        self,
        context: RunContext,
        topic: str
    ) -> str:
        """Perform web research focusing on community sentiment"""
        try:
            await context.session.say(f"Researching recent developments about {topic}...")
            
            web_research = self.openai.fetch_web_results(topic)
            self._last_web_research = web_research
            
            # Brief summary for voice
            summary = self.openai.get_brief_response(f"Summarize in one sentence: {web_research[:500]}")
            await context.session.say(summary)
            
            self.add_to_transcript(f"Research web about {topic}", summary)
            
            return web_research
            
        except Exception as e:
            logger.error(f"Web research error: {e}")
            return f"Research error: {str(e)}"
    
    async def _progress_speaker(self, context: RunContext, topic: str) -> None:
        """
        Periodically speak short progress blurbs and occasional fun facts
        until self._progress_stop is set.
        """
        # Friendly, short, non-intrusive lines; rotate through them
        progress_lines = [
            f"I'm pulling your documents on {topic}…",
            "Scanning files for the most relevant sections…",
            "Cross-checking with recent web context…",
            "Summarizing and structuring the report…",
            "Formatting for a clean Google Doc…",
            "Almost there…",
        ]
        line_i = 0
        tick = 0
        loop = asyncio.get_event_loop()

        while self._progress_stop and not self._progress_stop.is_set():
            msg = progress_lines[line_i % len(progress_lines)]
            line_i += 1

            # Every 3rd tick, try to add a 1-sentence fun fact (non-blocking pattern)
            if tick % 3 == 2:
                try:
                    # Run any blocking OpenAI call in executor to avoid freezing the loop
                    fact = await loop.run_in_executor(
                        None,
                        getattr(self.openai, "get_brief_response"),
                        f"Give one surprising but accurate fun fact about {topic} in a single short sentence. "
                        f"No preamble, no emojis."
                    )
                    if fact:
                        msg = f"{msg} Fun fact: {fact.strip()}"
                except Exception:
                    # If it fails, just skip the fact, keep progress speaking
                    pass

            try:
                await context.session.say(msg)
            except Exception:
                # If TTS queue is busy, silently continue
                pass

            tick += 1

            # Sleep-but-wake-early pattern: check stop every ~7s
            try:
                await asyncio.wait_for(self._progress_stop.wait(), timeout=7.0)
            except asyncio.TimeoutError:
                # loop again
                pass

    def _start_progress_speech(self, context: RunContext, topic: str) -> None:
        """Launch the background progress speaker if not already running."""
        if self._progress_task and not self._progress_task.done():
            return
        self._progress_stop = asyncio.Event()
        self._progress_task = asyncio.create_task(self._progress_speaker(context, topic))

    async def _stop_progress_speech(self) -> None:
        """Stop the background progress speaker cleanly."""
        if self._progress_stop:
            self._progress_stop.set()
        if self._progress_task:
            try:
                await asyncio.wait_for(self._progress_task, timeout=3.0)
            except Exception:
                with suppress(asyncio.CancelledError):
                    self._progress_task.cancel()
                    await self._progress_task
        self._progress_task = None
        self._progress_stop = None

    @function_tool(
        description="Start a human-in-the-loop discovery conversation for a topic. Asks the first question."
    )
    async def start_hitl_conversation(self, context: RunContext, topic: str) -> str:
        self._hitl_reset(topic=topic)
        self._hitl_active = True
        self._hitl_stage = "DISCOVERY"

        opener = (
            f"Let’s tailor this together. I’ll ask a few short questions so your report on {topic} fits exactly what you need."
        )
        try:
            await context.session.say(opener)
        except Exception:
            pass

        # build topic-specific list (non-blocking pattern)
        try:
            loop = asyncio.get_event_loop()
            qs = await loop.run_in_executor(None, self.openai.draft_topic_specific_questions, topic, self.openai.system_prompt)
            if qs:
                self._hitl_q_list = qs[:7]
        except Exception:
            pass


        q1 = self._hitl_next_question()
        await context.session.say(q1)
        self._hitl_pending_q = q1

        # log transcript if you have a helper
        try:
            if hasattr(self, "add_to_transcript"):
                self.add_to_transcript(f"Start HITL for {topic}", q1)
        except Exception:
            pass
        return q1

    @function_tool(
        description="Continue the HITL flow with the user's answer. Produces Objective/Assumptions/Paraphrase/Next step, and asks the next question or confirms."
    )

    async def _assess_answer_quality(self, question: str, answer: str) -> Tuple[bool, str]:
        """
        Returns (ok, followup). ok=False => ask follow-up and DO NOT advance.
        """
        a = (answer or "").strip()
        if not a:
            return False, "I didn’t catch that—could you share specific details?"
        if len(a) < 15:
            return False, "Could you add concrete details, numbers, examples or constraints?"

        # Optional model check using your brief helper
        try:
            if hasattr(self.openai, "get_brief_response"):
                prompt = (
                    "Is the user's answer specific and actionable for report drafting? "
                    "Reply exactly 'OK' or 'FOLLOWUP: <one-line micro-question>'.\n\n"
                    f"QUESTION: {question}\nANSWER: {a}"
                )
                res = self.openai.get_brief_response(prompt) or ""
                r = res.strip()
                if r.upper().startswith("OK"):
                    return True, ""
                if r.upper().startswith("FOLLOWUP"):
                    parts = r.split(":", 1)
                    if len(parts) == 2 and parts[1].strip():
                        return False, parts[1].strip()
                    return False, "Please add concrete specifics (values, examples, constraints)."
        except Exception:
            pass
        return True, ""

    async def continue_hitl_with_answer(self, context: RunContext, answer: str) -> str:
        # Safety: handle control words first
        a = (answer or "").strip().lower()
        if a in ("cancel", "abort"):
            self._hitl_reset(self._hitl_topic)
            msg = "Okay, pausing the report flow. What would you like to do next?"
            await context.session.say(msg)
            return msg
        if a in ("start again", "restart"):
            topic = self._hitl_topic or "your topic"
            self._hitl_reset(topic)
            self._hitl_active = True
            self._hitl_stage = "DISCOVERY"
            await context.session.say("Starting again from the top.")
            await context.session.say(self._hitl_next_question())
            return "restarted"
        
        # --- Confirmation handling for the "Are you done?" / "Should I make the report?" flow ---
        # If we are waiting for the user to answer "Are you done with it?" (we set _hitl_stage="CONFIRM_DONE")
        if getattr(self, "_hitl_stage", None) == "CONFIRM_DONE":
            if a in ("yes", "y", "i'm done", "im done", "done"):
                # Ask whether we should make the report now
                self._hitl_stage = "CONFIRM_REPORT"
                q = "Great — should I make the report now?"
                await context.session.say(q)
                return "asked_make_report"
            elif a in ("no", "not yet", "more"):
                await context.session.say("Okay — please add more details, or say 'start again' to restart the questions.")
                return "awaiting_more"
            # if user provides more content (not plain yes/no), fall through and treat it as additional answer

        # If we're already at the "Should I make the report?" stage
        if getattr(self, "_hitl_stage", None) == "CONFIRM_REPORT":
            if a in ("yes", "y", "please", "do it", "go ahead"):
                await context.session.say("Okay — I will generate the report now.")
                # call the report generator and bypass the HITL-start guard
                try:
                    result = await self.generate_and_upload_report(context, self._hitl_topic, include_web_research=True, skip_hitl_check=True)
                    return result
                except Exception as e:
                    logger.exception("Failed to run generate_and_upload_report from HITL confirm: %s", e)
                    await context.session.say("I encountered an error while generating the report.")
                    return f"error:{e}"
            elif a in ("no", "not now", "not yet"):
                await context.session.say("Understood — I will not generate the report. You can ask me to save the conversation or continue editing.")
                return "declined_report"
            # if user gives extra instruction, fall through to be recorded as additional info


        if not self._hitl_active or not self._hitl_topic:
            msg = "Let’s start by setting a topic first. Say: start a report on <topic>."
            await context.session.say(msg)
            return msg

        # Strict gate: require a satisfactory answer before advancing
        current_q = getattr(self, "_hitl_pending_q", None) or ""
        ok, followup = await self._assess_answer_quality(current_q, answer)
        if not ok:
            try:
                await context.session.say(followup)
            except Exception:
                pass
            return "need_more_detail"


        topic = self._hitl_topic
        step_name = "Discovery" if self._hitl_stage in (None, "DISCOVERY") else self._hitl_stage

        # Paraphrase the user's answer
        paraphrase = await self._paraphrase(answer)

        # Adjust assumptions from the latest input, if any heuristic needed
        # You can augment this by extracting 'assume'/'we will' phrases; keep minimal for now.
        # Example: if user mentions a hard constraint, record it once:
        if "assume" in a and a not in [s.lower() for s in self._hitl_assumptions]:
            self._hitl_assumptions.append(answer.strip())

        # Record the step
        block = self._reflect_block(topic, paraphrase, step_name)
        self._hitl_log.append({
            "q_index": max(self._hitl_q_index - 1, 0),
            "q": self._hitl_questions()[max(self._hitl_q_index - 1, 0)] if self._hitl_q_index > 0 else "",
            "user_raw": answer,
            "user_paraphrase": block["user_input_paraphrase"],
            "objective": block["objective"],
            "assumptions": block["assumptions"]
        })
        # Short-term note
        try:
            note = f"- Q: {current_q}\n  Paraphrase: {block['user_input_paraphrase']}\n"
            if block["assumptions"]:
                note += f"  {'; '.join(block['assumptions'])}\n"
            self._hitl_notes.append(note)
        except Exception:
            pass


        # Speak the 4 points briefly
        lines = [
            f"Objective: {block['objective']}",
            f"Assumptions: {', '.join(block['assumptions']) or 'None'}",
            f"Your input: {block['user_input_paraphrase']}",
        ]
        # compute Next step message
        if self._hitl_stage in ("CONFIRM_DONE", "CONFIRM_REPORT"):
            next_desc = "Confirm completion and permission to generate the report."
        else:
            next_desc = "Ask the next discovery question."
        lines.append(f"Next step: {next_desc}")
        # decide next step or confirm (with minimum question enforcement)
        ask_next = ""
        if self._hitl_stage in ("CONFIRM_DONE", "CONFIRM_REPORT"):
            pass
        else:
            answered = len(self._hitl_log)
            total_qs = len(self._hitl_questions())
            if (answered % 2) == 0:
                lines.append("Checkpoint: We’re aligned so far.")
            if self._hitl_q_index < max(self._hitl_min_questions, total_qs):
                ask_next = self._hitl_next_question()
                self._hitl_pending_q = ask_next  # don't advance until satisfied
            else:
                self._hitl_stage = "CONFIRM_DONE"
                ask_next = "Are you done with it, or do you want to add anything more?"



        # Speak the block
        for l in lines:
            try:
                await context.session.say(l)
            except Exception:
                pass

        # Ask next or confirm
        if ask_next:
            await context.session.say(ask_next)
            return "next"

        # Handle confirmation replies
        if self._hitl_stage == "CONFIRM_DONE":
            # If user later says "yes", we will ask should I make the report
            return "confirm_done"

        return "ok"

    @function_tool(description="Confirm discovery is done and ask for permission to make the report.")
    async def hitl_confirm_and_ask_make_report(self, context: RunContext) -> str:
        if not self._hitl_active or not self._hitl_topic:
            msg = "We haven’t started a topic yet."
            await context.session.say(msg)
            return msg
        self._hitl_stage = "CONFIRM_REPORT"
        q = "Should I make the report now?"
        await context.session.say(q)
        return q

    @function_tool(
        description="Generate a comprehensive evaluation report following the system prompt framework. Automatically analyzes documents, performs web research, generates formatted report, and uploads to Google Drive."
    )
    async def generate_and_upload_report(
        self,
        context: RunContext,
        topic: str,
        include_web_research: bool = True,
        skip_hitl_check: bool = False
    ) -> str:
        """Complete report generation pipeline with formatting"""

        # If user requested a report but HITL discovery has NOT run yet,
        # start the HITL discovery flow first and return control to the user.
        if not skip_hitl_check and not getattr(self, "_hitl_active", False) and not getattr(self, "_hitl_log", []):
            # Initialize HITL and ask the first discovery question
            self._hitl_reset(topic)
            self._hitl_active = True
            self._hitl_stage = "DISCOVERY"
            q1 = self._hitl_next_question()
            try:
                await context.session.say(
                    f"Before I generate the report, I’ll ask a few short questions to make sure it matches your needs. {q1}"
                )
            except Exception:
                pass
            # record the start in transcript so the conversation shows up
            try:
                self.add_to_transcript(f"Initiate HITL for {topic}", q1)
            except Exception:
                pass
            # Return early — the caller (user/LLM) should provide an answer next.
            return f"HITL_STARTED:{q1}"

        # >>> ADD: start background progress chatter
        self._start_progress_speech(context, topic)
        try:
            # Track this as the active topic
            self._transcript_topic = topic
            self._followup_active = True
            
            await context.session.say(f"Starting comprehensive report generation for {topic}. This will take a moment...")
            
            # Step 1: Analyze Google Drive documents
            await context.session.say("Analyzing your Google Drive documents...")
            files = self.gdrive.search_files(topic, limit=10)
            
            analysis_text = ""
            if files:
                for file_info in files[:5]:
                    content = self.gdrive.get_file_content(file_info['id'], file_info['mimeType'])
                    if content:
                        analysis_text += f"\n\nDocument: {file_info['name']}\n"
                        analysis_text += f"Modified: {file_info.get('modifiedTime', 'Unknown')}\n"
                        analysis_text += f"{content[:2000]}...\n"
            else:
                analysis_text = "No relevant documents found in Google Drive."
            
            self._last_analysis = analysis_text
            
            # Step 2: Web research
            web_research_text = ""
            if include_web_research:
                await context.session.say("Conducting web research and gathering community sentiment...")
                web_research_text = self.openai.fetch_web_results(topic)
                self._last_web_research = web_research_text
            
            # --- Inject human discovery notes (from the HITL flow) as a preface so the final report respects user inputs ---
            if getattr(self, "_hitl_log", None):
                notes_text = "USER DISCOVERY NOTES:\n" + "\n".join(self._hitl_notes[:12]) + "\n\n"
                analysis_text = (notes_text + (analysis_text or "")).strip()
                try:
                    notes_lines = []
                    # cap to avoid extremely long prefaces
                    for step in self._hitl_log[:10]:
                        q = step.get("q") or step.get("question") or ""
                        para = step.get("user_paraphrase") or step.get("user_paraphrase", "") or step.get("user_raw", "")[:300]
                        ass = step.get("assumptions") or []
                        # Add question and paraphrase
                        if q:
                            notes_lines.append(f"- Q: {q}")
                        if para:
                            notes_lines.append(f"  Paraphrase: {para}")
                        # Add assumptions (if any)
                        if ass:
                            # join short assumptions; limit size
                            short_ass = "; ".join([a if len(a) < 200 else a[:200] + "…" for a in ass])
                            notes_lines.append(f"  Assumptions: {short_ass}")
                    if notes_lines:
                        discovery_notes = "USER DISCOVERY NOTES:\n" + "\n".join(notes_lines) + "\n\n"
                        # Prepend discovery notes to analysis_text so the report generator sees them first
                        analysis_text = (discovery_notes + (analysis_text or "")).strip()
                except Exception:
                    # defensive: if anything goes wrong here, don't block report generation
                    pass

            
            # Step 3: Generate report using system prompt
            await context.session.say("Generating evaluation report using system prompt framework...")
            report_content = self.openai.generate_report(topic, analysis_text, web_research_text)
            
            # Store report context
            self._last_report_topic = topic
            self._last_report_content = report_content
            self._conversation_context.append({
                "timestamp": datetime.now(),
                "topic": topic,
                "report_snippet": report_content[:500]
            })
            
            # Step 4: Upload formatted report to Google Drive
            file_name = f"{topic} - Evaluation Report {datetime.now().strftime('%Y-%m-%d %H-%M')}"
            upload_result = self.gdrive.upload_report(report_content, file_name, self.config.google_drive_folder_id)
            
            if upload_result.get("error"):
                response = f"Report generated but upload failed: {upload_result['error']}"
                await context.session.say(response)
                self.add_to_transcript(f"Generate report about {topic}", response)
                return report_content
            
            # Success response
            link = upload_result.get('webViewLink', 'Google Drive')
            voice_response = (
                f"Report complete! Analyzed {len(files)} documents and conducted web research. "
                f"The formatted evaluation report has been uploaded to your Drive. "
                f"You can ask me follow-up questions about the findings."
            )
            await context.session.say(voice_response)
            
            self.add_to_transcript(f"Generate report about {topic}", voice_response)
            
            return f"Report uploaded successfully: {link}\n\nReport Preview:\n{report_content[:1000]}..."
            
        except Exception as e:
            logger.error(f"Report generation error: {e}")
            error_msg = f"Error generating report: {str(e)}"
            await context.session.say(error_msg)
            self.add_to_transcript(f"Generate report about {topic}", error_msg)
            return error_msg
        
        finally:
            # ALWAYS stop the progress speaker so it won't keep running
            try:
                await self._stop_progress_speech()
            except Exception:
                # defensive: don't let stopping the speaker hide the actual exception
                pass
    
    @function_tool(
        description="Answer follow-up questions about the most recent report or analysis"
    )
    async def answer_followup_question(
        self,
        context: RunContext,
        question: str
    ) -> str:
        """Handle follow-up questions about the last report"""
        try:
            if not self._last_report_content:
                response = "I don't have a recent report in memory. Would you like me to generate one?"
                await context.session.say(response)
                self.add_to_transcript(question, response)
                return response
            
            # Mark as follow-up active
            self._followup_active = True
            
            # Use OpenAI to answer based on report context
            prompt = f"""Based on this report about {self._last_report_topic}, answer the following question concisely:

Report content:
{self._last_report_content[:3000]}

Question: {question}

Provide a brief, specific answer based on the report content."""
            
            response = self.openai.client.chat.completions.create(
                model=self.config.openai_model,
                messages=[
                    {"role": "system", "content": "Answer questions based on the provided report. Be brief and specific."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.3
            )
            
            answer = response.choices[0].message.content
            await context.session.say(answer)
            
            self.add_to_transcript(question, answer)
            
            return answer
            
        except Exception as e:
            logger.error(f"Follow-up error: {e}")
            error_msg = "I encountered an error accessing the report context."
            self.add_to_transcript(question, error_msg)
            return error_msg
    
    def _render_transcript_analysis_text(self, analysis: dict) -> str:
        """
        Convert transcript-analysis JSON -> plain text with simple section markers
        that the Docs formatter will turn into headings/bullets.
        NOT a function_tool; internal helper only.
        """
        lines: List[str] = []
        lines.append("## CONVERSATION OVERVIEW")
        lines.append(analysis.get("overview", "").strip() or "Not discussed")
        lines.append("")

        def sec(title, items):
            lines.append(f"## {title}")
            if not items:
                lines.append("Not discussed")
                lines.append("")
                return
            for it in items:
                if isinstance(it, dict):
                    # key_questions entries
                    if "q" in it and "why_it_matters" in it:
                        lines.append(f"- Q: {it['q']} — {it.get('why_it_matters','')}")
                    # answers entries
                    elif {"q_ref", "summary"} <= set(it.keys()):
                        lines.append(f"- Answer to Q{it.get('q_ref')}:")
                        summary = it.get("summary") or []
                        if isinstance(summary, str):
                            lines.append(f"  - {summary}")
                        else:
                            for s in summary:
                                lines.append(f"  - {s}")
                    # action items
                    elif {"owner", "task"} <= set(it.keys()):
                        due = f" (due: {it.get('due')})" if it.get("due") else ""
                        notes = f" — {it.get('notes','')}" if it.get("notes") else ""
                        lines.append(f"- {it.get('owner','Unknown')}: {it.get('task','')}{due}{notes}")
                    else:
                        lines.append(f"- {json.dumps(it, ensure_ascii=False)}")
                else:
                    lines.append(f"- {str(it)}")
            lines.append("")

        sec("KEY QUESTIONS", analysis.get("key_questions") or [])
        sec("ANSWERS", analysis.get("answers") or [])
        sec("CLARIFICATIONS AND MISCONCEPTIONS", analysis.get("clarifications_and_misconceptions") or [])
        sec("DECISIONS AND AGREEMENTS", analysis.get("decisions_and_agreements") or [])
        sec("OPEN ISSUES", analysis.get("open_issues") or [])
        sec("ACTION ITEMS", analysis.get("action_items") or [])
        sec("TEACHING OUTLINE", analysis.get("teaching_outline") or [])
        sec("QUIZ", analysis.get("quiz") or [])
        sec("FOLLOW-UP RESOURCES", analysis.get("followup_resources") or [])

        cov = analysis.get("transcript_coverage") or {}
        if cov:
            lines.append("## COVERAGE")
            lines.append(f"- Turns total: {cov.get('turns_total', 0)}")
            lines.append(f"- Turns used: {cov.get('turns_used', 0)}")
            lines.append(f"- Coverage: {cov.get('coverage_pct', 0)}%")
            lines.append("")

        return "\n".join(lines).strip()

    
     
    @function_tool(
        description="Save the conversation transcript to Google Drive as a formatted Google Doc. Call this when user asks to save the conversation or transcript."
    )
    async def save_conversation_transcript(
        self, 
        context: RunContext,
        report_type: str = "raw"  # "raw" (default) or "analysis"
    ) -> str:

        """Save conversation transcript to Google Drive"""
        try:
            if not self._transcript:
                response = "No conversation to save yet. Have a conversation first, then ask me to save it."
                await context.session.say(response)
                return response
            
            # Determine topic and transcript text
            topic = self._transcript_topic or self._last_report_topic or "General Conversation"
            transcript_text = "\n".join([f"User — {u}\nZeno — {z}" for (u, z) in self._transcript])

            # ANALYSIS MODE: build a teaching-prep report from the transcript
            if (report_type or "").lower().startswith("analysis"):
                date_range = datetime.now().strftime("%Y-%m-%d")
                analysis = self.openai.analyze_transcript_for_insights(topic, transcript_text, date_range=date_range)
                pretty_text = self._render_transcript_analysis_text(analysis)
                title = analysis.get("title") or f"{topic} — Follow-up Conversation Report ({date_range})"

                meta = self.gdrive.create_google_doc_with_formatting(
                    title=title,
                    content=pretty_text,
                    folder_id=self.config.google_drive_folder_id
                )
                if meta.get("error"):
                    msg = f"Failed to save transcript analysis: {meta['error']}"
                    await context.session.say(msg)
                    return msg

                link = meta.get("webViewLink", meta.get("id", "Google Drive"))
                voice = "Transcript analysis report saved to Drive."
                await context.session.say(voice)
                self.add_to_transcript("Save our conversation (analysis)", voice)
                return f"Transcript analysis saved: {link}"
            
            # RAW MODE: save simple transcript
            else:
                meta = self.gdrive.create_transcript_doc(
                    topic,
                    list(self._transcript),
                    self.config.google_drive_folder_id
                )
                
                if meta.get("error"):
                    error_msg = f"Failed to save transcript: {meta['error']}"
                    await context.session.say(error_msg)
                    return error_msg
                
                link = meta.get('webViewLink', meta.get('id', 'Google Drive'))
                success_msg = f"Conversation transcript saved to Drive. It contains {len(self._transcript)} exchanges."
                await context.session.say(success_msg)
                
                # Add this save action to transcript
                self.add_to_transcript("Save our conversation", success_msg)
                
                return f"Transcript saved: {link}"
        except Exception as e:
            # Final catch to prevent crashes and surface a concise error
            logger.exception("save_conversation_transcript failed: %s", e)
            try:
                await context.session.say("Sorry — I couldn't save the transcript right now.")
            except Exception:
                pass
            return f"error_saving_transcript: {e}"

    
    @function_tool(
        description="Get a quick analysis or summary without generating a full report"
    )
    async def quick_analysis(
        self,
        context: RunContext,
        topic: str
    ) -> str:
        """Provide quick analysis without full report generation"""
        try:
            # Quick document check
            files = self.gdrive.search_files(topic, limit=3)
            
            if files:
                quick_summary = f"I found {len(files)} documents about {topic}. "
                quick_summary += f"Most recent: {files[0]['name']}. "
                
                # Get brief insight
                content = self.gdrive.get_file_content(files[0]['id'], files[0]['mimeType'])
                if content:
                    insight = self.openai.get_brief_response(
                        f"Give one key insight about {topic} from: {content[:500]}"
                    )
                    quick_summary += insight
            else:
                quick_summary = f"No documents found about {topic} in your Drive. "
                # Try web research
                web_insight = self.openai.get_brief_response(
                    f"Give one recent update about {topic}"
                )
                quick_summary += web_insight
            
            await context.session.say(quick_summary)
            self.add_to_transcript(f"Quick analysis of {topic}", quick_summary)
            
            return quick_summary
            
        except Exception as e:
            return f"Quick analysis error: {str(e)}"
    
    @function_tool(
        description="Get the conversation context and history"
    )
    async def get_conversation_context(
        self,
        context: RunContext
    ) -> str:
        """Return conversation history and context"""
        if not self._conversation_context:
            return "No conversation history yet."
        
        history = "Recent conversation topics:\n"
        for item in self._conversation_context[-5:]:  # Last 5 interactions
            history += f"- {item['topic']} at {item['timestamp'].strftime('%H:%M')}\n"
        
        if self._last_report_topic:
            history += f"\nLast report generated: {self._last_report_topic}"
        
        await context.session.say(f"We've discussed {len(self._conversation_context)} topics today.")
        return history

# =============================================================================
# LIVEKIT ENTRY POINT
# =============================================================================

async def entrypoint(ctx: agents.JobContext):
    """Main entry point for LiveKit agent"""
    
    # Connect to room
    await ctx.connect()
    
    # Initialize configuration
    config = Config()
    
    # Create agent session with voice components
    session = AgentSession(
        stt=lk_openai.STT(model="whisper-1"),
        llm=lk_openai.LLM(model=config.openai_model),
        tts=lk_openai.TTS(
            model="tts-1",
            voice="nova",
            instructions="Speak in a clear, friendly, and professional tone. Be concise unless providing detailed analysis. Also while the report generation and any other activity running in background enteratain the user with some funacfacts related to the topic of the query and keep the backend process running.",
            speed=1.3,
        ),
        vad=silero.VAD.load(),
        turn_detection="vad"
    )
    
    # Create and start the assistant
    assistant = ZenoAssistant(config)
    
    await session.start(
        room=ctx.room,
        agent=assistant,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        )
    )
    
    # Initial greeting
    await session.generate_reply(
        instructions="""Greet the user as ZENO. Let them know you can:
        1. Analyze documents from their Google Drive
        2. Research recent web developments
        3. Generate comprehensive formatted evaluation reports
        4. Save conversation transcripts
        5. Answer follow-up questions
        Keep it brief and friendlyand in one line only."""
    )

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
