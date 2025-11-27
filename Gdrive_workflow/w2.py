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
import time

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
        self.google_drive_token = os.getenv("GOOGLE_DRIVE_TOKEN", "token.json")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-5-mini")
        self.system_prompt_path = os.getenv("SYSTEM_PROMPT_PATH", "system_prompt.txt")
        self.google_drive_folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID", None)
        # New canonical paths (fall back to old envs for compatibility)
        self.index_doc_path = os.getenv("INDEX_DOC_PATH", os.getenv("CURRICULUM_INDEX_PATH", "index_with_tags_lines_only.txt"))
        self.data_doc_path  = os.getenv("DATA_DOC_PATH",  os.getenv("COHORT_NOTES_PATH",       "Data_Doc.txt"))


        # Validate critical configs
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found in .env")

# =============================================================================
# SERVICE CLASSES
# =============================================================================

class GoogleDriveService:
    """Handles all Google Drive operations with enhanced formatting"""
    
    SCOPES = [
        "https://www.googleapis.com/auth/drive",       # full Drive (fixes parent/visibility issues)
        "https://www.googleapis.com/auth/drive.file",
        "https://www.googleapis.com/auth/documents"
    ]
    
    def __init__(self, config: Config):
        self.config = config
        self.drive_service = None
        self.docs_service = None
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate with Google APIs"""
        from google.auth.exceptions import RefreshError
        creds = None
        token_path = Path(self.config.google_drive_token).expanduser().resolve()
        
        if token_path.exists():
            creds = Credentials.from_authorized_user_file(str(token_path), self.SCOPES)
        
        needs_upgrade = not creds or not set(self.SCOPES).issubset(set(getattr(creds, "scopes", []) or []))
        if not creds or not creds.valid or needs_upgrade:

            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
        needs_upgrade = not creds or not set(self.SCOPES).issubset(set(getattr(creds, "scopes", []) or []))
        if not creds or not creds.valid or needs_upgrade:
            if creds and creds.expired and creds.refresh_token and not needs_upgrade:
                try:
                    creds.refresh(Request())
                except RefreshError:
                    creds = None  # force new flow
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
        try:
            if not self.docs_service:
                return {"error": "Docs API unavailable"}

            # 1) Create a Google Doc file in Drive (parents set at creation)
            file_meta = {
                "name": title,
                "mimeType": "application/vnd.google-apps.document",
            }
            if folder_id:
                file_meta["parents"] = [folder_id]

            created = self.drive_service.files().create(
                body=file_meta,
                fields="id,name,webViewLink,parents",
                supportsAllDrives=True,
            ).execute()
            doc_id = created["id"]

            # 2) Insert and format the content using Docs API
            requests = self.format_as_google_doc_requests(title, content)
            logger.info("Docs formatter produced %d requests", len(requests) if requests else 0)
            if requests:
                try:
                    self.docs_service.documents().batchUpdate(
                        documentId=doc_id, body={"requests": requests}
                    ).execute()
                except Exception as e:
                    # Fallback: at least insert the raw content as plain text
                    try:
                        self.docs_service.documents().batchUpdate(
                            documentId=doc_id,
                            body={"requests":[{"insertText":{"location":{"index":1},"text": content}}]}
                        ).execute()
                    except Exception as e2:
                        logger.error(f"Docs API fallback failed: {e2}")
                        return {"error": f"formatting failed: {e}; fallback failed: {e2}"}

            # 3) Return consistent metadata
            meta = {
                "id": doc_id,
                "name": created.get("name", title),
                "webViewLink": created.get("webViewLink") or f"https://docs.google.com/document/d/{doc_id}/edit",
                "parents": created.get("parents", []),
                "documentId": doc_id,
            }
            return meta
        except Exception as e:
            logger.error(f"create_google_doc_with_formatting failed: {e}")
            return {"error": str(e)}
        # try:
        #     if not self.docs_service:
        #         return {"error": "Docs API unavailable"}

        #     # 1) Create the doc
        #     doc = self.docs_service.documents().create(body={"title": title}).execute()
        #     doc_id = doc.get("documentId")

        #     # 2) (optional) move to folder (Shared drives safe)
        #     prev_parents = []
        #     try:
        #         prev_meta = self.drive_service.files().get(
        #             fileId=doc_id,
        #             fields="parents",
        #             supportsAllDrives=True,
        #         ).execute()
        #         prev_parents = prev_meta.get("parents", []) or []
        #     except Exception as e:
        #         logger.debug(f"Could not fetch parents for {doc_id}: {e}")

        #     if folder_id:
        #         try:
        #             self.drive_service.files().update(
        #                 fileId=doc_id,
        #                 addParents=folder_id,
        #                 removeParents=",".join(prev_parents) if prev_parents else None,
        #                 supportsAllDrives=True,
        #                 fields="id, parents",
        #             ).execute()
        #         except Exception as e:
        #             logger.warning(f"Could not move doc to folder {folder_id}: {e}")

        #     # 3) Format content
        #     requests = self.format_as_google_doc_requests(title, content)
        #     logger.info("Docs formatter produced %d requests", len(requests) if requests else 0)
        #     if requests:
        #         self.docs_service.documents().batchUpdate(
        #             documentId=doc_id, body={"requests": requests}
        #         ).execute()

        #     # 4) Fetch metadata + ensure canonical Docs URL
        #     meta = self.drive_service.files().get(
        #         fileId=doc_id,
        #         fields="id,name,webViewLink,parents",
        #         supportsAllDrives=True,
        #     ).execute()
        #     meta["documentId"] = doc_id
        #     if not meta.get("webViewLink"):
        #         meta["webViewLink"] = f"https://docs.google.com/document/d/{doc_id}/edit"
        #     return meta

        # except Exception as e:
        #     logger.error(f"create_google_doc_with_formatting failed: {e}")
        #     return {"error": str(e)}


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
                fields="id, name, webViewLink",
                supportsAllDrives=True
            ).execute()
            
            return file
        except Exception as e:
            logger.error(f"Failed to create transcript doc: {e}")
            return {"error": str(e)}


class   IndexDocService:
    """
    Loads the Index Doc (topic = ... (line no = A-B) — tags: ...), then
    resolves those line ranges against the Data Doc to return exact slices.
    Replaces the old CurriculumIndexService.
    """
    _LINE_RE = re.compile(
        r"""^\s*topic\s*=\s*(?P<title>.+?)\s*
             \(\s*line\s*no\s*=\s*(?P<start>\d+)\s*-\s*(?P<end>\d+)\s*\)
             (?:\s*[—-]\s*tags:\s*(?P<tags>.*))?
             \s*$""",
        re.IGNORECASE | re.VERBOSE
    )

    def __init__(self, config: "Config"):
        self.config = config

    def _parse_index(self, raw: str) -> list[dict]:
        out = []
        for ln in raw.splitlines():
            m = self._LINE_RE.match(ln.strip())
            if not m:
                continue
            tags = [(t or "").strip().lower() for t in (m.group("tags") or "").split(",") if t.strip()]
            out.append({
                "title": m.group("title").strip(),
                "start": int(m.group("start")),
                "end":   int(m.group("end")),
                "tags":  tags,
                "raw":   ln.strip(),
            })
        return out

    def _score(self, entry: dict, terms: list[str]) -> int:
        title = entry["title"].lower()
        tags  = entry["tags"]
        # title hits are strongest, tag hits next, then loose substring overlap
        title_hit = sum(1 for t in terms if t and t in title)
        tag_hit   = sum(1 for t in terms if t and t in tags)
        loose     = sum(1 for t in terms if any(tok in title for tok in t.split()))
        return (title_hit * 100) + (tag_hit * 25) + loose

    def _extract_lines(self, start: int, end: int) -> str:
        p = Path(self.config.data_doc_path or "")
        if not p.exists():
            return ""
        lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
        s = max(1, start); e = min(len(lines), end)
        return "\n".join(lines[s-1:e]).strip()

    def get_relevant_sections(self, topic: str, max_chars: int = 3000) -> str:
        """
        SYSTEM PROMPT COMPLIANT: Expand topic → select 3-6 best Index entries → pull exact line ranges.
        """
        ipath = (self.config.index_doc_path or "").strip()
        if not ipath or not Path(ipath).exists():
            return ""

        raw_idx = Path(ipath).read_text(encoding="utf-8", errors="ignore")
        entries = self._parse_index(raw_idx)
        if not entries:
            return ""

        # Expand topic to synonyms/adjacent concepts for better matching
        expanded_terms = self._expand_topic_terms(topic)
        entries.sort(key=lambda e: self._score(e, expanded_terms), reverse=True)

        # Select 3-6 best matching entries as per system prompt
        selected_entries = entries[:6]

        chunks = []
        used = 0
        for e in selected_entries:
            body = self._extract_lines(e["start"], e["end"])
            if not body:
                continue
            # Clean format for lesson plan consumption
            header = f"## {e['title']}"
            if e["tags"]:
                header += f" (Tags: {', '.join(e['tags'])})"
            piece = f"{header}\n{body}"
            if used + len(piece) > max_chars:
                break
            chunks.append(piece)
            used += len(piece)
        return "\n\n".join(chunks)

    def _expand_topic_terms(self, topic: str) -> list[str]:
        """Expand topic to synonyms/adjacent concepts for better Index matching."""
        terms = [t for t in re.split(r"\s+", topic.lower().strip()) if t]

        # Add common synonyms for AI/ML topics
        expanded = list(terms)
        for term in terms:
            if term in ["agent", "agents"]: expanded.extend(["ai", "agentic", "multi-agent"])
            elif term in ["llm", "llms"]: expanded.extend(["language", "model", "openai", "gpt"])
            elif term in ["diffusion"]: expanded.extend(["stable", "image", "generation", "sdxl"])
            elif term in ["ui", "interface"]: expanded.extend(["gradio", "frontend", "api"])
            elif term in ["api", "apis"]: expanded.extend(["fastapi", "endpoint", "backend"])
            elif term in ["database", "db"]: expanded.extend(["supabase", "postgresql", "storage"])

        return list(set(expanded))


class DataDocService:
    """
    Reads the Data Doc (authoritative .txt). Returns topic-relevant snippets.
    Replaces the old CohortNotesService.
    """
    def __init__(self, config: "Config"):
        self.config = config

    def _score(self, text: str, terms: list[str]) -> int:
        low = text.lower()
        return sum(1 for t in set(terms) if t and t in low)

    def query_snippets(self, topic: str, k: int = 10, max_chars_per: int = 600) -> list[str]:
        p = Path(self.config.data_doc_path or "")
        if not p.exists():
            return []

        try:
            raw = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return []

        # Split into paragraphs by blank lines; fallback to grouped lines
        paras = [blk.strip() for blk in re.split(r"\n\s*\n+", raw) if blk.strip()]
        if not paras:
            lines = [l.strip() for l in raw.splitlines() if l.strip()]
            paras, buf = [], []
            for ln in lines:
                buf.append(ln)
                if len(" ".join(buf)) > 300 or len(buf) >= 4:
                    paras.append(" ".join(buf).strip()); buf = []
            if buf:
                paras.append(" ".join(buf).strip())

        terms = [t for t in re.split(r"\s+", topic.lower().strip()) if t]
        scored = sorted(((self._score(p, terms), len(p), p) for p in paras),
                        key=lambda x: (x[0], x[1]), reverse=True)

        positives = [p for s, L, p in scored if s > 0][:k]
        chosen = positives if positives else [p for _, _, p in scored[:k]]
        out = []
        seen = set()
        for ptxt in chosen:
            t = re.sub(r"\s+", " ", ptxt).strip()
            if not t or t in seen:
                continue
            seen.add(t)
            out.append(t[:max_chars_per])
            if len(out) >= k:
                break
        return out



class OpenAIService:
    """Handles OpenAI interactions"""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(api_key=config.openai_api_key)
        self.system_prompt = self._load_system_prompt()
        self._web_search_cache = {}
    
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
            resp = self._retry_without_newparams(
                self.client.chat.completions.create,
                model=self.config.openai_model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                **self._g5_args(verbosity="medium", reasoning_effort="low", max_ctok=4800)
            )
            return resp.choices[0].message.content
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return f"Report generation error: {str(e)}"



    def generate_lesson_plan(self, topic: str, sources: Dict[str, str]) -> str:
        """
        Produce a lesson plan (not a report) using the system prompt as rules/style ONLY.
        Content must come from the given sources (no copying literal text from the system prompt).
        sources keys expected: cohort, hitl, drive, web, curriculum, data doc, index
        """
        data_doc = sources.get("data_doc","")[:3000]
        hitl     = sources.get("hitl","")[:3000]
        drive    = sources.get("drive","")[:4000]
        web      = sources.get("web","")[:2000]
        index    = sources.get("index","")[:3000]


        user_msg = f"""
    Topic: {topic}

    Create a comprehensive, detailed lesson plan that is specific and actionable for the 100x Engineers cohort.
    STRICT VERBATIM RULES:
    - Treat the 'HITL CONTEXT' as CANONICAL input.
    - When you use the user's words from HITL CONTEXT, you MUST include them COMPULSORY WITH MINIMAL PARAPHRASING prefixed with (HITL). Example: (HITL) "Meaningful phrase..."
    - Do NOT paraphrase any text from HITL CONTEXT. You may reorganize sections, but quoted lines must remain identical.
    - If any source conflicts with HITL, prefer HITL and add a short 'Notes' line explaining the conflict.

        
    REQUIREMENTS:
    1. Reference these 5 sources by name: Data Doc, HITL Conversation, Drive Documents, Web Research, Index Doc
    2. Include specific module names and topics for connection points (use the Index Doc for exact module references)
    3. Make objectives concrete and measurable, not generic
    4. Create detailed, cohort-specific hooks and cliffhangers that relate to real engineering scenarios
    5. Use first-principles thinking with specific technical examples
    6. Include hands-on activities and practical exercises
    7. All content must be drawn from sources below - cite specific details.
    8. When citing "Connection Points", explicitly quote Index Doc module names verbatim and show the exact line header you used from Index Doc.


    SPECIAL FOCUS ON CONNECTION POINTS:
    - Identify exact previous modules/topics that connect to this lesson (cite module names from curriculum index)
    - Specify exact future modules/topics that will build on this lesson (cite module names from curriculum index)
    - Make connections specific: "This connects to Module X: Topic Y because..." and "This prepares for Module Z: Topic W by..."

    Sources (combine in priority order):

    === DATA DOC ===
    {data_doc}

    === HITL CONTEXT ===
    {hitl}

    === DRIVE ANALYSIS ===
    {drive}

    === WEB RESEARCH ===
    {web}

    === INDEX DOC ===
    {index}

    Structure your lesson plan with these sections:
    - Title & Overview
    - Learning Objectives (specific, measurable)
    - Connection Points (with exact module references)
    - First Principles Foundation (technical depth)
    - Core Content (detailed, not generic)
    - Hands-on Activities
    - Real-world Applications
    - Hooks & Cliffhangers (cohort-specific)
    - Assessment & Validation
    - Resources & Next Steps

    Remember: This is for experienced engineers, so be technical and specific, not introductory.
    """

        try:
            resp = self._retry_without_newparams(
                self.client.chat.completions.create,
                model=self.config.openai_model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_msg}
                ],
                **self._g5_args(verbosity="high", reasoning_effort="medium", max_ctok=6400)
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            logger.error(f"generate_lesson_plan failed: {e}")
            return f"Lesson plan generation error: {e}"

    def fetch_web_results(self, query: str) -> str:
        """
        Minimal Responses API web_search call (NO filters).
        Returns a single string: assistant text (if any) + SOURCES: list.
        Keeps the existing signature and return type so callers are unaffected.
        """
        import json, os

        # choose a model that supports the web_search tool (set via env if needed)
        model = os.getenv("OPENAI_WEBSEARCH_MODEL", getattr(self.config, "openai_model", None) or "gpt-4o")

        try:
            # 1) Call the Responses API using the hosted web_search tool (no filters)
            resp = self._retry_without_newparams(
                self.client.responses.create,
                model=model,
                input=query,
                tools=[{"type": "web_search"}],
                tool_choice="auto",
                include=["web_search_call.results"],
                **self._g5_args(verbosity="low", reasoning_effort="minimal", max_ctok=600)
            )
            # 2) Get assistant text (simple preferred path)
            assistant_text = (getattr(resp, "output_text", None) or "").strip()

            # 3) Extract web_search results (robust parsing)
            sources = []
            try:
                data = json.loads(resp.model_dump_json())
                for item in data.get("output", []):
                    if item.get("type") == "web_search_call":
                        for r in item.get("action", {}).get("results", []) or []:
                            url = r.get("url") or r.get("link") or r.get("uri")
                            title = r.get("title") or url
                            snippet = r.get("snippet") or ""
                            if url:
                                sources.append({"title": title, "url": url, "snippet": snippet})
                    # also capture assistant text if present in message items
                    if item.get("type") == "message" and not assistant_text:
                        for c in item.get("content", []) or []:
                            if c.get("type") == "output_text" and c.get("text"):
                                assistant_text = c.get("text").strip()
            except Exception:
                # best-effort fallback to attribute access
                try:
                    for out in getattr(resp, "output", []) or []:
                        if getattr(out, "type", "") == "web_search_call":
                            act = getattr(out, "action", None) or {}
                            for r in act.get("results", []) or []:
                                url = r.get("url") or r.get("link")
                                if url:
                                    sources.append({"title": r.get("title") or url, "url": url, "snippet": r.get("snippet") or ""})
                        if getattr(out, "type", "") == "message" and not assistant_text:
                            for c in getattr(out, "content", []) or []:
                                if c.get("type") == "output_text" and c.get("text"):
                                    assistant_text = c.get("text").strip()
                except Exception:
                    pass

            # 4) Format final string
            if not assistant_text and sources:
                assistant_text = f"Found {len(sources)} source(s) for: {query}"

            lines = [assistant_text] if assistant_text else []
            if sources:
                lines += ["", "SOURCES:"]
                for i, s in enumerate(sources, 1):
                    t = s.get("title") or s.get("url")
                    u = s.get("url")
                    snip = (s.get("snippet") or "").strip()
                    if snip:
                        lines.append(f"[{i}] {t} — {u}\n    {snip}")
                    else:
                        lines.append(f"[{i}] {t} — {u}")

            if not lines:
                return "No results."
            return "\n".join(lines)

        except Exception as e:
            logger.exception("fetch_web_results (Responses API) failed")
            return f"Web research error: {e}"

       




    
    def get_brief_response(self, query: str, context: str = "") -> str:
        """Generate brief conversational responses"""
        try:
            response = self.client.chat.completions.create(
                model=self.config.openai_model,
                messages=[
                    {"role": "system", "content": "You are ZENO, a helpful voice assistant. Give brief, conversational responses (1-2 sentences max)."},
                    {"role": "user", "content": f"Context: {context}\nQuery: {query}"}
                ],
                max_completion_tokens=100,
                #temperature=1
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
                max_completion_tokens=400,
                #temperature=1
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

    # Add inside OpenAIService
    def _g5_args(self, *, verbosity=None, reasoning_effort=None, max_ctok=None):
        args = {}
        if verbosity: args["verbosity"] = verbosity
        if reasoning_effort: args["reasoning_effort"] = reasoning_effort
        if max_ctok: args["max_completion_tokens"] = max_ctok
        # GPT-5 mini ignores custom temperature; omit to avoid errors
        return args

    def _retry_without_newparams(self, fn, **kwargs):
        """If GPT-5 rejects new params, retry without them."""
        try:
            return fn(**kwargs)
        except Exception as e:
            if "unsupported parameter" in str(e).lower():
                for k in ("verbosity", "reasoning_effort"):
                    kwargs.pop(k, None)
                return fn(**kwargs)
            raise
    def analyze_transcript_for_insights(self, topic: str, transcript_text: str, date_range: str = "") -> dict:
        """
        Turn a User↔Zeno transcript into a teaching-prep report (STRICT JSON).
        Returns a dict with keys:
        title, topic, date_range, participants, overview, key_questions, answers,
        clarifications_and_misconceptions, decisions_and_agreements, open_issues,
        action_items, teaching_outline, quiz, followup_resources, transcript_coverage
        """

        from datetime import datetime
        import json

        today = datetime.now().strftime("%Y-%m-%d")
        date_range_value = date_range or today
        title = f"{topic} — Follow-up Conversation Report ({today})"

        # Basic fallback structure (used on parse / API failure)
        def _fallback():
            turns_total = len([l for l in transcript_text.splitlines() if l.strip()])
            turns_used = turns_total if turns_total > 0 else 0
            coverage_pct = 100 if turns_total > 0 else 0
            return {
                "title": title,
                "topic": topic,
                "date_range": date_range_value,
                "participants": ["User", "Zeno"],
                "overview": (transcript_text[:600] if transcript_text else ""),
                "key_questions": [],
                "answers": [],
                "clarifications_and_misconceptions": [],
                "decisions_and_agreements": [],
                "open_issues": [],
                "action_items": [],
                "teaching_outline": [],
                "quiz": [],
                "followup_resources": [],
                "transcript_coverage": {"turns_total": turns_total, "turns_used": turns_used, "coverage_pct": coverage_pct},
            }

        system = (
            "You are a pedagogy-focused summarizer. Use ONLY transcript info. "
            "Return ONLY valid JSON matching the requested schema. No code fences."
        )

        user = f"""
    Build a “Follow-up Conversation Report” from the transcript below.

    Topic: {topic}
    Date range: {date_range_value}
    Participants: User, Zeno

    Transcript (verbatim lines):
    {transcript_text}

    Return ONLY JSON that exactly matches the requested schema. Do NOT include explanations.
    """

        # Define strict JSON Schema for the Responses API
        schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "topic": {"type": "string"},
                "date_range": {"type": "string"},
                "participants": {"type": "array", "items": {"type": "string"}},
                "overview": {"type": "string"},
                "key_questions": {"type": "array", "items": {"type": "string"}},
                "answers": {"type": "array", "items": {"type": "string"}},
                "clarifications_and_misconceptions": {"type": "array", "items": {"type": "string"}},
                "decisions_and_agreements": {"type": "array", "items": {"type": "string"}},
                "open_issues": {"type": "array", "items": {"type": "string"}},
                "action_items": {"type": "array", "items": {"type": "string"}},
                "teaching_outline": {"type": "array", "items": {"type": "string"}},
                "quiz": {"type": "array", "items": {"type": "string"}},
                "followup_resources": {"type": "array", "items": {"type": "string"}},
                "transcript_coverage": {
                    "type": "object",
                    "properties": {
                        "turns_total": {"type": "integer"},
                        "turns_used": {"type": "integer"},
                        "coverage_pct": {"type": "number"}
                    },
                    "required": ["turns_total", "turns_used", "coverage_pct"]
                }
            },
            "required": [
                "title", "topic", "date_range", "participants", "overview",
                "key_questions", "answers", "clarifications_and_misconceptions",
                "decisions_and_agreements", "open_issues", "action_items",
                "teaching_outline", "quiz", "followup_resources", "transcript_coverage"
            ]
        }

        # Build messages / input for Responses API
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]

        # Preferred: Responses API + structured json_schema output (GPT-5 mini friendly)
        try:
            # Safe construction of gpt5 args: prefer helper if present, otherwise inline params
            try:
                g5_kwargs = self._g5_args(verbosity="medium", reasoning_effort="low", max_ctok=3000)
            except Exception:
                # If your service uses different helper names, construct inline
                g5_kwargs = {"verbosity": "medium", "reasoning_effort": "low", "max_completion_tokens": 3000}

            # Prefer Responses API if available
            if hasattr(self.client, "responses") and callable(getattr(self.client, "responses").create):
                response_format = {"type": "json_schema", "json_schema": {"name": "FollowUpReport", "schema": schema}}

                # Wrap call in retry-on-unsupported-params
                def _call_responses(kwargs):
                    return self.client.responses.create(
                        model=self.config.openai_model,
                        input=messages,
                        response_format=response_format,
                        **kwargs
                    )

                # Try using helper retry if available
                try:
                    resp = self._retry_without_newparams(_call_responses, **g5_kwargs)
                except Exception:
                    # Fallback: try raw call removing new params
                    try:
                        stripped = {k: v for k, v in g5_kwargs.items() if k not in ("verbosity", "reasoning_effort")}
                        resp = _call_responses(stripped)
                    except Exception as e:
                        logger.error(f"Responses API failed (final fallback): {e}")
                        return _fallback()

                # Parse response: prefer output_parsed, else parse output_text as JSON
                parsed = None
                if hasattr(resp, "output_parsed") and resp.output_parsed:
                    parsed = resp.output_parsed
                else:
                    # Some SDKs return .output_text or .output[0].content
                    text = ""
                    if hasattr(resp, "output_text") and resp.output_text:
                        text = resp.output_text
                    else:
                        # Try common alternative structures
                        try:
                            # resp.output is a list of items with content
                            outputs = getattr(resp, "output", None)
                            if isinstance(outputs, (list, tuple)) and outputs:
                                # join any text content parts
                                parts = []
                                for it in outputs:
                                    if isinstance(it, dict) and "content" in it:
                                        # content can be string or list
                                        c = it.get("content")
                                        if isinstance(c, str):
                                            parts.append(c)
                                        elif isinstance(c, list):
                                            # look for text entries
                                            for x in c:
                                                if isinstance(x, dict) and x.get("type") == "output_text":
                                                    parts.append(x.get("text", ""))
                                text = "\n".join(parts).strip()
                        except Exception:
                            text = ""
                    if text:
                        try:
                            parsed = json.loads(text)
                        except Exception:
                            # maybe the API returned a top-level dict-like in str form; try regex-less parse
                            logger.error("Responses API returned text but JSON parse failed.")
                            parsed = None

                if isinstance(parsed, dict):
                    return parsed
                else:
                    logger.error("Responses API did not return parsed JSON; returning fallback.")
                    return _fallback()

            # If Responses API not available, fallback to chat.completions (legacy path)
            else:
                # prepare chat-completions kwargs (ensure GPT-5 compatibility: use max_completion_tokens, no temperature)
                try:
                    chat_kwargs = {}
                    if "max_completion_tokens" in g5_kwargs:
                        chat_kwargs["max_completion_tokens"] = g5_kwargs["max_completion_tokens"]
                    elif "max_ctok" in g5_kwargs:
                        chat_kwargs["max_completion_tokens"] = g5_kwargs["max_ctok"]
                    # Avoid sending temperature for GPT-5 models
                    resp = self.client.chat.completions.create(
                        model=self.config.openai_model,
                        messages=messages,
                        **chat_kwargs
                    )
                except Exception as e:
                    logger.error(f"Chat completions fallback failed: {e}")
                    return _fallback()

                content = ""
                try:
                    content = (resp.choices[0].message.content or "").strip()
                    return json.loads(content)
                except Exception as e:
                    logger.error(f"Chat completion JSON parse failed: {e}")
                    return _fallback()

        except Exception as e:
            logger.error(f"Transcript analysis failed: {e}")
            return _fallback()


#     def analyze_transcript_for_insights(self, topic: str, transcript_text: str, date_range: str = "") -> dict:
#         """
#         Turn a User↔Zeno transcript into a teaching-prep report (STRICT JSON).
#         Returns a dict with keys:
#           title, topic, date_range, participants, overview, key_questions, answers,
#           clarifications_and_misconceptions, decisions_and_agreements, open_issues,
#           action_items, teaching_outline, quiz, followup_resources, transcript_coverage
#         """
#         system = (
#             "You are a pedagogy-focused summarizer. Use ONLY transcript info. "
#             "Return ONLY valid JSON matching the requested schema. No code fences."
#         )
#         user = f"""
# Build a “Follow-up Conversation Report” from the transcript below.

# Topic: {topic}
# Date range: {date_range or datetime.now().strftime('%Y-%m-%d')}
# Participants: User, Zeno

# Transcript (verbatim lines):
# {transcript_text}

# Output format (STRICT JSON):
# {{
#   "title": "{topic} — Follow-up Conversation Report ({datetime.now().strftime('%Y-%m-%d')})",
#   "topic": "{topic}",
#   "date_range": "{date_range or datetime.now().strftime('%Y-%m-%d')}",
#   "participants": ["User","Zeno"],
#   "overview": "",
#   "key_questions": [],
#   "answers": [],
#   "clarifications_and_misconceptions": [],
#   "decisions_and_agreements": [],
#   "open_issues": [],
#   "action_items": [],
#   "teaching_outline": [],
#   "quiz": [],
#   "followup_resources": [],
#   "transcript_coverage": {{"turns_total": 0, "turns_used": 0, "coverage_pct": 0}}
# }}
# """
#         try:
#             resp = self.client.chat.completions.create(
#                 model=self.config.openai_model,
#                 messages=[
#                     {"role": "system", "content": system},
#                     {"role": "user", "content": user}
#                 ],
#                 max_completion_tokens=2000,
#                 #temperature=1,
#             )
#             payload = (resp.choices[0].message.content or "").strip()
#             return json.loads(payload)
#         except Exception as e:
#             logger.error(f"Transcript analysis JSON parse failed: {e}")
#             # fallback minimal structure
#             return {
#                 "title": f"{topic} — Follow-up Conversation Report ({datetime.now().strftime('%Y-%m-%d')})",
#                 "topic": topic,
#                 "date_range": date_range or datetime.now().strftime("%Y-%m-%d"),
#                 "participants": ["User", "Zeno"],
#                 "overview": transcript_text[:600],
#                 "key_questions": [],
#                 "answers": [],
#                 "clarifications_and_misconceptions": [],
#                 "decisions_and_agreements": [],
#                 "open_issues": [],
#                 "action_items": [],
#                 "teaching_outline": [],
#                 "quiz": [],
#                 "followup_resources": [],
#                 "transcript_coverage": {"turns_total": 0, "turns_used": 0, "coverage_pct": 0},
#             }



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


When asked to "save our conversation" or "save transcript", you will:
1. Format the conversation history
2. Upload it as a Google Doc
3. Provide the link to the saved transcript
When the user asks for a lesson plan in any wording, call generate_and_upload_lesson_plan. If HITL has not run, always run start_hitl_conversation 
and after each user reply call continue_hitl_with_answer until 6–7 answers are captured, then confirm completion and permission to generate.""")
        
        # Initialize services
        self.config = config
        self.gdrive = GoogleDriveService(config)
        self.openai = OpenAIService(config)
        self.index_doc = IndexDocService(config)
        self.data_doc  = DataDocService(config)
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
        self._hitl_target: str = "report"
        self._hitl_waiting_for_answer: bool = False
        # HITL verbatim storage + strict completion gating
        self._hitl_done: bool = False
        self._hitl_verbatim_lines: List[str] = []  # exact user strings, captured in order



    
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
        self._hitl_waiting_for_answer = False
        self._hitl_done = False
        self._hitl_verbatim_lines = []

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
        if self._hitl_q_index < len(qs) and self._hitl_q_index < self._hitl_min_questions:
            q = qs[self._hitl_q_index]
            self._hitl_q_index += 1
            self._hitl_waiting_for_answer = True  # Explicitly mark as waiting for input
            return q

        elif self._hitl_q_index >= self._hitl_min_questions:
            self._hitl_stage = "CONFIRM_DONE"
            return "Are you done with it, or is there more you want me to capture before I draft?"
        else:
            # If we've exhausted predefined questions but haven't met minimum, ask a generic follow-up
            self._hitl_q_index += 1
            return f"Question {self._hitl_q_index}: Is there anything else specific about {self._hitl_topic or 'this topic'} that will help me create exactly what you need?"


    def _compile_hitl_context(self, max_items: int = 24) -> str:
        """
        CRITICAL: Preserve detailed user responses for lesson plan customization.
        This is the MOST IMPORTANT part for personalizing the lesson plan.
        """
        if not getattr(self, "_hitl_log", []):
            return "HUMAN INPUT CONTEXT:\n(No user requirements captured yet)\n"

        lines = ["HUMAN INPUT CONTEXT (PRESERVE EXACTLY FOR LESSON CUSTOMIZATION):"]
        lines.append("=" * 60)

        # Include both questions and detailed user responses
        for i, entry in enumerate(self._hitl_log[:max_items], 1):
            question = entry.get("q", "").strip()
            user_response = entry.get("user_raw", "").strip()

            if question and user_response:
                lines.append(f"\nQ{i}: {question}")
                lines.append(f"USER ANSWER: {user_response}")

                # Include any additional context or assumptions
                if entry.get("assumptions"):
                    lines.append(f"NOTED ASSUMPTIONS: {'; '.join(entry['assumptions'])}")
                lines.append("-" * 40)

        # Add any additional user notes captured during conversation
        if getattr(self, "_hitl_notes", []):
            lines.append("\nADDITIONAL USER CONTEXT:")
            for note in self._hitl_notes[:5]:  # Latest 5 notes
                lines.append(f"- {note}")

        lines.append("\nIMPORTANT: Use this human input to customize the lesson plan exactly to user requirements.")
        return "\n".join(lines) + "\n"

    def _extract_hitl_keywords(self) -> list[str]:
        """Extract keywords from HITL responses for targeted searches."""
        if not getattr(self, "_hitl_log", []):
            return []

        keywords = []
        keyword_patterns = {
            'audience': r'\b(beginner|advanced|senior|junior|expert|student|engineer|developer|manager|team lead|cto|ceo)\b',
            'tech_stack': r'\b(aws|azure|gcp|kubernetes|docker|python|javascript|react|api|database|postgresql|mongodb)\b',
            'format': r'\b(hands.?on|practical|theory|slides|demo|exercise|code|build|tutorial|workshop)\b',
            'duration': r'\b(\d+\s*(?:hour|min|day|week)|short|long|quick|detailed|comprehensive)\b',
            'focus': r'\b(production|deployment|monitoring|testing|security|performance|scalability)\b',
            'level': r'\b(intro|introduction|basic|fundamental|advanced|deep.?dive|overview|detailed)\b'
        }

        for entry in self._hitl_log:
            user_text = (entry.get("user_raw", "") + " " + entry.get("user_paraphrase", "")).lower()

            for category, pattern in keyword_patterns.items():
                matches = re.findall(pattern, user_text, re.IGNORECASE)
                keywords.extend(matches)

        # Add topic-specific keywords
        if self._hitl_topic:
            topic_words = re.findall(r'\b\w{3,}\b', self._hitl_topic.lower())
            keywords.extend(topic_words)

        return list(set(keywords))  # Remove duplicates






    
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
                        f"Give a unique, surprising, and specific fun fact about {topic}, related to recent trends or discoveries. ALso generate a new funfcat every time. "
                        f"Ensure no preamble, and no emojis. The fact should be concise and specific to the topic, avoiding the common ones."
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


    async def _hitl_ask(self, context: RunContext, question: str) -> str:
        """
        Speak a HITL question and block progression by setting waiting flags.
        Nothing else should proceed until continue_hitl_with_answer() accepts an answer.
        """
        import time
        self._hitl_active = True
        self._hitl_pending_q = question
        self._hitl_waiting_for_answer = True   # <- hard gate
        self._hitl_last_question_at = time.time()
        try:
            await context.session.say(question)
        except Exception:
            pass
        return question



    @function_tool(
        description="Start a human-in-the-loop discovery conversation for a topic. Asks the first question."
    )
    async def start_hitl_conversation(self, context: RunContext, topic: str) -> str:
        """
        START HITL and ask the first question using the central _hitl_ask helper.
        This guarantees the assistant will wait for user input (no auto-advance).
        """
        self._hitl_reset(topic=topic)
        self._hitl_active = True
        self._hitl_stage = "DISCOVERY"

        # choose noun based on target
        noun = "lesson plan" if getattr(self, "_hitl_target", "") == "lesson_plan" else "report"
        opener = (
            f"Let’s tailor this together. I’ll ask a few short questions so your {noun} on {topic} fits exactly what you need."
        )

        try:
            await context.session.say(opener)
        except Exception:
            pass

        # build topic-specific list (non-blocking)
        try:
            loop = asyncio.get_event_loop()
            qs = await loop.run_in_executor(None, self.openai.draft_topic_specific_questions, topic, self.openai.system_prompt)
            if qs:
                self._hitl_q_list = qs[:7]
        except Exception:
            pass

        # Ask the first question via the helper that sets waiting flags.
        q1 = self._hitl_next_question()
        # Use the centralized ask helper so we never auto-advance until continue_hitl_with_answer is called.
        await self._hitl_ask(context, q1)

        # log transcript if you have a helper
        try:
            if hasattr(self, "add_to_transcript"):
                self.add_to_transcript(f"Start HITL for {topic}", q1)
        except Exception:
            pass

        return q1

        

    async def _assess_answer_quality(self, question: str, answer: str) -> Tuple[bool, str]:
        """
        Deterministic quality check. Returns (is_good, followup_prompt_or_empty).
        Criteria:
          - length threshold
          - presence of digits, commas, URLs, or keywords (audience, metric, budget, format)
        If not passing, returns False and a concise follow-up prompt.
        As a last-resort, calls the model with a strict instruction that MUST reply
        exactly "OK" or "FOLLOWUP: <short question>".
        """
        a = (answer or "").strip()
        if not a:
            return False, "I didn’t catch that — could you share a bit more detail?"
        # Quick deterministic checks
        if len(a) >= 40:
            low = a.lower()
            if any(ch.isdigit() for ch in a):
                return True, ""
            if "," in a:
                return True, ""
            if "http" in low or "www." in low:
                return True, ""
            if any(k in low for k in ("audience", "budget", "deadline", "format", "kpi", "metric", "example", "duration", "prereq", "prerequisite")):
                return True, ""
        if len(a) >= 20:
            # moderate-length answers that contain one of the keywords
            low = a.lower()
            if any(k in low for k in ("audience", "example", "metric", "format")):
                return True, ""

        # Deterministic failure case: give a concrete followup
        suggested = "Please add one concrete item: target audience, a metric (e.g. '90% pass'), or an example resource/URL."

        # Last-resort: strict model check (only if client available)
        try:
            if hasattr(self, "client") and getattr(self.config, "openai_model", None):
                sys_instruct = "You MUST return exactly either 'OK' or 'FOLLOWUP: <one short question>' with no extra commentary."
                user_prompt = f"Q: {question}\nA: {a}\n\nIf the answer is sufficient for drafting, reply OK. Otherwise reply FOLLOWUP: <short followup asking for a single missing concrete item>."
                res = self.client.responses.create(
                    model=self.config.openai_model,
                    input=user_prompt,
                    instructions=sys_instruct,
                    max_output_tokens=16,
                    #temperature=1,
                )
                text = getattr(res, "output_text", "") or ""
                text = text.strip()
                if text.upper().startswith("OK"):
                    return True, ""
                if text.upper().startswith("FOLLOWUP"):
                    return False, text.split(":", 1)[1].strip() or suggested
        except Exception:
            # ignore model failures and fall back to deterministic suggestion
            pass

        return False, suggested


    def _detect_context_request(self, user_input: str) -> Dict[str, Any]:
        """
        Detect if user is requesting context from specific sources like curriculum index or cohort notes.
        Returns dict with 'source_type' and 'context_text' if detected, empty dict otherwise.
        """
        input_lower = user_input.lower()
        
        # Index Doc requests
        if any(p in input_lower for p in ["index doc", "index file", "curriculum index", "from curriculum", "curriculum reference"]):
            if hasattr(self, "index_doc"):
                topic = self._hitl_topic or "current topic"
                context_text = self.index_doc.get_relevant_sections(topic, max_chars=1500)
                return {
                    "source_type": "index_doc",
                    "context_text": context_text,
                    "note": f"Retrieved context from index doc for {topic}"
                }

        
        # Data Doc requests
        if any(p in input_lower for p in ["data doc", "cohort notes", "from notes", "notes on", "cohort context", "course notes"]):
            if hasattr(self, "data_doc"):
                topic = self._hitl_topic or "current topic"
                snips = self.data_doc.query_snippets(topic, k=5, max_chars_per=250)
                context_text = "\n".join(snips)
                return {
                    "source_type": "data_doc",
                    "context_text": context_text,
                    "note": f"Retrieved data doc context for {topic}"
                }


    @function_tool(
        description="Record user's answer and continue HITL discovery with 4-point status update."
    )
    async def continue_hitl_with_answer(self, context: RunContext, answer: str) -> str:
        """Streamlined HITL flow: Record answer → 4 points → Next question"""

        # Handle control commands
        answer_lower = (answer or "").strip().lower()
        if answer_lower in ("cancel", "abort"):
            self._hitl_reset(self._hitl_topic)
            await context.session.say("HITL conversation cancelled.")
            return "cancelled"

        # Handle completion confirmation
        if getattr(self, "_hitl_stage", None) == "CONFIRM_DONE":
            if answer_lower in ("yes","y","done","i'm done","im done","ok","okay","sure",
                     "go ahead","proceed","please generate","generate","create","make it","do it","start"):
                self._hitl_done = True
                await context.session.say("Perfect! Generating your lesson plan now.")
                return await self.generate_and_upload_lesson_plan(context, self._hitl_topic)
            elif answer_lower in ("no", "not yet", "more"):
                self._hitl_stage = "DISCOVERY"
                await context.session.say("What else would you like to add?")
                return "continuing"

        # Validate HITL state
        if not self._hitl_active or not self._hitl_topic:
            await context.session.say("Let's start by setting a topic first.")
            return "not_active"

        # Record the user's answer
        current_q = getattr(self, "_hitl_pending_q", "") or ""
        paraphrase = await self._paraphrase(answer)

        # Store in log
        self._hitl_log.append({
            "q": current_q,
            "user_raw": answer.strip(),
            "user_paraphrase": paraphrase,
            "timestamp": datetime.now().strftime("%H:%M")
        })

        # Generate 4-point status update
        answered_count = len(self._hitl_log)
        four_points = [
            f"Objective: Capture requirements for {self._hitl_topic} lesson plan (Question {answered_count})",
            f"Assumptions: User has specific needs that will customize the lesson",
            f"Your input: {paraphrase}",
            f"Next step: {'Ask next question' if answered_count < 6 else 'Confirm completion'}"
        ]

        # Present status
        await context.session.say("Status update:")
        for point in four_points:
            await context.session.say(point)
            await asyncio.sleep(0.2)

        # Determine next action
        if answered_count >= 6:
            self._hitl_stage = "CONFIRM_DONE"
            await context.session.say("Great! We've gathered good information. Should I create the lesson plan now?")
            return "ready_to_generate"
        else:
            # Ask next question
            next_q = self._hitl_next_question()
            await self._hitl_ask(context, next_q)
            return "next_question"


    @function_tool(description="Confirm discovery is done and ask for permission to make the report.")
    async def hitl_confirm_and_ask_make_report(self, context: RunContext) -> str:
        if not self._hitl_active or not self._hitl_topic:
            msg = "We haven’t started a topic yet."
            await context.session.say(msg)
            return msg
        self._hitl_stage = "CONFIRM_REPORT"
        q = "Should I make the report now?"
        return await self._hitl_ask(context, q)


    # @function_tool(
    #     description="Generate a comprehensive evaluation report following the system prompt framework. Automatically analyzes documents, performs web research, generates formatted report, and uploads to Google Drive."
    # )
    # async def generate_and_upload_report(
    #     self,
    #     context: RunContext,
    #     topic: str,
    #     include_web_research: bool = True,
    #     skip_hitl_check: bool = False
    # ) -> str:
    #     """Complete report generation pipeline with formatting"""

    #     # If user requested a report but HITL discovery has NOT run yet,
    #     # start the HITL discovery flow first and return control to the user.
    #     if not skip_hitl_check and not getattr(self, "_hitl_active", False) and not getattr(self, "_hitl_log", []):
    #         # Use the existing start_hitl_conversation function
    #         return await self.start_hitl_conversation(context, topic)

    #     # >>> ADD: start background progress chatter
    #     self._start_progress_speech(context, topic)
    #     try:
    #         # Track this as the active topic
    #         self._transcript_topic = topic
    #         self._followup_active = True
            
    #         await context.session.say(f"Starting comprehensive report generation for {topic}. This will take a moment...")
            
    #         # Step 1: Analyze Google Drive documents
    #         await context.session.say("Analyzing your Google Drive documents...")
    #         files = self.gdrive.search_files(topic, limit=10)
            
    #         analysis_text = ""
    #         if files:
    #             for file_info in files[:5]:
    #                 content = self.gdrive.get_file_content(file_info['id'], file_info['mimeType'])
    #                 if content:
    #                     analysis_text += f"\n\nDocument: {file_info['name']}\n"
    #                     analysis_text += f"Modified: {file_info.get('modifiedTime', 'Unknown')}\n"
    #                     analysis_text += f"{content[:2000]}...\n"
    #         else:
    #             analysis_text = "No relevant documents found in Google Drive."
            
    #         self._last_analysis = analysis_text
            
    #         # Step 2: Web research
    #         web_research_text = ""
    #         if include_web_research:
    #             await context.session.say("Conducting web research and gathering community sentiment...")
    #             web_research_text = self.openai.fetch_web_results(topic)
    #             self._last_web_research = web_research_text
            
    #         # --- Inject human discovery notes (from the HITL flow) as a preface so the final report respects user inputs ---
    #         if getattr(self, "_hitl_log", None):
    #             notes_text = "USER DISCOVERY NOTES:\n" + "\n".join(self._hitl_notes[:12]) + "\n\n"
    #             analysis_text = (notes_text + (analysis_text or "")).strip()
    #             try:
    #                 notes_lines = []
    #                 # cap to avoid extremely long prefaces
    #                 for step in self._hitl_log[:10]:
    #                     q = step.get("q") or step.get("question") or ""
    #                     para = step.get("user_paraphrase") or step.get("user_paraphrase", "") or step.get("user_raw", "")[:300]
    #                     ass = step.get("assumptions") or []
    #                     # Add question and paraphrase
    #                     if q:
    #                         notes_lines.append(f"- Q: {q}")
    #                     if para:
    #                         notes_lines.append(f"  Paraphrase: {para}")
    #                     # Add assumptions (if any)
    #                     if ass:
    #                         # join short assumptions; limit size
    #                         short_ass = "; ".join([a if len(a) < 200 else a[:200] + "…" for a in ass])
    #                         notes_lines.append(f"  Assumptions: {short_ass}")
    #                 if notes_lines:
    #                     discovery_notes = "USER DISCOVERY NOTES:\n" + "\n".join(notes_lines) + "\n\n"
    #                     # Prepend discovery notes to analysis_text so the report generator sees them first
    #                     analysis_text = (discovery_notes + (analysis_text or "")).strip()
    #             except Exception:
    #                 # defensive: if anything goes wrong here, don't block report generation
    #                 pass

            
    #         # Step 3: Generate report using system prompt
    #         await context.session.say("Generating evaluation report using system prompt framework...")
    #         report_content = self.openai.generate_report(topic, analysis_text, web_research_text)
            
    #         # Store report context
    #         self._last_report_topic = topic
    #         self._last_report_content = report_content
    #         self._conversation_context.append({
    #             "timestamp": datetime.now(),
    #             "topic": topic,
    #             "report_snippet": report_content[:500]
    #         })
            
    #         # Step 4: Upload formatted report to Google Drive
    #         file_name = f"{topic} - Evaluation Report {datetime.now().strftime('%Y-%m-%d %H-%M')}"
    #         upload_result = self.gdrive.upload_report(report_content, file_name, self.config.google_drive_folder_id)
            
    #         if upload_result.get("error"):
    #             response = f"Report generated but upload failed: {upload_result['error']}"
    #             await context.session.say(response)
    #             self.add_to_transcript(f"Generate report about {topic}", response)
    #             return report_content
            
    #         # Success response
    #         link = upload_result.get('webViewLink', 'Google Drive')
    #         voice_response = (
    #             f"Report complete! Analyzed {len(files)} documents and conducted web research. "
    #             f"The formatted evaluation report has been uploaded to your Drive. "
    #             f"You can ask me follow-up questions about the findings."
    #         )
    #         await context.session.say(voice_response)
            
    #         self.add_to_transcript(f"Generate report about {topic}", voice_response)
            
    #         return f"Report uploaded successfully: {link}\n\nReport Preview:\n{report_content[:1000]}..."
            
    #     except Exception as e:
    #         logger.error(f"Report generation error: {e}")
    #         error_msg = f"Error generating report: {str(e)}"
    #         await context.session.say(error_msg)
    #         self.add_to_transcript(f"Generate report about {topic}", error_msg)
    #         return error_msg
        
    #     finally:
    #         # ALWAYS stop the progress speaker so it won't keep running
    #         try:
    #             await self._stop_progress_speech()
    #         except Exception:
    #             # defensive: don't let stopping the speaker hide the actual exception
    #             pass
    @function_tool(
        description="Generate a lesson plan (not a report) combining Cohort notes, HITL, Drive, Web, and Curriculum Index. Uploads a formatted Google Doc."
    )
    async def generate_and_upload_lesson_plan(
        self,
        context: RunContext,
        topic: str,
        include_web_research: bool = True
    ) -> str:
        
        # Check if HITL is truly complete - both conditions must be true
        hitl_answers = len(getattr(self, "_hitl_log", []))
        hitl_done_flag = getattr(self, "_hitl_done", False)
        # proceed if either explicit confirm OR enough discovery already (>=6)
        hitl_complete = (hitl_done_flag or hitl_answers >= 1)

        if not hitl_complete:   
            if hitl_answers > 0 and not hitl_done_flag:
                await context.session.say(f"HITL in progress ({hitl_answers} answers collected). Let's complete the discovery first.")
                return "hitl_in_progress"
            else:
                self._hitl_target = "lesson_plan"
                await context.session.say(f"Let me ask a few questions to customize the lesson plan for {topic}.")
                return await self.start_hitl_conversation(context, topic)
        self._start_progress_speech(context, topic)
        loop = asyncio.get_event_loop()
        try:
            # Extract HITL keywords for personalized searches
            hitl_keywords = self._extract_hitl_keywords()
            enhanced_topic = f"{topic} {' '.join(hitl_keywords[:5])}"  # Top 5 keywords

            # 1) Data Doc - PRIMARY SOURCE (enhanced with HITL keywords)
            data_snips = await loop.run_in_executor(None, self.data_doc.query_snippets, enhanced_topic, 10, 400)
            data_text = "\n".join(data_snips)

            # 2) HITL context - MOST IMPORTANT for personalization
            hitl_text = self._compile_hitl_context()

            # 3) Curriculum Index - Enhanced with HITL keywords
            await context.session.say("Mapping to exact curriculum modules using your requirements...")
            index_text = await loop.run_in_executor(None, self.index_doc.get_relevant_sections, enhanced_topic, 3000)

            # 4) Drive docs analysis - HITL-targeted search
            await context.session.say("Searching Drive for documents matching your specific needs...")
            drive_search_query = f"{topic} {' OR '.join(hitl_keywords[:3])}" if hitl_keywords else topic
            try:
                files = await loop.run_in_executor(None, lambda: self.gdrive.search_files(drive_search_query, limit=8))
            except TypeError:
                files = await loop.run_in_executor(None, lambda: self.gdrive.search_files(topic))

            drive_text = ""
            if files:
                parts = []
                for f in files[:5]:
                    txt = await loop.run_in_executor(None, self.gdrive.get_file_content, f["id"], f.get("mimeType",""))
                    if txt:
                        parts.append(f"\nDocument: {f.get('name','unnamed')}\n{txt[:1200]}")
                drive_text = "\n".join(parts)[:4000]

            # 5) Web research - Enhanced with user context
            web_text = ""
            if include_web_research:
                await context.session.say("Researching latest trends matching your requirements...")
                web_search_topic = f"{topic} {' '.join(hitl_keywords[:2])}" if hitl_keywords else topic
                web_text = await loop.run_in_executor(None, self.openai.fetch_web_results, web_search_topic)


            # Compose sources and generate lesson plan
            await context.session.say("Drafting the lesson plan from all sources...")
            sources = {
                "data_doc": data_text,
                "hitl":     hitl_text,
                "drive":    drive_text,
                "web":      web_text,
                "index":    index_text,
            }
            lesson = await loop.run_in_executor(None, self.openai.generate_lesson_plan, topic, sources)

            # Add source reference header with personalization info
            source_summary = f"## Personalized Lesson Plan for {topic}\n"
            if hitl_keywords:
                source_summary += f"**Customized using your requirements:** {', '.join(hitl_keywords[:8])}\n\n"
            source_summary += "## Sources Used\n"
            if data_text:  source_summary += "- **Data Doc**: Course-specific source of truth (PRIMARY)\n"
            if hitl_text:  source_summary += "- **HITL Conversation**: Your specific requirements and customizations\n"
            if index_text: source_summary += "- **Index Doc**: Exact curriculum module connections\n"
            if drive_text: source_summary += "- **Drive Documents**: Targeted materials matching your needs\n"
            if web_text:   source_summary += "- **Web Research**: Latest trends relevant to your context\n"
            source_summary += f"\n{lesson}"
            # Upload formatted lesson plan
            title = f"{topic} — Lesson Plan {datetime.now().strftime('%Y-%m-%d %H-%M')}"
            try:
                meta = await loop.run_in_executor(None, self.gdrive.create_google_doc_with_formatting, title, source_summary, self.config.google_drive_folder_id)
                logger.info(f"Upload result: {meta}")

                if isinstance(meta, dict) and meta.get("error"):
                    msg = f"Lesson plan generated but upload failed: {meta['error']}"
                    await context.session.say(msg)
                    self.add_to_transcript(f"Lesson plan for {topic}", msg)
                    return msg

                doc_id = meta.get("documentId") or meta.get("id")
                link = meta.get("webViewLink") or (f"https://docs.google.com/document/d/{doc_id}/edit" if doc_id else "")

                if not doc_id:
                    msg = f"Lesson plan upload unclear - received: {meta}"
                    await context.session.say(msg)
                    return msg

                voice = f"Lesson plan complete — uploaded to Drive. Document ID: {doc_id}"
                await context.session.say(voice)
                self.add_to_transcript(f"Lesson plan for {topic}", voice)

            except Exception as upload_error:
                logger.exception(f"Upload error: {upload_error}")
                msg = f"Lesson plan generated but upload failed: {str(upload_error)}"
                await context.session.say(msg)
                return msg

            # (Optional tiny post-flight: if search can't find it, we still gave a direct link)
            try:
                found = self.gdrive.search_files(title, limit=1)
                if not found:
                    await context.session.say("Note: I couldn't immediately find it via search; use the link I just gave you.")
            except Exception:
                pass

            return f"Lesson plan uploaded: {link}" if link else "Lesson plan uploaded."


        except Exception as e:
            logger.exception("generate_and_upload_lesson_plan failed")
            # In case of error, stop the progress speech
            msg = f"Error generating lesson plan: {e}"
            try:
                await context.session.say(msg)
            except Exception:
                pass
            return msg

        finally:
            # Ensure that progress speech is stopped regardless of success or failure
            try:
                await self._stop_progress_speech()
            except Exception:
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
                max_completion_tokens=300,
                #temperature=1
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
        #llm=lk_openai.LLM(model=config.openai_model),
        llm=lk_openai.LLM(model=config.openai_model, verbosity="low", reasoning_effort="minimal"),
        tts=lk_openai.TTS(
            model="tts-1",
            voice="sage",
            instructions=""" Personality and Tone

        Identity: Your AI voice copilot is an efficient, knowledgeable assistant who specializes in document analysis, research, and lesson plan generation. It is designed to be a helpful, yet succinct and practical assistant, always keeping the conversation moving forward while ensuring accurate outputs.

        Task: The voice copilot is responsible for analyzing documents from Google Drive, researching recent developments, generating evaluation reports, and uploading those reports. It also maintains the conversation context to provide intelligent follow-ups.

        Demeanor: The copilot is calm, concise, and focused, with a helpful yet professional tone.

        Tone: The voice style is conversational, informal, and easy to follow, allowing for smooth interactions.

        Level of Enthusiasm: The tone is neutral, with moderate enthusiasm to keep the conversation engaging without being overly energetic.

        Level of Formality: The language is conversational, but it remains professional and clear.

        Level of Emotion: Emotion is minimal; the focus is on efficient task execution.

        Filler Words: None, maintaining clarity and precision.

        Pacing: The pacing is steady and direct, prioritizing clarity over speed.

        Instructions

        Core Capabilities:

        Analyze documents from Google Drive for curriculum relevance.

        Research recent developments from the web.

        Generate reports following the system prompt framework.

        Upload reports to Google Drive.

        Maintain context for follow-up questions and intelligently track the conversation.

        Behavior:

        Keep responses brief and conversational.

        Only generate reports when explicitly asked.

        Prioritize Drive documents over web sources.

        Track conversation context for intelligent follow-ups.
        Conversation States:
        When asked for a lesson plan, initiate start_hitl_conversation if HITL hasn’t run.
        Capture 6–7 responses before confirming completion and generating the lesson plan.
        Also speak funfcats related to my topic query while the backend process keeps on running.""",
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
