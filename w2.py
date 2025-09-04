# #!/usr/bin/env python3
# """
# ZENO VOICE Copilot - 100xEngineers Second Brain
# Enhanced with transcript saving and formatted reports
# """

# import os
# import io
# import logging
# import asyncio
# import re
# from pathlib import Path
# from datetime import datetime
# from typing import Optional, Dict, Any, List, Tuple
# from dotenv import load_dotenv
# import json
# # Core dependencies
# import PyPDF2
# from docx import Document as DocxDocument

# # Google APIs
# from google.auth.transport.requests import Request
# from google.oauth2.credentials import Credentials
# from google_auth_oauthlib.flow import InstalledAppFlow
# from googleapiclient.discovery import build
# from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
# from googleapiclient.errors import HttpError

# # OpenAI
# from openai import OpenAI

# # LiveKit
# from livekit import agents
# from livekit.agents import (
#     Agent,
#     AgentSession, 
#     RunContext,
#     RoomInputOptions,
#     function_tool
# )
# from livekit.plugins import openai as lk_openai
# from livekit.plugins import silero, noise_cancellation

# # Load environment
# load_dotenv()

# # Setup logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# # =============================================================================
# # CONFIGURATION
# # =============================================================================

# class Config:
#     """Centralized configuration management"""
#     def __init__(self):
#         self.google_credentials_path = os.getenv("GOOGLE_CREDENTIALS_PATH")
#         self.google_drive_token = os.getenv("GOOGLE_DRIVE_TOKEN")
#         self.openai_api_key = os.getenv("OPENAI_API_KEY")
#         self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
#         self.system_prompt_path = os.getenv("SYSTEM_PROMPT_PATH", "sytem_prompt.md")
#         self.google_drive_folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID", None)
        
#         # Validate critical configs
#         if not self.openai_api_key:
#             raise ValueError("OpenAI API key not found in .env")

# # =============================================================================
# # SERVICE CLASSES
# # =============================================================================

# class GoogleDriveService:
#     """Handles all Google Drive operations with enhanced formatting"""
    
#     SCOPES = [
#         "https://www.googleapis.com/auth/drive.readonly",
#         "https://www.googleapis.com/auth/documents.readonly",
#         "https://www.googleapis.com/auth/drive.file"
#     ]
    
#     def __init__(self, config: Config):
#         self.config = config
#         self.drive_service = None
#         self.docs_service = None
#         self._authenticate()
    
#     def _authenticate(self):
#         """Authenticate with Google APIs"""
#         creds = None
#         token_path = Path(self.config.google_drive_token).expanduser().resolve()
        
#         if token_path.exists():
#             creds = Credentials.from_authorized_user_file(str(token_path), self.SCOPES)
        
#         if not creds or not creds.valid:
#             if creds and creds.expired and creds.refresh_token:
#                 creds.refresh(Request())
#             else:
#                 flow = InstalledAppFlow.from_client_secrets_file(
#                     self.config.google_credentials_path, self.SCOPES
#                 )
#                 creds = flow.run_local_server(port=0)
                
#                 # Save token for future use
#                 token_path.parent.mkdir(parents=True, exist_ok=True)
#                 token_path.write_text(creds.to_json(), encoding="utf-8")
        
#         self.drive_service = build('drive', 'v3', credentials=creds)
#         try:
#             self.docs_service = build("docs", "v1", credentials=creds)
#         except HttpError as e:
#             logger.warning(f"Docs API not accessible: {e}")
#             self.docs_service = None
    
#     def search_files(self, query: str, limit: int = 10) -> List[Dict]:
#         """Search files in Google Drive"""
#         try:
#             q = f"trashed = false and (name contains '{query}' or fullText contains '{query}')"
            
#             results = self.drive_service.files().list(
#                 q=q,
#                 pageSize=limit,
#                 fields="files(id, name, mimeType, modifiedTime)"
#             ).execute()
            
#             return results.get('files', [])
#         except Exception as e:
#             logger.error(f"Error searching files: {e}")
#             return []
    
#     def get_file_content(self, file_id: str, mime_type: str) -> str:
#         """Get content of a file from Google Drive"""
#         try:
#             if mime_type == "application/vnd.google-apps.document":
#                 data = self.drive_service.files().export(
#                     fileId=file_id, mimeType="text/plain"
#                 ).execute()
#                 return data.decode('utf-8') if isinstance(data, bytes) else str(data)
            
#             elif mime_type == "application/pdf":
#                 request = self.drive_service.files().get_media(fileId=file_id)
#                 buf = io.BytesIO()
#                 downloader = MediaIoBaseDownload(buf, request)
#                 done = False
#                 while not done:
#                     _, done = downloader.next_chunk()
#                 return self._extract_pdf_text(buf.getvalue())
            
#             elif "wordprocessingml" in mime_type:
#                 request = self.drive_service.files().get_media(fileId=file_id)
#                 buf = io.BytesIO()
#                 downloader = MediaIoBaseDownload(buf, request)
#                 done = False
#                 while not done:
#                     _, done = downloader.next_chunk()
#                 return self._extract_docx_text(buf.getvalue())
            
#             return ""
#         except Exception as e:
#             logger.error(f"Error getting file content: {e}")
#             return ""
    
#     def _extract_pdf_text(self, content: bytes) -> str:
#         """Extract text from PDF"""
#         try:
#             pdf_file = io.BytesIO(content)
#             pdf_reader = PyPDF2.PdfReader(pdf_file)
#             text = []
#             for page in pdf_reader.pages[:50]:
#                 text.append(page.extract_text())
#             return '\n'.join(text)
#         except:
#             return ""
    
#     def _extract_docx_text(self, content: bytes) -> str:
#         """Extract text from DOCX"""
#         try:
#             doc = DocxDocument(io.BytesIO(content))
#             return '\n'.join([p.text for p in doc.paragraphs])
#         except:
#             return ""

    
#         # ----- NEW: Docs API formatting (no HTML, no markdown) -----

#     def format_as_google_doc_requests(self, title: str, content: str) -> List[dict]:
#         """
#         Turn a plain text with headings/bullets into Docs API requests:
#         - Title as HEADING_1 centered
#         - Lines starting with '## ' or ALL CAPS (<=80 chars) => HEADING_2
#         - Lines starting with '-', '*', or '• ' => bullets
#         - Lines starting with '1. ' => we still render as bullets (simple & clean)
#         - Blank lines create spacing
#         """
#         ops: List[dict] = []
#         idx = 1

#         def ins(text: str):
#             nonlocal idx, ops
#             ops.append({"insertText": {"location": {"index": idx}, "text": text}})
#             idx += len(text)

#         def set_heading(start: int, end: int, level: str):
#             ops.append({
#                 "updateParagraphStyle": {
#                     "range": {"startIndex": start, "endIndex": end},
#                     "paragraphStyle": {"namedStyleType": level},
#                     "fields": "namedStyleType",
#                 }
#             })

#         def set_align_center(start: int, end: int):
#             ops.append({
#                 "updateParagraphStyle": {
#                     "range": {"startIndex": start, "endIndex": end},
#                     "paragraphStyle": {"alignment": "CENTER"},
#                     "fields": "alignment",
#                 }
#             })

#         # Title
#         t_start = idx
#         ins(title + "\n")
#         set_heading(t_start, idx, "HEADING_1")
#         set_align_center(t_start, idx)
#         ins("\n")

#         # Body parsing
#         lines = content.splitlines()
#         bullet_block_start = None
#         def flush_bullets():
#             nonlocal bullet_block_start
#             if bullet_block_start is not None:
#                 ops.append({
#                     "createParagraphBullets": {
#                         "range": {"startIndex": bullet_block_start, "endIndex": idx},
#                         "bulletPreset": "BULLET_DISC_CIRCLE_SQUARE"
#                     }
#                 })
#                 bullet_block_start = None
#                 ins("\n")

#         for raw in lines:
#             line = raw.rstrip()

#             if not line.strip():
#                 flush_bullets()
#                 ins("\n")
#                 continue

#             # Headings: '## ' or ALLCAPS short lines
#             if line.startswith("## "):
#                 flush_bullets()
#                 h_start = idx
#                 ins(line.replace("## ", "") + "\n")
#                 set_heading(h_start, idx, "HEADING_2")
#                 continue
#             if line.isupper() and len(line) <= 80:
#                 flush_bullets()
#                 h_start = idx
#                 ins(line + "\n")
#                 set_heading(h_start, idx, "HEADING_2")
#                 continue

#             # Bullets: -, *, •, 1.
#             if line.lstrip().startswith(("- ", "* ", "• ", "1. ", "2. ", "3. ")):
#                 if bullet_block_start is None:
#                     bullet_block_start = idx
#                 # normalize bullet text (strip prefix)
#                 text = line.lstrip()
#                 if text[:2] in ("- ", "* "):
#                     text = text[2:]
#                 elif text[:3] == "• ":
#                     text = text[2:]
#                 elif re.match(r"^\d+\.\s", text):
#                     text = re.sub(r"^\d+\.\s", "", text)
#                 ins(text + "\n")
#                 continue

#             # Paragraph
#             flush_bullets()
#             ins(line + "\n\n")

#         flush_bullets()
#         return ops

#     def create_google_doc_with_formatting(self, title: str, content: str, folder_id: Optional[str] = None) -> Dict:
#         """Create a Google Doc and format it using Docs API (no HTML)."""
#         try:
#             if not self.docs_service:
#                 return {"error": "Docs API unavailable"}
#             # 1) create doc
#             doc = self.docs_service.documents().create(body={"title": title}).execute()
#             doc_id = doc.get("documentId")
#             # 2) (optional) move to folder
#             if folder_id:
#                 try:
#                     self.drive_service.files().update(fileId=doc_id, addParents=folder_id, fields="id").execute()
#                 except Exception as e:
#                     logger.warning(f"Could not move doc to folder {folder_id}: {e}")
#             # 3) format content
#             requests = self.format_as_google_doc_requests(title, content)
#             if requests:
#                 self.docs_service.documents().batchUpdate(documentId=doc_id, body={"requests": requests}).execute()
#             # 4) metadata
#             meta = self.drive_service.files().get(fileId=doc_id, fields="id,name,webViewLink").execute()
#             meta["documentId"] = doc_id
#             return meta
#         except Exception as e:
#             logger.error(f"create_google_doc_with_formatting failed: {e}")
#             return {"error": str(e)}

#     def upload_report(self, content: str, file_name: str, folder_id: Optional[str] = None) -> Dict:
#         """
#         NEW: Upload using Google Docs API formatting (no markdown/HTML).
#         Keeps the same signature, but delegates to create_google_doc_with_formatting().
#         """
#         return self.create_google_doc_with_formatting(file_name, content, folder_id)

#     def create_transcript_doc(self, report_topic: str, transcript: List[Tuple[str, str]], folder_id: Optional[str] = None) -> Dict:
#         """Create a formatted Google Doc with conversation transcript"""
#         try:
#             now_str = datetime.now().strftime("%Y-%m-%d %H-%M")
#             title = f"Conversation - {report_topic} - {now_str}"
            
#             # Build HTML content for transcript
#             html_content = f"<b>Conversation Transcript</b><br>"
#             html_content += f"<b>Topic:</b> {report_topic}<br>"
#             html_content += f"<b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}<br><br>"
#             html_content += "<b>Conversation:</b><br><br>"
            
#             for user_text, zeno_text in transcript:
#                 html_content += f"<b>User:</b> {user_text}<br>"
#                 html_content += f"<b>Zeno:</b> {zeno_text}<br><br>"
            
#             # Upload as formatted document
#             metadata = {
#                 "name": title,
#                 "mimeType": "application/vnd.google-apps.document"
#             }
#             if folder_id:
#                 metadata["parents"] = [folder_id]
            
#             media = MediaIoBaseUpload(
#                 io.BytesIO(html_content.encode("utf-8")),
#                 mimetype="text/html",
#                 resumable=True
#             )
            
#             file = self.drive_service.files().create(
#                 body=metadata,
#                 media_body=media,
#                 fields="id, name, webViewLink"
#             ).execute()
            
#             return file
#         except Exception as e:
#             logger.error(f"Failed to create transcript doc: {e}")
#             return {"error": str(e)}

# class OpenAIService:
#     """Handles OpenAI interactions"""
    
#     def __init__(self, config: Config):
#         self.config = config
#         self.client = OpenAI(api_key=config.openai_api_key)
#         self.system_prompt = self._load_system_prompt()
    
#     def _load_system_prompt(self) -> str:
#         """Load system prompt from file"""
#         path = Path(self.config.system_prompt_path).expanduser().resolve()
#         try:
#             if path.exists():
#                 logger.info(f"Loading system prompt from: {path}")
#                 with open(path, "r", encoding="utf-8") as f:
#                     return f.read()
#             else:
#                 logger.warning(f"System prompt file not found at: {path}")
#         except Exception as e:
#             logger.warning(f"Could not load system prompt: {e}")
#         return "You are ZENO, an AI teaching Copilot focused on analyzing GenAI topics for curriculum relevance."
    
#     def generate_report(self, topic: str, analysis_text: str, web_research_text: str) -> str:
#         """Generate a comprehensive report based on system prompt framework"""
#         prompt = f"""
#         {self.system_prompt}
        
#         Topic to evaluate: {topic}
        
#         Document Analysis from Google Drive:
#         {analysis_text[:4000]}
        
#         Recent Web Research & Community Sentiment:
#         {web_research_text[:2000]}
        
#         Generate a complete evaluation report following the exact format in the system prompt.
#         Use clear headings and bullet points for better readability.
#         """
        
#         try:
#             response = self.client.chat.completions.create(
#                 model=self.config.openai_model,
#                 messages=[
#                     {"role": "system", "content": self.system_prompt},
#                     {"role": "user", "content": prompt}
#                 ],
#                 max_tokens=3000,
#                 temperature=0.2
#             )
#             return response.choices[0].message.content
#         except Exception as e:
#             logger.error(f"Report generation failed: {e}")
#             return f"Report generation error: {str(e)}"
    
#     def fetch_web_results(self, query: str) -> str:
#         """Fetch web results focusing on community sentiment"""
#         try:
#             response = self.client.chat.completions.create(
#                 model="gpt-4o-mini",
#                 messages=[
#                     {"role": "system", "content": "Search for recent developments and community discussions from Reddit, Twitter, LinkedIn, Medium about the topic. Focus on practical experiences and implementation feedback."},
#                     {"role": "user", "content": f"Research recent updates and community sentiment about: {query}"}
#                 ],
#                 max_tokens=1000,
#                 temperature=0.7
#             )
#             return response.choices[0].message.content
#         except Exception as e:
#             logger.error(f"Web research failed: {e}")
#             return ""
    
#     def get_brief_response(self, query: str, context: str = "") -> str:
#         """Generate brief conversational responses"""
#         try:
#             response = self.client.chat.completions.create(
#                 model=self.config.openai_model,
#                 messages=[
#                     {"role": "system", "content": "You are ZENO, a helpful voice assistant. Give brief, conversational responses (1-2 sentences max)."},
#                     {"role": "user", "content": f"Context: {context}\nQuery: {query}"}
#                 ],
#                 max_tokens=100,
#                 temperature=0.7
#             )
#             return response.choices[0].message.content
#         except Exception as e:
#             return "I encountered an issue processing that request."
    
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
#                 max_tokens=2000,
#                 temperature=0.2,
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


# # =============================================================================
# # MAIN AGENT CLASS WITH FUNCTION TOOLS
# # =============================================================================

# class ZenoAssistant(Agent):
#     """ZENO - The instructor's second brain for GenAI curriculum"""
    
#     def __init__(self, config: Config) -> None:
#         # Initialize with comprehensive instructions
#         super().__init__(instructions="""You are ZENO, an AI teaching Copilot and second brain for a GenAI instructor.

# Core Capabilities:
# - Analyze documents from Google Drive for curriculum relevance
# - Research recent developments and community sentiment from the web
# - Generate comprehensive evaluation reports following the system prompt framework
# - Upload formatted reports to Google Drive
# - Save conversation transcripts when requested
# - Maintain context for follow-up questions

# Behavior:
# - Keep voice responses brief and conversational
# - Only generate full reports when explicitly asked
# - Use the system prompt framework for evaluations
# - Prioritize Drive documents over web sources
# - Track conversation context for intelligent follow-ups

# When asked to "make a report" or "generate report", you will:
# 1. Search relevant documents in Google Drive
# 2. Perform web research for recent updates
# 3. Generate a comprehensive evaluation following the system prompt
# 4. Upload formatted report to Google Drive automatically
# 5. Provide a brief confirmation, not the full report content

# When asked to "save our conversation" or "save transcript", you will:
# 1. Format the conversation history
# 2. Upload it as a Google Doc
# 3. Provide the link to the saved transcript""")
        
#         # Initialize services
#         self.config = config
#         self.gdrive = GoogleDriveService(config)
#         self.openai = OpenAIService(config)
        
#         # State management
#         self._last_report_topic: Optional[str] = None
#         self._last_report_content: Optional[str] = None
#         self._last_analysis: Optional[str] = None
#         self._last_web_research: Optional[str] = None
#         self._conversation_context: List[Dict] = []
        
#         # Transcript tracking (FIX: Initialize these attributes)
#         self._transcript: List[Tuple[str, str]] = []  # List of (user_text, zeno_text) tuples
#         self._followup_active: bool = False  # Track if we're in follow-up mode
#         self._transcript_topic: Optional[str] = None  # Current topic being discussed
#         self._full_conversation: List[Dict] = []  # Full conversation history
    
#     def add_to_transcript(self, user_text: str, zeno_response: str):
#         """Add an interaction to the transcript"""
#         self._transcript.append((user_text, zeno_response))
#         self._full_conversation.append({
#             "timestamp": datetime.now(),
#             "user": user_text,
#             "zeno": zeno_response
#         })
#         # Keep only last 50 interactions in memory
#         if len(self._transcript) > 50:
#             self._transcript = self._transcript[-50:]
    
#     # =========================================================================
#     # FUNCTION TOOLS
#     # =========================================================================
    
#     @function_tool(
#         description="Search and analyze documents from Google Drive related to a topic. Returns key insights from found documents."
#     )
#     async def analyze_google_drive_documents(
#         self, 
#         context: RunContext, 
#         topic: str,
#         max_files: int = 5
#     ) -> str:
#         """Search and analyze Google Drive documents"""
#         try:
#             # Search for relevant files
#             files = self.gdrive.search_files(topic, limit=max_files)
            
#             if not files:
#                 response = f"No documents found in Google Drive for '{topic}'."
#                 self.add_to_transcript(f"Analyze documents about {topic}", response)
#                 return response
            
#             # Analyze documents
#             analysis_parts = []
#             analysis_parts.append(f"Found {len(files)} relevant documents:\n")
            
#             for file_info in files[:max_files]:
#                 content = self.gdrive.get_file_content(file_info['id'], file_info['mimeType'])
#                 if content:
#                     analysis_parts.append(f"\nDocument: {file_info['name']}")
#                     analysis_parts.append(f"Modified: {file_info.get('modifiedTime', 'Unknown')}")
#                     analysis_parts.append(f"Content preview: {content[:500]}...\n")
            
#             self._last_analysis = '\n'.join(analysis_parts)
            
#             # Brief voice response
#             voice_response = f"Analyzed {len(files)} documents from your Drive about {topic}."
#             await context.session.say(voice_response)
            
#             self.add_to_transcript(f"Analyze documents about {topic}", voice_response)
            
#             return self._last_analysis
            
#         except Exception as e:
#             logger.error(f"Document analysis error: {e}")
#             return f"Error analyzing documents: {str(e)}"
    
#     @function_tool(
#         description="Research recent web developments and community sentiment about a topic."
#     )
#     async def research_web_and_community(
#         self,
#         context: RunContext,
#         topic: str
#     ) -> str:
#         """Perform web research focusing on community sentiment"""
#         try:
#             await context.session.say(f"Researching recent developments about {topic}...")
            
#             web_research = self.openai.fetch_web_results(topic)
#             self._last_web_research = web_research
            
#             # Brief summary for voice
#             summary = self.openai.get_brief_response(f"Summarize in one sentence: {web_research[:500]}")
#             await context.session.say(summary)
            
#             self.add_to_transcript(f"Research web about {topic}", summary)
            
#             return web_research
            
#         except Exception as e:
#             logger.error(f"Web research error: {e}")
#             return f"Research error: {str(e)}"
    
#     @function_tool(
#         description="Generate a comprehensive evaluation report following the system prompt framework. Automatically analyzes documents, performs web research, generates formatted report, and uploads to Google Drive."
#     )
#     async def generate_and_upload_report(
#         self,
#         context: RunContext,
#         topic: str,
#         include_web_research: bool = True
#     ) -> str:
#         """Complete report generation pipeline with formatting"""
#         try:
#             # Track this as the active topic
#             self._transcript_topic = topic
#             self._followup_active = True
            
#             await context.session.say(f"Starting comprehensive report generation for {topic}. This will take a moment...")
            
#             # Step 1: Analyze Google Drive documents
#             await context.session.say("Analyzing your Google Drive documents...")
#             files = self.gdrive.search_files(topic, limit=10)
            
#             analysis_text = ""
#             if files:
#                 for file_info in files[:5]:
#                     content = self.gdrive.get_file_content(file_info['id'], file_info['mimeType'])
#                     if content:
#                         analysis_text += f"\n\nDocument: {file_info['name']}\n"
#                         analysis_text += f"Modified: {file_info.get('modifiedTime', 'Unknown')}\n"
#                         analysis_text += f"{content[:2000]}...\n"
#             else:
#                 analysis_text = "No relevant documents found in Google Drive."
            
#             self._last_analysis = analysis_text
            
#             # Step 2: Web research
#             web_research_text = ""
#             if include_web_research:
#                 await context.session.say("Conducting web research and gathering community sentiment...")
#                 web_research_text = self.openai.fetch_web_results(topic)
#                 self._last_web_research = web_research_text
            
#             # Step 3: Generate report using system prompt
#             await context.session.say("Generating evaluation report using system prompt framework...")
#             report_content = self.openai.generate_report(topic, analysis_text, web_research_text)
            
#             # Store report context
#             self._last_report_topic = topic
#             self._last_report_content = report_content
#             self._conversation_context.append({
#                 "timestamp": datetime.now(),
#                 "topic": topic,
#                 "report_snippet": report_content[:500]
#             })
            
#             # Step 4: Upload formatted report to Google Drive
#             file_name = f"{topic} - Evaluation Report {datetime.now().strftime('%Y-%m-%d %H-%M')}"
#             upload_result = self.gdrive.upload_report(report_content, file_name, self.config.google_drive_folder_id)
            
#             if upload_result.get("error"):
#                 response = f"Report generated but upload failed: {upload_result['error']}"
#                 await context.session.say(response)
#                 self.add_to_transcript(f"Generate report about {topic}", response)
#                 return report_content
            
#             # Success response
#             link = upload_result.get('webViewLink', 'Google Drive')
#             voice_response = (
#                 f"Report complete! Analyzed {len(files)} documents and conducted web research. "
#                 f"The formatted evaluation report has been uploaded to your Drive. "
#                 f"You can ask me follow-up questions about the findings."
#             )
#             await context.session.say(voice_response)
            
#             self.add_to_transcript(f"Generate report about {topic}", voice_response)
            
#             return f"Report uploaded successfully: {link}\n\nReport Preview:\n{report_content[:1000]}..."
            
#         except Exception as e:
#             logger.error(f"Report generation error: {e}")
#             error_msg = f"Error generating report: {str(e)}"
#             await context.session.say(error_msg)
#             self.add_to_transcript(f"Generate report about {topic}", error_msg)
#             return error_msg
    
#     @function_tool(
#         description="Answer follow-up questions about the most recent report or analysis"
#     )
#     async def answer_followup_question(
#         self,
#         context: RunContext,
#         question: str
#     ) -> str:
#         """Handle follow-up questions about the last report"""
#         try:
#             if not self._last_report_content:
#                 response = "I don't have a recent report in memory. Would you like me to generate one?"
#                 await context.session.say(response)
#                 self.add_to_transcript(question, response)
#                 return response
            
#             # Mark as follow-up active
#             self._followup_active = True
            
#             # Use OpenAI to answer based on report context
#             prompt = f"""Based on this report about {self._last_report_topic}, answer the following question concisely:

# Report content:
# {self._last_report_content[:3000]}

# Question: {question}

# Provide a brief, specific answer based on the report content."""
            
#             response = self.openai.client.chat.completions.create(
#                 model=self.config.openai_model,
#                 messages=[
#                     {"role": "system", "content": "Answer questions based on the provided report. Be brief and specific."},
#                     {"role": "user", "content": prompt}
#                 ],
#                 max_tokens=300,
#                 temperature=0.3
#             )
            
#             answer = response.choices[0].message.content
#             await context.session.say(answer)
            
#             self.add_to_transcript(question, answer)
            
#             return answer
            
#         except Exception as e:
#             logger.error(f"Follow-up error: {e}")
#             error_msg = "I encountered an error accessing the report context."
#             self.add_to_transcript(question, error_msg)
#             return error_msg
    
#     def _render_transcript_analysis_text(self, analysis: dict) -> str:
#         """
#         Convert transcript-analysis JSON -> plain text with simple section markers
#         that the Docs formatter will turn into headings/bullets.
#         NOT a function_tool; internal helper only.
#         """
#         lines: List[str] = []
#         lines.append("## CONVERSATION OVERVIEW")
#         lines.append(analysis.get("overview", "").strip() or "Not discussed")
#         lines.append("")

#         def sec(title, items):
#             lines.append(f"## {title}")
#             if not items:
#                 lines.append("Not discussed")
#                 lines.append("")
#                 return
#             for it in items:
#                 if isinstance(it, dict):
#                     # key_questions entries
#                     if "q" in it and "why_it_matters" in it:
#                         lines.append(f"- Q: {it['q']} — {it.get('why_it_matters','')}")
#                     # answers entries
#                     elif {"q_ref", "summary"} <= set(it.keys()):
#                         lines.append(f"- Answer to Q{it.get('q_ref')}:")
#                         summary = it.get("summary") or []
#                         if isinstance(summary, str):
#                             lines.append(f"  - {summary}")
#                         else:
#                             for s in summary:
#                                 lines.append(f"  - {s}")
#                     # action items
#                     elif {"owner", "task"} <= set(it.keys()):
#                         due = f" (due: {it.get('due')})" if it.get("due") else ""
#                         notes = f" — {it.get('notes','')}" if it.get("notes") else ""
#                         lines.append(f"- {it.get('owner','Unknown')}: {it.get('task','')}{due}{notes}")
#                     else:
#                         lines.append(f"- {json.dumps(it, ensure_ascii=False)}")
#                 else:
#                     lines.append(f"- {str(it)}")
#             lines.append("")

#         sec("KEY QUESTIONS", analysis.get("key_questions") or [])
#         sec("ANSWERS", analysis.get("answers") or [])
#         sec("CLARIFICATIONS AND MISCONCEPTIONS", analysis.get("clarifications_and_misconceptions") or [])
#         sec("DECISIONS AND AGREEMENTS", analysis.get("decisions_and_agreements") or [])
#         sec("OPEN ISSUES", analysis.get("open_issues") or [])
#         sec("ACTION ITEMS", analysis.get("action_items") or [])
#         sec("TEACHING OUTLINE", analysis.get("teaching_outline") or [])
#         sec("QUIZ", analysis.get("quiz") or [])
#         sec("FOLLOW-UP RESOURCES", analysis.get("followup_resources") or [])

#         cov = analysis.get("transcript_coverage") or {}
#         if cov:
#             lines.append("## COVERAGE")
#             lines.append(f"- Turns total: {cov.get('turns_total', 0)}")
#             lines.append(f"- Turns used: {cov.get('turns_used', 0)}")
#             lines.append(f"- Coverage: {cov.get('coverage_pct', 0)}%")
#             lines.append("")

#         return "\n".join(lines).strip()

    
     
#     @function_tool(
#         description="Save the conversation transcript to Google Drive as a formatted Google Doc. Call this when user asks to save the conversation or transcript."
#     )
#     async def save_conversation_transcript(
#         self, 
#         context: RunContext,
#         report_type: str = "raw"  # "raw" (default) or "analysis"
#     ) -> str:

#         """Save conversation transcript to Google Drive"""
#         try:
#             if not self._transcript:
#                 response = "No conversation to save yet. Have a conversation first, then ask me to save it."
#                 await context.session.say(response)
#                 return response
            
#             # Determine topic and transcript text
#             topic = self._transcript_topic or self._last_report_topic or "General Conversation"
#             transcript_text = "\n".join([f"User — {u}\nZeno — {z}" for (u, z) in self._transcript])

#             # ANALYSIS MODE: build a teaching-prep report from the transcript
#             if (report_type or "").lower().startswith("analysis"):
#                 date_range = datetime.now().strftime("%Y-%m-%d")
#                 analysis = self.openai.analyze_transcript_for_insights(topic, transcript_text, date_range=date_range)
#                 pretty_text = self._render_transcript_analysis_text(analysis)
#                 title = analysis.get("title") or f"{topic} — Follow-up Conversation Report ({date_range})"

#                 meta = self.gdrive.create_google_doc_with_formatting(
#                     title=title,
#                     content=pretty_text,
#                     folder_id=self.config.google_drive_folder_id
#                 )
#                 if meta.get("error"):
#                     msg = f"Failed to save transcript analysis: {meta['error']}"
#                     await context.session.say(msg)
#                     return msg

#                 link = meta.get("webViewLink", meta.get("id", "Google Drive"))
#                 voice = "Transcript analysis report saved to Drive."
#                 await context.session.say(voice)
#                 self.add_to_transcript("Save our conversation (analysis)", voice)
#                 return f"Transcript analysis saved: {link}"
            
#             # RAW MODE: save simple transcript
#             else:
#                 meta = self.gdrive.create_transcript_doc(
#                     topic,
#                     list(self._transcript),
#                     self.config.google_drive_folder_id
#                 )
                
#                 if meta.get("error"):
#                     error_msg = f"Failed to save transcript: {meta['error']}"
#                     await context.session.say(error_msg)
#                     return error_msg
                
#                 link = meta.get('webViewLink', meta.get('id', 'Google Drive'))
#                 success_msg = f"Conversation transcript saved to Drive. It contains {len(self._transcript)} exchanges."
#                 await context.session.say(success_msg)
                
#                 # Add this save action to transcript
#                 self.add_to_transcript("Save our conversation", success_msg)
                
#                 return f"Transcript saved: {link}"
#         except Exception as e:
#             # Final catch to prevent crashes and surface a concise error
#             logger.exception("save_conversation_transcript failed: %s", e)
#             try:
#                 await context.session.say("Sorry — I couldn't save the transcript right now.")
#             except Exception:
#                 pass
#             return f"error_saving_transcript: {e}"

    
#     @function_tool(
#         description="Get a quick analysis or summary without generating a full report"
#     )
#     async def quick_analysis(
#         self,
#         context: RunContext,
#         topic: str
#     ) -> str:
#         """Provide quick analysis without full report generation"""
#         try:
#             # Quick document check
#             files = self.gdrive.search_files(topic, limit=3)
            
#             if files:
#                 quick_summary = f"I found {len(files)} documents about {topic}. "
#                 quick_summary += f"Most recent: {files[0]['name']}. "
                
#                 # Get brief insight
#                 content = self.gdrive.get_file_content(files[0]['id'], files[0]['mimeType'])
#                 if content:
#                     insight = self.openai.get_brief_response(
#                         f"Give one key insight about {topic} from: {content[:500]}"
#                     )
#                     quick_summary += insight
#             else:
#                 quick_summary = f"No documents found about {topic} in your Drive. "
#                 # Try web research
#                 web_insight = self.openai.get_brief_response(
#                     f"Give one recent update about {topic}"
#                 )
#                 quick_summary += web_insight
            
#             await context.session.say(quick_summary)
#             self.add_to_transcript(f"Quick analysis of {topic}", quick_summary)
            
#             return quick_summary
            
#         except Exception as e:
#             return f"Quick analysis error: {str(e)}"
    
#     @function_tool(
#         description="Get the conversation context and history"
#     )
#     async def get_conversation_context(
#         self,
#         context: RunContext
#     ) -> str:
#         """Return conversation history and context"""
#         if not self._conversation_context:
#             return "No conversation history yet."
        
#         history = "Recent conversation topics:\n"
#         for item in self._conversation_context[-5:]:  # Last 5 interactions
#             history += f"- {item['topic']} at {item['timestamp'].strftime('%H:%M')}\n"
        
#         if self._last_report_topic:
#             history += f"\nLast report generated: {self._last_report_topic}"
        
#         await context.session.say(f"We've discussed {len(self._conversation_context)} topics today.")
#         return history

# # =============================================================================
# # LIVEKIT ENTRY POINT
# # =============================================================================

# async def entrypoint(ctx: agents.JobContext):
#     """Main entry point for LiveKit agent"""
    
#     # Connect to room
#     await ctx.connect()
    
#     # Initialize configuration
#     config = Config()
    
#     # Create agent session with voice components
#     session = AgentSession(
#         stt=lk_openai.STT(model="whisper-1"),
#         llm=lk_openai.LLM(model=config.openai_model),
#         tts=lk_openai.TTS(
#             model="tts-1",
#             voice="nova",
#             instructions="Speak in a clear, friendly, and professional tone. Be concise unless providing detailed analysis.",
#             speed=1.2,
#         ),
#         vad=silero.VAD.load(),
#         turn_detection="vad"
#     )
    
#     # Create and start the assistant
#     assistant = ZenoAssistant(config)
    
#     await session.start(
#         room=ctx.room,
#         agent=assistant,
#         room_input_options=RoomInputOptions(
#             noise_cancellation=noise_cancellation.BVC()
#         )
#     )
    
#     # Initial greeting
#     await session.generate_reply(
#         instructions="""Greet the user as ZENO. Let them know you can:
#         1. Analyze documents from their Google Drive
#         2. Research recent web developments
#         3. Generate comprehensive formatted evaluation reports
#         4. Save conversation transcripts
#         5. Answer follow-up questions
#         Keep it brief and friendlyand in one line only."""
#     )

# if __name__ == "__main__":
#     agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))

