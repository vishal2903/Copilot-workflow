#!/usr/bin/env python3
"""
Google Drive LLM Analyzer with Voice Support - Optimized Version
Teacher's AI Copilot for document analysis and research
"""

import os
import sys
import json
import asyncio
import logging
import time
import io
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from livekit.agents import UserInputTranscribedEvent, ConversationItemAddedEvent

# Core dependencies
import click
import PyPDF2
import pytz
from rich.console import Console
from rich.panel import Panel
from dotenv import load_dotenv

# Google APIs
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.auth.exceptions import RefreshError
# OpenAI
from openai import OpenAI
import tiktoken

# Document processing
from docx import Document as DocxDocument

# LiveKit Voice - Modern 2024+ API with Agent/AgentSession pattern
LIVEKIT_AVAILABLE = False
lk_openai = None
silero = None
Agent = None
JobContext = None
WorkerOptions = None
llm = None
AgentSession = None
RunContext = None

try:
    # Core livekit agents framework - modern pattern
    from livekit import agents
    from livekit.agents import (
        Agent,
        AgentSession,
        JobContext,
        RunContext,
        WorkerOptions,
        cli as lk_cli,
        llm,
        AutoSubscribe
    )
    
    # Plugins
    from livekit.plugins import openai as lk_openai
    from livekit.plugins import silero
    
    LIVEKIT_AVAILABLE = True
    print("[SUCCESS] LiveKit agents framework loaded (modern API)")
    
except ImportError as e:
    print(f"[ERROR] LiveKit import failed: {e}")
    print("Install with: pip install 'livekit-agents[openai,silero]' livekit-plugins-openai livekit-plugins-silero")
except Exception as e:
    print(f"[ERROR] Unexpected error loading LiveKit: {e}")

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
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
        self.google_docs_token = os.getenv("GOOGLE_DOCS_TOKEN")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.system_prompt_path = os.getenv("SYSTEM_PROMPT_PATH", "sytem_prompt.md")
        
        # LiveKit configuration
        self.livekit_url = os.getenv("LIVEKIT_URL")
        self.livekit_api_key = os.getenv("LIVEKIT_API_KEY")
        self.livekit_api_secret = os.getenv("LIVEKIT_API_SECRET")
        
        self._validate()
    
    def _validate(self):
        # Only validate OpenAI for voice agent (required for LLM)
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found in .env")

# =============================================================================
# CORE SERVICES
# =============================================================================

class GoogleDriveService:
    """Handles all Google Drive operations"""
    
    SCOPES = [
            "https://www.googleapis.com/auth/drive.readonly",
            "https://www.googleapis.com/auth/documents.readonly",
    ]
    
    def __init__(self, config: Config):
        self.config = config
        self.drive_service = None
        self.docs_service = None
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate with Google APIs and create Drive / Docs clients."""
        token_path_str = getattr(self.config, "google_drive_token", None)
        if not token_path_str:
            raise FileNotFoundError("Google Drive token path not set in config (google_drive_token).")

        token_path = Path(token_path_str)
        if token_path.exists():
            try:
                creds = Credentials.from_authorized_user_file(str(token_path), self.SCOPES)
            except Exception as exc:
                raise RuntimeError(f"Failed to load Google credentials from {token_path}: {exc}") from exc
        else:
            creds = None

        # If no creds or invalid, try refresh or run auth flow
        if not creds or not getattr(creds, "valid", False):
            # try refresh if possible
            try:
                if creds and getattr(creds, "expired", False) and getattr(creds, "refresh_token", None):
                    creds.refresh(Request())
                else:
                    # Need client secrets path to run OAuth flow
                    client_secrets = getattr(self.config, "google_credentials_path", None)
                    if not client_secrets or not Path(client_secrets).exists():
                        raise FileNotFoundError(
                            "Google client secrets not found (google_credentials_path). "
                            "Provide a valid path or pre-create the token file."
                        )
                    flow = InstalledAppFlow.from_client_secrets_file(client_secrets, self.SCOPES)
                    # NOTE: for headless servers you may want run_console() instead of run_local_server()
                    creds = flow.run_local_server(port=0)
            except Exception as exc:
                raise RuntimeError(f"Failed during Google OAuth/refresh: {exc}") from exc

            # Persist token file (create parent if needed)
            try:
                token_path.parent.mkdir(parents=True, exist_ok=True)
                with open(token_path, "w", encoding="utf-8") as fh:
                    fh.write(creds.to_json())
            except Exception as exc:
                # Token saving failed but we may still have valid creds in-memory
                logger.warning("Could not save Google token to %s: %s", token_path, exc)

        # Build API clients
        try:
            self.drive_service = build("drive", "v3", credentials=creds)
            self.docs_service = build("docs", "v1", credentials=creds)
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize Google API clients: {exc}") from exc

    def search_files(self, query: str = None, limit: int = 10) -> List[Dict]:
        """Search for files in Google Drive"""
        try:
            q = "trashed = false"
            if query:
                q += f" and (name contains '{query}' or fullText contains '{query}')"
            
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
        """Get content from a file"""
        try:
            if mime_type == 'application/vnd.google-apps.document':
                # Google Docs - get as plain text
                content = self.drive_service.files().export_media(
                    fileId=file_id,
                    mimeType='text/plain'
                ).execute()
                return content.decode('utf-8')
            else:
                # Other files - download and process
                request = self.drive_service.files().get_media(fileId=file_id)
                file_io = io.BytesIO()
                downloader = MediaIoBaseDownload(file_io, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
                return self._extract_text(file_io.getvalue(), mime_type)
        except Exception as e:
            logger.error(f"Error getting file content: {e}")
            return ""
    
    def _extract_text(self, content: bytes, mime_type: str) -> str:
        """Extract text from various file formats"""
        try:
            if mime_type == 'application/pdf':
                return self._extract_pdf_text(content)
            elif 'wordprocessingml' in mime_type:
                return self._extract_docx_text(content)
            elif mime_type == 'text/plain':
                return content.decode('utf-8', errors='ignore')
            else:
                return ""
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            return ""
    
    def _extract_pdf_text(self, content: bytes) -> str:
        """Extract text from PDF"""
        try:
            pdf_file = io.BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = []
            for page in pdf_reader.pages[:50]:  # Limit to 50 pages
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

class OpenAIService:
    """Handles OpenAI interactions"""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(api_key=config.openai_api_key)
        self.system_prompt = self._load_system_prompt()
    
    def _load_system_prompt(self) -> str:
        """Load system prompt from file with UTF-8 encoding"""
        try:
            if os.path.exists(self.config.system_prompt_path):
             # Add encoding='utf-8' to handle special characters
                with open(self.config.system_prompt_path, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            logger.warning(f"Could not load system prompt: {e}")
        return "You are a helpful AI assistant for document analysis."
    
    def analyze_documents(self, documents: List[str], topic: str) -> str:
        """Analyze documents using GPT"""
        try:
            combined_text = '\n\n'.join(documents[:5])  # Limit for token management
            
            prompt = f"""
            Analyze these documents about: {topic}
            
            Documents:
            {combined_text[:8000]}  # Token limit
            
            Provide:
            1. Key insights
            2. Main themes
            3. Actionable recommendations
            """
            
            response = self.client.chat.completions.create(
                model=self.config.openai_model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI analysis error: {e}")
            return f"Analysis failed: {str(e)}"
    
    def web_research(self, topic: str) -> str:
        """Perform web research using Responses API"""
        try:
            response = self.client.responses.create(
                model=self.config.openai_model,
                tools=[{"type": "web_search"}],
                input=f"Research the latest developments about: {topic}"
            )
            return response.output_text
        except:
            # Fallback to regular completion
            return self.analyze_documents([], topic)

class AnalysisService:
    """Main business logic orchestration"""
    
    def __init__(self, config: Config):
        self.config = config
        self.gdrive = GoogleDriveService(config)
        self.openai = OpenAIService(config)
        self.memory = ConversationMemory()
    
    async def analyze_topic(self, topic: str, query: str = None) -> Dict:
        """Main analysis function"""
        try:
            # Search for documents
            files = self.gdrive.search_files(query or topic)
            
            if not files:
                return {
                    'status': 'no_documents',
                    'message': f"No documents found for '{topic}'",
                    'analysis': self.openai.web_research(topic)
                }
            
            # Get document contents
            documents = []
            for file in files[:5]:  # Process top 5 files
                content = self.gdrive.get_file_content(file['id'], file['mimeType'])
                if content:
                    documents.append(content)
            
            # Analyze with OpenAI
            analysis = self.openai.analyze_documents(documents, topic)
            
            # Store in memory for follow-ups
            self.memory.add_interaction(topic, analysis)
            
            return {
                'status': 'success',
                'files_analyzed': len(documents),
                'analysis': analysis,
                'topic': topic
            }
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }

class ConversationMemory:
    """Simple conversation state management"""
    
    def __init__(self):
        self.history = []
        self.current_topic = None
    
    def add_interaction(self, topic: str, response: str):
        self.history.append({
            'timestamp': datetime.now(),
            'topic': topic,
            'response': response[:500]  # Store summary
        })
        self.current_topic = topic
        
        # Keep only last 10 interactions
        if len(self.history) > 10:
            self.history = self.history[-10:]
    
    def get_context(self) -> str:
        if not self.history:
            return ""
        return f"Previous topic: {self.current_topic}"

# =============================================================================
# VOICE AGENT (LIVEKIT)
# =============================================================================

if LIVEKIT_AVAILABLE:
    
    async def entrypoint(ctx: JobContext):
        """Minimal LiveKit agent entry point"""
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        
        # Lazy initialization - only create when needed
        config = Config()
        service = None
        wake_word_active = [False]
        
        # Create agent session
        session = AgentSession(
            vad=silero.VAD.load(),
            stt=lk_openai.STT(model="whisper-1"),
            llm=lk_openai.LLM(model="gpt-4o-mini"),
            tts=lk_openai.TTS(voice="alloy", speed=1.0),
        )
        # --- helpers: spawn + async text handler (small, reusable) ----------
        def _spawn(coro):
            """Fire-and-forget wrapper that logs exceptions from background tasks."""
            async def _wrap():
                try:
                    await coro
                except Exception as e:
                    logger.exception("Background task crashed: %s", e)
            asyncio.create_task(_wrap())

        async def _handle_user_text(text: str):
            """Main async handler for final user transcripts."""
            nonlocal service
            t = (text or "").lower().strip()

            # wake word
            if "hey zeno" in t:
                wake_word_active[0] = True
                await session.say("Yes, I'm listening. Ask me to analyze or research topics.")
                return

            # analyze / research commands
            if wake_word_active[0] and any(w in t for w in ("analyze", "research")):
                topic = t.replace("analyze", "").replace("research", "").strip()

                # lazy-init the analysis service
                if service is None:
                    try:
                        service = AnalysisService(config)
                        logger.info("Google Drive service initialized")
                    except Exception as e:
                        await session.say(f"Sorry, I can't access Google Drive right now. Error: {str(e)[:50]}")
                        return

                try:
                    result = await asyncio.wait_for(service.analyze_topic(topic), 45)
                    response = f"I analyzed {result.get('files_analyzed', 0)} documents about {topic}."
                    await session.say(response)
                except asyncio.TimeoutError:
                    await session.say("Analysis timed out. Try a narrower topic.")
                except Exception as e:
                    await session.say(f"Analysis failed: {str(e)[:50]}")
                return

        # --- sync event binding (must be sync) -------------------------------
        # Use the v1 transcript event and only act on final transcripts.
        from livekit.agents import UserInputTranscribedEvent, ConversationItemAddedEvent

        @session.on("user_input_transcribed")
        def _on_user_input(ev: UserInputTranscribedEvent):
            if hasattr(ev, "is_final") and not ev.is_final:
                return
            _spawn(_handle_user_text(ev.transcript or ""))

        @session.on("conversation_item_added")
        def _on_item_added(ev: ConversationItemAddedEvent):
            # Fires for both user and agent items; filter to user text
            if getattr(ev.item, "role", "") == "user":
                _spawn(_handle_user_text(getattr(ev.item, "text_content", "") or ""))


        # Start agent (unchanged)
        agent = Agent(instructions="You are Zeno, an AI teaching assistant. Say 'Hey Zeno' to activate me for document analysis.")
        await session.start(agent=agent, room=ctx.room)
        await session.say("Zeno is ready. Say 'Hey Zeno' to wake me, or type in the console.")


        
      

# =============================================================================
# CLI INTERFACE
# =============================================================================

console = Console()

@click.group()
def cli():
    """Google Drive Analyzer with Voice Support"""
    pass

@cli.command()
@click.option('--topic', '-t', help='Topic to analyze')
@click.option('--query', '-q', help='Search query for documents')
def analyze(topic: str, query: str):
    """Analyze documents from Google Drive"""
    config = Config()
    service = AnalysisService(config)
    
    console.print(Panel(f"Analyzing: {topic or query}", style="blue"))
    
    # Run async function
    result = asyncio.run(service.analyze_topic(topic or query, query))
    
    if result['status'] == 'success':
        console.print(f"\n[green]Analysis Complete![/green]")
        console.print(f"Files analyzed: {result['files_analyzed']}")
        console.print("\n[bold]Analysis:[/bold]")
        console.print(result['analysis'])
    else:
        console.print(f"[red]Error: {result.get('message')}[/red]")


@cli.command()
def test():
    """Test connections"""
    try:
        config = Config()
        console.print("[green]OK Configuration loaded[/green]")
        
        # Test Google Drive
        gdrive = GoogleDriveService(config)
        files = gdrive.search_files(limit=1)
        console.print(f"[green]OK Google Drive connected ({len(files)} files)[/green]")
        
        # Test OpenAI
        openai = OpenAIService(config)
        console.print("[green]OK OpenAI connected[/green]")
        
        # Test LiveKit
        if LIVEKIT_AVAILABLE:
            console.print("[green]OK LiveKit available[/green]")
        else:
            console.print("[yellow]WARNING LiveKit not installed[/yellow]")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")

# Standard LiveKit CLI integration  
if __name__ == "__main__":
    if LIVEKIT_AVAILABLE and len(sys.argv) > 1 and sys.argv[1] in ['dev', 'console', 'start']:
        lk_cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
    else:
        cli()