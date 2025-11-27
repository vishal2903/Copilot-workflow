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
