from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional
import os
import json
import httpx
import re
import logging
from pathlib import Path
from datetime import datetime
import time
from contextlib import asynccontextmanager

class Settings:
    PROJECT_NAME: str = "AI Flowchart Generator"
    VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "TRUE").lower() == "true"
    
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = "gemini-2.5-flash"
    GEMINI_TIMEOUT: int = 30
    
    ALLOWED_HOSTS: List[str] = ["*"] if DEBUG else os.getenv("ALLOWED_HOSTS", "").split(",")
    CORS_ORIGINS: List[str] = ["*"] if DEBUG else os.getenv("CORS_ORIGINS", "").split(",")
    
    MAX_REQUEST_SIZE: int = 10000
    REQUEST_TIMEOUT: int = 30

    LOG_LEVEL: str = "DEBUG" if DEBUG else "INFO"

settings = Settings()


logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log') if not settings.DEBUG else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)


class Decision(BaseModel):
    question: str = Field(..., min_length=1, max_length=200)
    yes: str = Field(..., min_length=1, max_length=200)
    no: str = Field(..., min_length=1, max_length=200)

class FlowchartResponse(BaseModel):
    steps: List[str] = Field(..., min_items=1)
    decisions: List[Decision] = Field(default_factory=list)
    mermaid: str = Field(..., min_length=10)

class GenerateRequest(BaseModel):
    text: str = Field(..., min_length=5, max_length=10000)
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty or whitespace only')
        return v.strip()

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
    gemini_configured: bool

class ErrorResponse(BaseModel):
    detail: str
    timestamp: str
    path: Optional[str] = None


class GeminiClient:
    
    def __init__(self):
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        self.api_key = settings.GEMINI_API_KEY
        self.timeout = settings.GEMINI_TIMEOUT
        self._client: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()
    
    async def generate_flowchart(self, prompt: str) -> Dict:
        url = f"{self.base_url}/{settings.GEMINI_MODEL}:generateContent?key={self.api_key}"
        
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.2,                  
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 4096,
            }
        }
        
        try:
            response = await self._client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Gemini API HTTP error: {e.response.status_code} - {e.response.text}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"External API error: {e.response.status_code}"
            )
        except httpx.RequestError as e:
            logger.error(f"Gemini API request error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="External API unavailable"
            )


class FlowchartService:
    
    PROMPT_TEMPLATE = """You are an expert in flowchart design and Mermaid syntax.

Convert the following text into a structured flowchart:

**User Input:**
{user_text}

**Requirements:**
1. Extract clear sequential steps
2. Identify all decision points (yes/no branches)
3. Generate valid Mermaid flowchart code using "flowchart TD" syntax
4. Use simple, descriptive node labels
5. Ensure every decision has both Yes and No branches
6. Use proper Mermaid syntax: arrows (-->), decision diamonds {{}}, rectangles []

**Output Format:**
Return ONLY a valid JSON object (no markdown, no additional text):

{{
  "steps": ["step1", "step2", "step3"],
  "decisions": [
    {{"question": "decision text?", "yes": "action if yes", "no": "action if no"}}
  ],
  "mermaid": "flowchart TD\\n    Start[Start]\\n    ..."
}}

**Mermaid Syntax Examples:**
- Rectangle: A[Label]
- Diamond: B{{Decision?}}
- Arrow: A --> B
- Labeled arrow: A -->|Yes| B

Generate the flowchart now:"""

    @staticmethod
    def extract_json_from_response(content: str) -> Dict:
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r'(\{.*\})', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                raise ValueError("No valid JSON found in response")
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}. Content: {json_str[:500]}")
            raise ValueError(f"Invalid JSON structure: {str(e)}")
    
    @staticmethod
    def validate_mermaid_syntax(mermaid_code: str) -> bool:
        if not mermaid_code.strip().startswith('flowchart'):
            return False
        if '-->' not in mermaid_code:
            return False
        return True
    
    @classmethod
    async def generate(cls, text: str) -> FlowchartResponse:
        start_time = time.time()
        logger.info(f"Generating flowchart for text: {text[:100]}...")
        
        # Create prompt
        prompt = cls.PROMPT_TEMPLATE.format(user_text=text)
        
        # Call Gemini API
        async with GeminiClient() as client:
            response_data = await client.generate_flowchart(prompt)
        
        if 'candidates' not in response_data or not response_data['candidates']:
            logger.error(f"Gemini response missing 'candidates': {response_data}")
            raise ValueError("No candidates found in AI model response. This might be due to safety filters or an empty response.")
        
        first_candidate = response_data['candidates'][0]
        if 'content' not in first_candidate or 'parts' not in first_candidate['content'] or not first_candidate['content']['parts']:
            logger.error(f"Gemini candidate missing 'content' or 'parts': {first_candidate}")
            raise ValueError("AI model response content is malformed or empty.")
            
        content = first_candidate['content']['parts'][0]['text']
        logger.debug(f"Raw LLM response: {content[:500]}")
        
        result = cls.extract_json_from_response(content)
        
        if not all(key in result for key in ["steps", "decisions", "mermaid"]):
            raise ValueError("Missing required fields in response")
        
        if not cls.validate_mermaid_syntax(result['mermaid']):
            raise ValueError("Invalid Mermaid syntax generated")
        
        flowchart = FlowchartResponse(**result)
        
        duration = time.time() - start_time
        logger.info(f"Flowchart generated successfully in {duration:.2f}s")
        
        return flowchart


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    logger.info(f"Gemini API configured: {bool(settings.GEMINI_API_KEY)}")
    
    yield
    
    logger.info("Shutting down application")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Transform natural language into beautiful Mermaid flowcharts using AI",
    lifespan=lifespan,
    docs_url="/api/docs" if settings.DEBUG else None,
    redoc_url="/api/redoc" if settings.DEBUG else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

if not settings.DEBUG and settings.ALLOWED_HOSTS != ["*"]:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.ALLOWED_HOSTS)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Duration: {duration:.3f}s"
    )
    
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            detail=exc.detail,
            timestamp=datetime.utcnow().isoformat(),
            path=str(request.url.path)
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.exception("Unexpected error occurred")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            detail="An unexpected error occurred" if not settings.DEBUG else str(exc),
            timestamp=datetime.utcnow().isoformat(),
            path=str(request.url.path)
        ).dict()
    )


@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_path = Path(__file__).parent / "index.html"
    try:
        return html_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="index.html not found")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        timestamp=datetime.utcnow().isoformat(),
        gemini_configured=bool(settings.GEMINI_API_KEY)
    )

@app.post("/generate", response_model=FlowchartResponse)
async def generate_flowchart(request: GenerateRequest):
    
    try:
        return await FlowchartService.generate(request.text)
    except ValueError as e:
        logger.warning(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error generating flowchart: {str(e)}", exc_info=True)
        raise

@app.get("/api/info")
async def api_info():
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "debug": settings.DEBUG,
        "endpoints": {
            "generate": "/generate",
            "health": "/health",
            "docs": "/api/docs" if settings.DEBUG else None,
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
