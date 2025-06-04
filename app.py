from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Request
import os
import json
import logging
from typing import List, Dict, Optional
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
from pinecone import Pinecone
from datetime import datetime
from collections import defaultdict

# ======================== CONFIG ========================
class Config:
    # Twilio configuration
    TWILIO_ACCOUNT_SID: str = os.getenv("TWILIO_ACCOUNT_SID", "")
    TWILIO_AUTH_TOKEN: str = os.getenv("TWILIO_AUTH_TOKEN", "")
    TWILIO_PHONE_NUMBER: str = os.getenv("TWILIO_PHONE_NUMBER", "+15138132458")

    # OpenAI configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # Pinecone configuration
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "")
    TREATMENT_INDEX_NAME: str = os.getenv("TREATMENT_INDEX_NAME", "treatment-index")

    # Emergency hotline
    NIGERIA_EMERGENCY_HOTLINE: str = os.getenv("NIGERIA_EMERGENCY_HOTLINE", "112")

    # FastAPI settings
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"


config = Config()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======================== PINECONE CLIENT ========================
class PineconeClient:
    def __init__(self):
        if not config.PINECONE_API_KEY or not config.PINECONE_ENVIRONMENT:
            raise ValueError("PINECONE_API_KEY and PINECONE_ENVIRONMENT must be set")
        os.environ["PINECONE_API_KEY"] = config.PINECONE_API_KEY
        os.environ["PINECONE_ENVIRONMENT"] = config.PINECONE_ENVIRONMENT
        try:
            self._treatment_index = Pinecone().Index(name=config.TREATMENT_INDEX_NAME)
            stats = self._treatment_index.describe_index_stats()
            logger.info(f"✅ Treatment index connected successfully - {stats.total_vector_count} vectors")
        except Exception as e:
            logger.error(f"❌ Failed to initialize treatment index: {e}")
            self._treatment_index = None

    async def query_treatment(self, symptom_text: str, top_k: int = 3) -> List[str]:
        """
        Query Pinecone for similar treatments based on symptom text.
        """
        if not self._treatment_index:
            return []

        openai = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        embedding_resp = await openai.embeddings.create(input=symptom_text, model="text-embedding-3-small")
        vector = embedding_resp.data[0].embedding

        query_result = self._treatment_index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True
        )
        treatments = []
        for match in query_result.matches:
            metadata = match.metadata or {}
            if "treatment" in metadata:
                treatments.append(metadata["treatment"])
        return treatments


# ======================== OPENAI TRIAGE SERVICE ========================
class TriageRequest(BaseModel):
    description: str
    thread_id: Optional[str] = None
    city: Optional[str] = None


class TriageResponse(BaseModel):
    text: str
    possible_conditions: List[str] = Field(default_factory=list)
    safety_measures: List[str] = Field(default_factory=list)
    send_sos: bool = False
    follow_up_questions: List[str] = Field(default_factory=list)
    thread_id: Optional[str] = None


class TriageService:
    def __init__(self):
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY must be set")
        self.openai = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        try:
            self.pinecone_client = PineconeClient()
        except Exception as e:
            logger.warning(f"⚠️ Pinecone unavailable: {e}")
            self.pinecone_client = None

        # In-memory thread storage
        self.threads: Dict[str, List[str]] = defaultdict(list)

    async def analyze(self, request: TriageRequest) -> TriageResponse:
        """
        Send prompt to OpenAI and parse response for triage.
        """
        prompt = self._build_prompt(request)
        try:
            chat_resp = await self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a medical triage assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=512,
                temperature=0.3
            )
        except Exception as e:
            logger.error(f"❌ OpenAI API error: {e}")
            raise HTTPException(status_code=502, detail="OpenAI API request failed")

        content = chat_resp.choices[0].message.content.strip()
        return self._parse_response(content, request.thread_id)

    def _build_prompt(self, request: TriageRequest) -> str:
        """
        Construct the user prompt for OpenAI including thread context.
        """
        history = ""
        if request.thread_id and request.thread_id in self.threads:
            history = "\n".join(self.threads[request.thread_id]) + "\n"
        prompt = (
            f"{history}"
            f"Patient Location: {request.city or 'Unknown'}\n"
            f"Symptoms: {request.description}\n"
            f"Provide possible conditions, safety measures, and whether emergency services are needed. "
            f"Format response as JSON with keys: text, possible_conditions, safety_measures, send_sos, follow_up_questions."
        )
        return prompt

    def _parse_response(self, content: str, thread_id: Optional[str]) -> TriageResponse:
        """
        Parse OpenAI response assumed to be in JSON. If parsing fails, wrap raw content into text.
        """
        try:
            data = json.loads(content)
            response = TriageResponse(**data)
        except json.JSONDecodeError:
            response = TriageResponse(text=content)

        if thread_id:
            self.threads[thread_id].append(f"User: {response.text}")
        return response


# ======================== FASTAPI APP SETUP ========================
app = FastAPI(
    title="Medical Triage Assistant API",
    description="AI-powered medical triage over SMS and WhatsApp",
    version="5.0.0",
)

# Enable CORS if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

triage_service = TriageService()

# Initialize Twilio client once
if config.TWILIO_ACCOUNT_SID and config.TWILIO_AUTH_TOKEN:
    twilio_client = Client(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)
    logger.info("✅ Twilio client initialized")
else:
    twilio_client = None
    logger.warning("⚠️ Twilio credentials not found; SMS/WhatsApp features disabled")


# ======================== UTILITY FUNCTIONS ========================
def is_whatsapp_message(from_number: str) -> bool:
    """Check if message is from WhatsApp"""
    return from_number.startswith("whatsapp:")


def format_response_for_sms(response: TriageResponse) -> str:
    """Format the triage response for SMS/WhatsApp"""
    message_parts = []

    # Main response
    message_parts.append(f"🩺 MEDICAL ASSESSMENT:\n{response.text}")

    # Emergency check
    if response.send_sos:
        message_parts.append(
            f"\n🚨 EMERGENCY: Call {config.NIGERIA_EMERGENCY_HOTLINE} immediately!"
        )

    # Possible conditions
    if response.possible_conditions:
        conditions = ", ".join(response.possible_conditions)
        message_parts.append(f"\n⚕️ Possible: {conditions}")

    # Safety measures
    if response.safety_measures:
        measures = "; ".join(response.safety_measures)
        message_parts.append(f"\n🛡️ Safety: {measures}")

    # Follow-up questions
    if response.follow_up_questions:
        questions = " ".join(response.follow_up_questions)
        message_parts.append(f"\n❓ {questions}")

    return "\n".join(message_parts).strip()


# ======================== API ENDPOINTS ========================
@app.get("/health")
async def health_check():
    return {
        "status": "running",
        "version": app.version,
        "features": [
            "AI-powered medical triage",
            "SMS and WhatsApp support",
            "Conversation memory",
            "Emergency detection",
            "Multi-platform accessibility"
        ]
    }


@app.post("/webhook/twilio/sms")
async def handle_twilio_sms(request: Request):
    """
    Handle incoming SMS messages posted by Twilio.
    """
    if not twilio_client:
        logger.error("Twilio client not configured")
        raise HTTPException(status_code=500, detail="SMS service unavailable")

    form_data = await request.form()
    from_number = form_data.get("From", "")
    body = form_data.get("Body", "").strip()

    logger.info(f"📱 SMS received from {from_number}: '{body}'")

    triage_request = TriageRequest(
        description=body,
        thread_id=form_data.get("SmsSid", None),
        city=None
    )

    try:
        triage_response = await triage_service.analyze(triage_request)
    except HTTPException as e:
        logger.error(f"Error during triage: {e.detail}")
        twiml_error = MessagingResponse()
        twiml_error.message("⚠️ Sorry, something went wrong. Please try again later.")
        return Response(content=str(twiml_error), media_type="application/xml")

    formatted = format_response_for_sms(triage_response)
    twiml = MessagingResponse()
    twiml.message(formatted)

    logger.info(f"✅ SMS response sent to {from_number}")
    return Response(content=str(twiml), media_type="application/xml")


@app.post("/webhook/twilio/whatsapp")
async def handle_twilio_whatsapp(request: Request):
    """
    Handle incoming WhatsApp messages posted by Twilio.
    """
    if not twilio_client:
        logger.error("Twilio client not configured")
        raise HTTPException(status_code=500, detail="WhatsApp service unavailable")

    form_data = await request.form()
    from_number = form_data.get("From", "")
    body = form_data.get("Body", "").strip()

    if not is_whatsapp_message(from_number):
        raise HTTPException(status_code=400, detail="Not a WhatsApp message")

    logger.info(f"💬 WhatsApp received from {from_number}: '{body}'")

    triage_request = TriageRequest(
        description=body,
        thread_id=form_data.get("SmsSid", None),
        city=None
    )

    try:
        triage_response = await triage_service.analyze(triage_request)
    except HTTPException as e:
        logger.error(f"Error during triage: {e.detail}")
        twiml_error = MessagingResponse()
        twiml_error.message("⚠️ Sorry, something went wrong. Please try again later.")
        return Response(content=str(twiml_error), media_type="application/xml")

    formatted = format_response_for_sms(triage_response)
    twiml = MessagingResponse()
    twiml.message(formatted)

    logger.info(f"✅ WhatsApp response sent to {from_number}")
    return Response(content=str(twiml), media_type="application/xml")


@app.get("/docs")
async def docs_redirect():
    return JSONResponse({"detail": "See /docs for API documentation"})


@app.get("/")
async def root():
    return {
        "message": "🩺 Medical Triage Assistant API is running. See /docs for usage.",
        "endpoints": {
            "sms_webhook": "/webhook/twilio/sms",
            "whatsapp_webhook": "/webhook/twilio/whatsapp",
            "health": "/health",
            "docs": "/docs"
        }
    }


# ======================== RUNNING VIA Uvicorn ========================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=config.DEBUG)
