from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Request
import os
import json
import logging
import re
import hashlib
import asyncio
from datetime import datetime
from typing import List, Dict, Optional

# Twilio imports
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import Pinecone (make it optional for Vercel)
try:
    from pinecone import Pinecone
    PINECONE_AVAILABLE = True
    logger.info("✅ Pinecone client available")
except ImportError:
    PINECONE_AVAILABLE = False
    logger.warning("⚠️ Pinecone not available - continuing without it")

# ========================================
# Configuration from Environment Variables
# ========================================


class Config:
    def __init__(self):
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        # For treatment processing & citations
        self.OPENAI_API_KEY_2 = os.getenv("OPENAI_API_KEY_2")
        self.PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        self.PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east1-gcp")
        self.ASSISTANT_ID = os.getenv(
            "ASSISTANT_ID", "asst_pAhSF6XJsj60efD9GEVdEG5n")
        self.EMBEDDING_MODEL = os.getenv(
            "EMBEDDING_MODEL", "text-embedding-ada-002")
        self.INDEX_NAME = os.getenv("INDEX_NAME", "triage-index")
        self.TREATMENT_INDEX_NAME = os.getenv(
            "TREATMENT_INDEX_NAME", "triage-index-treatment")
        self.MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
        self.MIN_SYMPTOMS_FOR_PINECONE = int(
            os.getenv("MIN_SYMPTOMS_FOR_PINECONE", "3"))
        self.MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "3"))
        self.NIGERIA_EMERGENCY_HOTLINE = os.getenv("EMERGENCY_HOTLINE", "112")
        self.PINECONE_SCORE_THRESHOLD = float(
            os.getenv("PINECONE_SCORE_THRESHOLD", "0.88"))
        self.PORT = int(os.getenv("PORT", "8000"))

        # Twilio Configuration
        self.TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
        self.TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
        self.TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
        self.TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")

    def validate(self):
        """Validate required environment variables"""
        required_vars = ["OPENAI_API_KEY"]
        missing = [var for var in required_vars if not getattr(self, var)]
        if missing:
            raise ValueError(
                f"Missing required environment variables: {missing}")


config = Config()

# Validate configuration
try:
    config.validate()
    logger.info("Configuration validated successfully")
except ValueError as e:
    logger.error(f"Configuration error: {e}")
    raise

# ========================================
# Global Constants
# ========================================

RED_FLAGS = [
    "bullet wound", "gunshot", "profuse bleeding", "crushing chest pain",
    "sudden shortness of breath", "loss of consciousness", "slurred speech", "seizure",
    "head trauma", "neck trauma", "high fever with stiff neck", "uncontrolled vomiting",
    "severe allergic reaction", "anaphylaxis", "difficulty breathing", "persistent cough with blood",
    "severe abdominal pain", "sudden vision loss", "chest tightness with sweating",
    "blood in urine", "inability to pass urine", "sharp abdominal pain", "intermenstrual bleeding"
]

# ========================================
# FastAPI App
# ========================================

app = FastAPI(
    title="Medical Triage Assistant API",
    description="AI-powered medical triage assistant for symptom assessment, condition suggestions, and CITATIONS with Twilio SMS/WhatsApp support",
    version="1.2.0-twilio"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================================
# Client Management (Singleton Pattern)
# ========================================


class ClientManager:
    def __init__(self):
        self._openai_client = None
        self._openai_client_2 = None  # For treatment processing & citations
        self._pinecone_client = None
        self._pinecone_index = None
        self._treatment_index = None
        self._twilio_client = None
        self._user_threads = {}  # In-memory storage for user->thread mapping

    async def get_openai_client(self):
        """Get OpenAI client - always create fresh for serverless"""
        try:
            logger.info("🔑 Creating fresh OpenAI client for serverless...")
            client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
            logger.info("🧪 Testing OpenAI connection...")
            await client.models.list()
            logger.info("✅ OpenAI client created and tested successfully")
            return client
        except Exception as e:
            logger.error(f"❌ Failed to create OpenAI client: {e}")
            raise HTTPException(
                status_code=503, detail=f"OpenAI connection failed: {str(e)}")

    async def get_openai_client_2(self):
        """Get secondary OpenAI client for treatment & citations processing"""
        try:
            api_key = config.OPENAI_API_KEY_2 or config.OPENAI_API_KEY
            logger.info(
                "🔑 Creating fresh OpenAI client 2 for treatment/citations processing...")
            client = AsyncOpenAI(api_key=api_key)
            await client.models.list()
            logger.info("✅ OpenAI client 2 created and tested successfully")
            return client
        except Exception as e:
            logger.error(f"❌ Failed to create OpenAI client 2: {e}")
            raise HTTPException(
                status_code=503, detail=f"Treatment/citations OpenAI connection failed: {str(e)}")

    def get_pinecone_index(self):
        """Get Pinecone index - cache for performance"""
        if not PINECONE_AVAILABLE or not config.PINECONE_API_KEY:
            return None
        if self._pinecone_index is None:
            try:
                logger.info("🔍 Initializing Pinecone client...")
                self._pinecone_client = Pinecone(
                    api_key=config.PINECONE_API_KEY)
                logger.info(
                    f"🔗 Connecting to Pinecone index: {config.INDEX_NAME}")
                self._pinecone_index = self._pinecone_client.Index(
                    name=config.INDEX_NAME)
                stats = self._pinecone_index.describe_index_stats()
                logger.info(
                    f"✅ Pinecone connected successfully - {stats.total_vector_count} vectors")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Pinecone: {e}")
                return None
        return self._pinecone_index

    def get_treatment_index(self):
        """Get Pinecone treatment index - cache for performance"""
        if not PINECONE_AVAILABLE or not config.PINECONE_API_KEY:
            return None
        if self._treatment_index is None:
            try:
                if self._pinecone_client is None:
                    logger.info(
                        "🔍 Initializing Pinecone client for treatment index...")
                    self._pinecone_client = Pinecone(
                        api_key=config.PINECONE_API_KEY)
                logger.info(
                    f"🔗 Connecting to treatment index: {config.TREATMENT_INDEX_NAME}")
                self._treatment_index = self._pinecone_client.Index(
                    name=config.TREATMENT_INDEX_NAME)
                stats = self._treatment_index.describe_index_stats()
                logger.info(
                    f"✅ Treatment index connected successfully - {stats.total_vector_count} vectors")
            except Exception as e:
                logger.error(f"❌ Failed to initialize treatment index: {e}")
                return None
        return self._treatment_index

    def get_twilio_client(self):
        """Get Twilio client - cache for performance"""
        if self._twilio_client is None:
            try:
                logger.info("📱 Initializing Twilio client...")
                self._twilio_client = Client(
                    config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)
                logger.info("✅ Twilio client initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Twilio: {e}")
                return None
        return self._twilio_client

    def get_user_thread(self, user_id: str) -> Optional[str]:
        """Get thread ID for a user"""
        return self._user_threads.get(user_id)

    def set_user_thread(self, user_id: str, thread_id: str):
        """Set thread ID for a user"""
        self._user_threads[user_id] = thread_id
        logger.info(f"🔗 Mapped user {user_id[:10]}... to thread {thread_id}")

    async def get_or_create_thread(self, user_id: str) -> str:
        """Get existing thread or create new one for user"""
        existing_thread = self.get_user_thread(user_id)

        if existing_thread:
            # Validate existing thread
            if await validate_thread(existing_thread):
                logger.info(
                    f"♻️ Using existing thread {existing_thread} for user {user_id[:10]}...")
                return existing_thread
            else:
                logger.warning(
                    f"⚠️ Invalid thread {existing_thread} for user {user_id[:10]}... - creating new one")

        # Create new thread
        try:
            client = await self.get_openai_client()
            thread = await client.beta.threads.create()
            self.set_user_thread(user_id, thread.id)
            logger.info(
                f"🆕 Created new thread {thread.id} for user {user_id[:10]}...")
            return thread.id
        except Exception as e:
            logger.error(
                f"❌ Failed to create thread for user {user_id[:10]}...: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to create conversation thread")


client_manager = ClientManager()

# ========================================
# Pydantic Models
# ========================================


class TriageRequest(BaseModel):
    description: str = Field(..., min_length=1, max_length=1000)
    thread_id: Optional[str] = Field(None)


class ConditionInfo(BaseModel):
    name: str
    description: str
    file_citation: str


class TriageInfo(BaseModel):
    type: str
    location: str


class TriageResponse(BaseModel):
    text: str
    possible_conditions: List[ConditionInfo] = []
    safety_measures: List[str] = []
    disease_names: List[str] = []
    citations: List[str] = []
    triage: TriageInfo
    send_sos: bool = False
    follow_up_questions: List[str] = []
    thread_id: str
    symptoms_count: int = 0
    should_query_pinecone: bool = False


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    services: Dict[str, str]

# ========================================
# Utility Functions
# ========================================


def extract_symptoms_fallback(text: str) -> List[str]:
    """Fallback keyword-based symptom extraction"""
    text_lower = text.lower()
    symptoms = []
    symptom_patterns = {
        "headache": ["headache", "head pain", "migraine"],
        "nausea": ["nausea", "nauseous", "feeling sick", "sick to stomach"],
        "fever": ["fever", "temperature", "hot", "feverish"],
        "cough": ["cough", "coughing"],
        "sore throat": ["sore throat", "throat pain", "throat hurts"],
        "stomach pain": ["stomach pain", "stomach ache", "abdominal pain", "belly pain"],
        "diarrhea": ["diarrhea", "loose stool", "watery stool"],
        "constipation": ["constipation", "not stooling", "can't poop", "no bowel movement"],
        "vomiting": ["vomiting", "throwing up", "vomit"],
        "dizziness": ["dizzy", "dizziness", "lightheaded"],
        "fatigue": ["tired", "fatigue", "exhausted", "weak"],
        "chest pain": ["chest pain", "chest hurts"],
        "shortness of breath": ["shortness of breath", "hard to breathe", "can't breathe"],
        "back pain": ["back pain", "backache"],
        "joint pain": ["joint pain", "joints hurt"],
        "runny nose": ["runny nose", "stuffy nose", "congestion"],
        "sour taste": ["sour taste", "bitter taste", "metallic taste"]
    }
    for symptom, patterns in symptom_patterns.items():
        if any(pattern in text_lower for pattern in patterns):
            symptoms.append(symptom)
    return symptoms


async def validate_thread(thread_id: str, client=None) -> bool:
    """Check if an OpenAI thread ID is valid - with retry logic"""
    if not thread_id or not thread_id.strip():
        return False
    if client is None:
        try:
            client = await client_manager.get_openai_client()
        except Exception as e:
            logger.error(
                f"❌ Could not get OpenAI client for thread validation: {e}")
            return False
    for attempt in range(2):
        try:
            await client.beta.threads.retrieve(thread_id=thread_id.strip())
            logger.info(
                f"✅ Thread {thread_id} validated successfully (attempt {attempt + 1})")
            return True
        except Exception as e:
            logger.error(
                f"❌ Thread validation attempt {attempt + 1} failed for {thread_id}: {e}")
            if attempt == 0:
                await asyncio.sleep(0.5)
    return False


async def get_thread_context(thread_id: str, client=None) -> Dict:
    """Retrieve and analyze thread context"""
    try:
        if client is None:
            client = await client_manager.get_openai_client()
        logger.info(f"📚 Retrieving context for thread: {thread_id}")
        messages = await client.beta.threads.messages.list(thread_id=thread_id, order='asc', limit=100)
        user_messages, all_symptoms, max_severity = [], [], 0
        logger.info(f"📊 Found {len(messages.data)} total messages in thread")
        for i, msg in enumerate(messages.data):
            if msg.role == "user":
                content = ""
                if msg.content and hasattr(msg.content[0], "text"):
                    content = msg.content[0].text.value
                if content:
                    user_messages.append(content)
                    logger.info(f"👤 User message {i+1}: '{content[:50]}...'")
                    try:
                        symptom_data = await extract_symptoms_comprehensive(content, client)
                        extracted_symptoms = symptom_data["symptoms"]
                        severity = symptom_data["severity"]
                        logger.info(
                            f"🩺 Extracted from message {i+1}: {extracted_symptoms} (severity: {severity})")
                        all_symptoms.extend(extracted_symptoms)
                        max_severity = max(max_severity, severity)
                    except Exception as e:
                        logger.error(
                            f"❌ Failed to extract symptoms from message {i+1}: {e}")
                        fallback_symptoms = extract_symptoms_fallback(content)
                        all_symptoms.extend(fallback_symptoms)

        return {
            "user_messages": user_messages,
            "symptoms": list(set(all_symptoms)),  # Remove duplicates
            "severity": max_severity,
            "conversation_length": len(user_messages)
        }
    except Exception as e:
        logger.error(f"❌ Failed to get thread context: {e}")
        return {"user_messages": [], "symptoms": [], "severity": 0, "conversation_length": 0}


async def extract_symptoms_comprehensive(text: str, client=None) -> Dict:
    """Enhanced symptom extraction using OpenAI"""
    try:
        if client is None:
            client = await client_manager.get_openai_client()

        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": """You are a medical symptom extraction expert. Extract symptoms from the user's description and assess severity.

Return JSON with:
- "symptoms": list of specific symptoms mentioned
- "severity": integer 1-10 (1=mild, 10=life-threatening)
- "red_flags": list of any emergency indicators

Focus on medical symptoms, not general complaints."""
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            temperature=0.1
        )

        result = json.loads(response.choices[0].message.content)
        logger.info(
            f"🩺 Extracted symptoms: {result.get('symptoms', [])} (severity: {result.get('severity', 0)})")
        return result

    except Exception as e:
        logger.error(f"❌ Symptom extraction failed: {e}")
        # Fallback to keyword extraction
        symptoms = extract_symptoms_fallback(text)
        return {
            "symptoms": symptoms,
            "severity": len(symptoms),  # Simple severity based on count
            "red_flags": [flag for flag in RED_FLAGS if flag.lower() in text.lower()]
        }

# ========================================
# Twilio Helper Functions
# ========================================


def generate_user_id(from_number: str) -> str:
    """Generate consistent user ID from phone number"""
    # Hash the phone number for privacy
    return hashlib.sha256(from_number.encode()).hexdigest()[:16]


def clean_phone_number(phone_number: str) -> str:
    """Clean and format phone number"""
    # Remove whatsapp: prefix if present
    if phone_number.startswith('whatsapp:'):
        phone_number = phone_number.replace('whatsapp:', '')
    return phone_number.strip()


def is_whatsapp_message(from_number: str) -> bool:
    """Check if message is from WhatsApp"""
    return from_number.startswith('whatsapp:')


def format_response_for_sms(response: TriageResponse) -> str:
    """Format the triage response for SMS/WhatsApp"""
    message_parts = []

    # Main response
    message_parts.append(f"🩺 MEDICAL ASSESSMENT:\n{response.text}")

    # Emergency check
    if response.send_sos:
        message_parts.append(
            f"\n🚨 EMERGENCY: Call {config.NIGERIA_EMERGENCY_HOTLINE} immediately!")

    # Possible conditions (limit to top 3 for SMS)
    if response.possible_conditions:
        message_parts.append("\n📋 POSSIBLE CONDITIONS:")
        for i, condition in enumerate(response.possible_conditions[:3]):
            message_parts.append(
                f"{i+1}. {condition.name}: {condition.description[:100]}...")

    # Safety measures (limit to top 3)
    if response.safety_measures:
        message_parts.append("\n⚠️ IMMEDIATE STEPS:")
        for i, measure in enumerate(response.safety_measures[:3]):
            message_parts.append(f"{i+1}. {measure}")

    # Follow-up questions (limit to 2)
    if response.follow_up_questions:
        message_parts.append("\n❓ TO HELP YOU MORE:")
        for i, question in enumerate(response.follow_up_questions[:2]):
            message_parts.append(f"• {question}")

    # Triage info
    triage_emoji = "🔴" if response.triage.type == "emergency" else "🟡" if response.triage.type == "urgent" else "🟢"
    message_parts.append(
        f"\n{triage_emoji} URGENCY: {response.triage.type.upper()}")
    message_parts.append(f"📍 RECOMMENDED: {response.triage.location}")

    # Citations (abbreviated)
    if response.citations:
        message_parts.append(
            f"\n📚 Sources: {len(response.citations)} medical references used")

    full_message = "\n".join(message_parts)

    # SMS has 1600 char limit, WhatsApp is more flexible
    if len(full_message) > 1500:
        # Truncate and add continuation message
        truncated = full_message[:1400]
        last_newline = truncated.rfind('\n')
        if last_newline > 1200:
            truncated = truncated[:last_newline]
        full_message = truncated + "\n\n📱 Reply for more details or ask follow-up questions."

    return full_message

# ========================================
# Core Processing Functions
# ========================================


async def parse_assistant_response(response_text: str, client=None) -> Dict:
    """Parse assistant response into structured data"""
    try:
        if client is None:
            client = await client_manager.get_openai_client()

        parsing_response = await client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": """Parse the medical assessment into structured JSON with:

{
  "possible_conditions": [{"name": "condition", "description": "brief description", "file_citation": "medical source"}],
  "safety_measures": ["immediate step 1", "immediate step 2"],
  "citations": ["source 1", "source 2"],
  "triage_type": "routine|urgent|emergency",
  "triage_location": "Primary Care|Urgent Care|Emergency Room",
  "follow_up_questions": ["question 1", "question 2"]
}

Extract key medical information accurately."""
                },
                {
                    "role": "user",
                    "content": f"Parse this medical assessment:\n\n{response_text}"
                }
            ],
            temperature=0.1
        )

        result = json.loads(parsing_response.choices[0].message.content)

        # Convert conditions to ConditionInfo objects
        conditions = []
        for c in result.get("possible_conditions", []):
            conditions.append(ConditionInfo(
                name=c.get("name", "Unknown Condition"),
                description=c.get("description", "No description available"),
                file_citation=c.get("file_citation", "Medical Database")
            ))

        result["possible_conditions"] = conditions
        return result

    except Exception as e:
        logger.error(f"❌ Response parsing failed: {e}")
        return {
            "possible_conditions": [],
            "safety_measures": ["Consult with a healthcare provider"],
            "citations": [],
            "triage_type": "routine",
            "triage_location": "Primary Care",
            "follow_up_questions": []
        }


async def query_pinecone_conditions(symptoms: List[str]) -> List[ConditionInfo]:
    """Query Pinecone for similar conditions based on symptoms"""
    try:
        index = client_manager.get_pinecone_index()
        if not index:
            logger.info("🔍 Pinecone not available, skipping condition search")
            return []

        # Create embedding for symptoms
        client = await client_manager.get_openai_client()
        symptoms_text = ", ".join(symptoms)

        embedding_response = await client.embeddings.create(
            model=config.EMBEDDING_MODEL,
            input=symptoms_text
        )

        query_embedding = embedding_response.data[0].embedding

        # Query Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=3,
            include_metadata=True,
            filter={"type": "condition"}
        )

        conditions = []
        for match in results.matches:
            if match.score >= config.PINECONE_SCORE_THRESHOLD:
                metadata = match.metadata
                conditions.append(ConditionInfo(
                    name=metadata.get("name", "Unknown Condition"),
                    description=metadata.get(
                        "description", "No description available"),
                    file_citation=metadata.get("source", "Medical Database")
                ))

        logger.info(
            f"🔍 Found {len(conditions)} matching conditions from Pinecone")
        return conditions

    except Exception as e:
        logger.error(f"❌ Pinecone query failed: {e}")
        return []


async def process_triage_request(request: TriageRequest) -> TriageResponse:
    """Main triage processing function with enhanced Twilio support"""
    try:
        logger.info(
            f"🩺 Processing triage request: '{request.description[:50]}...'")

        # Get or create thread
        if not request.thread_id:
            client = await client_manager.get_openai_client()
            thread = await client.beta.threads.create()
            thread_id = thread.id
            logger.info(f"🆕 Created new thread: {thread_id}")
        else:
            thread_id = request.thread_id
            # Validate existing thread
            if not await validate_thread(thread_id):
                logger.warning(
                    f"⚠️ Invalid thread {thread_id}, creating new one")
                client = await client_manager.get_openai_client()
                thread = await client.beta.threads.create()
                thread_id = thread.id

        # Extract symptoms for better context
        client = await client_manager.get_openai_client()
        symptom_data = await extract_symptoms_comprehensive(request.description, client)
        symptoms = symptom_data.get("symptoms", [])
        severity = symptom_data.get("severity", 0)
        red_flags = symptom_data.get("red_flags", [])

        # Check for immediate emergency
        send_sos = severity >= 8 or len(red_flags) > 0
        if send_sos:
            logger.warning(
                f"🚨 Emergency detected - Severity: {severity}, Red flags: {red_flags}")

        # Add message to thread
        await client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=request.description
        )

        # Create and run assistant
        run = await client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=config.ASSISTANT_ID,
            instructions="""You are a medical triage assistant. Provide professional symptom assessment with:

1. Clear, empathetic medical guidance
2. Specific possible conditions with descriptions  
3. Immediate safety measures
4. Appropriate urgency level (routine/urgent/emergency)
5. Recommended care location
6. Follow-up questions to gather more information

Be thorough but concise. Always prioritize patient safety."""
        )

        # Wait for completion
        max_wait_time = 30  # seconds
        wait_time = 0
        while wait_time < max_wait_time:
            run = await client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
            if run.status == "completed":
                break
            elif run.status in ["failed", "cancelled", "expired"]:
                logger.error(
                    f"❌ Assistant run failed with status: {run.status}")
                raise HTTPException(
                    status_code=500, detail=f"Assistant processing failed: {run.status}")

            await asyncio.sleep(1)
            wait_time += 1

        if run.status != "completed":
            logger.error(f"❌ Assistant run timed out after {max_wait_time}s")
            raise HTTPException(
                status_code=408, detail="Medical assessment timed out")

        # Get response messages
        messages = await client.beta.threads.messages.list(thread_id=thread_id, order="desc", limit=1)

        if not messages.data:
            raise HTTPException(
                status_code=500, detail="No response from medical assistant")

        assistant_message = messages.data[0].content[0].text.value
        logger.info(
            f"✅ Assistant response received: {len(assistant_message)} characters")

        # Parse structured response using AI
        structured_response = await parse_assistant_response(assistant_message, client)

        # Enhance with Pinecone if available and enough symptoms
        pinecone_conditions = []
        if len(symptoms) >= config.MIN_SYMPTOMS_FOR_PINECONE:
            pinecone_conditions = await query_pinecone_conditions(symptoms)

        # Combine conditions
        all_conditions = structured_response.get(
            "possible_conditions", []) + pinecone_conditions
        # Remove duplicates and limit to top 5
        unique_conditions = []
        seen_names = set()
        for condition in all_conditions:
            if condition.name.lower() not in seen_names:
                unique_conditions.append(condition)
                seen_names.add(condition.name.lower())
                if len(unique_conditions) >= 5:
                    break

        # Determine triage level
        triage_type = "emergency" if send_sos else structured_response.get(
            "triage_type", "routine")
        triage_location = "Emergency Room" if send_sos else structured_response.get(
            "triage_location", "Primary Care")

        return TriageResponse(
            text=assistant_message,
            possible_conditions=unique_conditions,
            safety_measures=structured_response.get("safety_measures", []),
            disease_names=[c.name for c in unique_conditions],
            citations=structured_response.get("citations", []),
            triage=TriageInfo(type=triage_type, location=triage_location),
            send_sos=send_sos,
            follow_up_questions=structured_response.get(
                "follow_up_questions", []),
            thread_id=thread_id,
            symptoms_count=len(symptoms),
            should_query_pinecone=len(
                symptoms) >= config.MIN_SYMPTOMS_FOR_PINECONE
        )

    except Exception as e:
        logger.error(f"❌ Triage processing failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Medical assessment failed: {str(e)}")

# ========================================
# API Endpoints
# ========================================


@app.post("/triage", response_model=TriageResponse)
async def triage_endpoint(request: TriageRequest):
    """Main triage endpoint - now Twilio-enhanced"""
    return await process_triage_request(request)


@app.post("/webhook/twilio/sms")
async def handle_twilio_sms(request: Request):
    """Handle incoming SMS messages from Twilio"""
    try:
        form_data = await request.form()
        from_number = form_data.get("From", "")
        to_number = form_data.get("To", "")
        message_body = form_data.get("Body", "").strip()

        logger.info(
            f"📱 SMS received from {from_number}: '{message_body[:50]}...'")

        if not message_body:
            twiml = MessagingResponse()
            twiml.message("Please send a message describing your symptoms.")
            return Response(content=str(twiml), media_type="application/xml")

        # Generate user ID and get/create thread
        user_id = generate_user_id(from_number)
        thread_id = await client_manager.get_or_create_thread(user_id)

        # Process the medical query
        triage_request = TriageRequest(
            description=message_body, thread_id=thread_id)
        triage_response = await process_triage_request(triage_request)

        # Format response for SMS
        formatted_response = format_response_for_sms(triage_response)

        # Send response via Twilio
        twiml = MessagingResponse()
        twiml.message(formatted_response)

        logger.info(f"✅ SMS response sent to {from_number}")
        return Response(content=str(twiml), media_type="application/xml")

    except Exception as e:
        logger.error(f"❌ Error handling SMS: {e}")
        twiml = MessagingResponse()
        twiml.message(
            "Sorry, I'm having technical difficulties. Please try again or call emergency services if urgent.")
        return Response(content=str(twiml), media_type="application/xml")


@app.post("/webhook/twilio/whatsapp")
async def handle_twilio_whatsapp(request: Request):
    """Handle incoming WhatsApp messages from Twilio"""
    try:
        form_data = await request.form()
        from_number = form_data.get("From", "")
        to_number = form_data.get("To", "")
        message_body = form_data.get("Body", "").strip()

        logger.info(
            f"💬 WhatsApp received from {from_number}: '{message_body[:50]}...'")

        if not message_body:
            twiml = MessagingResponse()
            twiml.message(
                "Hello! 👋 I'm your AI medical assistant. Please describe your symptoms and I'll help assess your condition.")
            return Response(content=str(twiml), media_type="application/xml")

        # Handle special commands
        if message_body.lower() in ['restart', 'reset', 'new', 'start over']:
            user_id = generate_user_id(from_number)
            # Clear existing thread to start fresh
            if user_id in client_manager._user_threads:
                del client_manager._user_threads[user_id]
            thread_id = await client_manager.get_or_create_thread(user_id)

            twiml = MessagingResponse()
            twiml.message(
                "🔄 Starting a new conversation. Please describe your current symptoms.")
            return Response(content=str(twiml), media_type="application/xml")

        # Generate user ID and get/create thread
        user_id = generate_user_id(from_number)
        thread_id = await client_manager.get_or_create_thread(user_id)

        # Process the medical query
        triage_request = TriageRequest(
            description=message_body, thread_id=thread_id)
        triage_response = await process_triage_request(triage_request)

        # Format response for WhatsApp (can be longer than SMS)
        formatted_response = format_response_for_sms(triage_response)

        # Send response via Twilio
        twiml = MessagingResponse()
        twiml.message(formatted_response)

        # If emergency, send additional urgent message
        if triage_response.send_sos:
            urgent_msg = f"🚨 URGENT: Your symptoms require immediate medical attention!\n\nCall {config.NIGERIA_EMERGENCY_HOTLINE} right now or go to the nearest emergency room."
            twiml.message(urgent_msg)

        logger.info(f"✅ WhatsApp response sent to {from_number}")
        return Response(content=str(twiml), media_type="application/xml")

    except Exception as e:
        logger.error(f"❌ Error handling WhatsApp: {e}")
        twiml = MessagingResponse()
        twiml.message(
            "Sorry, I'm experiencing technical issues. Please try again in a moment or contact emergency services if this is urgent.")
        return Response(content=str(twiml), media_type="application/xml")


@app.post("/test/twilio")
async def test_twilio_integration(request: TriageRequest):
    """Test endpoint to simulate Twilio integration"""
    try:
        # Simulate a phone number
        test_phone = "+2348012345678"
        user_id = generate_user_id(test_phone)

        # Get or create thread
        thread_id = await client_manager.get_or_create_thread(user_id)

        # Update request with thread
        request.thread_id = thread_id

        # Process request
        response = await process_triage_request(request)

        # Format for SMS/WhatsApp
        formatted_response = format_response_for_sms(response)

        return {
            "user_id": user_id,
            "thread_id": thread_id,
            "original_response": response,
            "formatted_for_sms": formatted_response,
            "character_count": len(formatted_response)
        }

    except Exception as e:
        logger.error(f"❌ Test error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Enhanced health check including Twilio connectivity"""
    services = {
        "api": "healthy",
        "openai": "unknown",
        "pinecone": "unknown",
        "twilio": "unknown"
    }

    # Test OpenAI
    try:
        client = await client_manager.get_openai_client()
        await client.models.list()
        services["openai"] = "healthy"
    except:
        services["openai"] = "unhealthy"

    # Test Pinecone
    try:
        index = client_manager.get_pinecone_index()
        if index:
            services["pinecone"] = "healthy"
        else:
            services["pinecone"] = "disabled"
    except:
        services["pinecone"] = "unhealthy"

    # Test Twilio
    try:
        twilio_client = client_manager.get_twilio_client()
        if twilio_client and config.TWILIO_ACCOUNT_SID:
            # Test with a simple API call
            twilio_client.api.accounts(config.TWILIO_ACCOUNT_SID).fetch()
            services["twilio"] = "healthy"
        else:
            services["twilio"] = "disabled"
    except:
        services["twilio"] = "unhealthy"

    return HealthResponse(
        status="healthy" if all(s in ["healthy", "disabled"]
                                for s in services.values()) else "degraded",
        timestamp=datetime.now().isoformat(),
        version="1.2.0-twilio",
        services=services
    )


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "🩺 Medical Triage Assistant API with Twilio SMS/WhatsApp Support",
        "version": "1.2.0-twilio",
        "endpoints": {
            "triage": "/triage - Main medical assessment endpoint",
            "sms_webhook": "/webhook/twilio/sms - SMS webhook for Twilio",
            "whatsapp_webhook": "/webhook/twilio/whatsapp - WhatsApp webhook for Twilio",
            "health": "/health - Service health check",
            "docs": "/docs - Interactive API documentation",
            "test": "/test/twilio - Test Twilio integration"
        },
        "features": [
            "AI-powered medical triage",
            "SMS and WhatsApp support",
            "Conversation memory",
            "Emergency detection",
            "Multi-platform accessibility"
        ]
    }
