from pydantic import BaseModel, Field
from typing import Optional, Literal, List


class UserQuery(BaseModel):
    """User input for the assistant."""
    text: str


class QueryClassification(BaseModel):
    """Model to classify if a query is Selcom-related or not."""
    is_selcom_related: bool = Field(
        description="True if the query is related to Selcom Pesa, Selcom services, banking, payments, or financial services"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence level in the classification (0.0 to 1.0)"
    )
    reason: str = Field(
        description="Brief explanation of why the query is classified as Selcom-related or not"
    )


class AssistantResponse(BaseModel):
    """The assistant's validated response to the user."""
    response: str = Field(
        description="A natural, conversational response that directly answers the user's question or addresses their concern. Should be friendly, helpful, and informative based on the Selcom knowledge base. For greetings, introduce yourself as the Selcom bot and ask how you can help."
    )
    language: Literal["en", "sw"] = Field(
        default="en",
        description="The language of the response (English or Swahili)"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence level in the response accuracy (0.0 to 1.0)"
    )
    escalation_required: bool = Field(
        default=False,
        description="True if the request needs escalation to human support"
    )
    related_topics: Optional[List[str]] = Field(
        default=None,
        description="Related topics or services that might be helpful to the user"
    )


class KnowledgeBasedResponse(BaseModel):
    """A more comprehensive response model for knowledge-based interactions."""
    greeting: Optional[str] = Field(
        default=None,
        description="Friendly greeting if this is the first interaction or user said hello"
    )
    main_response: str = Field(
        description="The main answer to the user's question, based on Selcom knowledge base"
    )
    additional_info: Optional[str] = Field(
        default=None,
        description="Additional helpful information or tips related to the query"
    )
    contact_info: Optional[str] = Field(
        default=None,
        description="Relevant contact information if needed (e.g., customer support numbers)"
    )
    next_steps: Optional[str] = Field(
        default=None,
        description="Suggested next steps for the user if applicable"
    )
    language: Literal["en", "sw"] = Field(
        default="en",
        description="The language of the response"
    )


class EscalationLog(BaseModel):
    """Information required for human support handoff."""
    user_id: str
    issue_summary: str
    escalation_reason: str
    timestamp: Optional[str]


class TranscriptionInput(BaseModel):
    """Voice note transcription payload."""
    base64_audio: str
    format: Literal["ogg", "mp3", "wav"]
    language: Literal["en", "sw"] = "sw"


class TransactionSummary(BaseModel):
    """Model for transaction history summaries."""
    account_id: str
    period: Literal["daily", "weekly", "monthly"]
    total_sent: float
    total_received: float
    savings: Optional[float]
    transaction_count: int


class RegistrationStep(BaseModel):
    """A model to guide the user through registration."""
    current_step: Literal["language", "phone", "OTP", "NIDA", "biometric"]
    prompt: str
    next_action: Optional[str]
    example_data: Optional[str]
    
    
