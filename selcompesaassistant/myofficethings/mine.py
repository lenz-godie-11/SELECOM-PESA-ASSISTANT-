import re
import os
import time
import asyncio
from aiofiles import open as aio_open
from dotenv import load_dotenv
from collections import defaultdict, deque
from typing import List, Optional
from pydantic_ai import Agent, ModelRetry, RunContext, Tool
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.deepseek import DeepSeekProvider
from services.models import QueryClassification, KnowledgeBasedResponse
from httpx import Client, Limits
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

# Initialize environment and model
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Precompile regex patterns for performance
SWAHILI_PATTERN = re.compile(
    r"\b(?:habari|safi|karibu|asante|samahani|sawa|ndiyo|hapana|pole|shule|dawa|chakula|mji|bei|rafiki|siku)\b", 
    re.IGNORECASE
)
GREETING_PATTERN = re.compile(
    r"\b(?:hello|hi|hey|hey there|hi there|hello there|good morning|good afternoon|good evening|good day|morning|afternoon|evening|habari|karibu|jambo|mambo|poa|nzuri|asante|thanks|thank you|greetings|salutations)\b",
    re.IGNORECASE
)
FORMATTING_PATTERNS = [
    (re.compile(r'(?<!\\)[\*#_\[\]()]+'), ''),  # Remove markdown
    (re.compile(r'(https?://[^\s]+)\s'), r'\1'),  # Fix trailing spaces in links
    (re.compile(r'([a-zA-Z])(https?://)'), r'\1 \2'),  # Add space before links
    (re.compile(r'(\d+\.\s)'), r'\n\1')  # Add newlines before steps
]

# Predefine system prompts
SYSTEM_PROMPTS = {
    "sw": (
        "Wewe ni msaidizi wa wateja wa Selcom Pesa. Toa majibu mafupi, rahisi na yanayoeleweka kwa Kiswahili. "
        "Fupisha maelezo na uepuke muundo tata. Toa hatua muhimu tu bila maelezo ya ziada. "
        "Weka viungo muhimu kama vinahitajika. Hakikisha viungo havina nafasi zisizohitajika. "
        "Kama habari haipo, sema 'Samahani, sina taarifa kuhusu hilo.'"
    ),
    "en": (
        "You are a Selcom Pesa customer support assistant. Provide short, simple responses in plain English. "
        "Summarize information and avoid complex formatting. Give only essential steps without extra details. "
        "Include important links when needed. Ensure links have no unnecessary spaces. "
        "If information is unavailable, say 'Sorry, I don't have information about that.'"
    )
}

# Initialize OpenAI client with connection pooling
client = OpenAI(
    api_key=DEEPSEEK_API_KEY or "sk-e5a9973a512644148e4f862537063f66",
    base_url="https://api.deepseek.com",
    http_client=Client(
        http2=True,
        limits=Limits(max_connections=10, max_keepalive_connections=5)
    )
)
user_contexts = {}
conversation_memory = defaultdict(lambda: deque(maxlen=4))

# Initialize Pydantic AI model
model = OpenAIModel(
    'deepseek-chat',
    provider=DeepSeekProvider(api_key=DEEPSEEK_API_KEY or "sk-e5a9973a512644148e4f862537063f66")
)

# Initialize embedding model and Chroma client
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path="./chroma_db")
# Cache for knowledge base
_KNOWLEDGE_CACHE = None
_ENGLISH_COLLECTION = None
_KISWAHILI_COLLECTION = None

# ------------------------------------------------------------------------------
# 1. Knowledge Base Loading with Vector Database
# ------------------------------------------------------------------------------
async def load_knowledge():
    global _KNOWLEDGE_CACHE, _ENGLISH_COLLECTION, _KISWAHILI_COLLECTION
    if _KNOWLEDGE_CACHE is not None:
        return _KNOWLEDGE_CACHE

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    knowledge_dir = os.path.join(base_dir, 'myofficethings')
    
    try:
        # Load English knowledge
        async with aio_open(os.path.join(knowledge_dir, "Knowledge.txt"), "r", encoding="utf-8") as file:
            english_text = await file.read()
            english_chunks = [chunk.strip() for chunk in english_text.split('\n\n') if chunk.strip()]
        
        # Load Kiswahili knowledge
        async with aio_open(os.path.join(knowledge_dir, "kiswahiliKnowledge.txt"), "r", encoding="utf-8") as file:
            kiswahili_text = await file.read()
            kiswahili_chunks = [chunk.strip() for chunk in kiswahili_text.split('\n\n') if chunk.strip()]
        
        # Store in Chroma
        _ENGLISH_COLLECTION = chroma_client.get_or_create_collection(name="english_knowledge")
        _KISWAHILI_COLLECTION = chroma_client.get_or_create_collection(name="kiswahili_knowledge")
        
        # Generate embeddings and store in Chroma
        if english_chunks:
            english_embeddings = embedding_model.encode(english_chunks, show_progress_bar=False)
            _ENGLISH_COLLECTION.upsert(
                ids=[f"en_{i}" for i in range(len(english_chunks))],
                embeddings=english_embeddings.tolist(),
                documents=english_chunks
            )
        
        if kiswahili_chunks:
            kiswahili_embeddings = embedding_model.encode(kiswahili_chunks, show_progress_bar=False)
            _KISWAHILI_COLLECTION.upsert(
                ids=[f"sw_{i}" for i in range(len(kiswahili_chunks))],
                embeddings=kiswahili_embeddings.tolist(),
                documents=kiswahili_chunks
            )
        
        _KNOWLEDGE_CACHE = (english_chunks, kiswahili_chunks)
        return _KNOWLEDGE_CACHE
    except Exception as e:
        print(f"Error loading knowledge: {e}")
        return [], []

# Initialize knowledge base lazily
async def ensure_knowledge_loaded():
    global ENGLISH_KNOWLEDGE, KISWAHILI_KNOWLEDGE, _ENGLISH_COLLECTION, _KISWAHILI_COLLECTION
    if _KNOWLEDGE_CACHE is None:
        ENGLISH_KNOWLEDGE, KISWAHILI_KNOWLEDGE = await load_knowledge()
        if _ENGLISH_COLLECTION is None:
            _ENGLISH_COLLECTION = chroma_client.get_or_create_collection(name="english_knowledge")
        if _KISWAHILI_COLLECTION is None:
            _KISWAHILI_COLLECTION = chroma_client.get_or_create_collection(name="kiswahili_knowledge")
    else:
        ENGLISH_KNOWLEDGE, KISWAHILI_KNOWLEDGE = _KNOWLEDGE_CACHE

# ------------------------------------------------------------------------------
# 2. Retrieval-Augmented Generation (RAG) with Vector Database
# ------------------------------------------------------------------------------
async def retrieve_relevant_chunks(ctx: RunContext[dict] = None, query: str = None, language: str = None, top_k: int = 3) -> List[str]:
    """Retrieve the top_k most relevant knowledge base chunks using vector similarity."""
    await ensure_knowledge_loaded()
    
    # Handle language: use ctx.deps if provided, else fall back to language parameter
    if ctx is not None:
        language = ctx.deps.get('language', 'en')
    elif language is None:
        raise ValueError("Language must be provided when ctx is not used")
    
    # Handle query: use ctx.deps if provided, else use query parameter
    if ctx is not None and query is None:
        query = ctx.deps.get('user_input', '')
    if not query:
        return []
    
    # Select the appropriate collection
    collection = _KISWAHILI_COLLECTION if language == "sw" else _ENGLISH_COLLECTION
    
    # Generate query embedding
    query_embedding = embedding_model.encode([query], show_progress_bar=False).tolist()[0]
    
    # Query the vector database
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    # Extract documents from results
    documents = results.get('documents', [[]])[0]
    return documents if documents else []

# ------------------------------------------------------------------------------
# 3. Language Detection and Conversation Management
# ------------------------------------------------------------------------------
def contains_kiswahili(text):
    return bool(SWAHILI_PATTERN.search(text)) if text else False

def is_greeting(text):
    return bool(GREETING_PATTERN.search(text)) if text else False

def add_to_conversation_memory(user_id, message, is_user=True):
    """Add a message to the user's conversation memory."""
    role = "user" if is_user else "assistant"
    conversation_memory[user_id].append({"role": role, "content": message})

def get_conversation_context(user_id):
    """Get the last 4 messages as conversation context."""
    if user_id not in conversation_memory or not conversation_memory[user_id]:
        return ""
    
    return "\n".join(f"{msg['role'].capitalize()}: {msg['content']}" for msg in conversation_memory[user_id])

def clear_conversation_memory(user_id):
    """Clear the conversation memory for a user."""
    if user_id in conversation_memory:
        conversation_memory[user_id].clear()

# ------------------------------------------------------------------------------
# 4. Agent with Query Classification
# ------------------------------------------------------------------------------
classification_agent = Agent(
    model=model,
    result_type=QueryClassification,
    retries=3,
    system_prompt=(
        "You are a query classifier for a Selcom Pesa customer support bot. "
        "Determine if user queries are related to Selcom Pesa, banking, payments, or financial services. "
        "Always allow greetings and follow-up questions about previously discussed Selcom topics."
    )
)

@classification_agent.tool_plain()
def get_conversation_context_tool(user_id: str) -> str:
    """Retrieve conversation context for classification."""
    return get_conversation_context(user_id)

async def classify_query(user_id, user_input, is_greeting=False):
    """Classify if a query is Selcom-related or not."""
    try:
        await ensure_knowledge_loaded()
        if is_greeting:
            return QueryClassification(
                is_selcom_related=True,
                confidence=1.0,
                reason="Greeting detected - always allowed for customer service"
            )
        
        classification_prompt = (
            f"Determine if the user query is related to Selcom Pesa, Selcom services, banking, payments, or financial services.\n"
            f"Context:\n{get_conversation_context(user_id)}\n\n"
            f"SELCOM-RELATED TOPICS INCLUDE:\n"
            f"- Selcom Pesa app and services\n- Banking and financial services\n- Money transfers and payments\n"
            f"- Mobile money and digital payments\n- Selcom cards and products\n- Selcom Huduma and agency services\n"
            f"- Selcom Pay and merchant services\n- Registration and account management\n- Transaction fees and charges\n"
            f"- Customer support for Selcom services\n- Financial technology and digital banking\n"
            f"- Greetings and introductions\n- Follow-up questions about Selcom topics\n"
            f"- Questions about Selcom's social media, YouTube, or online presence\n\n"
            f"NON-SELCOM TOPICS INCLUDE:\n"
            f"- General questions about other companies\n- Entertainment, jokes, or casual conversation\n"
            f"- Questions about celebrities, politics, or non-financial topics\n- Technical questions unrelated to Selcom\n"
            f"- Personal advice not related to financial services\n\n"
            f"User query: \"{user_input}\"\n\nClassify this query."
        )

        response = await classification_agent.run(classification_prompt)
        return response.output
        
    except Exception as e:
        print(f"Error in query classification: {e}")
        return QueryClassification(
            is_selcom_related=True,
            confidence=0.5,
            reason="Classification failed, defaulting to Selcom-related"
        )

# ------------------------------------------------------------------------------
# 5. Agent with Structured Response (RAG-Integrated)
# ------------------------------------------------------------------------------
response_agent = Agent(
    model=model,
    result_type=KnowledgeBasedResponse,
    retries=3,
    system_prompt=(
        "You are a friendly and helpful Selcom Pesa customer support assistant. "
        "Provide short, simple responses in the user's language (English or Kiswahili). "
        "Use the retrieved knowledge base content and conversation context to answer questions conversationally. "
        "If the query is a greeting, introduce yourself warmly. "
        "If information is unavailable, respond politely and suggest contacting support."
    ),
    tools=[
        Tool(lambda ctx: ENGLISH_KNOWLEDGE if ctx.deps.get('language', 'en') == "en" else KISWAHILI_KNOWLEDGE, takes_ctx=True, name="get_full_knowledge_base"),
        Tool(retrieve_relevant_chunks, name="retrieve_knowledge_tool", takes_ctx=True)
    ]
)

@response_agent.system_prompt
async def add_context_prompt(ctx: RunContext[dict]) -> str:
    """Add conversation context and user details to the system prompt."""
    user_id = ctx.deps.get('user_id', '')
    language = ctx.deps.get('language', 'en')
    is_greeting = ctx.deps.get('is_greeting', False)
    context = get_conversation_context(user_id)
    return f"User language: {language}\nIs greeting: {is_greeting}\n{context}"

@response_agent.tool_plain()
def validate_response(response: str) -> str:
    """Validate and format the response."""
    try:
        for pattern, replacement in FORMATTING_PATTERNS[:3]:
            response = pattern.sub(replacement, response)
        
        if len(response) > 1000:
            for break_char in ('. ', '\n', '; ', ', '):
                pos = response.rfind(break_char, 0, 800)
                if pos > 400:
                    response = response[:pos] + "..."
                    break
            else:
                response = response[:800] + "..."

        if FORMATTING_PATTERNS[3][0].search(response):
            response = FORMATTING_PATTERNS[3][0].sub(FORMATTING_PATTERNS[3][1], response)
        
        return response
    except Exception as e:
        raise ModelRetry(f"Response formatting failed: {e}. Please retry with simplified formatting.")

async def get_structured_ai_response(user_id, user_input):
    """Get a structured, knowledge-based response using RAG."""
    start_time = time.time()
    try:
        await ensure_knowledge_loaded()
        add_to_conversation_memory(user_id, user_input, is_user=True)
        
        use_kiswahili = contains_kiswahili(user_input)
        is_greeting_msg = is_greeting(user_input)
        language = 'sw' if use_kiswahili else 'en'
        deps = {
            'user_id': user_id,
            'language': language,
            'is_greeting': is_greeting_msg,
            'user_input': user_input
        }
        
        classification = await classify_query(user_id, user_input, is_greeting=is_greeting_msg)
        
        if not classification.is_selcom_related:
            response = KnowledgeBasedResponse(
                greeting=None,
                main_response=(
                    "Samahani, mimi ni msaidizi wa Selcom Pesa tu. Ninaweza kukusaidia tu na masuala yanayohusiana na Selcom Pesa, huduma za benki, au malipo ya kidijitali. Kama una swali kuhusu Selcom Pesa, ninafuraha kukusaidia!" 
                    if use_kiswahili else 
                    "Hey! I'm actually just here to help with Selcom Pesa stuff - you know, banking, payments, and financial services. If you've got any questions about Selcom Pesa, I'd be happy to help!"
                ),
                language=language,
                additional_info=None,
                contact_info=None,
                next_steps=None
            )
            add_to_conversation_memory(user_id, response.main_response, is_user=False)
            return response
        
        # Retrieve relevant knowledge
        relevant_knowledge = await retrieve_relevant_chunks(None, user_input, language)
        enhanced_prompt = f"User query: {user_input}\n\nRelevant knowledge:\n{' '.join(relevant_knowledge) if relevant_knowledge else 'No relevant information found.'}"
        
        response = await response_agent.run(user_prompt=enhanced_prompt, deps=deps)
        add_to_conversation_memory(user_id, response.output.main_response, is_user=False)
        return response.output
        
    except Exception as e:
        print(f"Error in structured AI response: {e}")
        response = KnowledgeBasedResponse(
            greeting="Hi there! I'm your Selcom Pesa assistant. How can I help you today?" if is_greeting_msg else None,
            main_response="Sorry, I'm having some trouble right now. Please try again or reach out to our support team.",
            language="en",
            additional_info=None,
            contact_info=None,
            next_steps=None
        )
        add_to_conversation_memory(user_id, response.main_response, is_user=False)
        return response
    finally:
        end_time = time.time()
        print(f"AI response time: {end_time - start_time:.2f} seconds")

# ------------------------------------------------------------------------------
# 6. Agent with Direct AI Response (RAG-Integrated)
# ------------------------------------------------------------------------------
async def get_ai_response(user_id, user_input):
    start_time = time.time()
    try:
        await ensure_knowledge_loaded()
        use_kiswahili = contains_kiswahili(user_input)
        lang_key = "sw" if use_kiswahili else "en"
        
        # Retrieve relevant knowledge
        relevant_knowledge = await retrieve_relevant_chunks(None, user_input, lang_key)
        
        if user_id not in user_contexts:
            user_contexts[user_id] = [
                {"role": "system", "content": SYSTEM_PROMPTS[lang_key]},
                {"role": "user", "content": f"Relevant knowledge:\n{' '.join(relevant_knowledge) if relevant_knowledge else 'No relevant information found.'}"}
            ]

        user_contexts[user_id].append({"role": "user", "content": user_input})

        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=user_contexts[user_id],
            max_tokens=300
        )

        ai_reply = response.choices[0].message.content.strip()
        
        for pattern, replacement in FORMATTING_PATTERNS[:3]:
            ai_reply = pattern.sub(replacement, ai_reply)
        
        if len(ai_reply) > 1000:
            for break_char in ('. ', '\n', '; ', ', '):
                pos = ai_reply.rfind(break_char, 0, 800)
                if pos > 400:
                    ai_reply = ai_reply[:pos] + "..."
                    break
            else:
                ai_reply = ai_reply[:800] + "..."

        if FORMATTING_PATTERNS[3][0].search(ai_reply):
            ai_reply = FORMATTING_PATTERNS[3][0].sub(FORMATTING_PATTERNS[3][1], ai_reply)
        
        user_contexts[user_id].append({"role": "assistant", "content": ai_reply})
        add_to_conversation_memory(user_id, ai_reply, is_user=False)
        
        return ai_reply
    except Exception as e:
        print(f"AI Error: {e}")
        return SYSTEM_PROMPTS["sw"].split('.')[-1] if use_kiswahili else SYSTEM_PROMPTS["en"].split('.')[-1]
    finally:
        end_time = time.time()
        print(f"AI response time: {end_time - start_time:.2f} seconds")

# ------------------------------------------------------------------------------
# 7. CLI Interface for Testing
# ------------------------------------------------------------------------------
async def run_cli():
    """Run the Selcom Pesa Assistant in a CLI for testing."""
    print("Welcome to the Selcom Pesa Customer Support Assistant!")
    print("Type 'exit' or 'quit' to stop, or 'clear' to reset conversation history.")
    print("Please enter a user ID for this session (or press Enter for default 'test_user'):")
    
    user_id = input("> ").strip() or "test_user"
    print(f"Starting session for user: {user_id}")
    print("Enter your query (English or Kiswahili):")

    while True:
        try:
            user_input = input("> ").strip()
            
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            
            if user_input.lower() == "clear":
                clear_conversation_memory(user_id)
                print("Conversation history cleared. Enter your next query:")
                continue
            
            if not user_input:
                print("Please enter a valid query.")
                continue
            
            # Get structured response
            response = await get_structured_ai_response(user_id, user_input)
            
            # Display response
            print("\nAssistant Response:")
            if response.greeting:
                print(f"Greeting: {response.greeting}")
            print(f"Response: {response.main_response}")
            if response.additional_info:
                print(f"Additional Info: {response.additional_info}")
            if response.contact_info:
                print(f"Contact Info: {response.contact_info}")
            if response.next_steps:
                print(f"Next Steps: {response.next_steps}")
            print(f"Language: {response.language}")
            print("\nEnter your next query:")
            
        except KeyboardInterrupt:
            print("\nInterrupted! Exiting...")
            break
        except Exception as e:
            print(f"Error processing query: {e}")
            print("Please try again.")

def main():
    """Main function to run the CLI."""
    asyncio.run(run_cli())

if __name__ == "__main__":
    main()