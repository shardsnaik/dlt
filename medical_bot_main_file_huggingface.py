from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, asyncio, uuid, json, tempfile
from datetime import datetime
from pathlib import Path

from contextlib import asynccontextmanager
from langchain.chains import ConversationChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory

from langchain_community.llms import LlamaCpp

from langchain.schema import BaseOutputParser, HumanMessage, AIMessage
from concurrent.futures import ThreadPoolExecutor

# Hugging Face imports
from huggingface_hub import hf_hub_download
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
    
class MedicalChatBotOutputparser(BaseOutputParser):

    def parse(self, text: str)->str:
            cleaned_text = text.strip()
            return cleaned_text
    
    @property
    def _type(self)-> str:
         return 'medgemma_output_parser'


class MedicalBot:
    def __init__(self):
        self.index = None
        # Hugging Face model configuration
        self.repo_id = "sharadsnaik/medgemma-4b-it-medical-gguf"  # Replace with your actual model repo
        self.filename = "medgemma-4b-it-finnetunned-merged_new_for_cpu_q5_k_m.gguf"  # Replace with actual filename
        
        # Use /tmp for Lambda (ephemeral storage)
        self.cache_dir = "/tmp/huggingface_cache" if os.path.exists("/tmp") else "./model_cache"
        self.model_path = None
        
        self.num_threads = min(os.cpu_count() or 4, 4)  # Limit threads for Lambda

        self.chat_history = InMemoryChatMessageHistory()
        self.session_id = self.generate_session_id()
        self.history_file = f'chat_history_file_{self.session_id}_.json'
        self.session_start_time = datetime.now()
        self.executor = ThreadPoolExecutor(max_workers=2)  # Reduced for Lambda

        # Download model and initialize services
        self._download_model()
        self.initialize_services()

    def _download_model(self):
        """Download model from Hugging Face Hub"""
        try:
            logger.info(f"Downloading model from Hugging Face: {self.repo_id}")
            
            # Ensure cache directory exists
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Download the model file
            self.model_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=self.filename,
                cache_dir=self.cache_dir,
                token=os.getenv("HF_TOKEN")  # Optional: set HF_TOKEN in environment for private models
            )
            
            logger.info(f"Model downloaded successfully to: {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to download model: {str(e)}")
            # Fallback to local path if download fails
            local_fallback = f"./{self.filename}"
            if os.path.exists(local_fallback):
                self.model_path = local_fallback
                logger.info(f"Using local fallback model: {self.model_path}")
            else:
                raise RuntimeError(f"Could not download model and no local fallback found: {str(e)}")
    
    def generate_session_id(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"medical_session_{timestamp}_{unique_id}"
    
    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Get or create chat history for a session"""
        return self.chat_history
    
    def initialize_services(self):
        """Initialize LLM and conversation chain"""
        if not self.model_path or not os.path.exists(self.model_path):
            raise RuntimeError("Model path not available or model file doesn't exist")
            
        logger.info(f"Initializing LLM with model: {self.model_path}")
        
        self.llm = LlamaCpp(
            model_path=self.model_path,
            n_ctx=8192,       # Reduced context size for Lambda memory constraints
            n_threads=self.num_threads,
            n_batch=256,      # Reduced batch size
            verbose=False,
            temperature=0.7,
            top_p=0.9,
            max_tokens=256,
            stop=["<end_of_turn>"])

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful medical assistant. Use the conversation history to provide contextual responses"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "<start_of_turn>user \n{input}<end_of_turn>\n<start_of_turn>assistant\n"),
        ])

        self.chain = self.prompt_template | self.llm | MedicalChatBotOutputparser()

        self.conversation_chain = RunnableWithMessageHistory(
            self.chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history"
        )

    def chat(self, user_input: str, session_id: str = 'default') -> str:
        """Process user input and return AI response"""
        try:
            logger.info(f"Processing message for session {self.session_id}: {user_input[:50]}...")

            response = self.conversation_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": self.session_id}}
            )
            
            cleaned_response = response
            return cleaned_response
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            return f"Sorry, I encountered an error: {str(e)}"
    
    def _clean_response(self, response: str) -> str:
        """Clean up the model response"""
        cleaned_text = response.strip()
        if "<end_of_turn>" in cleaned_text:
            cleaned_text = cleaned_text.split("<end_of_turn>")[0].strip()
        return cleaned_text
    
    async def chat_async(self, user_input: str, session_id: str = 'default') -> str:
        """Async wrapper for chat function"""
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                self.executor,
                self.chat,
                user_input,
                session_id
            )
            return response
        except Exception as e:
            error_message = f'Error encountered in the chat_async function: {str(e)}'
            logger.error(error_message)
            return error_message
        
    def get_memory_chat(self, session_id: str = 'default'):
        """Function to get current conversation history"""
        try:
            messages = self.chat_history.messages
            if not messages:
                return 'No Messages are stored'
            
            formatted_history = []
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    role = "Human"
                elif isinstance(msg, AIMessage):
                    role = "Assistant"
                formatted_history.append(f"{role}: {msg.content}")

            self.chat_history_file()
            return '\n'.join(formatted_history)
        
        except Exception as e:
            logger.error(f"Error retrieving memory: {str(e)}")
            return f"Error retrieving memory: {str(e)}"
        
    def chat_history_file(self):
        """Function to save chat history as json file"""
        logger.info('Initiating the chat_history file')
        try:
            messages = self.chat_history.messages
            formatted_history = []
            
            for i, msg in enumerate(messages):
                if isinstance(msg, HumanMessage):
                    role = "Human"
                elif isinstance(msg, AIMessage):
                    role = "Assistant"
                
                data = {
                    'id': i + 1,
                    'role': role,
                    'content': msg.content
                }
                formatted_history.append(data)
            
            chat_data = {
                'session_id': self.session_id,
                'created_at': self.session_start_time.isoformat(),
                'messages': formatted_history
            }
            
            # Use /tmp for Lambda
            log_dir = '/tmp/logs' if os.path.exists('/tmp') else 'logs'
            os.makedirs(log_dir, exist_ok=True)
            file_path = os.path.join(log_dir, self.history_file)
            
            with open(file=file_path, mode='w', encoding='utf-8') as f:
                json.dump(chat_data, f, indent=2)

            logger.info(f"Chat history saved to: {file_path}")

        except Exception as e:
            logger.error(f"Error occurred while saving the chat_history file: {str(e)}")
            return f"Error occurred while saving the chat_history file: {str(e)}"
    
    def clear_memory(self):
        """Clear memory for a specific session"""
        try:
            if self.chat_history is not None:
                self.chat_history.clear()
                logger.info(f"Conversation memory cleared for session: {self.session_id}")
            else:
                logger.info(f"No memory found for session: {self.session_id}")
        except Exception as e:
            logger.error(f"Error clearing memory: {str(e)}")

    def get_all_sessions(self):
        """Function to display all sessions"""
        return {
            "session_id": self.session_id,
            "start_time": self.session_start_time.isoformat(),
            "message_count": len(self.chat_history.messages),
            "uptime_minutes": round((datetime.now() - self.session_start_time).total_seconds() / 60, 2)
        }


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("Starting Medical Bot initialization...")
        app.state.medicalbot = MedicalBot()
        logger.info("Medical bot startup completed")
        yield
    except Exception as e:
        error_message = f"Failed or error occurred in lifespan function: {str(e)}"
        logger.error(error_message)
        app.state.startup_error = error_message
        yield
    finally:
        logger.info("Shutting down Medical Bot")


app = FastAPI(lifespan=lifespan, title='Medical_bot')

# Configure CORS for Lambda
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"],  # More permissive for Lambda
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Lambda handler
from mangum import Mangum
handler = Mangum(app)


class QueryHandler(BaseModel):
    query: str


class ChatResponsetype(BaseModel):
    response: str
    status: str = 'success'


class MemoryResponsetype(BaseModel):
    memory: str
    status: str = 'success'


@app.get('/')
async def home_page():
    return {
        'message': "Medical Bot API is running",
        'status': 'healthy',
        'endpoints': {
            'chat': '/invocation',
            'memory': '/memory',
            'clear_memory': '/clear-memory'
        }
    }


@app.post('/invocation', response_model=ChatResponsetype)
async def chat_page(req: QueryHandler, request: Request):
    """Main Chat endpoint"""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail='Query cannot be empty')
    
    try:
        medicalbot = request.app.state.medicalbot
        response = await medicalbot.chat_async(req.query)

        return ChatResponsetype(
            response=response,
            status='success'
        )
    except Exception as e:
        error_msg = f'Error in /invocation endpoint: {str(e)}'
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)


@app.get('/memory')
async def get_memory(request: Request):
    """Function to get conversation history"""
    try:
        medicalbot = request.app.state.medicalbot
        memory_content = medicalbot.get_memory_chat()
        return memory_content
    except Exception as e:
        logger.error(f'Failed in getting memory: {str(e)}')
        raise HTTPException(status_code=500, detail=f'Failed in getting the memory: {str(e)}')


@app.get('/clear-memory')
async def clear_memory(request: Request):
    """Clear conversation memory"""
    if not hasattr(request.app.state, 'medicalbot') or request.app.state.medicalbot is None:
        raise HTTPException(status_code=503, detail="Medical bot not initialized")
    
    try:
        medicalbot = request.app.state.medicalbot
        medicalbot.clear_memory()

        return {
            'message': 'Conversation history cleared',
            'status': 'success'
        }
    except Exception as e:
        logger.error(f'Error in clear-memory: {str(e)}')
        raise HTTPException(status_code=500, detail=f'Error occurred in /clear-memory endpoint: {str(e)}')


@app.get('/get_session_id')
async def get_all_session_id(request: Request):
    """Function to get all corresponding session ids"""
    medicalbot = request.app.state.medicalbot
    all_sessions = medicalbot.get_all_sessions()
    return all_sessions


@app.get('/health')
async def health_checker(request: Request):
    med_bot_initialized = hasattr(request.app.state, 'medicalbot') and request.app.state.medicalbot is not None
    startup_error = getattr(request.app.state, 'startup_error', None)
    
    return {
        'status': 'healthy' if med_bot_initialized else 'unhealthy',
        'medical_bot_initialized': med_bot_initialized,
        'startup_error': startup_error,
        'api_version': '1.0.0'
    }


# For local development
if __name__ == '__main__':
    import uvicorn
    uvicorn.run('medical_bot_main_file:app', host='127.0.0.1', port=8000, reload=True)