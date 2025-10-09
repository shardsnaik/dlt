from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, asyncio, uuid, json
from datetime import datetime

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
        # self.model_path = 'medgemma-4b-it-finnetunned-merged_new_for_cpu_q5_k_m.gguf'
        self.model_path = os.getenv("MODEL_PATH")

# Change this to Linux-friendly relative path:
        # if os.path.exists("/opt/ml/model/medgemma-4b-it-finnetunned-merged_new_for_cpu_q5_k_m.gguf"):
        #     self.model_path = "/opt/ml/model/medgemma-4b-it-finnetunned-merged_new_for_cpu_q5_k_m.gguf"   ##For running in sagemaker 
        # else: 
        #     self.model_path = "./medgemma-4b-it-finnetunned-merged_new_for_cpu_q5_k_m.gguf"   ## For running docker in loacal
        # self.model_path = "medgemma-4b-it-finnetunned-merged_new_for_cpu_q5_k_m.gguf"   ## For running docker in loacal
        self.num_threads = os.cpu_count()

        # self.store :Dict[str: InMemoryChatMessageHistory] = {}
        self.chat_history = InMemoryChatMessageHistory()

        self.session_id = self.generate_session_id()
        self.history_file = f'chat_history_file_{self.session_id}_.json'
        self.session_start_time = datetime.now()
        self.executor = ThreadPoolExecutor(max_workers=4)

        self.initialize_services()

    
    def generate_session_id(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"medical_session_{timestamp}_{unique_id}"
    
    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Get or create chat history for a session"""
        # if session_id not in self.store:
        #     self.store[session_id] = InMemoryChatMessageHistory()
        #     print(f"🆕 Created new session: {session_id}")
        # return self.store[session_id]
        return self.chat_history
    
    def initialize_services(self):
        self.llm = LlamaCpp(
            model_path=self.model_path,
            n_ctx=5000,       # Context size
            n_threads=self.num_threads,
            n_batch=512,      # Increases throughput
            verbose=False,
            temperature=0.7,
            top_p=0.9,
            max_tokens=256,
            stop=["<end_of_turn>"])
        
        # self.memory = ConversationBufferWindowMemory(
        #      k = 5,
        #      return_messages=True,
        #      memory_key='chat_history'
        # )

#         self.prompt_template = PromptTemplate(
#             input_variables=["chat_history", "input"],
#             template="""You are a helpful medical assistant. Use the conversation history to provide contextual responses.

# Chat History:
# {chat_history}

# <start_of_turn>user
# {input}<end_of_turn>
# <start_of_turn>assistant
# """
#         )

        self.prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful medical assistant. Use the conversation history to provide contextual responses"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "<start_of_turn>user \n{input}<end_of_turn>\n<start_of_turn>assistant\n"),
])

        self.chain = self.prompt_template | self.llm |MedicalChatBotOutputparser()


        # self.conversation_chain = ConversationChain(
        #     llm=self.llm,
        #     memory=self.memory,
        #     prompt=self.prompt_template,
        #     verbose=True,
        #     output_parser=MedicalChatBotOutputparser()
        # )
        self.conversation_chain = RunnableWithMessageHistory(
            self.chain,
            self.get_session_history,
            input_messages_key= "input",
            history_messages_key="chat_history"
        )

    def chat(self, user_input:str, session_id: str='default')-> str:
        """Process user input and return AI response"""
        try:
            print(f"📝 Processing message for session {self.session_id}: {user_input[:50]}...")

            response = self.conversation_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": self.session_id}}
            )
            # Clean up response
            cleaned_response = response

            return cleaned_response
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"
    
    def _clean_response(self, response: str) -> str:
        """Clean up the model response"""
        cleaned_text = response.strip()
        if "<end_of_turn>" in cleaned_text:
            cleaned_text = cleaned_text.split("<end_of_turn>")[0].strip()
        return cleaned_text
    
    async def chat_async(self, user_input: str, session_id:str ='default')-> str:
        '''
        Async wraper for chat function
        '''
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
            error_messages = f'Error encountered in the chat_async function: {str(e)}'
            return error_messages
        
    def get_memory_chat(self, session_id:str ='default'):
        '''
        Function to get current conversation history
        '''
        try:
            # if session_id in self.store:
            messages = self.chat_history.messages
            if not messages:
                return str('No Messages are stored')
            formated_history = []
            print(messages)
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    role = "Human"
                elif isinstance(msg, AIMessage):
                    role = "Assistant"
                formated_history.append(f"{role} : {msg.content}")

                print(formated_history)
            self.chat_history_file()

            return '\n'.join(formated_history)
            # return str('No Messages are stored')
        
        #  {'type': 'model_attributes_type', 'loc': ('response',), 'msg': 'Input should be a valid dictionary or object to extract fields from', 'input': [HumanMessage(content='h', additional_kwargs={}, response_metadata={}), AIMessage(content="Hi,I am sorry to hear about your mother's illness. I hope she is feeling better. Are you looking for some medical information or a diagnosis? If so, please describe her symptoms and any tests that have been performed. You can also tell me what medications she is taking and if there is any other relevant information.", additional_kwargs={}, response_metadata={})]}
        except Exception as e:
            return f"Error retrieving memory: {str(e)}"
        
    def chat_history_file(self):
        '''
        Function to save chat history as json file 
        '''
        print('Initiating the chat_history file')
        try:
            messages = self.chat_history.messages
            formated_history = []
            for i, msg in enumerate(messages):
                if isinstance(msg, HumanMessage):
                    role = "Human"
                elif isinstance(msg, AIMessage):
                    role = "Assistant"
                # formated_history.append(f"{role} : {msg.content}")
                data = {
                    'id': i + 1,
                    'role': role,
                    'content': msg.content

                }
                formated_history.append(data)
            
            chat_data ={
                'session_id': self.session_id,
                'created_at': self.session_start_time.isoformat(),
                'messages': formated_history
            }
            os.makedirs('logs', exist_ok=True)
            file = os.path.join('logs', self.history_file)
            with open(file=file, mode='w', encoding=' utf-8') as f:
                json.dump(chat_data, f, indent=2 )

            print(f"💾 Chat history saved to: {file}")

        except Exception as e:
            return f"Error occured while saving the chat_histoy file {str(e)}"
    
    def clear_memory(self):
        """Clear memory for a specific session"""
        try:
            # if session_id in self.store:
            if self.chat_history is not None:
                self.chat_history.clear()
                # self.store[session_id].clear()
                print(f"✅ Conversation memory cleared for session: {self.session_id}")
            else:
                print(f"ℹ️  No memory found for session: {self.session_id}")
        except Exception as e:
            print(f"❌ Error clearing memory: {str(e)}")

    def get_all_sessions(self):
        ''' 
        Function to display all sessions
        '''
        # return self.store
        return {
            "session_id": self.session_id,
            "start_time": self.session_start_time.isoformat(),
            "message_count": len(self.chat_history.messages),
            "uptime_minutes": round((datetime.now() - self.session_start_time).total_seconds() / 60, 2)
        }

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.medicalbot = MedicalBot()
        print("medical bot startup completed")
        yield
    except Exception as e:
        error_message = f"Failed or error occurred in lifespan function: {str(e)}"
        print(f"❌ {error_message}")
        yield

    finally:
        print("Shutting down Medical Bot")

app = FastAPI(lifespan= lifespan,title='Medical_bot')
app.add_middleware(
    CORSMiddleware, 
    allow_origins =[
        'http://localhost:3000'
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Include OPTIONS for preflight
    allow_headers=["Content-Type", "Authorization"],  # List expected headers
)

# wrap with Mangum so Lambda can serve FastAPI
from mangum import Mangum
handler = Mangum(app)

class QueryHandler(BaseModel):
    query: str

class ChatResponsetype(BaseModel):
    response: str
    status: str = 'success'

class MemoryResponsetype(BaseModel):
    memory : str
    status: str = 'success'

@app.get('/')
async def home_page():
    return {
         'message': "Medical Bot API is running",
        'status': 'healthy',
        'endpoints': {
            'chat': '/chat',
            'memory': '/memory',
            'clear_memory': '/clear-memory'
        }
    }

@app.post('/invocation', response_model=ChatResponsetype)
async def chat_page(req: QueryHandler, request: Request):
    ''' Main Chat endpoint'''
    if not req.query.strip():
        raise HTTPException(status_code=400, detail= 'Query connot be empty')
    # Pass and Exception
    try:
        medicalbot = request.app.state.medicalbot
        response =  await medicalbot.chat_async(req.query)
        # res =  med.chat(req.query)

        return ChatResponsetype(
            response= response,
            status='success'
        )
    except Exception as e:
        error_msg = f'Error in /chat endpoint: {str(e)}'
        raise HTTPException(status_code=500, detail=error_msg)
    
@app.get('/memory')
async def get_memory(request: Request):
    '''
    Function to get conversation history
    '''
    try:
        medicalbot = request.app.state.medicalbot
        memory_content = medicalbot.get_memory_chat()

        # return MemoryResponsetype(
        #     memory=memory_content, 
        #     status='success'
        # )
        return memory_content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed in Getting the memory in /memory endpoint {str(e)}')
    
    
@app.get('/clear-memory')
async def clear_memory(request: Request):
    """Clear conversation memory"""

    if not hasattr(request.app.state, 'medicalbot') or request.app.state.medicalbot is None:
        raise HTTPException(status_code=503, detail="Medical bot not initialized in /clear-memory endpoint")
    try:
        medicalbot = request.app.state.medicalbot
        medicalbot.clear_memory()

        return {
            'message': 'Conversation history cleared',
            'status': 'success'
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error occured in /clear-memory Endpoint: {str(e)}')

@app.get('/get_session_id')
async def get_all_session_id(request: Request):
    ''' Function to get all corresponding session id's'''
    medicalbot = request.app.state.medicalbot
    all_sessions = medicalbot.get_all_sessions()
    return all_sessions


@app.get('/health')
async def health_checker(request: Request):
    med_bot_initialzed = hasattr(request.app.state, 'medicalbot') and request.app.state.medicalbot is not None
    startup_error = getattr(request.app.state, 'startup_error', None)
    
    return {
        'status': 'healthy' if med_bot_initialzed else 'unhealthy',
        'medical_bot_initialized': med_bot_initialzed,
        'startup_error': startup_error,
        'api_version': '1.0.0'
    }


import uvicorn
import nest_asyncio
nest_asyncio.apply()

if __name__ =='__main__':
    uvicorn.run('medical_bot_main_file:app', host='0.0.0.0', port=8000, reload= True)
# it is a classic issue when you're trying to run uvicorn.run() inside an environment that already has an active event loop—like Jupyter Notebook, IPython, or certain IDEs.
