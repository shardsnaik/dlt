from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, asyncio, uuid, json
from datetime import datetime

from contextlib import asynccontextmanager
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory

# Replace LlamaCpp import with OpenAIu
from openai import OpenAI
from motor.motor_asyncio import AsyncIOMotorClient

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

def create_openai_llm_function(client, model_name,temperature=0.7,max_token=256 ):
        '''
        Creating the function for runnable lambda
        '''
        def openai_llm_invoke(prompt):  
            '''
            Need to convert Langchain message formate to OpenAImessage formate for Inference throgh hugging face endpoint
            '''
            # print('langchain messages formate',type(prompt))
            messages = prompt
            openai_messages =[]

            for message in messages:
                if hasattr(message, 'content'):
                    if isinstance(message, AIMessage) or getattr(message, 'type', None) == 'ai':
                        openai_messages.append({"role": "assistant", "content": message.content})
                        print('fgwgwg',openai_messages)
                    else:
                        openai_messages.append({'role': 'user', 'content':message.content})
                        print('mesagesssss',openai_messages)
            
            print('mesagesssss',openai_messages)

            try:
                chat_completion = client.chat.completions.create(
                     model = model_name,
	            messages = openai_messages,
                temperature=temperature,
                max_tokens=max_token,
                stream=False)
                
                return chat_completion.choices[0].message.content

            except Exception as e:
                return f'Error occured while calling the main endpoint in openAIWraper invoke function {str(e)}'

        return openai_llm_invoke

class MedicalBot:
    def __init__(self):
        self.index = None
        self.client = OpenAI(
            base_url="https://h561hdi5s1r2d2tz.us-east-1.aws.endpoints.huggingface.cloud/v1/",
            api_key=os.getenv('huggingface_api_key')
        )
        self.model_name = "sharadsnaik/medgemma-4b-it-medical-gguf"
        self.num_threads = os.cpu_count()
        self.chat_history = InMemoryChatMessageHistory()

        self.session_id = self.generate_session_id()
        self.history_file = f'chat_history_file_{self.session_id}_.json'
        self.session_start_time = datetime.now()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.Database_name_MongoDB = 'Sever_FastAPI_Medical_bot'
        self.initialize_services()
        self.collection_name_mongodb= None

    
    def generate_session_id(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"medical_session_{timestamp}_{unique_id}"
    
    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Get or create chat history for a session"""
        return self.chat_history
    
    def initialize_services(self):
        openai_llm_function = create_openai_llm_function(
            client=self.client,
            model_name=self.model_name,
            temperature=0.7,
            max_token=256
        )
        self.llm = RunnableLambda(openai_llm_function)        

        self.prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful medical assistant. Use the conversation history to provide contextual responses"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "<start_of_turn>user \n{input}<end_of_turn>\n<start_of_turn>assistant\n"),
])

        self.chain = self.prompt_template | self.llm | MedicalChatBotOutputparser()

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
            cleaned_response = self._clean_response(response)
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
            
            # saving all conversations to MONGODB
            await self.database_mongoDB()

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
    
# MONGODB FUNCTIONNSSSSS
    async def connect_mongoBD(self):
        try:
            self.mongodb_client = AsyncIOMotorClient(os.getenv("mongodb_uri"))
            await self.mongodb_client.admin.command('ping')
            self.Database_name = self.mongodb_client[self.Database_name_MongoDB]
            
            # Optional
            self.collection_name_mongodb = self.Database_name['chat_history_collection']

            await self.collection_name_mongodb.create_index("session_id")
            await self.collection_name_mongodb.create_index("created_at")
            # Optional

            print(f"✅ Connected to MongoDB: {self.Database_name}")
            return True
        except Exception as e:
            print(f'Error in connection MONGODB Databases => 👎{e}')

    async def disconnect_mongodb(self):
        '''
        Disconnecting mongoDB after the yelid of life span funtionn (FASTAPI)

        '''
        if self.mongodb_client:
            self.mongodb_client.close()
            print('MongoDB connection client closed')

    async def database_mongoDB(self):
        try:
            messages = self.chat_history.messages
            formated_history = []
            for i, msg in enumerate(messages):
                if isinstance(msg, HumanMessage):
                    role = "Human"
                elif isinstance(msg, AIMessage):
                    role = "Assistant"
                formated_history.append(
                    {
                    'id': i + 1,
                    'role': role,
                    'content': msg.content,
                    'timestamp': datetime.now().isoformat()
                    })
                

            conversation_doc = {
                "session_id": self.session_id,
                "messages": formated_history,
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
                "metadata": {
                    'start_time': self.session_start_time.isoformat(),
                    'message_count': len(formated_history),
                    'model': self.model_name
                }
            }
            
            await self.collection_name_mongodb.update_one(
                {"session_id": self.session_id},
                {"$set": conversation_doc},
                upsert=True)
            
            # await self.collection_name_mongodb.insert_one(conversation_doc)
            # print(f"💾 Conversation saved to MongoDB for session: {self.session_id}")
            return True
        except Exception as e:
            print(f"❌ Error saving conversation to MongoDB: {str(e)}")
            return False
            

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.medicalbot = MedicalBot()
        await app.state.medicalbot.connect_mongoBD()
        # await is mandatory mongodb only works in await and async 

        print("medical bot startup completed")
        yield
    except Exception as e:
        error_message = f"Failed or error occurred in lifespan function: {str(e)}"
        print(f"❌ {error_message}")
        yield

    finally:
        print("Shutting down Medical Bot")
        await app.state.medicalbot.disconnect_mongodb()
        
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

@app.post('/chat', response_model=ChatResponsetype)
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
    uvicorn.run('medical_bot_main_file_V2:app', host='127.0.0.1', port=8000, reload= True)
