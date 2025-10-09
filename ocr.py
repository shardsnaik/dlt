from fastapi import FastAPI, File, UploadFile, HTTPException
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import PyPDF2
import pytesseract
from io import BytesIO
from PIL import Image
from transformers import LayoutLMv3Processor

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins= ['https://127.0.0.1'],
    allow_credentials = True,
   allow_methods=["*"],
    allow_headers=["*"],
)
def extract_content(file):
    # with open(file, 'r', encoding='utf-8') as d:  
    input_data = BytesIO(file)
    input_data = PyPDF2.PdfReader(input_data)  
    first_page = input_data.pages[0]  
    text = first_page.extract_text()  
    print(text)

def read_ocr(image_):
    '''
    Read the text data from image 

    '''
    # image_ = Image.open(image_).convert('RGB')
    pytesseract.pytesseract.tesseract_cmd= r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

    # Get ocr result with layout information
    # ocr_result = pytesseract.image_to_data(image_, output_type = pytesseract.OUTPUT.DICT)
    precessor = LayoutLMv3Processor 
    image_ = Image.open(BytesIO(image_)).convert('RGB')
    text = pytesseract.image_to_string(image_)
    return text


@app.post('/upload')
async def read_file(files: UploadFile = File(...)):
    try: 
        file_bytes = await files.read()
        filename_extension = files.filename.split('.')[-1].lower()
        if filename_extension == 'pdf':
            data = extract_content(file_bytes)
            return JSONResponse(content=data)
        elif filename_extension in ['jpg', 'jpeg', 'png']:
            text = read_ocr(file_bytes)
            return JSONResponse(content={"status": "success", "format": "image", "text": text})
        else:
            return 'Invalid formate'
    except Exception as e:
        raise HTTPException(status_code=500 , detail=str(e))

 
uvicorn.run(app, host='127.0.0.1', port= 8000)

