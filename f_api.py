from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from MOM_Generation import mom

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to a specific origin or origins if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/file")
async def create_upload_file(file: UploadFile = File(...)):
    file_content = await file.read()  # Read file content directly from UploadFile object
    
    text =  file_content.decode()
    result = mom(text)

    return result
