from fastapi import FastAPI, UploadFile, File
import os
from task_1 import main as process_document

app = FastAPI()

UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def health_check():
    return {"status": "API is running"}

@app.post("/uploadDocument")
async def upload_document(file: UploadFile = File(...)):
    contents = await file.read()

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(contents)

    result = process_document(file_path)

    if result is None:
        return {"error": "Failed to process document"}

    return {
        "doc_id": result["doc_id"],
        "filename": file.filename,
        "message": "Document uploaded and processed successfully"
    }