# Document Chat API

FastAPI service for uploading PDFs and chatting with individual documents.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your-groq-api-key
jwt_secret_key=a-long-random-jwt-signing-secret
DOCUMENTS_API_KEY=a-long-random-api-key
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
# Optional, only when web search is enabled:
TAVILY_API_KEY=your-tavily-api-key
```

Start the server:

```powershell
uvicorn app:app --reload
```

Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

## Authentication

`/uploadDocument`, `/getAllUploadedDocuments`, and `/document/{doc_id}` require:

```http
X-API-Key: <DOCUMENTS_API_KEY>
```

`/initiateChat`, `/sendChat`, and `/refreshToken` require:

```http
Authorization: Bearer <chat_token>
```

## API examples

Health check:

```powershell
curl http://127.0.0.1:8000/
```

Upload a PDF:

```powershell
curl -X POST http://127.0.0.1:8000/uploadDocument `
  -H "X-API-Key: <DOCUMENTS_API_KEY>" `
  -F "file=@C:\path\to\document.pdf"
```

List uploaded documents:

```powershell
curl http://127.0.0.1:8000/getAllUploadedDocuments `
  -H "X-API-Key: <DOCUMENTS_API_KEY>"
```

Get a chat token for a document:

```powershell
curl http://127.0.0.1:8000/document/<doc_id> `
  -H "X-API-Key: <DOCUMENTS_API_KEY>"
```

Start a new chat session:

```powershell
curl -X POST http://127.0.0.1:8000/initiateChat `
  -H "Authorization: Bearer <chat_token>"
```

Send a chat message:

```powershell
curl -X POST http://127.0.0.1:8000/sendChat `
  -H "Authorization: Bearer <chat_token>" `
  -H "Content-Type: application/json" `
  -d '{"session_id":"<session_id>","query":"What is the objective of this paper?"}'
```

Refresh an expired chat token within its refresh window:

```powershell
curl -X POST http://127.0.0.1:8000/refreshToken `
  -H "Authorization: Bearer <expired_chat_token>"
```

Chat tokens expire after `TOKEN_EXPIRE_MINUTES`; refreshed tokens cannot exceed `MAX_REFRESH_AGE_HOURS` from the first token issuance.
