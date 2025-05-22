# LightRAG Experimental API

A FastAPI-based RAG (Retrieval-Augmented Generation) API service that provides chat functionality and document management with error handling.

## Why LightRAG?

LightRAG combines the best of both worlds - the speed and efficiency of naive RAG systems with the sophisticated relationship mapping of graph-based RAG approaches. It offers:

- **Lightweight**: Minimal dependencies and efficient resource usage
- **Fast Processing**: Optimized document chunking and retrieval
- **Hybrid Approach**: Combines traditional chunking with relationship mapping
- **Flexible Architecture**: Easy to integrate and extend

For detailed information, visit the [LightRAG Documentation](https://github.com/HKUDS/LightRAG).

Prerequisites

- Python 3.8+
- OpenAI API key

## Installation

1. Clone the repository:

```bash
git clone https://github.com/damieee/LightRAG_Experimental.git
cd LightRAG_Experimental
cd api
```

2. Install dependencies:

```bash
pip install uv
uv sync
```

3. Set up your OpenAI API key:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Running the Application

Start the server:

```bash
uv run app.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Chat Endpoints

#### POST /chat

Regular chat endpoint that returns a complete response.

Request body:

_With History_

```json
{
  "user_input": "How do I print hello world in Python?",
  "message_history": [
    { "role": "user", "content": "Hi" },
    { "role": "assistant", "content": "Hello! How can I help you?" }
  ]
}
```

_Without History_

```json
{
  "user_input": "How do I print hello world in Python?"
}
```

#### POST /chat/stream

Streaming chat endpoint that returns response chunks.

Request body: Same as /chat

### Document Management Endpoints

#### POST /docs/insert

Insert a new document into the RAG system.

Request body:

```json
{
  "content": "Your document content here"
}
```

#### POST /docs/update

Update an existing document.

Request body:

```json
{
  "doc_id": "document_id",
  "content": "Updated content here"
}
```

#### POST /docs/remove

Remove a document from the system.

Request body:

```json
{
  "doc_id": "document_id"
}
```

## Error Handling

All endpoints include error handling with structured responses:

```json
{
  "error": "Error Type",
  "details": "Detailed error message"
}
```

## Project Structure

```
LightRAG_Experimental/
├── api/
│   ├── app.py              # Main FastAPI application
│   ├── schemas/
│   │   ├── chat.py        # Pydantic models for chat
│   │   └── docs.py        # Document operation schemas
│   └── services/
│       ├── lightrag_service.py    # RAG operations
│       └── pydantic_ai_service.py # AI service integration
└── pydantic-docs/         # Document storage directory
```

## Features

- Real-time chat with AI using RAG
- Streaming responses support
- Document management (insert/update/remove)
- CORS enabled for frontend integration
- Comprehensive error handling
- Pydantic model validation

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
