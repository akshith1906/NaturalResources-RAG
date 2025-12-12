# 🤖 SME RAG Agent: Complete System Overview

## Executive Summary
This is a Subject Matter Expert (SME) RAG system specialized in K-12 Geography and Natural Resources education. It combines advanced retrieval techniques with agentic workflows to provide intelligent document generation, Q&A, and automation capabilities.

## 🎯 Key Capabilities

### Core Features
- **Intelligent Q&A**: Context-aware responses using hybrid retrieval
- **Document Generation**: Create quizzes, reports, and presentations (PDF/DOCX/PPTX)
- **Email Automation**: Send generated content directly to recipients
- **Multi-Step Reasoning**: Chain multiple operations in a single query
- **Streaming Responses**: Real-time SSE feedback via FastAPI

### Advanced Features (Bonus)
- **Hybrid Retrieval**: Dense (semantic) + Sparse (BM25 keyword) search
- **Reranking**: BAAI/bge-reranker-base for top-10 precision
- **Delta Ingestion**: Only process new/modified documents with change detection

## 🏗️ System Architecture

### Three-Layer Design
```
┌─────────────────────────────────────────────────────────┐
│                    LAYER 3: Frontend                    │
│  Streamlit UI (sme_frontend.py) → User Interaction     │
└─────────────────────────────────────────────────────────┘
                          ↕️ HTTP/SSE
┌─────────────────────────────────────────────────────────┐
│                 LAYER 2: Agent Workflow                 │
│  LangGraph State Machine → Multi-Step Orchestration    │
│  • Contextualize → Plan → Execute → Route → Respond    │
└─────────────────────────────────────────────────────────┘
                          ↕️ Function Calls
┌─────────────────────────────────────────────────────────┐
│               LAYER 1: Data & Retrieval                 │
│  Pinecone Vector DB + Hybrid Search + Reranker         │
└─────────────────────────────────────────────────────────┘
```

## 🔄 Agent Workflow (LangGraph Nodes)

### State Machine Flow
| Node | Purpose | Key Logic |
|------|---------|-----------|
| **Contextualize** | Query rewriting | Resolves pronouns using chat history |
| **Planner** | Tool orchestration | Generates JSON plan with tool sequence |
| **Executor** | Action execution | Runs tools, resolves dynamic arguments |
| **Router** | Flow control | Decides: continue loop or finalize |
| **Final Response** | Result packaging | Formats answer for user |

### Dynamic Argument Resolution
The Executor supports references like `$results.step_0.file_path`, enabling tools to use outputs from previous steps.

## 🛠️ Available Tools

### 1. run_chat
- **Purpose**: RAG-powered Q&A
- **Process**: Hybrid search → Rerank → LLM synthesis
- **Use Case**: "What are the benefits of afforestation?"

### 2. generate_quiz
- **Purpose**: Create educational assessments
- **Outputs**: PDF/DOCX/PPTX with questions
- **Use Case**: "Generate a 10-question quiz on soil erosion"

### 3. generate_report
- **Purpose**: Produce comprehensive documents
- **Formats**: PDF, DOCX, or PPTX
- **Use Case**: "Create a presentation on natural resources"

### 4. send_email
- **Purpose**: Automated email delivery
- **SMTP**: Configurable via .env
- **Use Case**: "Email the quiz to student@school.edu"

## 🔍 Retrieval Pipeline Deep Dive

### Hybrid Search Strategy
```
User Query → Embeddings (Dense) + BM25 Tokens (Sparse)
                      ↓
            Pinecone Query (α=0.5 blend)
                      ↓
            Top 50 Parent Chunks Retrieved
                      ↓
         Reranker (BAAI/bge-reranker-base)
                      ↓
              Top 10 Final Context
                      ↓
                  LLM Answer
```

### Why Hybrid?
- **Dense**: Captures semantic similarity ("pollution" ↔ "contamination")
- **Sparse**: Exact keyword matching ("CO₂ emissions" appears verbatim)
- **Result**: Higher recall and precision for educational content

## 📊 Data Ingestion Process

### Multi-Level Chunking
- **Parent Chunks**: 2048 tokens (for retrieval)
- **Child Chunks**: 512 tokens (for context)
- **Supports**: PDF, DOCX, TXT, MD, PPTX

### Delta Ingestion (Smart Updates)
- **Hash Calculation**: SHA-256 for each document
- **Manifest Tracking**: `logs/ingestion_manifest.json` stores metadata
- **Change Detection**: Only process modified files
- **Cleanup**: Delete old vectors from Pinecone
- **BM25 Fitting**: Update sparse index with new vocabulary

### Automatic Monitoring
`watcher.py` uses the Watchdog library to trigger ingestion on file changes in `./Docs`.

## 🚀 Quick Start Commands

### Complete Setup (3 Steps)

**Step 1: Configure .env file**
```bash
cat > .env << EOF
GOOGLE_API_KEY="your_gemini_key"
PINECONE_API_KEY="your_pinecone_key"
SMTP_SERVER="smtp.gmail.com"
SMTP_PORT=587
SMTP_SENDER_EMAIL="your_email@gmail.com"
SMTP_SENDER_PASSWORD="your_app_password"
EOF
```

**Step 2: Start everything**
```bash
python watcher.py           # Terminal 1: API + File Watcher
python -m streamlit run sme_frontend.py  # Terminal 2: Frontend
```

### Access Points
- **Frontend**: http://localhost:8501
- **API**: http://localhost:8000/docs (FastAPI Swagger)

## 💡 Usage Examples

### Example 1: Simple Q&A
**User**: "What is the significance of afforestation?"  
**System**:
1. Contextualizes query
2. Plans: `run_chat`
3. Executes hybrid search + rerank
4. Streams detailed answer

### Example 2: Chained Operations
**User**: "Generate a quiz on soil erosion and email it to teacher@school.edu"  
**System**:
1. Plans: Step 1 = `generate_quiz`, Step 2 = `send_email`
2. Executes Step 1 → gets `file_path`
3. Executes Step 2 using `$results.step_0.file_path`
4. Confirms email sent

### Example 3: Document Generation with Fallback
**User**: "Create a PPTX report on water conservation"  
**System**:
1. Attempts PPTX generation
2. If fails → tries DOCX
3. If fails → falls back to PDF
4. Returns generated file path

## 🔑 Key Configuration Files

| File | Purpose |
|------|---------|
| `config.py` | API keys, chunk sizes, model names |
| `.env` | Secrets (API keys, SMTP credentials) |
| `logs/ingestion_manifest.json` | Document version tracking |

## 🎓 Educational Domain Focus

### Optimized For
Natural Resources topics

### Example Topics Covered
- Afforestation and deforestation
- Soil erosion and conservation
- Water resources management
- Climate and ecosystems
- Sustainable development

## 🔧 Error Handling & Robustness

### File Generation Fallback
```python
# Automatic format cascade
try_pptx() → try_docx() → try_pdf() → raise_error()
```

### Ingestion Safety
- Hash verification prevents duplicate processing
- Graceful handling of corrupted files
- Manifest rollback on failure

### API Resilience
- Memory persistence via LangChain MemorySaver
- Conversation history retrieval
- SSE connection management

## 📈 Performance Features

### Efficiency Gains
- **Delta Ingestion**: Only process changed files
- **Namespace Isolation**: Separate vector spaces in Pinecone
- **Reranking**: Reduce LLM context from 50→10 chunks
- **Streaming**: Real-time partial responses

### Cost Optimization
- Avoid re-embedding unchanged documents
- Minimal Pinecone operations
- Efficient token usage in prompts

## 🔐 Security Considerations

### Credential Management
- All secrets in `.env` (not committed)
- SMTP uses app-specific passwords
- API keys never exposed to frontend

### Data Privacy
- Local file storage in `./generated_files`
- No external data leakage
- Pinecone namespace isolation

## 🛣️ Future Enhancement Opportunities

### Potential Additions
- Multi-domain support (beyond Geography)
- User authentication system
- Document versioning UI
- Analytics dashboard
- Mobile-responsive design
- Voice interaction support

## 📚 Technology Stack Summary

| Component | Technology |
|-----------|------------|
| **LLM** | Google Gemini API |
| **Vector DB** | Pinecone (Serverless) |
| **Orchestration** | LangGraph |
| **Backend** | FastAPI + SSE |
| **Frontend** | Streamlit |
| **Embeddings** | Sentence Transformers |
| **Reranker** | BAAI/bge-reranker-base |
| **Document Processing** | python-docx, reportlab, python-pptx |


## Demo Video

[Demo](https://drive.google.com/drive/folders/1e-59spcjYCljral9IKtl8dk43MXzupF1?usp=sharing)
