import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal, List, Dict, Any, AsyncGenerator 
from sse_starlette.sse import EventSourceResponse
import uvicorn
import json
import uuid


from core.graph import workflow 
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("rag.api")

app = FastAPI(title="SME RAG Agent")

memory = MemorySaver()
agent_app = workflow.compile(checkpointer=memory)

class AgentQuery(BaseModel):
    query: str
    model_name: Literal["all-mpnet-base-v2", "BAAI/bge-base-en-v1.5"]
    conversation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

def get_content(msg: Any) -> str:
    if isinstance(msg, dict): return msg.get("content", "")
    return getattr(msg, "content", str(msg))

def get_type(msg: Any) -> str:
    if isinstance(msg, dict): return msg.get("type", "")
    return getattr(msg, "type", "")

async def stream_graph_events(query: AgentQuery) -> AsyncGenerator[str, None]:
    run_config = {"configurable": {"thread_id": query.conversation_id}}
    graph_input = {
        "messages": [HumanMessage(content=query.query)],
        "original_query": query.query,
        "model_name": query.model_name,
    }
    
    logger.info(f"Invoking graph for thread_id: {query.conversation_id}")
    
    is_chat_answer_streamed = False 
    
    async for event in agent_app.astream_events(graph_input, config=run_config, version="v1"):
        event_type = event["event"]
        
        if event_type == "on_chat_model_stream" and event["name"] == "planner":
            chunk = event["data"].get("chunk")
            if chunk:
                content = get_content(chunk)
                if content and ("{" in content or "[" in content):
                    yield f"data: {json.dumps({'type': 'plan', 'content': content})}\n\n"

        elif event_type == "on_tool_end":
            tool_output = event["data"].get("output")
            tool_name = event["name"]
            
            if tool_name == "run_chat" and isinstance(tool_output, dict):
                thought = tool_output.get("thought", "")
                answer = tool_output.get("answer", "")
                
                if thought:
                    yield f"data: {json.dumps({'type': 'thought', 'content': thought})}\n\n"
                
                if answer:
                    yield f"data: {json.dumps({'type': 'answer', 'content': answer})}\n\n"
                    is_chat_answer_streamed = True
            
            elif isinstance(tool_output, dict):
                error = tool_output.get("error", "")
                file_path = tool_output.get("file_path", "")
                
                if error:
                    yield f"data: {json.dumps({'type': 'error', 'content': error})}\n\n"
                if file_path:
                    yield f"data: {json.dumps({'type': 'file_generated', 'content': file_path})}\n\n"
                
                if tool_name == "send_email":
                    status = tool_output.get("content", str(tool_output)) if isinstance(tool_output, dict) else str(tool_output)
                    yield f"data: {json.dumps({'type': 'email_status', 'content': status})}\n\n"
            
            if isinstance(tool_output, dict):
                 display_content = json.dumps(tool_output)
            else:
                 display_content = str(tool_output)
            
            yield f"data: {json.dumps({'type': 'tool_result', 'tool': tool_name, 'content': display_content})}\n\n"


        elif event_type == "on_chain_end" and event["name"] == agent_app.name:
            final_state = event["data"].get("output")
            
            if not is_chat_answer_streamed:
                final_answer = "No answer generated."
                
                if isinstance(final_state, dict) and "messages" in final_state:
                    messages = final_state["messages"]
                    for msg in reversed(messages):
                        msg_type = get_type(msg)
                        msg_content = get_content(msg)
                        
                        if msg_type == "ai" and msg_content and not msg_content.startswith("Plan:"):
                            final_answer = msg_content
                            break

                yield f"data: {json.dumps({'type': 'final_answer', 'content': final_answer})}\n\n"

@app.post("/agent/invoke_stream")
async def invoke_agent_stream(query: AgentQuery):
    try:
        return EventSourceResponse(stream_graph_events(query))
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agent/history")
async def get_agent_history(conversation_id: str):
    try:
        config = {"configurable": {"thread_id": conversation_id}}
        state_snapshot = await memory.aget(config=config)
        if not state_snapshot: return {"messages": []}
        messages = state_snapshot.channel_values.get('messages', [])
        return {"messages": [{"type": get_type(m), "content": get_content(m)} for m in messages]}
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "SME RAG Agent Running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)