import json
import logging
import re
from typing import Dict, Any, Literal

from langchain_core.messages import ToolMessage, AIMessage, HumanMessage

from . import tools
from .models import llm_invoke
from prompts import PLANNER_PROMPT, CONTEXTUALIZE_Q_SYSTEM_PROMPT
from .state import AgentState

logger = logging.getLogger("rag.nodes")

tool_map = {
    "run_chat": tools.run_chat,
    "generate_quiz": tools.generate_quiz,
    "generate_report": tools.generate_report,
    "send_email": tools.send_email 
}

def get_tool_descriptions() -> str:
    descriptions = []
    for name, func in tool_map.items():
        doc = func.__doc__ or f"{name}()"
        doc_cleaned = re.sub(r'\n\s+', '\n', doc).strip()
        descriptions.append(f"- `{name}`: {doc_cleaned}")
    return "\n".join(descriptions)

def contextualize_node(state: AgentState) -> Dict[str, Any]:
    logger.info("--- Entering Contextualize Node ---")
    original_query = state['original_query']
    chat_history = state.get('messages', [])
    
    if len(chat_history) <= 1:
        return {"rewritten_query": original_query}

    history_str = ""
    recent_msgs = chat_history[:-1][-5:] 
    for msg in recent_msgs:
        role = "Assistant" if isinstance(msg, AIMessage) else "Human"
        history_str += f"{role}: {msg.content}\n"
        
    prompt = CONTEXTUALIZE_Q_SYSTEM_PROMPT.format(chat_history=history_str, question=original_query)
    rewritten = llm_invoke(prompt).strip()
    
    if not rewritten or "Error" in rewritten: 
        rewritten = original_query
        
    logger.info(f"Original: '{original_query}' -> Rewritten: '{rewritten}'")
    return {"rewritten_query": rewritten}

def planner_node(state: AgentState) -> Dict[str, Any]:
    logger.info("--- Entering Planner Node ---")
    tool_descriptions = get_tool_descriptions()
    query_to_plan_for = state.get("rewritten_query", state['original_query'])
    
    prompt = PLANNER_PROMPT.format(user_input=query_to_plan_for, tools=tool_descriptions)
    llm_response_str = llm_invoke(prompt)
    
    try:
        if "```" in llm_response_str:
            llm_response_str = llm_response_str.split("```")[1].replace("json\n", "")
            
        plan_data = json.loads(llm_response_str)
        plan = plan_data.get("plan", [])
        
        return {
            "plan": plan, 
            "current_step": 0, 
            "intermediate_results": [], 
            "messages": [AIMessage(content=f"Plan: {json.dumps(plan, indent=2)}")]
        }

    except Exception as e:
        logger.error(f"Failed to parse plan: {e}")
        fallback_plan = [{"tool": "run_chat", "args": {"query": query_to_plan_for, "model_name": state['model_name']}}]
        return {
            "plan": fallback_plan, 
            "current_step": 0, 
            "intermediate_results": [],  
            "messages": [AIMessage(content="I'll try to answer directly.")]
        }

def execute_tools_node(state: AgentState) -> Dict[str, Any]:
    logger.info(f"--- Entering Executor Node (Step {state['current_step']}) ---")
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)
    
    if current_step >= len(plan):
        return {}

    intermediate_results = state.get("intermediate_results", [])
    
    if intermediate_results and current_step > 0:
        last_result = intermediate_results[-1]
        if isinstance(last_result, dict) and "error" in last_result:
            error_msg = f"Stopping execution because previous step failed: {last_result['error']}"
            logger.error(error_msg)
            return {
                 "intermediate_results": [{"error": error_msg}],
                 "current_step": len(plan), # Jump to end
                 "messages": [ToolMessage(content=error_msg, tool_call_id="system_halt")]
            }
            
    step = plan[current_step]
    tool_name = step.get("tool")
    tool_args = step.get("args", {})
    
    if tool_name not in tool_map:
        error_result = {"error": f"Unknown tool: {tool_name}"}
        return {
            "intermediate_results": [error_result], 
            "current_step": current_step + 1,
            "messages": [ToolMessage(content=json.dumps(error_result), tool_call_id=f"step_{current_step}")]
        }
    
    if tool_name in ["run_chat", "generate_quiz", "generate_report"]:
        tool_args["model_name"] = state["model_name"]
        
    final_args = tool_args.copy()
    
    for key, value in final_args.items():
        if isinstance(value, str) and value.startswith("$results.step_"):
            try:
                parts = value.split(".")
                ref_idx = int(parts[1].split("_")[1])
                ref_key = parts[2]
                
                if ref_idx < len(intermediate_results):
                    resolved_value = intermediate_results[ref_idx].get(ref_key)
                    
                    if resolved_value is None:
                        error_msg = f"Error: Dependency '{value}' resolved to None. Step {ref_idx} likely failed."
                        logger.error(error_msg)
                        error_result = {"error": error_msg}
                        return {
                             "intermediate_results": [error_result],
                             "current_step": len(plan), 
                             "messages": [ToolMessage(content=error_msg, tool_call_id=f"step_{current_step}")]
                        }
                    
                    final_args[key] = resolved_value
                else:
                    pass
            except Exception as e:
                logger.error(f"Ref resolution failed: {e}")

    logger.info(f"Executing {tool_name} with {final_args}")
    
    try:
        tool_func = tool_map[tool_name]
        result = tool_func.invoke(final_args)
        msg_content = json.dumps(result) if not isinstance(result, str) else result
        result_message = ToolMessage(content=msg_content, tool_call_id=f"step_{current_step}")
        
        return {
            "intermediate_results": [result], 
            "current_step": current_step + 1,
            "messages": [result_message]
        }
    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        error_result = {"error": str(e)}
        return {
            "intermediate_results": [error_result],
            "current_step": current_step + 1,
            "messages": [ToolMessage(content=f"Error: {e}", tool_call_id=f"step_{current_step}")]
        }

def router_node(state: AgentState) -> Literal["continue", "final_response"]:
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)
    if current_step < len(plan):
        return "continue"
    return "final_response"

def final_response_node(state: AgentState) -> Dict[str, Any]:
    logger.info("--- Entering Final Response Node ---")
    intermediate_results = state.get("intermediate_results", [])
    final_msg = "Task completed."
    
    if intermediate_results:
        last_result = intermediate_results[-1]
        if isinstance(last_result, dict):
            if "answer" in last_result:
                final_msg = last_result["answer"]
            elif "file_path" in last_result:
                final_msg = f"I have generated the file: {last_result['file_path']}"
            elif "error" in last_result:
                final_msg = f"Something went wrong: {last_result['error']}"
            elif last_result and "Successfully sent" in str(last_result):
                final_msg = str(last_result)
        elif isinstance(last_result, str):
            final_msg = last_result
            
    return {"messages": [AIMessage(content=final_msg)]}