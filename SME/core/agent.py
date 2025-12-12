import logging, json
from typing import Dict, Any

import config, prompts
from core.models import llm_invoke
from core.tools import AgentTools
from core.vector_store import PineconeRetriever

logger = logging.getLogger("rag.agent")

class RAGAgent:
    def __init__(self):
        self.retriever = PineconeRetriever()
        self.tools = AgentTools(self.retriever)
        self.tool_map = {
            "run_chat": self.tools.run_chat,
            "generate_quiz": self.tools.generate_quiz,
            "generate_report": self.tools.generate_report,
        }
        logger.info("RAGAgent initialized with tools and Pinecone retriever.")

    def _parse_router_output(self, llm_output: str) -> Dict[str, Any]:
        """Safely parses the LLM's JSON output."""
        try:
            if "```" in llm_output:
                llm_output = llm_output.split("```")[1].replace("json\n", "")
            return json.loads(llm_output)
        except Exception as e:
            logger.error(f"Failed to parse router JSON: {e}\nRaw output: {llm_output}")
            return {
                "action": "run_chat",
                "args": {"query": f"I'm sorry, I had trouble understanding your request. (Error: {e})"}
            }

    def route(self, user_input: str, model_name: str) -> Dict[str, Any]:
        """
        1. Routes user input to the correct tool using the LLM.
        2. Executes the tool and returns its result.
        """
        if model_name not in config.SUPPORTED_EMBEDDING_MODELS:
            return {"error": f"Invalid embedding model name. Supported: {config.SUPPORTED_EMBEDDING_MODELS}"}
        

        router_prompt = prompts.ROUTER_PROMPT.replace("{{user_input}}", user_input)
        llm_json_output = llm_invoke(router_prompt)
        decision = self._parse_router_output(llm_json_output)
        
        action = decision.get("action")
        args = decision.get("args", {})
        args["model_name"] = model_name
        
        if action in self.tool_map:
            try:
                tool_function = self.tool_map[action]
                result = tool_function(**args)
                return {"action": action, "result": result}
            except Exception as e:
                logger.error(f"Error executing tool '{action}' with args {args}: {e}")
                return {"error": f"An error occurred while running the {action} tool."}
        else:
            logger.warning(f"Router decided on an unknown action: {action}")
            return {"action": "run_chat", "result": self.tools.run_chat(user_input, model_name)}