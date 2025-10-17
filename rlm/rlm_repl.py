"""
Simple Recursive Language Model (RLM) with REPL environment.
"""

from typing import Dict, List, Optional, Any 

from rlm import RLM
from rlm.repl import REPLEnv
from rlm.utils.llm import OpenAIClient
from rlm.utils.prompts import DEFAULT_QUERY, next_action_prompt, build_system_prompt
import rlm.utils.utils as utils
import json

from rlm.logger.root_logger import ColorfulLogger
from rlm.logger.repl_logger import REPLEnvLogger


class RLM_REPL(RLM):
    """
    LLM Client that can handle long contexts by recursively calling itself.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 model: str = "gpt-5",
                 recursive_model: str = "gpt-5",
                 max_iterations: int = 20,
                 depth: int = 0,
                 enable_logging: bool = False,
                 allow_ollama_tools: bool = False,
                 ):
        self.api_key = api_key
        self.model = model
        self.recursive_model = recursive_model
        self.allow_ollama_tools = allow_ollama_tools
        self.llm = OpenAIClient(api_key, model) # Replace with other client
        
        # Track recursive call depth to prevent infinite loops
        self.repl_env = None
        self.depth = depth # Unused in this version.
        self._max_iterations = max_iterations
        
        # Initialize colorful logger
        self.logger = ColorfulLogger(enabled=enable_logging)
        self.repl_env_logger = REPLEnvLogger(enabled=enable_logging)
        
        self.messages = [] # Initialize messages list
        self.query = None
    
    def setup_context(self, context: List[str] | str | List[Dict[str, str]], query: Optional[str] = None):
        """
        Setup the context for the RLMClient.

        Args:
            context: The large context to analyze in the form of a list of messages, string, or Dict
            query: The user's question
        """
        if query is None:
            query = DEFAULT_QUERY

        self.query = query
        self.logger.log_query_start(query)

        # Initialize the conversation with the REPL prompt
        self.messages = build_system_prompt()
        # If using Ollama and tools are not allowed, add an anti-tool directive
        if getattr(self.llm, "provider", None) == "ollama" and not getattr(self, "allow_ollama_tools", False):
            self.messages.insert(0, {
                "role": "system",
                "content": "IMPORTANT: Do not use structured tool/function calls or JSON tool calls. Only respond in plain text. To run code, emit fenced blocks using ```repl ...``` and nothing else. Never call functions in JSON."
            })
        self.logger.log_initial_messages(self.messages)
        
        # Initialize REPL environment with context data
        context_data, context_str = utils.convert_context_for_repl(context)
        
        self.repl_env = REPLEnv(
            context_json=context_data, 
            context_str=context_str, 
            recursive_model=self.recursive_model,
        )
        
        return self.messages

    def completion(self, context: List[str] | str | List[Dict[str, str]], query: Optional[str] = None) -> str:
        """
        Given a query and a (potentially long) context, recursively call the LM
        to explore the context and provide an answer using a REPL environment.
        """
        self.messages = self.setup_context(context, query)
        
        # Main loop runs for fixed # of root LM iterations
        for iteration in range(self._max_iterations):
            
            # Query root LM to interact with REPL environment
            msgs = self.messages + [next_action_prompt(query, iteration)]
            response = ""
            used_tools = False

            if getattr(self.llm, "provider", None) == "ollama" and getattr(self, "allow_ollama_tools", False):
                # Initialize Ollama client if needed
                try:
                    self.llm._init_ollama()
                except Exception:
                    pass

                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": "repl",
                            "description": "Execute Python code in the current REPL environment. You can access `context`, use `llm_query(prompt: str)`, and return a value via FINAL(...) or FINAL_VAR(...). Return the stdout/stderr text.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "code": {"type": "string", "description": "Python code to execute inside the REPL"}
                                },
                                "required": ["code"]
                            }
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "llm_query",
                            "description": "Query the sub‑LLM from the REPL to summarize or reason over chunks.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "prompt": {"type": "string", "description": "Prompt for the sub‑LLM"}
                                },
                                "required": ["prompt"]
                            }
                        }
                    }
                ]

                # Work on a local copy while we interact with tools
                working_msgs = list(msgs)
                options = {"temperature": 0.2, "num_predict": 1024}
                try:
                    resp = self.llm._ollama_client.chat(model=self.model, messages=working_msgs, tools=tools, options=options, keep_alive="5m")
                    # Tool-use loop
                    while True:
                        msg_obj = None
                        tool_calls = []
                        content_text = None
                        try:
                            # ChatResponse path
                            msg_obj = getattr(resp, "message", None)
                            if msg_obj is not None:
                                tool_calls = getattr(msg_obj, "tool_calls", None) or []
                                content_text = getattr(msg_obj, "content", None)
                        except Exception:
                            msg_obj = None
                        if msg_obj is None and isinstance(resp, dict):
                            msg_obj = resp.get("message") or {}
                            tool_calls = msg_obj.get("tool_calls") or []
                            content_text = msg_obj.get("content")

                        if tool_calls:
                            used_tools = True
                            for call in tool_calls:
                                # Extract name and arguments
                                try:
                                    fn = getattr(call, "function", None) or call.get("function")
                                    name = getattr(fn, "name", None) or (fn.get("name") if isinstance(fn, dict) else None)
                                    args_raw = getattr(fn, "arguments", None) or (fn.get("arguments") if isinstance(fn, dict) else None)
                                    if isinstance(args_raw, str):
                                        try:
                                            args = json.loads(args_raw)
                                        except Exception:
                                            args = {"code": args_raw} if name == "repl" else {"prompt": args_raw}
                                    elif isinstance(args_raw, dict):
                                        args = args_raw
                                    else:
                                        args = {}
                                except Exception:
                                    name = None
                                    args = {}
                                result_text = ""
                                if name == "repl":
                                    code = args.get("code", "")
                                    repl_result = self.repl_env.code_execution(code)
                                    # Format REPL output
                                    result_text = utils.format_execution_result(repl_result.stdout, repl_result.stderr, repl_result.locals, truncate_length=200)
                                elif name == "llm_query":
                                    prompt_arg = args.get("prompt", "")
                                    try:
                                        result_text = str(self.repl_env.globals["llm_query"](prompt_arg))
                                    except Exception as e:
                                        result_text = f"Error in llm_query: {str(e)}"
                                else:
                                    result_text = f"Unsupported tool: {name}"
                                # Append tool result message
                                working_msgs.append({"role": "tool", "name": name or "unknown", "content": result_text})
                            # Continue the chat after providing tool results
                            resp = self.llm._ollama_client.chat(model=self.model, messages=working_msgs, tools=tools, options=options, keep_alive="5m")
                            continue

                        # No tool calls; capture assistant content and break
                        response = content_text or ""
                        # Save assistant message to history
                        if response:
                            working_msgs.append({"role": "assistant", "content": response})
                        # Persist working messages back to conversation for next iterations
                        self.messages = working_msgs
                        break
                except Exception as e:
                    # Fallback: retry once without tools using an explicit anti-tool directive
                    safe_msgs = list(msgs)
                    safe_msgs.insert(0, {
                        "role": "system",
                        "content": "IMPORTANT: Do not use structured tool/function calls or JSON tool calls. Only respond in plain text. To run code, emit fenced blocks using ```repl ...``` and nothing else. Never call functions in JSON."
                    })
                    try:
                        resp2 = self.llm._ollama_client.chat(model=self.model, messages=safe_msgs, options=options, keep_alive="5m")
                        # Extract assistant content
                        response = ""
                        try:
                            response = getattr(getattr(resp2, "message", None), "content", "") or ""
                        except Exception:
                            pass
                        if not response and isinstance(resp2, dict):
                            response = (resp2.get("message") or {}).get("content", "") or ""
                        # Persist messages
                        if response:
                            safe_msgs.append({"role": "assistant", "content": response})
                        self.messages = safe_msgs
                    except Exception:
                        # Last-resort fallback: call generic client with anti-tool directive
                        response = self.llm.completion(safe_msgs)
            else:
                response = self.llm.completion(msgs)

            # Check for code blocks
            code_blocks = utils.find_code_blocks(response)
            self.logger.log_model_response(response, has_tool_calls=bool(code_blocks) or used_tools)
            
            # Process code execution or add assistant message
            if code_blocks:
                self.messages = utils.process_code_execution(
                    response, self.messages, self.repl_env, 
                    self.repl_env_logger, self.logger
                )
            else:
                # Add assistant message when there are no code blocks and not already appended in tool loop
                if not used_tools:
                    assistant_message = {"role": "assistant", "content": "You responded with:\n" + response}
                    self.messages.append(assistant_message)
            
            # Check that model produced a final answer
            final_answer = utils.check_for_final_answer(
                response, self.repl_env, self.logger,
            )

            # In practice, you may need some guardrails here.
            if final_answer:
                self.logger.log_final_response(final_answer)
                return final_answer

            
        # If we reach here, no final answer was found in any iteration
        print("No final answer found in any iteration")
        self.messages.append(next_action_prompt(query, iteration, final_answer=True))
        final_answer = self.llm.completion(self.messages)
        self.logger.log_final_response(final_answer)

        return final_answer
    
    def cost_summary(self) -> Dict[str, Any]:
        """Get the cost summary of the Root LM + Sub-RLM Calls."""
        raise NotImplementedError("Cost tracking not implemented for RLM REPL.")

    def reset(self):
        """Reset the (REPL) environment and message history."""
        self.repl_env = REPLEnv()
        self.messages = []
        self.query = None


if __name__ == "__main__":
    pass
