"""
Unified LLM client with provider routing (OpenAI, xAI Grok, Anthropic Claude, Google Gemini).
Performance-focused, with lazy init and minimal payload conversions.
"""

import os
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

# Lazy imports for optional providers
try:
    from openai import OpenAI as _OpenAI
except Exception:
    _OpenAI = None

load_dotenv()


def _normalize_messages(messages: List[Dict[str, str]] | Dict[str, str] | str) -> List[Dict[str, str]]:
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    if isinstance(messages, dict):
        return [messages]
    return messages or []


class OpenAIClient:
    """
    Backward-compatible name for the unified LLM client. Routes by model name:
    - OpenAI (default): model names like gpt-5, gpt-4o, etc.
    - xAI Grok (OpenAI-compatible): model includes 'grok' (e.g., 'grok-4-latest')
    - Anthropic Claude: model includes 'claude' (e.g., 'claude-4.5-sonnet' or 'claude-sonnet-4.5')
    - Google Gemini: model includes 'gemini' (e.g., 'gemini-2.5-pro')
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-5"):
        self.model = model
        ml = model.lower()

        if "gemini" in ml:
            self.provider = "google"
        elif "claude" in ml:
            self.provider = "anthropic"
        elif "grok" in ml:
            self.provider = "xai"
        else:
            self.provider = "openai"

        # Defer heavy client creation until first use
        self._client = None
        self._api_key = api_key  # may be None; resolved per provider lazily
        self._gemini_model = None  # cached GenerativeModel

    # Provider-specific lazy initializers
    def _init_openai(self):
        if self._client is not None:
            return
        if _OpenAI is None:
            raise ImportError("openai package is required. Install 'openai'.")
        api_key = self._api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI provider.")
        self._client = _OpenAI(api_key=api_key)

    def _init_xai(self):
        if self._client is not None:
            return
        if _OpenAI is None:
            raise ImportError("openai package is required for xAI. Install 'openai'.")
        api_key = self._api_key or os.getenv("XAI_API_KEY")
        if not api_key:
            raise ValueError("XAI_API_KEY is required for xAI Grok provider.")
        base_url = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
        self._client = _OpenAI(api_key=api_key, base_url=base_url)

    def _init_anthropic(self):
        if self._client is not None:
            return
        try:
            from anthropic import Anthropic  # type: ignore
        except Exception:
            raise ImportError("anthropic package is required. Install 'anthropic'.")
        api_key = self._api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for Anthropic provider.")
        self._client = Anthropic(api_key=api_key)

    def _init_gemini(self, system_instruction: Optional[str] = None):
        if self._gemini_model is not None:
            return
        try:
            import google.generativeai as genai  # type: ignore
        except Exception:
            raise ImportError("google-generativeai package is required. Install 'google-generativeai'.")
        api_key = self._api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is required for Google Gemini provider.")
        genai.configure(api_key=api_key)
        # Cache the GenerativeModel; system_instruction helps performance by pushing system prompt server-side
        if system_instruction:
            self._gemini_model = genai.GenerativeModel(self.model, system_instruction=system_instruction)
        else:
            self._gemini_model = genai.GenerativeModel(self.model)

    def completion(
        self,
        messages: List[Dict[str, str]] | Dict[str, str] | str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        msgs = _normalize_messages(messages)
        temp = 0.2 if temperature is None else temperature
        max_tokens = max_tokens or 1024

        if self.provider == "openai":
            self._init_openai()
            # Use OpenAI Chat Completions
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=msgs,
                max_completion_tokens=max_tokens,
                temperature=temp,
                # OpenAI client handles keep-alive internally; avoid extra kwargs for performance
            )
            return resp.choices[0].message.content

        if self.provider == "xai":
            self._init_xai()
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=msgs,
                max_completion_tokens=max_tokens,
                temperature=temp,
            )
            return resp.choices[0].message.content

        if self.provider == "anthropic":
            self._init_anthropic()
            # Convert messages to Anthropic format
            system_parts: List[str] = []
            anthro_msgs: List[Dict[str, str]] = []
            for m in msgs:
                role = m.get("role")
                content = m.get("content", "")
                if role == "system":
                    system_parts.append(content)
                elif role in ("user", "assistant"):
                    anthro_msgs.append({"role": role, "content": content})
            system_prompt = "\n\n".join(system_parts) if system_parts else None

            # Call Anthropic Messages API
            message = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temp,
                messages=anthro_msgs,
                system=system_prompt,
            )
            # Extract text content
            texts: List[str] = []
            try:
                for block in getattr(message, "content", []) or []:
                    if getattr(block, "type", "") == "text":
                        texts.append(getattr(block, "text", ""))
            except Exception:
                pass
            return "".join(texts) or getattr(getattr(message, "content", None), "text", "") or ""

        if self.provider == "google":
            # For Gemini, prefer server-side system_instruction for perf
            sys_parts = [m["content"] for m in msgs if m.get("role") == "system"]
            system_instruction = "\n\n".join(sys_parts) if sys_parts else None
            self._init_gemini(system_instruction=system_instruction)

            # Convert to Gemini chat history format
            gemini_history: List[Dict[str, Any]] = []
            role_map = {"user": "user", "assistant": "model"}
            for m in msgs:
                role = role_map.get(m.get("role"), "user")
                content = m.get("content", "")
                # Skip system here (already used as system_instruction when available)
                if m.get("role") == "system":
                    continue
                gemini_history.append({"role": role, "parts": [content]})

            try:
                generation_config = {"max_output_tokens": max_tokens, "temperature": temp}
                resp = self._gemini_model.generate_content(
                    contents=gemini_history if gemini_history else [
                        {"role": "user", "parts": [""]}
                    ],
                    generation_config=generation_config,
                )
                # google-generativeai returns text in .text
                return getattr(resp, "text", "") or ""
            except Exception as e:
                raise RuntimeError(f"Error generating completion (Gemini): {str(e)}")

        raise RuntimeError(f"Unsupported provider for model '{self.model}'.")