"""
LLM client wrapper.
Uses the OpenAI-format API uniformly.
"""

import json
import re
from typing import Optional, Dict, Any, List
from openai import OpenAI

from ..config import Config
from .logger import log_llm_interaction


class LLMClient:
    """LLM client."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model = model or Config.LLM_MODEL_NAME
        
        if not self.api_key:
            raise ValueError("LLM_API_KEY is not configured")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None,
        should_log: bool = True
    ) -> str:
        """
        Send a chat request.
        
        Args:
            messages: Message list
            temperature: Temperature parameter
            max_tokens: Maximum token count
            response_format: Response format (e.g., JSON mode)
            should_log: Whether to save request/response to log file
            
        Returns:
            Model response text
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if response_format:
            kwargs["response_format"] = response_format
        
        response = self.client.chat.completions.create(**kwargs)
        content_raw = response.choices[0].message.content or ""
        # Some models (e.g., MiniMax M2.5) include <think> content that should be removed
        content_cleaned = re.sub(r'<think>[\s\S]*?</think>', '', content_raw).strip()

        if should_log:
            log_llm_interaction(
                source_file="llm_client.py",
                messages=messages,
                response_text=content_cleaned,
            )

        return content_cleaned
    
    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        Send a chat request and return JSON.
        
        Args:
            messages: Message list
            temperature: Temperature parameter
            max_tokens: Maximum token count
            
        Returns:
            Parsed JSON object
        """
        response = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            should_log=False,
        )
        # Clean markdown code fence markers
        cleaned_response = response.strip()
        cleaned_response = re.sub(r'^```(?:json)?\s*\n?', '', cleaned_response, flags=re.IGNORECASE)
        cleaned_response = re.sub(r'\n?```\s*$', '', cleaned_response)
        cleaned_response = cleaned_response.strip()

        try:
            parsed_json = json.loads(cleaned_response)
            log_llm_interaction(
                source_file="llm_client.py",
                messages=messages,
                response_text=cleaned_response,
            )
            return parsed_json
        except json.JSONDecodeError:
            log_llm_interaction(
                source_file="llm_client.py",
                messages=messages,
                response_text=cleaned_response,
            )
            raise ValueError(f"Invalid JSON returned by LLM: {cleaned_response}")

