import time
from typing import Any

import openai
from openai import OpenAI

from ..eval_types import MessageList, SamplerBase, SamplerResponse


class GPT5WebSampler(SamplerBase):
    """
    Sample from OpenAI's GPT-5 with web search enabled
    """

    def __init__(
        self,
        model: str = "gpt-5",
        system_message: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
    ):
        self.api_key_name = "OPENAI_API_KEY"
        self.client = OpenAI()
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_format = "url"

    def _handle_image(
        self,
        image: str,
        encoding: str = "base64",
        format: str = "png",
        fovea: int = 768,
    ):
        new_image = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{format};{encoding},{image}",
            },
        }
        return new_image

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        if self.system_message:
            message_list = [
                self._pack_message("system", self.system_message)
            ] + message_list
        
        trial = 0
        while True:
            try:
                # Try with web search first
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=message_list,
                        temperature=1.0,  # GPT-5 only supports temperature=1
                        max_completion_tokens=self.max_tokens,
                        tools=[{"type": "web_search"}],
                    )
                except Exception as web_error:
                    print(f"Web search not available, falling back to regular GPT-5: {web_error}")
                    # Fall back to regular GPT-5 without tools
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=message_list,
                        temperature=1.0,  # GPT-5 only supports temperature=1
                        max_completion_tokens=self.max_tokens,
                    )
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("OpenAI API returned empty response; retrying")
                return SamplerResponse(
                    response_text=content,
                    response_metadata={"usage": response.usage},
                    actual_queried_message_list=message_list,
                )
            except openai.BadRequestError as e:
                print("Bad Request Error", e)
                return SamplerResponse(
                    response_text="No response (bad request).",
                    response_metadata={"usage": None},
                    actual_queried_message_list=message_list,
                )
            except Exception as e:
                exception_backoff = 2**trial  # exponential back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1