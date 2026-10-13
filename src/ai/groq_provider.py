from __future__ import annotations

import os
import time

from groq import APIError, Groq, RateLimitError


class GroqProviderError(RuntimeError):
    """Raised when the Groq provider cannot produce a response."""


DEFAULT_MODEL = "openai/gpt-oss-20b"


class GroqReasoningProvider:
    """
    Thin provider wrapper around the Groq API.

    The provider is responsible only for communicating with Groq.
    Parsing, grounding, semantic validation, and governance remain
    in the downstream AI reasoning layer.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        resolved_api_key = (
            api_key
            or os.getenv("GROQ_API_KEY")
        )

        if not resolved_api_key:
            raise GroqProviderError(
                "GROQ_API_KEY is not configured."
            )

        self.model = (
            model
            or os.getenv(
                "GROQ_MODEL",
                DEFAULT_MODEL,
            )
        )

        self.timeout = timeout

        self._client = Groq(
            api_key=resolved_api_key,
            timeout=timeout,
        )

    def generate_json(
        self,
        prompt: str,
    ) -> str:
        """
        Request a JSON-formatted response and return its raw text.

        Transient Groq rate-limit responses are retried with bounded
        backoff. The caller remains responsible for parsing JSON,
        validating the reasoning contract, and checking grounding.
        """

        if not isinstance(prompt, str) or not prompt.strip():
            raise GroqProviderError(
                "prompt must be a non-empty string."
            )

        max_retries = 3

        for attempt in range(max_retries + 1):
            try:
                response = (
                    self._client
                    .chat
                    .completions
                    .create(
                        model=self.model,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a constrained reasoning "
                                    "component in a governed "
                                    "decision-support system. "
                                    "Use only the evidence supplied "
                                    "in the prompt. "
                                    "Never invent facts, metrics, "
                                    "causes, or outcomes. "
                                    "Do not execute actions or claim "
                                    "that an action has been approved. "
                                    "Return valid JSON only."
                                ),
                            },
                            {
                                "role": "user",
                                "content": prompt,
                            },
                        ],
                        temperature=0,
                        max_completion_tokens=2500,
                        response_format={
                            "type": "json_object"
                        },
                    )
                )

                content = (
                    response
                    .choices[0]
                    .message
                    .content
                )

                if not content or not content.strip():
                    raise GroqProviderError(
                        "Groq returned an empty response."
                    )

                return content

            except RateLimitError as exc:
                if attempt >= max_retries:
                    raise GroqProviderError(
                        "Groq API rate limit persisted after "
                        f"{max_retries} retries."
                    ) from exc

                retry_after = getattr(
                    exc,
                    "retry_after",
                    None,
                )

                if retry_after is None:
                    response = getattr(
                        exc,
                        "response",
                        None,
                    )

                    if response is not None:
                        headers = getattr(
                            response,
                            "headers",
                            {},
                        )

                        retry_after = headers.get(
                            "retry-after"
                        )

                try:
                    wait_seconds = float(
                        retry_after
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    wait_seconds = (
                        2.0 * (attempt + 1)
                    )

                wait_seconds = max(
                    1.0,
                    min(wait_seconds, 10.0),
                )

                time.sleep(wait_seconds)

            except GroqProviderError:
                raise

            except APIError as exc:
                raise GroqProviderError(
                    "Groq API request failed: "
                    f"{type(exc).__name__}"
                ) from exc

            except Exception as exc:
                raise GroqProviderError(
                    "Unexpected failure while requesting "
                    "Groq reasoning: "
                    f"{type(exc).__name__}"
                ) from exc

        raise GroqProviderError(
            "Groq reasoning request failed unexpectedly."
        )