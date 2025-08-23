# providers.py
import os, json, httpx
from typing import AsyncGenerator
from dotenv import load_dotenv
load_dotenv()

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from models import ProviderEndpoint

TIMEOUT = int(os.getenv("TIMEOUT_SECONDS","60"))

async def _get_provider(session: AsyncSession, name: str) -> dict:
    q = await session.execute(select(ProviderEndpoint).where(ProviderEndpoint.name==name))
    pe = q.scalar_one_or_none()
    if pe:
        return {"base_url": pe.base_url, "api_key": pe.api_key}
    # Fallback: .env
    if name == "openai_compat":
        return {"base_url": os.getenv("OPENAI_BASE_URL"), "api_key": os.getenv("OPENAI_API_KEY")}
    if name == "gemini":
        # Gemini nutzt kein base_url
        return {"base_url": None, "api_key": os.getenv("GEMINI_API_KEY")}
    if name == "tts_provider":
        return {"base_url": os.getenv("TTS_PROVIDER_BASE_URL"), "api_key": os.getenv("TTS_PROVIDER_API_KEY")}
    if name == "asr_provider":
        return {"base_url": os.getenv("ASR_PROVIDER_BASE_URL"), "api_key": os.getenv("ASR_PROVIDER_API_KEY")}
    return {"base_url": None, "api_key": None}

async def openai_chat_stream(session: AsyncSession, payload: dict) -> AsyncGenerator[str, None]:
    prov = await _get_provider(session, "openai_compat")
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        r = await client.post(f"{prov['base_url']}/chat/completions",
                              headers={"Authorization": f"Bearer {prov['api_key']}",
                                       "Content-Type": "application/json"},
                              json=payload)
        r.raise_for_status()
        async for line in r.aiter_lines():
            if line and line.startswith("data: "):
                yield line[6:]

async def gemini_stream(session: AsyncSession, contents: list, system_instruction: dict | None, generation_config: dict | None) -> AsyncGenerator[str, None]:
    prov = await _get_provider(session, "gemini")
    model = "gemini-2.5-pro"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse&key={prov['api_key']}"
    body = {"contents": contents}
    if system_instruction: body["systemInstruction"] = system_instruction
    if generation_config: body["generationConfig"] = generation_config

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        r = await client.post(url, headers={"Content-Type":"application/json"}, json=body)
        r.raise_for_status()
        async for line in r.aiter_lines():
            if line and line.startswith("data: "):
                yield line[6:]


async def tts_forward(session: AsyncSession, model:str, text:str, provider: str) -> bytes:
    # Get the provider's base_url and api_key from the database
    prov_details = await _get_provider(session, provider)
    base_url = prov_details.get("base_url")
    api_key = prov_details.get("api_key")

    if not base_url or not api_key:
        raise ValueError(f"Provider '{provider}' is missing base_url or api_key in configuration.")

    # --- START OF NEW LOGIC ---
    # We choose the payload format based on the provider's name.
    # It's a good practice to name your Fish Audio provider "tts_fish_audio" or similar in the admin panel.
    if "fish.audio" in base_url or "fish" in provider.lower():
        # This is the payload format that Fish Audio expects
        payload = {
            "text": text,
            "reference_id": model,
            "normalize": True,
            "format": "mp3",
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        # The URL for Fish is just the base URL, not base_url + /audio/speech
        final_url = base_url
    else:
        # This is the standard OpenAI-compatible format that your other providers (and the middleware itself) use.
        payload = {
            "model": model,
            "input": text,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        # For OpenAI-compatible providers, we append the standard path
        final_url = f"{base_url.rstrip('/')}/audio/speech"
    # --- END OF NEW LOGIC ---

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        # Note we are now sending a JSON payload for both cases
        r = await client.post(final_url, headers=headers, json=payload)
        
        # This will raise an error if the status is 4xx or 5xx, which is what we want.
        r.raise_for_status()
        return r.content


async def asr_forward(session: AsyncSession, model:str, file_bytes: bytes, filename: str, provider: str) -> dict:
    prov = await _get_provider(session, provider)
    base_url = prov.get("base_url", "")
    api_key = prov.get("api_key")

    # --- START OF NEW ROBUST LOGIC ---
    # This logic now intelligently constructs the final URL.
    
    # Clean up the base URL by removing trailing slashes.
    clean_base_url = base_url.rstrip('/')
    
    # If the user accidentally included '/chat/completions' in the base URL,
    # we intelligently replace it with the correct path.
    if '/chat/completions' in clean_base_url:
        final_url = clean_base_url.replace('/chat/completions', '/audio/transcriptions')
    else:
        # Otherwise, we just append the correct path as normal.
        final_url = f"{clean_base_url}/audio/transcriptions"
    # --- END OF NEW ROBUST LOGIC ---

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        files = {"file": (filename, file_bytes, "audio/wav")}
        data = {"model": model}
        
        # Use the newly constructed final_url
        r = await client.post(
            final_url,
            headers={"Authorization": f"Bearer {api_key}"},
            files=files, 
            data=data
        )
        r.raise_for_status()
        return r.json()