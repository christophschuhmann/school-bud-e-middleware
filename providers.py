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

async def tts_forward(session: AsyncSession, model: str, text: str, provider: str) -> bytes:
    # Get the provider's base_url and api_key from the database / .env
    prov_details = await _get_provider(session, provider)
    base_url = prov_details.get("base_url")
    api_key = prov_details.get("api_key")

    if not base_url or not api_key:
        raise ValueError(f"Provider '{provider}' is missing base_url or api_key in configuration.")

    # --- helpers -------------------------------------------------------------
    def _parse_model_options(raw: str) -> dict:
        """
        Allow passing options via the Routes 'model' column, e.g.:
          - "auto"
          - "en-US-Chirp3-HD-Achernar"
          - "voice=en-US-Wavenet-D"
          - "voice=de-DE-Chirp3-HD-Achernar;lang=de-DE;format=mp3"
        Returns a dict possibly containing: voice_name, language_code, audio_format
        """
        out = {}
        if not raw:
            return out
        s = raw.strip()
        if s.lower() == "auto":
            return out
        # key=value;key=value ... OR just a bare voice name
        if ("=" not in s) and (";" not in s):
            out["voice_name"] = s
            return out
        for part in s.split(";"):
            if not part.strip():
                continue
            if "=" in part:
                k, v = part.split("=", 1)
                k = k.strip().lower()
                v = v.strip()
                if k in ("voice", "voice_name"):
                    out["voice_name"] = v
                elif k in ("lang", "language", "language_code"):
                    out["language_code"] = v
                elif k in ("format", "audio_format"):
                    out["audio_format"] = v
        return out

    clean_base = base_url.rstrip("/")

    # ---------------- FISH AUDIO (unchanged) --------------------------------
    if "fish.audio" in base_url or "fish" in provider.lower():
        payload = {
            "text": text,
            "reference_id": model,   # you already use model to carry reference_id
            "normalize": True,
            "format": "mp3",
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        final_url = clean_base  # Fish uses the base URL directly

    # ---------------- VERTEX PROXY (new branch) -----------------------------
    # Detect by provider name OR by your proxy base_url (e.g. http://localhost:8001/v1)
    elif "vertex" in provider.lower() or "localhost:8001" in clean_base:
        # --- Vertex proxy expects: {"text", optional "voice_name", optional "language_code", "audio_format"} ---
        opts = _parse_model_options(model or "")

        voice_name = opts.get("voice_name")
        language_code = opts.get("language_code")  # user may omit this – we’ll infer from voice_name

        # If user passed a full Google voice name (e.g., "de-DE-Chirp3-HD-Achernar"
        # or "en-US-Wavenet-D") but NO language_code, auto-derive it from the prefix.
        if voice_name and not language_code:
            import re
            m = re.match(r"^([a-z]{2,3}-[A-Z]{2})-", voice_name)
            if m:
                language_code = m.group(1)

        payload = {
            "text": text,
            **({"voice_name": voice_name} if voice_name else {}),
            **({"language_code": language_code} if language_code else {}),  # leave out ? proxy will detect from text
            "audio_format": opts.get("audio_format", "mp3"),
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        final_url = f"{clean_base}/audio/speech"


    # ---------------- DEFAULT (OpenAI-compatible) ---------------------------
    else:
        payload = {
            "model": model,
            "input": text,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        final_url = f"{clean_base}/audio/speech"

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        r = await client.post(final_url, headers=headers, json=payload)
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