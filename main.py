# main.py
import os
import json
import logging
from logging.handlers import RotatingFileHandler
from typing import Any, AsyncGenerator, Dict, List, Tuple, Optional
from pydantic import BaseModel 

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Depends, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from fastapi.exceptions import RequestValidationError
from sqlalchemy.exc import IntegrityError, MissingGreenlet
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
# --- TRACE SETUP (safe even if DATA_DIR isn't imported yet) -------------------
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Any

# Try to use DATA_DIR if it's already defined; otherwise fall back to ./data or env
try:
    BASE_DATA_DIR = DATA_DIR  # type: ignore[name-defined]
except NameError:
    BASE_DATA_DIR = Path(
        os.getenv("BUDDY_DATA_DIR", str(Path(__file__).resolve().parent / "data"))
    )

if isinstance(BASE_DATA_DIR, str):
    BASE_DATA_DIR = Path(BASE_DATA_DIR)

TRACE_DIR = BASE_DATA_DIR / "logs"
TRACE_DIR.mkdir(parents=True, exist_ok=True)

_reqlog = logging.getLogger("reqtrace")
if not _reqlog.handlers:
    fh = RotatingFileHandler(TRACE_DIR / "traffic.log", maxBytes=5_000_000, backupCount=4)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s")
    fh.setFormatter(fmt)
    _reqlog.addHandler(fh)
_reqlog.setLevel(logging.INFO)

def _trace(msg: str) -> None:
    # Always log AND print to terminal
    _reqlog.info(msg)
    print(msg)

def _redact_headers(h: Dict[str, str]) -> Dict[str, str]:
    if not isinstance(h, dict):
        return {}
    out = dict(h)
    for k in list(out.keys()):
        if k.lower() == "authorization":
            out[k] = "***"
    return out

def _preview(obj: Any, limit: int = 800) -> str:
    try:
        s = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    except Exception:
        s = str(obj)
    if len(s) > limit:
        return s[:limit] + f"... <{len(s)-limit} more chars>"
    return s
# ---------------------------------------------------------------------------

def _compose_openai_url(base_url: str, path: str = "/v1/chat/completions") -> str:
    """
    Compose a correct OpenAI-compatible URL, avoiding duplicate path segments.
    This version is robust against full URLs being used as the base_url.
    """
    b = (base_url or "").strip().rstrip("/")
    p = path.strip("/")

    # If the base_url already contains the final path segment, just return it.
    # This handles cases where the user enters the full endpoint URL.
    if "chat/completions" in b:
        return b

    # If the base URL already ends with a version suffix (e.g., /v1)
    if b.endswith("/v1") or b.endswith("/openai/v1"):
        # And the path also starts with it, strip the duplicate from the path.
        if p.startswith("v1/"):
            p = p[len("v1/"):].strip("/")
        return f"{b}/{p}"

    # If the base URL is like "https://api.groq.com/openai"
    if b.endswith("/openai"):
        # Ensure the v1 segment is present.
        if not p.startswith("v1/"):
            p = f"v1/{p}"
        return f"{b}/{p}"

    # Default case (e.g., https://api.openai.com)
    return f"{b}/{p}"

from db import (
    get_session,
    init_models,
    start_backup_task,
    stop_backup_task,
    DATA_DIR,
    IS_SQLITE,
)
from models import (
    User,
    ModelType,
    RoutePref,
    RouteKind,
    ProviderEndpoint,   # NEU: für Base-URL + API-Key
)
from security import get_current_user
from billing import (
    approx_tokens_from_text,
    charge_llm,
    charge_tts,
    charge_asr,
    log_usage,
)
from providers import (
    openai_chat_stream,  # belassen (wird hier nicht mehr benötigt, aber kompatibel)
    gemini_stream,       # weiterhin nutzbar, wir streamen aber unten via Passthrough
    tts_forward,         # forward TTS
    asr_forward,         # forward ASR
)

import httpx  # NEU: für echten SSE-Passthrough

# -----------------------------------------------------------------------------
# App + CORS + static admin
# -----------------------------------------------------------------------------

app = FastAPI(title="Buddy Universal API")

class SpeechRequest(BaseModel):
    model: str
    input: str
    voice: str | None = None


def install_error_handlers(app: FastAPI) -> None:
    """Friendly exceptions for Admin UI and API clients."""
    async def _integrity_handler(request: Request, exc: IntegrityError):
        detail = str(getattr(exc, "orig", exc))
        if "FOREIGN KEY constraint failed" in detail:
            msg = (
                "You referenced an ID that does not exist (foreign key). "
                "Create the referenced record first (e.g., create the project, then assign the user)."
            )
            code = 400
        elif "UNIQUE constraint failed" in detail:
            msg = "A record with this unique value already exists. Change the value and try again."
            code = 400
        else:
            msg = "Database constraint error."
            code = 400
        return JSONResponse(status_code=code, content={"detail": msg, "tech": detail})

    async def _validation_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "detail": exc.errors(),
                "hint": "One or more required fields are missing or have the wrong type. Please check the form values.",
            },
        )

    async def _greenlet_handler(request: Request, exc: MissingGreenlet):
        return JSONResponse(
            status_code=500,
            content={
                "detail": (
                    "Internal async database error (lazy load in wrong context). "
                    "The server uses eager loading to prevent this; if you still see it, please reload and try again."
                ),
            },
        )

    async def _generic_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal Server Error. Please retry. If it persists, check server logs for details.",
            },
        )

    app.add_exception_handler(IntegrityError, _integrity_handler)
    app.add_exception_handler(RequestValidationError, _validation_handler)
    app.add_exception_handler(MissingGreenlet, _greenlet_handler)
    app.add_exception_handler(Exception, _generic_handler)

install_error_handlers(app)

allow = [o.strip() for o in os.getenv("ALLOW_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static Admin UI (tabs, backup/restore, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

# -----------------------------------------------------------------------------
# Provider failure logging (rotating)
# -----------------------------------------------------------------------------

LOG_DIR = DATA_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
_errlog = logging.getLogger("provider_failures")
if not _errlog.handlers:
    h = RotatingFileHandler((LOG_DIR / "provider_failures.log"), maxBytes=2_000_000, backupCount=5)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s")
    h.setFormatter(fmt)
    _errlog.setLevel(logging.INFO)
    _errlog.addHandler(h)

# -----------------------------------------------------------------------------
# Lifecycle
# -----------------------------------------------------------------------------

@app.on_event("startup")
async def startup() -> None:
    await init_models()
    # periodic sqlite snapshot every 10 min, keep last 10
    start_backup_task(interval_sec=600, keep=10)

@app.on_event("shutdown")
async def shutdown() -> None:
    await stop_backup_task()

# -----------------------------------------------------------------------------
# Health
# -----------------------------------------------------------------------------

@app.get("/healthz")
async def healthz() -> Dict[str, Any]:
    return {"ok": True, "data_dir": str(DATA_DIR), "sqlite": IS_SQLITE}

# -----------------------------------------------------------------------------
# Routing helpers
# -----------------------------------------------------------------------------

async def _ordered_routes(session: AsyncSession, kind: RouteKind) -> List[RoutePref]:
    q = await session.execute(
        select(RoutePref).where(RoutePref.kind == kind, RoutePref.enabled == True).order_by(RoutePref.priority.asc(), RoutePref.id.asc())
    )
    return list(q.scalars().all())

async def _provider_map(session: AsyncSession) -> Dict[str, ProviderEndpoint]:
    q = await session.execute(select(ProviderEndpoint))
    items = q.scalars().all()
    return {p.name: p for p in items}

# -----------------------------------------------------------------------------
# SSE/JSON Hilfsfunktionen für Passthrough
# -----------------------------------------------------------------------------

def _looks_like_vlm(messages: List[Dict[str, Any]]) -> bool:
    try:
        for m in messages or []:
            c = m.get("content")
            if isinstance(c, list):
                for p in c:
                    if isinstance(p, dict) and (p.get("type") in ("image_url", "pdf") or "image_url" in p):
                        return True
    except Exception:
        pass
    return False

def _extract_assistant_text(payload: Dict[str, Any]) -> str:
    # OpenAI
    try:
        choices = payload.get("choices") or []
        if choices:
            c0 = choices[0]
            if isinstance(c0.get("message", {}).get("content"), str):
                return c0["message"]["content"]
            if isinstance(c0.get("text"), str):
                return c0["text"]
            if isinstance(c0.get("delta", {}).get("content"), str):
                return c0["delta"]["content"]
    except Exception:
        pass
    # Gemini
    try:
        cands = payload.get("candidates") or []
        if cands:
            parts = (cands[0].get("content") or {}).get("parts") or []
            txt = "".join([p.get("text", "") for p in parts if isinstance(p, dict)])
            if txt:
                return txt
    except Exception:
        pass
    # Fallback
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    if isinstance(payload.get("content"), str):
        return payload["content"]
    if isinstance(payload.get("content"), list):
        txt = "".join([str(p.get("text", "")) for p in payload["content"] if isinstance(p, dict)])
        return txt
    return ""
async def _json_proxy(url: str, headers: Dict[str, str], body: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
    _trace(f"[UPSTREAM][JSON REQ] POST {url}")
    _trace(f"[UPSTREAM][JSON REQ HEADERS] {_redact_headers(headers)}")
    _trace(f"[UPSTREAM][JSON REQ BODY] {_preview(body)}")
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        r = await client.post(url, headers=headers, json=body)
        status = r.status_code
        ctype = r.headers.get("content-type", "")
        raw = await r.aread()
        _trace(f"[UPSTREAM][JSON RESP] status={status} content-type={ctype}")
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception:
            data = {"raw": raw.decode(errors="ignore")}
        _trace(f"[UPSTREAM][JSON RESP BODY PREVIEW] {_preview(data)}")
        return status, data

async def _sse_passthrough_and_bill(
    url: str,
    headers: Dict[str, str],
    body: Dict[str, Any],
) -> AsyncGenerator[bytes, None]:
    """1:1 SSE passthrough with full, explicit tracing."""
    enc_done = b"data: [DONE]\n\n"

    _trace(f"[UPSTREAM][SSE REQ] POST {url}")
    _trace(f"[UPSTREAM][SSE REQ HEADERS] {_redact_headers(headers)}")
    _trace(f"[UPSTREAM][SSE REQ BODY] {_preview(body)}")

    async with httpx.AsyncClient(timeout=None, follow_redirects=True) as client:
        async with client.stream("POST", url, headers=headers, json=body) as resp:
            status = resp.status_code
            ctype = resp.headers.get("content-type", "")
            _trace(f"[UPSTREAM][SSE RESP] status={status} content-type={ctype}")

            if status >= 400:
                blob = await resp.aread()
                msg = blob.decode(errors="ignore")[:800]
                _trace(f"[UPSTREAM][SSE RESP ERROR BODY] {_preview(msg)}")
                raise HTTPException(status_code=502, detail=f"Upstream {status}: {msg}")

            if "text/event-stream" in (ctype or "").lower():
                buf = ""
                async for chunk in resp.aiter_raw():
                    if not chunk:
                        continue
                    # Log (truncated) each SSE line and count tokens for billing
                    try:
                        s = chunk.decode("utf-8", errors="ignore")
                        buf += s
                        lines = buf.split("\n")
                        buf = lines.pop() or ""
                        for line in lines:
                            if line.startswith("data: "):
                                if line == "data: [DONE]":
                                    _trace("[UPSTREAM][SSE <<] [DONE]")
                                else:
                                    _trace(f"[UPSTREAM][SSE <<] {line[:200]}")
                                    try:
                                        j = json.loads(line[6:])
                                        delta = (j.get("choices") or [{}])[0].get("delta", {})
                                        t = delta.get("content")
                                        if t:
                                            _billing_ctx["out_tokens"] += approx_tokens_from_text(t)
                                    except Exception:
                                        pass
                    except Exception:
                        pass
                    _trace(f"[CLIENT][SSE >>] {len(chunk)} bytes")
                    yield chunk
                _trace("[CLIENT][SSE >>] sending synthetic [DONE]")
                yield enc_done
                return

            # No SSE ? log JSON fallback and produce minimal SSE to client
            raw = await resp.aread()
            try:
                data = json.loads(raw.decode("utf-8"))
                text = _extract_assistant_text(data)
                _trace(f"[UPSTREAM][JSON FALLBACK BODY PREVIEW] {_preview(data)}")
            except Exception:
                text = raw.decode(errors="ignore")
                _trace(f"[UPSTREAM][JSON FALLBACK RAW PREVIEW] {_preview(text)}")

            start = f"data: {json.dumps({'choices':[{'delta':{'role':'assistant'}}]})}\n\n".encode()
            yield start
            if text:
                _billing_ctx["out_tokens"] += approx_tokens_from_text(text)
                body_bytes = f"data: {json.dumps({'choices':[{'delta':{'content':text}}]})}\n\n".encode()
                _trace(f"[CLIENT][SSE >> MINI] {len(body_bytes)} bytes (content)")
                yield body_bytes
            _trace("[CLIENT][SSE >> MINI] [DONE]")
            yield enc_done
            return


# Eine sehr kleine "Kontextbox" für die laufende Anfrage (nur innerhalb des Request-Tasks verwendet)
_billing_ctx: Dict[str, int] = {"out_tokens": 0}

# -----------------------------------------------------------------------------
# Chat Completions (LLM/VLM) – SSE streaming with provider failover
# -----------------------------------------------------------------------------

@app.post("/v1/chat/completions")
async def chat_completions(
    payload: dict,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """
    OpenAI-compatible chat with **priority failover** across providers
    based on RoutePref (VLM if the message contains images/PDFs, else LLM).
    If payload.model is present, it is tried first; then RoutePref defaults.
    Default behaviour: stream passthrough (SSE). If stream=false ? JSON.
    """
    requested_model = (payload.get("model") or "").strip()
    wants_stream: bool = bool(payload.get("stream", True))

    # VLM wenn in Messages Bild/PDF vorkommt
    messages = payload.get("messages") or []
    is_vlm = _looks_like_vlm(messages)

    route_kind = RouteKind.VLM if is_vlm else RouteKind.LLM
    routes = await _ordered_routes(session, route_kind)
    if not routes and route_kind == RouteKind.VLM:
        # fallback to LLM if no VLM route configured
        routes = await _ordered_routes(session, RouteKind.LLM)


    # --- PDF detection + Gemini-only narrowing (NEW) ---
    def _has_pdf(msgs: list[dict]) -> bool:
        try:
            for m in msgs:
                c = m.get("content")
                if isinstance(c, list):
                    for p in c:
                        if isinstance(p, dict) and p.get("type") == "pdf":
                            return True
        except Exception:
            pass
        return False

    has_pdf = _has_pdf(messages)
    if has_pdf:
        # Only Gemini providers for PDFs
        routes = [r for r in routes if r.provider.lower() == "gemini"]
        if not routes:
            raise HTTPException(
                400,
                "PDF input requires a Gemini VLM route. Configure provider 'gemini' (Google API key) in Admin ? Routes."
            )


    # Token-Schaetzung für Input (nur User-Text)
    user_texts: List[str] = []
    for m in messages:
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str):
                user_texts.append(c)
            elif isinstance(c, list):
                for p in c:
                    if p.get("type") == "text":
                        user_texts.append(p.get("text", ""))
    est_in = approx_tokens_from_text(" ".join(user_texts))

    # Kandidatenliste: requested_model zuerst (über Provider in Prioritaet), dann Default
    candidates: List[Tuple[str, str]] = []
    if requested_model:
        for r in routes:
            candidates.append((r.provider, requested_model))
    for r in routes:
        candidates.append((r.provider, r.model))

    # de-duplizieren
    seen = set()
    ordered: List[Tuple[str, str]] = []
    for t in candidates:
        if t not in seen:
            seen.add(t)
            ordered.append(t)

    provs = await _provider_map(session)

    # --- OpenAI?Gemini adapters (NEW) -------------------------------------------
    def _data_url_to_inline_data(url: str) -> dict | None:
        # expects data:...;base64,XXXX
        if not (isinstance(url, str) and url.startswith("data:") and ";base64," in url):
            return None
        head, b64 = url.split(",", 1)
        mime = head[5:head.find(";")] or "application/octet-stream"
        return {"inlineData": {"mimeType": mime, "data": b64}}

    def _openai_to_gemini(msgs: list[dict]) -> tuple[list, dict | None]:
        """
        Convert OpenAI chat 'messages' into Gemini {contents, systemInstruction}.
        - text parts -> {text: "..."}
        - image_url (data URL) -> {inlineData: {mimeType, data}}
        - pdf parts -> {inlineData: {mimeType: "application/pdf", data}}
        - roles: user/assistant/system -> user/model/systemInstruction
        """
        contents: list = []
        system_instruction: dict | None = None

        for m in msgs or []:
            role = m.get("role", "user")
            gr = "user" if role in ("user", "system") else "model"
            parts = []

            c = m.get("content")
            if isinstance(c, str):
                if c.strip():
                    parts.append({"text": c})
            elif isinstance(c, list):
                for p in c:
                    if not isinstance(p, dict):
                        continue
                    t = p.get("type")
                    if t == "text":
                        txt = p.get("text", "")
                        if txt.strip():
                            parts.append({"text": txt})
                    elif t == "image_url":
                        img = p.get("image_url")
                        url = img.get("url") if isinstance(img, dict) else (img if isinstance(img, str) else "")
                        idata = _data_url_to_inline_data(url)
                        if idata:
                            parts.append(idata)
                    elif t == "pdf":
                        pdata = p.get("data", "")
                        if pdata:
                            parts.append({"inlineData": {
                                "mimeType": p.get("mime_type") or "application/pdf",
                                "data": pdata
                            }})

            if role == "system":
                # **Match frontend**: role + parts structure
                text = "".join([pp.get("text", "") for pp in parts if isinstance(pp, dict) and "text" in pp]) or ""
                if text:
                    system_instruction = {
                        "role": "system",
                        "parts": [{"text": text}],
                    }
                continue

            if parts:
                contents.append({"role": gr, "parts": parts})

        return contents, system_instruction


    def _collect_texts(obj) -> str:
        """Recursively collect all 'text' fields from a Gemini response tree."""
        out = []
        def walk(o):
            if isinstance(o, dict):
                if "text" in o and isinstance(o["text"], str):
                    out.append(o["text"])
                for v in o.values():
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
        walk(obj)
        return "".join(out)



    def _collect_texts(obj) -> str:
        """Recursively collect all 'text' fields from a Gemini response tree."""
        out = []
        def walk(o):
            if isinstance(o, dict):
                if "text" in o and isinstance(o["text"], str):
                    out.append(o["text"])
                for v in o.values():
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
        walk(obj)
        return "".join(out)
    async def do_json_call(provider: str, model: str) -> Tuple[int, Dict[str, Any]]:
        """
        JSON (non-stream) call.
        - For PDFs routed to Gemini: translate OpenAI messages -> Gemini body and POST :generateContent
        - Otherwise: standard OpenAI-compatible JSON passthrough to /v1/chat/completions
        """
        # PDF ? Gemini JSON branch
        if has_pdf:
            pe = provs.get(provider)
            base = (pe.base_url or "").lower() if pe else ""
            if (provider.lower() == "gemini") or ("generativelanguage.googleapis.com" in base):
                if not pe or not pe.api_key:
                    raise HTTPException(503, "Provider 'gemini' incomplete configuration")
                final_model = (model or routes[0].model or "gemini-2.5-pro")

                contents, system_instruction = _openai_to_gemini(messages)
                body: Dict[str, Any] = {
                    "contents": contents,
                    # **Mirror frontend**:
                    "generationConfig": {"thinkingConfig": {"thinkingBudget": -1}},
                    "tools": [{"googleSearch": {}}],
                }
                temp = payload.get("temperature")
                if isinstance(temp, (int, float)):
                    body.setdefault("generationConfig", {})["temperature"] = temp
                if system_instruction:
                    body["systemInstruction"] = system_instruction

                url = f"https://generativelanguage.googleapis.com/v1beta/models/{final_model}:generateContent?key={pe.api_key}"
                _trace(f"[ROUTE][JSON TRY] gemini:{final_model} -> {url}")

                async with httpx.AsyncClient(timeout=120.0) as client:
                    r = await client.post(
                        url,
                        headers={"Content-Type": "application/json", "Accept": "application/json"},
                        json=body
                    )
                    status = r.status_code
                    try:
                        data = r.json()
                    except Exception:
                        data = {"raw": (await r.aread()).decode(errors="ignore")}
                    return status, data

        # Default OpenAI-compatible JSON passthrough
        pe = provs.get(provider)
        if not pe or not pe.base_url or not pe.api_key:
            raise HTTPException(503, f"Provider '{provider}' incomplete configuration")
        url = _compose_openai_url(pe.base_url, "/v1/chat/completions")
        headers = {"Authorization": f"Bearer {pe.api_key}", "Content-Type": "application/json"}
        body = dict(payload)
        body["model"] = model or payload.get("model") or "auto"
        body["stream"] = False
        _trace(f"[ROUTE][JSON TRY] {provider}:{body['model']} -> {url}")
        return await _json_proxy(url, headers, body)

    async def do_stream_call(provider: str, model: str) -> AsyncGenerator[bytes, None]:
        """
        Streaming (SSE) call.
        - For PDFs routed to Gemini: translate to Gemini body and POST :streamGenerateContent?alt=sse
          Then bridge Gemini SSE into OpenAI-style delta events for the client.
        - Otherwise: standard OpenAI-compatible SSE passthrough.
        """
        # PDF ? Gemini SSE branch
        if has_pdf:
            pe = provs.get(provider)
            base = (pe.base_url or "").lower() if pe else ""
            if (provider.lower() == "gemini") or ("generativelanguage.googleapis.com" in base):
                if not pe or not pe.api_key:
                    raise HTTPException(503, "Provider 'gemini' incomplete configuration")
                final_model = (model or routes[0].model or "gemini-2.5-pro")

                contents, system_instruction = _openai_to_gemini(messages)
                body: Dict[str, Any] = {
                    "contents": contents,
                    # **Mirror frontend**:
                    "generationConfig": {"thinkingConfig": {"thinkingBudget": -1}},
                    "tools": [{"googleSearch": {}}],
                }
                temp = payload.get("temperature")
                if isinstance(temp, (int, float)):
                    body.setdefault("generationConfig", {})["temperature"] = temp
                if system_instruction:
                    body["systemInstruction"] = system_instruction

                url = f"https://generativelanguage.googleapis.com/v1beta/models/{final_model}:streamGenerateContent?alt=sse&key={pe.api_key}"
                _trace(f"[ROUTE][SSE TRY] gemini:{final_model} -> {url}")

                async def gen() -> AsyncGenerator[bytes, None]:
                    # initial assistant-role delta
                    yield f"data: {json.dumps({'choices':[{'delta':{'role':'assistant'}}]})}\n\n".encode()
                    async with httpx.AsyncClient(timeout=None) as client:
                        async with client.stream(
                            "POST",
                            url,
                            headers={
                                "Content-Type": "application/json",
                                "Accept": "text/event-stream",
                            },
                            json=body
                        ) as resp:
                            if resp.status_code >= 400:
                                blob = await resp.aread()
                                msg = blob.decode(errors="ignore")[:800]
                                raise HTTPException(status_code=502, detail=f"Gemini upstream {resp.status_code}: {msg}")

                            async for line in resp.aiter_lines():
                                if not line or not line.startswith("data: "):
                                    continue
                                if line.strip() == "data: [DONE]":
                                    continue
                                try:
                                    j = json.loads(line[6:])
                                except Exception:
                                    continue
                                text = _collect_texts(j)
                                if text:
                                    _billing_ctx["out_tokens"] += approx_tokens_from_text(text)
                                    yield f"data: {json.dumps({'choices':[{'delta':{'content':text}}]})}\n\n".encode()
                    yield b"data: [DONE]\n\n"

                return gen()

        # Default OpenAI-compatible SSE passthrough
        pe = provs.get(provider)
        if not pe or not pe.base_url or not pe.api_key:
            raise HTTPException(503, f"Provider '{provider}' incomplete configuration")
        url = _compose_openai_url(pe.base_url, "/v1/chat/completions")
        headers = {"Authorization": f"Bearer {pe.api_key}", "Content-Type": "application/json"}
        body = dict(payload)
        body["model"] = model or payload.get("model") or "auto"
        body["stream"] = True
        _trace(f"[ROUTE][SSE TRY] {provider}:{body['model']} -> {url}")
        return _sse_passthrough_and_bill(url, headers, body)



    last_error: Optional[Exception] = None
    tried: List[str] = []

    # --- STREAM-PFAD ---
    if wants_stream:
        async def stream_wrapper() -> AsyncGenerator[bytes, None]:
            # lokale Abrechnung
            _billing_ctx["out_tokens"] = 0
            for provider, model in ordered:
                tried.append(f"{provider}:{model}")
                try:
                    agen = await do_stream_call(provider, model)
                    # Erfolgreich: 1:1 SSE zu Client streamen
                    async for chunk in agen:
                        yield chunk
                    # Wenn wir hier sind ? Stream beendet, Abrechnung/Log unten
                    last_error = None
                    break
                except Exception as e:
                    _errlog.info(f"chat stream failed provider={provider} model={model} err={e}")
                    last_error = e
                    continue

            # Billing + Usage
            cost = await charge_llm(
                session,
                user,
                model=requested_model or (ordered[0][1] if ordered else "unknown"),
                provider=(ordered[0][0] if ordered else "unknown"),
                input_tokens=est_in,
                output_tokens=_billing_ctx["out_tokens"],
            )
            await log_usage(
                session,
                user,
                model=requested_model or "auto",
                provider="auto",
                model_type=ModelType.VLM if is_vlm else ModelType.LLM,
                input_count=est_in,
                output_count=_billing_ctx["out_tokens"],
                billed_credits=cost,
                response_meta={"tried": tried, "error": str(last_error) if last_error else None},
            )
            await session.commit()

            if last_error:
                errline = f'data: {json.dumps({"error": {"message": "All providers failed. See server logs."}})}\n\n'
                yield errline.encode()
            # DONE wird im Passthrough gesendet; falls alle fehlschlugen, schicken wir hier kein extra DONE mehr.

        return StreamingResponse(stream_wrapper(), media_type="text/event-stream")

    # --- NON-STREAM-PFAD (JSON) ---
    # wir versuchen nacheinander Provider; der erste erfolgreiche JSON-Call wird geliefert
    for provider, model in ordered:
        tried.append(f"{provider}:{model}")
        try:
            status, data = await do_json_call(provider, model)
            # Output-Token grob schaetzen
            out_text = _extract_assistant_text(data)
            out_tokens = approx_tokens_from_text(out_text)

            cost = await charge_llm(
                session,
                user,
                model=requested_model or model or "auto",
                provider=provider,
                input_tokens=est_in,
                output_tokens=out_tokens,
            )
            await log_usage(
                session,
                user,
                model=requested_model or model or "auto",
                provider=provider,
                model_type=ModelType.VLM if is_vlm else ModelType.LLM,
                input_count=est_in,
                output_count=out_tokens,
                billed_credits=cost,
                response_meta={"tried": tried, "json": True},
            )
            await session.commit()
            return JSONResponse(data, status_code=status)
        except Exception as e:
            _errlog.info(f"chat json failed provider={provider} model={model} err={e}")
            last_error = e
            continue

    raise HTTPException(status_code=503, detail=f"Providers unavailable. Last error: {last_error}")

# -----------------------------------------------------------------------------
# Text-to-Speech with priority failover
# -----------------------------------------------------------------------------
@app.post("/v1/audio/speech")
async def audio_speech(
    request: SpeechRequest, # <-- USE THE NEW SpeechRequest class
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    # The 'model', 'input', and 'voice' are now accessed from the request object
    model = request.model
    input_text = request.input

    routes = await _ordered_routes(session, RouteKind.TTS)
    last_error = None
    for r in routes:
        try:
            # Use the model from the request, or the route's default
            final_model = r.model or model
            cost = await charge_tts(session, user, model=final_model, provider=r.provider, characters=len(input_text or ""))
            
            # Pass the correct data to the forwarder
            data = await tts_forward(session, final_model, input_text, provider=r.provider)
            
            await log_usage(session, user, model=final_model, provider=r.provider, model_type=ModelType.TTS,
                            input_count=len(input_text or ""), output_count=0, billed_credits=cost)
            await session.commit()
            return StreamingResponse(iter([data]), media_type="audio/mpeg")
        except Exception as e:
            _errlog.info(f"tts route failed provider={r.provider} model={r.model} err={e}")
            last_error = e
            continue
    raise HTTPException(status_code=503, detail=f"TTS providers unavailable. Last error: {last_error}")



# -----------------------------------------------------------------------------
# Speech-to-Text with priority failover
# -----------------------------------------------------------------------------

@app.post("/v1/audio/transcriptions")
async def audio_transcriptions(
    file: UploadFile = File(...),
    model: str = Form(...),
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    routes = await _ordered_routes(session, RouteKind.ASR)
    blob = await file.read()
    last_error = None
    for r in routes:
        try:
            asr_json = await asr_forward(session, r.model or model, blob, file.filename or "audio.wav", provider=r.provider)
            seconds = max(1, int(len(blob) / (16000 * 2)))  # 16kHz * 16-bit mono approx
            cost = await charge_asr(session, user, model=r.model or model, provider=r.provider, seconds=seconds)
            await log_usage(session, user, model=r.model or model, provider=r.provider, model_type=ModelType.ASR,
                            input_count=seconds, output_count=0, billed_credits=cost)
            await session.commit()
            return JSONResponse(asr_json)
        except Exception as e:
            _errlog.info(f"asr route failed provider={r.provider} model={r.model} err={e}")
            last_error = e
            continue
    raise HTTPException(status_code=503, detail=f"ASR providers unavailable. Last error: {last_error}")
