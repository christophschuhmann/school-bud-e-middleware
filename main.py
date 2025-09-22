# -*- coding: utf-8 -*-
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
import base64  

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
    ProviderEndpoint,   # for Base-URL + API-Key
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
    openai_chat_stream,  # (kept for compatibility)
    gemini_stream,       # still available; we stream via passthrough below
    tts_forward,         # forward TTS
    asr_forward,         # forward ASR
)

import httpx  # for real SSE passthrough

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
# SSE/JSON Helpers for passthrough
# -----------------------------------------------------------------------------
def _looks_like_vlm(messages: List[Dict[str, Any]]) -> bool:
    try:
        for m in messages or []:
            c = m.get("content")
            if isinstance(c, list):
                for p in c:
                    if not isinstance(p, dict):
                        continue
                    t = (p.get("type") or "").lower()
                    mime = (p.get("mime_type") or "").lower()
                    if t in ("image_url", "pdf"):
                        return True
                    # NEW: treat PDFs/images sent as input_file/file as VLM
                    if t in ("input_file", "file") and (
                        mime.startswith("image/") or "pdf" in mime or mime == "application/pdf"
                    ):
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


# A tiny "billing context" for the running request (used only inside request task)
_billing_ctx: Dict[str, int] = {"out_tokens": 0}

def _gemini_capable_vlm(routes):
    """
    Accept Gemini via provider 'gemini' OR 'vertex' (when model is gemini-*) for PDF.
    """
    out = []
    for r in routes:
        prov = (r.provider or "").lower()
        model = (r.model or "").lower()
        if prov == "gemini":
            out.append(r)
        elif prov == "vertex" and model.startswith("gemini-"):
            out.append(r)
    return out


# -----------------------------------------------------------------------------
# Chat Completions (LLM/VLM)  SSE streaming with provider failover
# -----------------------------------------------------------------------------
@app.post("/v1/chat/completions")
async def chat_completions(
    payload: dict,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """
    OpenAI-compatible chat with priority/failover across providers via RoutePref.

    Accurate usage:
    - OpenAI-compatible:
        * Non-stream: read response["usage"]
        * Stream: send stream_options.include_usage = true and parse final SSE 'usage' event
    - Gemini:
        * Non-stream: read response["usageMetadata"]
        * Stream: parse final SSE event with 'usageMetadata'
        * Optional preflight :countTokens to get per-modality counts (image tokens, etc.)

    Routing behavior is unchanged (PDF -> Gemini adapter; image/text -> OpenAI-compat).
    """
    requested_model = (payload.get("model") or "").strip()
    wants_stream: bool = bool(payload.get("stream", True))

    # Messages & kind detection
    messages = payload.get("messages") or []

    # === TRACE: incoming payload summary (no secrets) ===
    def _summarize_messages(msgs: list[dict]) -> list[dict]:
        out = []
        for m in msgs or []:
            item = {"role": m.get("role"), "string_len": None, "parts": []}
            c = m.get("content")
            if isinstance(c, str):
                item["string_len"] = len(c)
            elif isinstance(c, list):
                for p in c:
                    if not isinstance(p, dict):
                        continue
                    t = (p.get("type") or "").lower()
                    mime = (p.get("mime_type") or "").lower()
                    data_len = len(p.get("data") or "") if isinstance(p.get("data"), str) else None
                    item["parts"].append({
                        "type": t,
                        "mime": mime,
                        "has_data": isinstance(p.get("data"), (str, bytes, bytearray)),
                        "data_len": data_len,
                        "has_image_url": bool(p.get("image_url")),
                        "image_url_is_dataurl": isinstance(p.get("image_url"), dict) and str(p.get("image_url", {}).get("url", "")).startswith("data:"),
                    })
            out.append(item)
        return out

    _trace("[CHAT] incoming model=%s stream=%s" % (payload.get("model"), payload.get("stream", True)))
    _trace("[CHAT] messages summary: " + _preview(_summarize_messages(messages)))

    is_vlm = _looks_like_vlm(messages)

    route_kind = RouteKind.VLM if is_vlm else RouteKind.LLM
    routes = await _ordered_routes(session, route_kind)
    if not routes and route_kind == RouteKind.VLM:
        routes = await _ordered_routes(session, RouteKind.LLM)

    _trace(f"[ROUTE] _looks_like_vlm={is_vlm} -> route_kind={getattr(route_kind,'name',route_kind)}")

    # Detect PDFs specifically (forces Gemini provider)
    def _has_pdf(messages: list[dict]) -> bool:
          for m in messages or []:
              c = m.get("content")
              if isinstance(c, list):
                  for p in c:
                      if not isinstance(p, dict): 
                          continue
                      t = (p.get("type") or "").lower()
                      mime = (p.get("mime_type") or "").lower()
                      if t == "pdf":
                          return True
                      if t in {"input_file", "file"} and ("pdf" in mime or mime == "application/pdf"):
                          return True
          return False

    has_pdf = _has_pdf(messages)
    _trace(f"[ROUTE] _has_pdf={has_pdf}")
    _trace(f"[ROUTE] initial routes found={len(routes)} -> {[f'{r.provider}:{r.model}' for r in routes]}")

    if has_pdf:

        # Ensure we have VLM routes available if we started on LLM
        if route_kind == RouteKind.LLM:
          vlm_routes = await _ordered_routes(session, RouteKind.VLM)
          _trace(f"[ROUTE] swapping to VLM routes for PDF; VLM routes={len(vlm_routes)} -> {[f'{r.provider}:{r.model}' for r in vlm_routes]}")
          # Prefer VLM routes for PDFs; if none defined, keep current 'routes'
          if vlm_routes:
            routes = vlm_routes


        # accept either provider='gemini' OR provider='vertex' with a gemini-* model
        before = [f"{r.provider}:{r.model}" for r in routes]
        routes = _gemini_capable_vlm(routes)
        after = [f"{r.provider}:{r.model}" for r in routes]
        _trace(f"[ROUTE] _gemini_capable_vlm filtered {before} -> {after}")
        if not routes:
            raise HTTPException(
                400,
                ("PDF input requires a Gemini-capable VLM route. "
                "Add provider 'gemini' OR 'vertex' with a gemini-* model in Admin ? Routes.")
            )

    # Build candidate list: requested_model first (if not 'auto'), then route defaults
    candidates: List[Tuple[str, str]] = []
    if requested_model and requested_model.lower() != "auto":  # <<< FIX: skip 'auto' to honor priorities
        for r in routes:
            candidates.append((r.provider, requested_model))
    for r in routes:
        candidates.append((r.provider, r.model))
    # de-dup in order
    seen = set(); ordered: List[Tuple[str, str]] = []
    for t in candidates:
        if t not in seen:
            seen.add(t); ordered.append(t)

    provs = await _provider_map(session)
    _trace(f"[ROUTE] ordered candidates={ordered}")
    _trace(f"[PROV] provider map: {list(provs.keys())}")

    # ------------- token helpers (still used as fallback) ----------------
    def approx_tokens_from_text(text: str) -> int:
        return max(1, int(len(text) / 4) + text.count(" "))

    def _extract_assistant_text(obj: Any) -> str:
        # OpenAI JSON or similar: choices[].message/content
        try:
            ch = (obj.get("choices") or [{}])[0]
            msg = ch.get("message") or {}
            cnt = msg.get("content")
            if isinstance(cnt, str): return cnt
            if isinstance(cnt, list):
                return "".join([p.get("text","") for p in cnt if isinstance(p, dict) and p.get("type")=="text"])
        except Exception:
            pass
        return ""

    # ---------- Gemini helpers (adapter + usage) ----------
    def _data_url_to_inline_data(url: str) -> dict | None:
        try:
            if not isinstance(url, str) or not url.startswith("data:") or ";base64," not in url:
                return None
            head, b64 = url.split(",", 1)
            mime = head[5:head.find(";")] or "application/octet-stream"
            return {"inlineData": {"mimeType": mime, "data": b64}}
        except Exception:
            return None

    def _openai_to_gemini(msgs: list[dict]) -> tuple[list, dict | None]:
        """
        Convert OpenAI-style messages -> Gemini 'contents' + optional systemInstruction.
        Supports: text, image_url data:URL, and base64 PDFs via input_file/file.
        """
        contents: list = []
        system_instruction: dict | None = None

        def _data_url_to_inline_data(url: str) -> dict | None:
            try:
                if not isinstance(url, str) or not url.startswith("data:") or ";base64," not in url:
                    return None
                head, b64 = url.split(",", 1)
                mime = head[5:head.find(";")] or "application/octet-stream"
                return {"inlineData": {"mimeType": mime, "data": b64}}
            except Exception:
                return None

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
                    t = (p.get("type") or "").lower()

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
                        b64 = p.get("data") or ""
                        if b64:
                            parts.append({"inlineData": {
                                "mimeType": p.get("mime_type") or "application/pdf",
                                "data": b64 if isinstance(b64, str) else base64.b64encode(b64).decode("utf-8"),
                            }})

                    elif t in ("input_file", "file"):
                        b64 = p.get("data", "")
                        mime = p.get("mime_type") or "application/octet-stream"
                        if isinstance(b64, (bytes, bytearray)):
                            b64 = base64.b64encode(b64).decode("utf-8")
                        if isinstance(b64, str) and b64:
                            parts.append({"inlineData": {"mimeType": mime, "data": b64}})

            if role == "system" and parts:
                # Gemini supports a top-level systemInstruction; we fold system text there.
                # If both system & user text exist, we still include everything as user parts too.
                sys_txt = "\n".join([pp.get("text","") for pp in parts if "text" in pp]).strip()
                if sys_txt:
                    system_instruction = {"role": "system", "parts": [{"text": sys_txt}]}
            if parts:
                contents.append({"role": gr, "parts": parts})

        # TRACE: concise summary of what we will send to Gemini
        try:
            def _flat_summary(contents_in):
                s = []
                for c in contents_in:
                    role = c.get("role")
                    parts = c.get("parts") or []
                    s_parts = []
                    for p in parts:
                        if "text" in p:
                            txt = p["text"]
                            s_parts.append({"text_len": len(txt)})
                        elif "inlineData" in p:
                            idt = p["inlineData"]
                            s_parts.append({"inlineData": {"mime": idt.get("mimeType"), "data_len": len(idt.get("data") or "")}})
                    s.append({"role": role, "parts": s_parts})
                return s
            _trace("[GEMINI] contents summary: " + _preview(_flat_summary(contents)))
            if system_instruction:
                _trace("[GEMINI] systemInstruction len=%d" % sum(len(pp.get("text","")) for pp in (system_instruction.get("parts") or [])))
        except Exception:
            pass

        return contents, system_instruction


    def _collect_texts(obj: Any) -> str:
        out: List[str] = []
        def walk(o):
            if isinstance(o, dict):
                if "text" in o and isinstance(o["text"], str): out.append(o["text"])
                for v in o.values(): walk(v)
            elif isinstance(o, list):
                for v in o: walk(v)
        walk(obj)
        return "".join(out)

    # --- helper: rewrite {type:"pdf"} -> {type:"input_file"} for OpenAI-style proxies ---
    def _rewrite_pdf_to_input_file(msgs: list[dict]) -> list[dict]:
        out_msgs: List[dict] = []
        for m in msgs or []:
            c = m.get("content")
            if isinstance(c, list):
                new_parts = []
                for p in c:
                    if isinstance(p, dict) and (p.get("type") or "").lower() == "pdf":
                        b64 = p.get("data")
                        mime = p.get("mime_type") or "application/pdf"
                        # Normalize bytes to b64 if needed
                        if isinstance(b64, (bytes, bytearray)):
                            b64 = base64.b64encode(b64).decode("utf-8")
                        new_parts.append({
                            "type": "input_file",
                            "mime_type": mime,
                            "data": b64,
                        })
                    else:
                        new_parts.append(p)
                nm = dict(m)
                nm["content"] = new_parts
                out_msgs.append(nm)
            else:
                out_msgs.append(m)
        return out_msgs

    # ---------- OpenAI-compatible STREAM proxy with usage (revised) ----------
    async def _openai_stream_with_usage(
        url: str, headers: dict, body: dict
    ) -> Tuple[AsyncGenerator[bytes, None], dict]:
        """
        Streams upstream SSE back to the client *unchanged* and captures usage
        from the final chunk (when stream_options.include_usage=True).

        NEW: If upstream omits completion_tokens, we approximate it by counting
        the streamed assistant deltas (text only), so output tokens are never 0.
        """
        # ensure include_usage in stream
        so = dict(body.get("stream_options") or {})
        so["include_usage"] = True
        body["stream_options"] = so

        usage: Dict[str, Any] = {}
        out_parts: List[str] = []  # collect assistant deltas

        async def gen() -> AsyncGenerator[bytes, None]:
            try:
                async with httpx.AsyncClient(timeout=None) as client:
                    _trace(f"[OPENAI] STREAM to {url}")
                    async with client.stream("POST", url, headers=headers, json=body) as resp:
                        if resp.status_code >= 400:
                            raw = await resp.aread()
                            payload = {
                                "provider": "openai-compat",
                                "model": body.get("model"),
                                "status": resp.status_code,
                                "message": raw.decode(errors="ignore")[:1000],
                            }
                            yield f"event: error\ndata: {json.dumps(payload)}\n\n".encode()
                            yield b"data: [DONE]\n\n"
                            return

                        async for raw_line in resp.aiter_lines():
                            if raw_line is None:
                                continue
                            line = raw_line.strip("\r")

                            # Forward exactly what upstream sends.
                            # We parse usage and collect assistant delta from "data: " lines that contain JSON.
                            if line.startswith("data: "):
                                data_str = line[6:]

                                # Capture usage + collect delta text
                                try:
                                    j = json.loads(data_str)
                                    if isinstance(j, dict):
                                        if isinstance(j.get("usage"), dict):
                                            usage.update(j["usage"])
                                        ch = (j.get("choices") or [{}])[0]
                                        delta = ch.get("delta") or {}
                                        cnt = delta.get("content")
                                        if isinstance(cnt, str):
                                            out_parts.append(cnt)
                                        elif isinstance(cnt, list):
                                            out_parts.extend(
                                                p.get("text", "")
                                                for p in cnt
                                                if isinstance(p, dict) and p.get("type") == "text"
                                            )
                                except Exception:
                                    pass

                                # Forward the data line (normalize with \n\n separator)
                                yield f"data: {data_str}\n\n".encode()

                            elif line == "":
                                # Upstream keep-alive; we re-add separators above for data lines
                                continue
                            else:
                                # Pass through other SSE fields (e.g., "event: ...", "id: ...")
                                yield (line + "\n\n").encode()

            except Exception as e:
                # Transport / parsing error ? forward as SSE error
                payload = {
                    "provider": "openai-compat",
                    "model": body.get("model"),
                    "status": getattr(e, "status_code", 502),
                    "message": str(e)[:1000],
                }
                yield f"event: error\ndata: {json.dumps(payload)}\n\n".encode()
                yield b"data: [DONE]\n\n"
                return

            # Fallback for missing completion_tokens
            if "completion_tokens" not in usage:
                joined = "".join(out_parts)
                if joined:
                    usage["completion_tokens"] = approx_tokens_from_text(joined)

            yield b"data: [DONE]\n\n"

        return gen(), usage


    # ---------- Gemini STREAM bridge with usage ----------
    async def _gemini_stream_bridge_with_usage(
        final_model: str,
        pe: ProviderEndpoint,
        contents: list,
        system_instruction: dict | None,
        gen_cfg: dict,
    ) -> Tuple[AsyncGenerator[bytes, None], dict, dict]:
        """
        Calls :streamGenerateContent and converts responses to OpenAI-style delta events.
        Also captures final usageMetadata for accurate token counts.
        Returns (generator, usage_metadata, prompt_modality_details_dict).

        This revised version ALSO forwards upstream errors as SSE `event: error`.
        """
        # Optional: preflight countTokens (good for per-modality image/PDF token details)
        prompt_modality_details: Dict[str, Any] = {}
        try:
            count_url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{final_model}:countTokens?key={pe.api_key}"
            )
            count_body = {
                "generateContentRequest": {
                    "contents": contents,
                    **({"systemInstruction": system_instruction} if system_instruction else {}),
                    "generationConfig": gen_cfg or {},
                    "tools": [{"googleSearch": {}}],
                }
            }
            async with httpx.AsyncClient(timeout=60.0) as cclient:
                cr = await cclient.post(
                    count_url,
                    headers={"Content-Type": "application/json"},
                    json=count_body,
                )
                if cr.status_code < 400:
                    cjson = cr.json()
                    prompt_modality_details = cjson
        except Exception:
            # best-effort; not required
            pass

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{final_model}:streamGenerateContent?alt=sse&key={pe.api_key}"
        )
        usage_md: Dict[str, Any] = {}
        collected_txt: List[str] = []  # NEW: collect assistant text for fallback

        async def gen() -> AsyncGenerator[bytes, None]:
            # Emit initial role delta to keep client consistent
            yield f"data: {json.dumps({'choices': [{'delta': {'role': 'assistant'}}]})}\n\n".encode()

            try:
                async with httpx.AsyncClient(timeout=None) as client:
                    _trace(f"[GEMINI] STREAM model={final_model} (PDF path) calling :streamGenerateContent")
                    async with client.stream(
                        "POST",
                        url,
                        headers={
                            "Content-Type": "application/json",
                            "Accept": "text/event-stream",
                        },
                        json={
                            "contents": contents,
                            "tools": [{"googleSearch": {}}],
                            "generationConfig": gen_cfg or {},
                            **({"systemInstruction": system_instruction} if system_instruction else {}),
                        },
                    ) as resp:
                        if resp.status_code >= 400:
                            # Forward upstream error as SSE error event (do not raise)
                            raw = await resp.aread()
                            payload = {
                                "provider": "gemini",
                                "model": final_model,
                                "status": resp.status_code,
                                "message": raw.decode(errors="ignore")[:1000],
                            }
                            yield f"event: error\ndata: {json.dumps(payload)}\n\n".encode()
                            yield b"data: [DONE]\n\n"
                            return

                        async for raw_line in resp.aiter_lines():
                            if raw_line is None:
                                continue
                            line = raw_line.strip("\r")
                            if not line.startswith("data: "):
                                # Ignore non-data lines from upstream SSE
                                continue

                            data_str = line[6:]
                            if data_str == "[DONE]":
                                # Upstream finished; we'll send our own DONE below
                                continue

                            try:
                                j = json.loads(data_str)
                            except Exception:
                                # Ignore malformed chunks
                                continue

                            # capture usageMetadata if present (usually on the last chunk)
                            if isinstance(j, dict) and isinstance(j.get("usageMetadata"), dict):
                                usage_md.update(j["usageMetadata"])

                            # collect any text and emit as OpenAI-style delta
                            txt = _collect_texts(j)
                            if txt:
                                collected_txt.append(txt)  # NEW
                                yield f"data: {json.dumps({'choices': [{'delta': {'content': txt}}]})}\n\n".encode()

            except Exception as e:
                # Network/transport/parsing error ? forward as SSE error
                payload = {
                    "provider": "gemini",
                    "model": final_model,
                    "status": getattr(e, "status_code", 502),
                    "message": str(e)[:1000],
                }
                yield f"event: error\ndata: {json.dumps(payload)}\n\n".encode()
                yield b"data: [DONE]\n\n"
                return

            # If Gemini omitted candidatesTokenCount, approximate from streamed text
            if "candidatesTokenCount" not in usage_md and collected_txt:
                usage_md["candidatesTokenCount"] = approx_tokens_from_text("".join(collected_txt))

            # Final DONE for client
            yield b"data: [DONE]\n\n"

        return gen(), usage_md, prompt_modality_details

    # ----------------- upstream call builders -----------------
    async def do_json_call(provider: str, model: str) -> Tuple[int, Dict[str, Any]]:
        _trace(f"[UPSTREAM] do_json_call provider={provider} model={model} has_pdf={has_pdf}")
        # Gemini JSON (PDFs or explicit gemini provider)
        if has_pdf:
            pe = provs.get(provider)
            base = (pe.base_url or "").lower() if pe else ""
            if (provider.lower() == "gemini") or ("generativelanguage.googleapis.com" in base):
                if not pe or not pe.api_key:
                    raise HTTPException(503, "Provider 'gemini' incomplete configuration")
                final_model = (model or (routes[0].model if routes else None) or "gemini-2.5-pro")
                contents, system_instruction = _openai_to_gemini(messages)
                gen_cfg: Dict[str, Any] = {"thinkingConfig": {"thinkingBudget": -1}}
                temp = payload.get("temperature")
                if isinstance(temp, (int, float)):
                    gen_cfg["temperature"] = temp
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{final_model}:generateContent?key={pe.api_key}"
                _trace(f"[GEMINI] JSON model={final_model} (PDF path) calling :generateContent")
                async with httpx.AsyncClient(timeout=120.0) as client:
                    r = await client.post(url, headers={"Content-Type": "application/json"}, json={
                        "contents": contents,
                        "tools": [{"googleSearch": {}}],
                        "generationConfig": gen_cfg,
                        **({"systemInstruction": system_instruction} if system_instruction else {}),
                    })
                    status = r.status_code
                    try:
                        data = r.json()
                    except Exception:
                        data = {"raw": (await r.aread()).decode(errors="ignore")}
                    return status, data

        # OpenAI-compatible JSON
        pe = provs.get(provider)
        if not pe or not pe.base_url or not pe.api_key:
            raise HTTPException(503, f"Provider '{provider}' incomplete configuration")
        url = _compose_openai_url(pe.base_url, "/v1/chat/completions")
        headers = {"Authorization": f"Bearer {pe.api_key}", "Content-Type": "application/json"}
        body = dict(payload)
        body["model"] = model or payload.get("model") or "auto"
        body["stream"] = False

        # If we have a PDF and we're going to an OpenAI-style proxy, convert "pdf" -> "input_file"
        if has_pdf:
            try:
                body["messages"] = _rewrite_pdf_to_input_file(body.get("messages") or [])
                _trace("[OPENAI] rewritten messages (pdf->input_file) summary: " +
                       _preview([{"role": m.get("role"), "parts":[{"type": (p.get("type") if isinstance(p,dict) else None),
                                                                   "mime_type": (p.get("mime_type") if isinstance(p,dict) else None),
                                                                   "has_data": isinstance((p or {}).get('data'), (str, bytes, bytearray))}
                                                                  for p in (m.get("content") or []) if isinstance(p, dict)]}
                               for m in body["messages"] ]))
            except Exception as _:
                pass

        _trace(f"[OPENAI] JSON to {url}")
        # (non-stream JSON responses usually include `usage` directly)
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(url, headers=headers, json=body)
            status = r.status_code
            try:
                data = r.json()
            except Exception:
                data = {"raw": (await r.aread()).decode(errors="ignore")}
            return status, data

    async def do_stream_call(provider: str, model: str) -> Tuple[AsyncGenerator[bytes, None], Dict[str, Any], Dict[str, Any]]:
        """
        Returns (generator, usage_totals_dict, usage_details_dict)
        - OpenAI: usage_totals_dict = {'prompt_tokens':..., 'completion_tokens':...}
                  usage_details_dict = {'prompt_tokens_details': {...}, 'completion_tokens_details': {...}} if present
        - Gemini : usage_totals_dict = {'promptTokenCount':..., 'candidatesTokenCount':..., 'totalTokenCount':...}
                  usage_details_dict = result from countTokens (may include per-modality arrays)
        """
        _trace(f"[UPSTREAM] do_stream_call provider={provider} model={model} has_pdf={has_pdf}")
        # Gemini stream (PDF)
        if has_pdf:
            pe = provs.get(provider)
            base = (pe.base_url or "").lower() if pe else ""
            if (provider.lower() == "gemini") or ("generativelanguage.googleapis.com" in base):
                if not pe or not pe.api_key:
                    raise HTTPException(503, "Provider 'gemini' incomplete configuration")
                final_model = (model or (routes[0].model if routes else None) or "gemini-2.5-pro")
                contents, system_instruction = _openai_to_gemini(messages)
                gen_cfg: Dict[str, Any] = {"thinkingConfig": {"thinkingBudget": -1}}
                temp = payload.get("temperature")
                if isinstance(temp, (int, float)): gen_cfg["temperature"] = temp
                agen, usage_md, prompt_details = await _gemini_stream_bridge_with_usage(final_model, pe, contents, system_instruction, gen_cfg)
                return agen, usage_md, prompt_details

        # OpenAI-compatible stream
        pe = provs.get(provider)
        if not pe or not pe.base_url or not pe.api_key:
            raise HTTPException(503, f"Provider '{provider}' incomplete configuration")
        url = _compose_openai_url(pe.base_url, "/v1/chat/completions")
        headers = {"Authorization": f"Bearer {pe.api_key}", "Content-Type": "application/json"}
        body = dict(payload)
        body["model"] = model or payload.get("model") or "auto"
        body["stream"] = True

        # If we have a PDF and we're going to an OpenAI-style proxy, convert "pdf" -> "input_file"
        if has_pdf:
            try:
                body["messages"] = _rewrite_pdf_to_input_file(body.get("messages") or [])
                _trace("[OPENAI] rewritten messages (pdf->input_file) summary: " +
                       _preview([{"role": m.get("role"), "parts":[{"type": (p.get("type") if isinstance(p,dict) else None),
                                                                   "mime_type": (p.get("mime_type") if isinstance(p,dict) else None),
                                                                   "has_data": isinstance((p or {}).get('data'), (str, bytes, bytearray))}
                                                                  for p in (m.get("content") or []) if isinstance(p, dict)]}
                               for m in body["messages"] ]))
            except Exception as _:
                pass

        agen, usage = await _openai_stream_with_usage(url, headers, body)

        # usage details (if provided)
        usage_details: Dict[str, Any] = {}
        if "prompt_tokens_details" in usage or "completion_tokens_details" in usage:
            usage_details = {
                "prompt_tokens_details": usage.get("prompt_tokens_details", {}),
                "completion_tokens_details": usage.get("completion_tokens_details", {}),
            }
        return agen, usage, usage_details

    # ----------------- run candidates with failover -----------------
    last_error: Optional[Exception] = None
    tried: List[str] = []

    if wants_stream:
        async def stream_wrapper() -> AsyncGenerator[bytes, None]:
            used_provider: Optional[str] = None
            used_model: Optional[str] = None
            in_tokens: Optional[int] = None
            out_tokens: Optional[int] = None
            extra_details: Dict[str, Any] = {}

            for provider, model in ordered:
                tried.append(f"{provider}:{model}")
                try:
                    agen, usage_totals, usage_details = await do_stream_call(provider, model)
                    first = True
                    async for chunk in agen:
                        if first:
                            used_provider = provider
                            used_model = model
                            _trace(f"[RUN] streaming from provider={used_provider} model={used_model}")
                            first = False
                        yield chunk
                    # After stream finished, capture usage for billing/logging
                    if usage_totals:
                        # OpenAI format
                        if "prompt_tokens" in usage_totals or "completion_tokens" in usage_totals:
                            in_tokens = usage_totals.get("prompt_tokens")
                            out_tokens = usage_totals.get("completion_tokens")
                        # Gemini format
                        elif "promptTokenCount" in usage_totals or "candidatesTokenCount" in usage_totals:
                            in_tokens = usage_totals.get("promptTokenCount")
                            out_tokens = usage_totals.get("candidatesTokenCount")
                        extra_details = usage_details or {}
                    last_error = None
                    break
                except Exception as e:
                    _errlog.info(f"chat stream failed provider={provider} model={model} err={e}")
                    last_error = e
                    continue

            # Fallback to approximate if provider gave no usage
            if in_tokens is None:
                # include ALL system+user text this time (as requested)
                all_texts: List[str] = []
                for m in messages:
                    c = m.get("content")
                    if isinstance(c, str): all_texts.append(c)
                    elif isinstance(c, list):
                        for p in c:
                            if p.get("type") == "text":
                                all_texts.append(p.get("text",""))
                in_tokens = approx_tokens_from_text(" ".join(all_texts))
            if out_tokens is None:
                # we did not collect deltas here; treat as 0 if provider omitted usage
                out_tokens = 0

            # finalize billing + usage row
            bill_model = (used_model or requested_model or (ordered[0][1] if ordered else "unknown"))
            bill_provider = (used_provider or (ordered[0][0] if ordered else "unknown"))

            _trace(f"[BILL] provider={bill_provider} model={bill_model} in_tokens={in_tokens} out_tokens={out_tokens} tried={tried}")

            cost = await charge_llm(
                session, user,
                model=bill_model,
                provider=bill_provider,
                input_tokens=int(in_tokens or 0),
                output_tokens=int(out_tokens or 0),
            )
            await log_usage(
                session, user,
                model=bill_model,
                provider=bill_provider,
                model_type=ModelType.VLM if is_vlm else ModelType.LLM,
                input_count=int(in_tokens or 0),
                output_count=int(out_tokens or 0),
                billed_credits=cost,
                response_meta={"tried": tried, "stream": True, "usage_details": extra_details},
            )
            await session.commit()

            if last_error and not used_provider:
                payload = {
                    "status": getattr(last_error, "status_code", 503),
                    "message": str(last_error),
                    "tried": tried,
                }
                yield f"event: error\ndata: {json.dumps(payload)}\n\n".encode()
                yield b"data: [DONE]\n\n"
                return

        return StreamingResponse(stream_wrapper(), media_type="text/event-stream")

    # --------- JSON (non-stream) path ----------
    for provider, model in ordered:
        tried.append(f"{provider}:{model}")
        try:
            status, data = await do_json_call(provider, model)

            # Prefer provider-reported usage
            in_tokens: Optional[int] = None
            out_tokens: Optional[int] = None
            meta_usage: Dict[str, Any] = {}

            # OpenAI JSON usage
            if isinstance(data, dict) and "usage" in data and isinstance(data["usage"], dict):
                u = data["usage"]
                in_tokens = u.get("prompt_tokens")
                out_tokens = u.get("completion_tokens")
                meta_usage = {
                    "prompt_tokens_details": u.get("prompt_tokens_details", {}),
                    "completion_tokens_details": u.get("completion_tokens_details", {}),
                }
            # Gemini JSON usage
            if isinstance(data, dict) and "usageMetadata" in data and isinstance(data["usageMetadata"], dict):
                u = data["usageMetadata"]
                in_tokens = u.get("promptTokenCount", in_tokens)
                out_tokens = u.get("candidatesTokenCount", out_tokens)
                meta_usage = {**meta_usage, "usageMetadata": u}

            # Fallback if missing: count ALL text (system+user) for input, assistant text for output
            if in_tokens is None:
                all_texts: List[str] = []
                for m in messages:
                    c = m.get("content")
                    if isinstance(c, str): all_texts.append(c)
                    elif isinstance(c, list):
                        for p in c:
                            if p.get("type") == "text":
                                all_texts.append(p.get("text",""))
                in_tokens = approx_tokens_from_text(" ".join(all_texts))
            if out_tokens is None:
                out_tokens = approx_tokens_from_text(_extract_assistant_text(data))

            _trace(f"[RUN][JSON] provider={provider} model={model} status={status} in_tokens={in_tokens} out_tokens={out_tokens} tried={tried}")

            cost = await charge_llm(
                session, user,
                model=requested_model or model or "auto",
                provider=provider,
                input_tokens=int(in_tokens or 0),
                output_tokens=int(out_tokens or 0),
            )
            await log_usage(
                session, user,
                model=requested_model or model or "auto",
                provider=provider,
                model_type=ModelType.VLM if is_vlm else ModelType.LLM,
                input_count=int(in_tokens or 0),
                output_count=int(out_tokens or 0),
                billed_credits=cost,
                response_meta={"tried": tried, "json": True, "usage_details": meta_usage},
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
    """
    Forward ASR to the first healthy provider according to RoutePref.

    Billing:
       Preferred: bill by tokens if upstream returns usage (OpenAI/Gemini style).
       Fallback: bill by HOURS when usage is absent (Whisper-style endpoints).
    """
    routes = await _ordered_routes(session, RouteKind.ASR)
    blob = await file.read()
    filename = file.filename or "audio.wav"
    last_error = None

    # Duration heuristic (16 kHz * 16-bit mono) for fallback time billing
    seconds = max(1, int(len(blob) / (16000 * 2)))  # ~1 sec per 32kB

    for r in routes:
        try:
            final_model = r.model or model

            # Call upstream ASR
            asr_json = await asr_forward(session, final_model, blob, filename, provider=r.provider)

            # -------- Try token billing if provider gave usage --------
            in_tok: int | None = None
            out_tok: int | None = None
            meta_usage: dict = {}

            if isinstance(asr_json, dict):
                u = asr_json.get("usage")
                # OpenAI-like
                if isinstance(u, dict) and ("prompt_tokens" in u or "completion_tokens" in u):
                    in_tok = int(u.get("prompt_tokens") or 0)
                    out_tok = int(u.get("completion_tokens") or 0)
                    meta_usage = {"usage": u}
                # Gemini-like (if you ever pass it through unchanged)
                um = asr_json.get("usageMetadata")
                if isinstance(um, dict) and (("promptTokenCount" in um) or ("candidatesTokenCount" in um)):
                    in_tok = int(um.get("promptTokenCount") or 0)
                    out_tok = int(um.get("candidatesTokenCount") or 0)
                    meta_usage = {"usageMetadata": um}

            if (in_tok is not None) or (out_tok is not None):
                # Per-million token pricing via charge_llm
                cost = await charge_llm(
                    session,
                    user,
                    model=final_model,
                    provider=r.provider,
                    input_tokens=int(in_tok or 0),
                    output_tokens=int(out_tok or 0),
                )
                await log_usage(
                    session,
                    user,
                    model=final_model,
                    provider=r.provider,
                    model_type=ModelType.ASR,
                    input_count=int(in_tok or 0),
                    output_count=int(out_tok or 0),
                    billed_credits=cost,
                    response_meta=meta_usage,
                )
                await session.commit()
                return JSONResponse(asr_json)

            # -------- Fallback: hourly billing --------
            # (We still record output characters for reference.)
            transcript = ""
            try:
                if isinstance(asr_json, dict):
                    if isinstance(asr_json.get("text"), str):
                        transcript = asr_json["text"]
                    elif isinstance(asr_json.get("transcript"), str):
                        transcript = asr_json["transcript"]
                    elif isinstance(asr_json.get("segments"), list):
                        transcript = " ".join(s.get("text", "") for s in asr_json["segments"] if isinstance(s, dict))
                    elif isinstance(asr_json.get("results"), list):
                        parts = []
                        for res in asr_json["results"]:
                            if not isinstance(res, dict): continue
                            alts = res.get("alternatives") or []
                            if alts and isinstance(alts[0], dict):
                                parts.append(alts[0].get("transcript", ""))
                        transcript = " ".join(parts)
            except Exception:
                transcript = ""
            out_chars = len(transcript or "")

            cost = await charge_asr(
                session,
                user,
                model=final_model,
                provider=r.provider,
                seconds=seconds,  # charge_asr now converts to hours internally
            )
            await log_usage(
                session,
                user,
                model=final_model,
                provider=r.provider,
                model_type=ModelType.ASR,
                input_count=seconds,  # stored raw seconds as input_count for diagnostics
                output_count=out_chars,
                billed_credits=cost,
            )
            await session.commit()
            return JSONResponse(asr_json)

        except Exception as e:
            _errlog.info(f"asr route failed provider={r.provider} model={r.model} err={e}")
            last_error = e
            continue

    raise HTTPException(status_code=503, detail=f"ASR providers unavailable. Last error: {last_error}")
