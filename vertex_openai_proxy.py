#!/usr/bin/env python3
"""
Vertex AI (Gemini) ? OpenAI-compatible proxy server
- /v1/audio/transcriptions  (Whisper-compatible)
- /v1/chat/completions      (OpenAI Chat Completions; supports text + base64 images/PDFs)
- /v1/responses             (minimal OpenAI Responses compatibility)
- /v1/models                (tiny models listing for clients)
- /admin/settings (GET/POST) to view/change default region/model/project/SA JSON

Updates in this version:
- Standard OpenAI usage fields are returned in non-streaming responses (unchanged behavior),
  and now also included in the **final streaming chunk** (OpenAI-style) before [DONE].
- For Vertex streaming, we pull `usage_metadata` from the stream object after iteration.
  If unavailable, we approximate via `count_tokens()` as a fallback.
- NEW: /v1/audio/transcriptions returns OpenAI-style `usage` in JSON responses and
  always sets `X-Usage-*` headers for all formats (text/srt/vtt/json).
"""

import os, base64, mimetypes, uuid, time, json, threading
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse

# ===========
# Defaults
# ===========

DEFAULTS = {
    "REGION":       os.getenv("VERTEX_REGION",      "europe-west4"),
    "MODEL":        os.getenv("VERTEX_MODEL_NAME",  "gemini-2.0-flash-001"),
    "PROJECT_ID":   os.getenv("VERTEX_PROJECT_ID",  "gemini-school-471209"),
    "SA_JSON":      os.getenv("VERTEX_SA_JSON",     "/etc/gcp/gemini-school-471209-3972614b5ec6.json"),
}

SUPPORTED_AUDIO = {
    ".m4a": "audio/mp4",
    ".mp4": "audio/mp4",
    ".mp3": "audio/mp3",
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".opus": "audio/opus",
    ".aac": "audio/aac",
    ".webm": "audio/webm",
    ".aiff": "audio/aiff",
}

def _sse_chat_delta(model_name: str, delta_text: str, idx: int = 0, finish_reason: Optional[str] = None) -> str:
    evt = {
        "id": _new_id("chatcmpl"),
        "object": "chat.completion.chunk",
        "created": _now_unix(),
        "model": model_name,
        "choices": [{
            "index": idx,
            "delta": ({"content": delta_text} if delta_text else {}),
            "finish_reason": finish_reason,
        }],
    }
    return f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"

def _sse_chat_final(model_name: str, usage: Optional[Dict[str, int]], idx: int = 0, finish_reason: str = "stop") -> str:
    # Final chunk that includes OpenAI-shaped usage (if available)
    evt = {
        "id": _new_id("chatcmpl"),
        "object": "chat.completion.chunk",
        "created": _now_unix(),
        "model": model_name,
        "choices": [{
            "index": idx,
            "delta": {},
            "finish_reason": finish_reason,
        }],
    }
    if usage:
        evt["usage"] = usage
    return f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"

def normalize_audio_mime(filename: str, content_type: Optional[str]) -> str:
    ct = (content_type or "").lower().strip()
    if ct and ct != "application/octet-stream" and ct.startswith("audio/"):
        if ct in {"audio/x-m4a", "audio/mp4a-latm"}:
            return "audio/mp4"
        return ct
    ext = Path(filename or "").suffix.lower()
    if ext in SUPPORTED_AUDIO:
        return SUPPORTED_AUDIO[ext]
    guess, _ = mimetypes.guess_type(filename or "")
    if guess and guess.startswith("audio/"):
        return guess
    return "audio/wav"

def set_usage_headers(resp: Response, usage: Optional[object]):
    if not usage:
        return
    # Dict style (AI Studio/REST)
    if isinstance(usage, dict):
        if "promptTokenCount" in usage: resp.headers["X-Usage-Prompt-Tokens"] = str(usage["promptTokenCount"])
        if "candidatesTokenCount" in usage: resp.headers["X-Usage-Candidates-Tokens"] = str(usage["candidatesTokenCount"])
        if "totalTokenCount" in usage: resp.headers["X-Usage-Total-Tokens"] = str(usage["totalTokenCount"])
        if "cachedContentTokenCount" in usage: resp.headers["X-Usage-Cached-Content-Tokens"] = str(usage["cachedContentTokenCount"])
        return
    # Vertex SDK usage object
    pt = getattr(usage, "prompt_token_count", None)
    ct = getattr(usage, "candidates_token_count", None)
    tt = getattr(usage, "total_token_count", None)
    cc = getattr(usage, "cached_content_token_count", None)
    if pt is not None: resp.headers["X-Usage-Prompt-Tokens"] = str(pt)
    if ct is not None: resp.headers["X-Usage-Candidates-Tokens"] = str(ct)
    if tt is not None: resp.headers["X-Usage-Total-Tokens"] = str(tt)
    if cc is not None: resp.headers["X-Usage-Cached-Content-Tokens"] = str(cc)

# ====================
# Lazy dep management
# ====================

def _ensure(pkg: str, mod: Optional[str] = None):
    try:
        __import__(mod or pkg.replace("-", "_"))
    except ImportError:
        import subprocess, sys as _s
        subprocess.check_call([_s.executable, "-m", "pip", "install", pkg])
        __import__(mod or pkg.replace("-", "_"))

_ensure("google-cloud-aiplatform", "google.cloud.aiplatform")

# Vertex SDK
import vertexai
from vertexai.generative_models import GenerativeModel, Part

# ===========
# App setup
# ===========

app = FastAPI(title="Vertex Gemini ? OpenAI-compatible proxy")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

STATE = {
    "region":     DEFAULTS["REGION"],
    "model":      DEFAULTS["MODEL"],
    "project_id": DEFAULTS["PROJECT_ID"],
    "sa_json":    DEFAULTS["SA_JSON"],
}
_STATE_LOCK = threading.Lock()

_MODEL_CACHE: Dict[Tuple[str, str, str, str], GenerativeModel] = {}

def _init_vertex(region: str, project_id: str, sa_json: str):
    if sa_json and os.path.isfile(sa_json):
        from google.oauth2 import service_account
        creds = service_account.Credentials.from_service_account_file(
            sa_json, scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        vertexai.init(project=project_id, location=region, credentials=creds)
    else:
        vertexai.init(project=project_id, location=region)

def _get_model(model_name: str) -> GenerativeModel:
    key = (STATE["region"], STATE["project_id"], STATE["sa_json"], model_name)
    m = _MODEL_CACHE.get(key)
    if m is not None:
        return m
    _init_vertex(STATE["region"], STATE["project_id"], STATE["sa_json"])
    gm = GenerativeModel(model_name)
    _MODEL_CACHE[key] = gm
    return gm

def _clear_model_cache():
    _MODEL_CACHE.clear()

# =========================
# Helpers: OpenAI payloads
# =========================

def _now_unix() -> int:
    return int(time.time())

def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"

def _parse_data_url(url: str) -> Tuple[bytes, str]:
    if not url.startswith("data:"):
        raise ValueError("Only data: URLs are supported for images/files.")
    head, b64 = url.split(",", 1)
    if ";base64" not in head:
        raise ValueError("Only base64-encoded data: URLs are supported.")
    mime = head[5:].split(";")[0].strip() or "application/octet-stream"
    return base64.b64decode(b64), mime

def _collect_text_and_binary_from_messages(messages: List[Dict[str, Any]]) -> Tuple[str, List[Part]]:
    text_blocks: List[str] = []
    binaries: List[Part] = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            if content:
                text_blocks.append(content)
            continue
        if isinstance(content, list):
            line_chunks: List[str] = []
            for part in content:
                ptype = part.get("type")
                if ptype in {"text", "input_text"}:
                    t = part.get("text") or part.get("input_text") or ""
                    if t:
                        line_chunks.append(t)
                elif ptype in {"image_url", "input_image"}:
                    data_url = None
                    if "image_url" in part and isinstance(part["image_url"], dict):
                        data_url = part["image_url"].get("url")
                    elif "image_url" in part and isinstance(part["image_url"], str):
                        data_url = part["image_url"]
                    if data_url:
                        blob, mime = _parse_data_url(data_url)
                        binaries.append(Part.from_data(mime_type=mime, data=blob))
                elif ptype in {"input_file", "file"}:
                    b64 = part.get("data")
                    mime = part.get("mime_type") or "application/octet-stream"
                    if b64:
                        blob = base64.b64decode(b64 if isinstance(b64, (bytes, bytearray)) else str(b64).encode("utf-8"))
                        binaries.append(Part.from_data(mime_type=mime, data=blob))
            if line_chunks:
                text_blocks.append("\n".join(line_chunks))
    text_prompt = "\n\n".join([t for t in text_blocks if t]).strip()
    return text_prompt, binaries

def _gen_config_from_openai_params(body: Dict[str, Any]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    if "temperature" in body and body["temperature"] is not None:
        cfg["temperature"] = float(body["temperature"])
    if "top_p" in body and body["top_p"] is not None:
        cfg["top_p"] = float(body["top_p"])
    if "max_tokens" in body and body["max_tokens"] is not None:
        cfg["max_output_tokens"] = int(body["max_tokens"])
    # Add more mappings (e.g., top_k) here if you like.
    return cfg

def _usage_dict_from_vertex(usage_obj: Any) -> Dict[str, int]:
    if not usage_obj:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    pt = getattr(usage_obj, "prompt_token_count", None)
    ct = getattr(usage_obj, "candidates_token_count", None)
    tt = getattr(usage_obj, "total_token_count", None)
    # Prefer exact: completion = total - prompt (when both present). Else use candidates.
    completion = None
    if tt is not None and pt is not None:
        completion = int(tt) - int(pt)
    elif ct is not None:
        completion = int(ct)
    return {
        "prompt_tokens": int(pt or 0),
        "completion_tokens": int(completion or 0),
        "total_tokens": int(tt or ((pt or 0) + (ct or 0))),
    }

def _approx_usage_via_count_tokens(gm: GenerativeModel, prompt_parts: List[Any], completion_text: str) -> Dict[str, int]:
    # Fallback if streaming usage_metadata is not exposed by the SDK.
    try:
        pin = gm.count_tokens(prompt_parts)
        pin_total = int(getattr(pin, "total_tokens", 0) or 0)
    except Exception:
        pin_total = 0
    try:
        pout = gm.count_tokens([completion_text] if completion_text else [""])
        pout_total = int(getattr(pout, "total_tokens", 0) or 0)
    except Exception:
        pout_total = 0
    return {
        "prompt_tokens": pin_total,
        "completion_tokens": pout_total,
        "total_tokens": pin_total + pout_total,
    }

# ==========================
# OpenAI-compatible routes
# ==========================

@app.get("/v1/models")
def list_models():
    models = {STATE["model"], *[k[3] for k in _MODEL_CACHE.keys()]}
    return {
        "object": "list",
        "data": [{"id": m, "object": "model", "created": _now_unix(), "owned_by": "google-vertex"} for m in sorted(models)],
    }

@app.post("/v1/audio/transcriptions")
async def transcribe_audio(
    response: Response,
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),  # json | text | srt | vtt
    temperature: Optional[float] = Form(None),
    language: Optional[str] = Form(None),
    translate: Optional[str] = Form(None),
):
    """
    Whisper-compatible endpoint that returns:
    - text transcript in requested format
    - and (for JSON format) OpenAI-style `usage` with prompt/completion/total tokens.
    """
    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file.")
        mime = normalize_audio_mime(file.filename, file.content_type)

        transcribe_guard = (
            "ROLE: You are a strict speech-to-text transcriber.\n"
            "TASK: TRANSCRIBE VERBATIM the spoken words in the provided audio.\n"
            "RULES: Do NOT answer questions, do NOT summarize, do NOT add prefixes like 'Assistant:' or 'User:'. "
            "If 'translate' is true, translate to English; otherwise keep original language. "
            "Return ONLY the transcript text (or the requested caption format)."
        )

        parts = []
        if prompt:
            parts.append(f"Context: {prompt}")
        if language:
            parts.append(f"Audio language hint: {language}.")
        if translate and str(translate).strip().lower() in {"1", "true", "yes", "on"}:
            parts.append("Translate to English; otherwise transcribe verbatim.")
        if response_format.lower() == "srt":
            parts.append("Return a valid SRT file (no commentary).")
        elif response_format.lower() == "vtt":
            parts.append("Return a valid WebVTT starting with 'WEBVTT' (no commentary).")
        else:
            parts.append("Return plain text only (no prefixes, no commentary).")
        instr = " ".join(parts) if parts else "Transcribe the audio; return plain text only."

        model_name = model or STATE["model"]
        gm = _get_model(model_name)

        gen_cfg = {
            "temperature": 0.0,
            "top_p": 0.0,
            "response_mime_type": "text/plain",
        }

        # Generate
        res = gm.generate_content(
            [transcribe_guard, instr, Part.from_data(mime_type=mime, data=audio_bytes)],
            generation_config=gen_cfg,
        )

        # Extract text
        transcript = getattr(res, "text", "") or ""
        if not transcript:
            try:
                cand = res.candidates[0]
                ps = getattr(cand, "content", None)
                if ps and getattr(ps, "parts", None):
                    transcript = "".join([getattr(p, "text", "") or "" for p in ps.parts])
            except Exception:
                transcript = ""
        t_strip = transcript.lstrip()
        if t_strip.lower().startswith("assistant:"):
            transcript = t_strip.split(":", 1)[1].lstrip()

        # Usage: headers + JSON usage (OpenAI-style)
        usage_meta = getattr(res, "usage_metadata", None)
        set_usage_headers(response, usage_meta)

        # OpenAI-style usage dict (fallback to count_tokens if missing)
        try:
            usage_dict = _usage_dict_from_vertex(usage_meta) if usage_meta else None
        except Exception:
            usage_dict = None
        if usage_dict is None:
            try:
                prompt_parts = [transcribe_guard, instr, Part.from_data(mime_type=mime, data=audio_bytes)]
                usage_dict = _approx_usage_via_count_tokens(gm, prompt_parts, transcript)
            except Exception:
                usage_dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        # Format-specific returns
        fmt = response_format.lower()
        if fmt == "text":
            return PlainTextResponse(transcript, status_code=200, headers=response.headers)
        if fmt == "srt":
            return PlainTextResponse(transcript, status_code=200, media_type="application/x-subrip", headers=response.headers)
        if fmt == "vtt":
            out = transcript if transcript.strip().lower().startswith("webvtt") else "WEBVTT\n\n" + transcript
            return PlainTextResponse(out, status_code=200, media_type="text/vtt", headers=response.headers)

        # JSON (OpenAI Whisper-compatible)
        return JSONResponse({"text": transcript, "usage": usage_dict}, status_code=200, headers=response.headers)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Vertex error in /v1/audio/transcriptions: {e}")

@app.post("/v1/chat/completions")
def chat_completions(body: Dict[str, Any] = Body(...)):
    """
    OpenAI Chat Completions compatibility (+ SSE when stream=true).
    """
    try:
        model_name = body.get("model") or STATE["model"]
        messages = body.get("messages") or []
        if not isinstance(messages, list) or not messages:
            raise HTTPException(status_code=400, detail="Missing 'messages'.")

        text_prompt, binaries = _collect_text_and_binary_from_messages(messages)
        system_texts = [
            m.get("content", "")
            for m in messages
            if m.get("role") == "system" and isinstance(m.get("content"), str)
        ]
        anti_prefix = "Reply directly without any speaker labels or prefixes (no 'Assistant:' or 'User:')."

        # Build final prompt parts for Gemini
        prompt_parts: List[Any] = []
        preamble = " ".join([anti_prefix] + [s for s in system_texts if s])
        if preamble:
            prompt_parts.append(preamble)
        if text_prompt:
            prompt_parts.append(text_prompt)
        prompt_parts.extend(binaries)

        gm = _get_model(model_name)
        gen_cfg = _gen_config_from_openai_params(body)

        # STREAMING
        if body.get("stream"):
            include_usage = True
            so = body.get("stream_options") or {}
            if "include_usage" in so:
                include_usage = bool(so.get("include_usage"))

            stream = gm.generate_content(
                prompt_parts,
                generation_config=gen_cfg if gen_cfg else None,
                stream=True,
            )

            # We also buffer text so that, if needed, we can approximate output tokens.
            out_buf: List[str] = []

            def iter_sse():
                try:
                    for ch in stream:
                        txt = getattr(ch, "text", None)
                        if not txt:
                            try:
                                cand = ch.candidates[0]
                                ps = getattr(cand, "content", None)
                                if ps and getattr(ps, "parts", None):
                                    txt = "".join([(getattr(p, "text", "") or "") for p in ps.parts])
                            except Exception:
                                txt = ""
                        if txt:
                            t = txt.lstrip()
                            if t.lower().startswith("assistant:"):
                                txt = t.split(":", 1)[1].lstrip()
                            out_buf.append(txt)
                            yield _sse_chat_delta(model_name, txt)

                    # After stream ends, attach usage in a final chunk (OpenAI style)
                    usage_dict: Optional[Dict[str, int]] = None
                    if include_usage:
                        try:
                            usage = getattr(stream, "usage_metadata", None)
                            if usage:
                                usage_dict = _usage_dict_from_vertex(usage)
                        except Exception:
                            usage_dict = None
                        if usage_dict is None:
                            # Fallback approximate usage if SDK didn't expose usage on stream
                            try:
                                usage_dict = _approx_usage_via_count_tokens(gm, prompt_parts, "".join(out_buf))
                            except Exception:
                                usage_dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

                    # Final chunk with finish_reason and (optionally) usage
                    yield _sse_chat_final(model_name, usage_dict, finish_reason="stop")
                    yield "data: [DONE]\n\n"
                except Exception:
                    # Send an error-shaped finalization to keep clients happy
                    yield _sse_chat_final(model_name, None, finish_reason="error")
                    yield "data: [DONE]\n\n"

            headers = {
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                # Note: we can't set token-usage headers at the very end of a stream.
            }
            return StreamingResponse(iter_sse(), media_type="text/event-stream", headers=headers)

        # NON-STREAMING
        res = gm.generate_content(prompt_parts, generation_config=gen_cfg if gen_cfg else None)
        output_text = getattr(res, "text", None) or ""
        if output_text:
            t = output_text.lstrip()
            if t.lower().startswith("assistant:"):
                output_text = t.split(":", 1)[1].lstrip()

        usage = getattr(res, "usage_metadata", None)
        usage_dict = _usage_dict_from_vertex(usage)

        return {
            "id": _new_id("chatcmpl"),
            "object": "chat.completion",
            "created": _now_unix(),
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": output_text},
                "finish_reason": "stop",
            }],
            "usage": usage_dict,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Vertex error: {e}")

@app.post("/v1/responses")
def responses_api(body: Dict[str, Any] = Body(...)):
    """
    Minimal compatibility with OpenAI 'Responses' API.
    """
    try:
        model_name = body.get("model") or STATE["model"]
        gm = _get_model(model_name)

        input_ = body.get("input")
        parts: List[Any] = []
        binaries: List[Part] = []
        text_prompt = ""

        if isinstance(input_, str):
            text_prompt = input_
        elif isinstance(input_, list):
            text_prompt, binaries = _collect_text_and_binary_from_messages([{"role":"user","content":input_}])
        else:
            raise HTTPException(status_code=400, detail="Unsupported 'input' type.")

        if text_prompt:
            parts.append(text_prompt)
        parts.extend(binaries)

        gen_cfg = _gen_config_from_openai_params(body)
        res = gm.generate_content(parts, generation_config=gen_cfg if gen_cfg else None)
        output_text = getattr(res, "text", None) or ""
        usage = getattr(res, "usage_metadata", None)
        usage_dict = _usage_dict_from_vertex(usage)

        return {
            "id": _new_id("resp"),
            "object": "response",
            "created": _now_unix(),
            "model": model_name,
            "output": [{"type": "output_text", "text": output_text}],
            "usage": usage_dict,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Vertex error: {e}")

# ====================
# Admin configuration
# ====================

@app.get("/admin/settings")
def get_settings():
    with _STATE_LOCK:
        return dict(STATE)

@app.post("/admin/settings")
def set_settings(body: Dict[str, Any] = Body(...)):
    """
    Change defaults at runtime. Any of: region, model, project_id, sa_json
    """
    changed = False
    with _STATE_LOCK:
        for k in ("region","model","project_id","sa_json"):
            if k in body and body[k] and body[k] != STATE[k]:
                STATE[k] = body[k]
                changed = True
        if changed:
            _clear_model_cache()
    return {"ok": True, "changed": changed, "state": dict(STATE)}

# ============
# Root ping
# ============

@app.get("/")
def root():
    return {
        "ok": True,
        "msg": "Vertex Gemini ? OpenAI proxy is up.",
        "region": STATE["region"],
        "model": STATE["model"],
        "project_id": STATE["project_id"],
    }
