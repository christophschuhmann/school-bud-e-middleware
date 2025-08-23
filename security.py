# security.py
from __future__ import annotations

import hashlib
from fastapi import Depends, Header, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db import get_session
from models import ApiKey, User


# --- helpers ---------------------------------------------------------

def hash_api_key(key: str) -> str:
    """Return hex sha256(key)."""
    return hashlib.sha256((key or "").encode("utf-8")).hexdigest()


def _extract_api_key(
    request: Request,
    x_api_key: str | None,
    authorization: str | None,
) -> str | None:
    """
    Accept any of:
      - X-API-Key: <key>
      - Authorization: Bearer <key>
      - ?api_key=<key> (query param)
    """
    if x_api_key:
        return x_api_key.strip()

    if authorization:
        parts = authorization.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1].strip()

    q = request.query_params.get("api_key")
    if q:
        return q.strip()

    return None


# --- dependency used by API endpoints --------------------------------

async def get_current_user(
    request: Request,
    session: AsyncSession = Depends(get_session),
    x_api_key: str | None = Header(default=None, convert_underscores=False),
    authorization: str | None = Header(default=None),
) -> User:
    """
    Resolve and validate the caller from an API key.
    Raises:
      - 401 if key is missing/invalid
      - 403 if user is inactive
    """
    key = _extract_api_key(request, x_api_key, authorization)
    if not key:
        raise HTTPException(
            status_code=401,
            detail="API key required. Use X-API-Key header, Authorization: Bearer <key>, or api_key=",
        )

    key_hash = hash_api_key(key)

    q = await session.execute(
        select(ApiKey).where(ApiKey.key_hash == key_hash, ApiKey.is_active.is_(True))
    )
    ak = q.scalar_one_or_none()
    if not ak:
        raise HTTPException(status_code=401, detail="Invalid or inactive API key.")

    user = await session.get(User, ak.user_id)
    if not user or not user.is_active:
        raise HTTPException(status_code=403, detail="User is inactive or not found.")

    return user
