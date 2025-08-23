# admin.py
import os
import io
import csv
import sqlite3
import tempfile
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_DOWN
from typing import Any, Dict, List, Tuple, Optional
import random, string

from fastapi import APIRouter, Depends, HTTPException, Form, UploadFile, File, Query
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, text
from sqlalchemy.orm import selectinload

from db import get_session, IS_SQLITE, DATA_DIR
from models import (
    User,
    ApiKey,
    ModelPricing,
    ModelType,
    ProviderEndpoint,
    Project,
    ProjectShare,
    SplitStrategy,
    UsageLog,
    CreditLedger,
    RoutePref,
    RouteKind,
    ProjectAllowance,     # legacy row store of remaining allowance
    AllowanceInterval,
    generate_api_key,
)

router = APIRouter(prefix="/admin", tags=["admin"])

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _dec(x) -> str:
    return str(x if x is not None else 0)

def _parse_ids(csv_ids: str) -> List[int]:
    return [int(s) for s in (csv_ids or "").split(",") if s.strip().isdigit()]

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _compute_next(now: datetime, interval: Optional[AllowanceInterval]) -> Optional[datetime]:
    if not interval:
        return None
    if interval == AllowanceInterval.DAILY:
        return now + timedelta(days=1)
    if interval == AllowanceInterval.WEEKLY:
        return now + timedelta(weeks=1)
    if interval == AllowanceInterval.MONTHLY:
        return now + timedelta(days=30)
    return None

def _next_from_interval(now: datetime, interval: Optional[AllowanceInterval]) -> Optional[datetime]:
    # Thin alias for backwards-compat; reuse existing logic
    return _compute_next(now, interval)



# built-in simple username generator (no external dependency)
_ADJ = ["bright","quick","calm","brave","clever","fresh","merry","pure","swift","kind",
        "bold","sharp","chill","eager","happy","jolly","neat","proud","slick","solid"]
_ANM = ["otter","lynx","panda","eagle","tiger","wolf","swan","koala","hare","orca",
        "fox","seal","owl","yak","mole","crab","ibis","kiwi","boar","yak"]

def _username_for(i: int, pid: int) -> str:
    idx = (i * 37 + pid * 101) % (len(_ADJ) * len(_ANM))
    a = _ADJ[idx % len(_ADJ)]
    b = _ANM[(idx // len(_ADJ)) % len(_ANM)]
    return f"{a}-{b}-{i:02d}"

def _rand_name(n: int = 6) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(random.choices(alphabet, k=max(3, n)))

# -----------------------------------------------------------------------------
# USERS
# -----------------------------------------------------------------------------
@router.get("/users")
async def list_users(
    project_id: int | None = None,
    q: str | None = None,
    sort: str = "id",
    order: str = "asc",
    session: AsyncSession = Depends(get_session),
):
    # ---- load real users (+ their projects) ----
    uq = select(User).options(selectinload(User.project))
    if project_id is not None:
        uq = uq.where(User.project_id == project_id)
    ures = await session.execute(uq)
    users = list(ures.scalars().all())

    # project shares for effective credit calc
    shares = {}
    qs = await session.execute(select(ProjectShare))
    for s in qs.scalars().all():
        shares[(s.project_id, s.user_id)] = Decimal(s.share_ratio or 0)

    # allowances (remaining)
    allowances = {}
    qa = await session.execute(select(ProjectAllowance))
    for a in qa.scalars().all():
        allowances[(a.project_id, a.user_id)] = Decimal(a.remaining or 0)

    def effective_credits(u: User) -> Decimal:
        eff = Decimal(u.credits or 0)
        if u.project:
            if u.project.has_common_pool:
                # allowance only; keep behavior as before (common pool not included in user eff)
                allow = allowances.get((u.project_id, u.id))
                if allow is None:
                    allow = Decimal(u.project.allowance_per_user or 0)
                eff = Decimal(u.credits or 0) + allow
            else:
                ratio = shares.get((u.project_id, u.id), Decimal(0))
                eff = Decimal(u.credits or 0) + (ratio * Decimal(u.project.credits or 0))
        return eff

    out: list[dict[str, Any]] = []
    # real users
    for u in users:
        row = {
            "id": u.id,
            "email": u.email,
            "first_name": u.first_name,
            "last_name": u.last_name,
            "project_id": u.project_id,
            "is_active": u.is_active,
            "credits": _dec(effective_credits(u)),
        }

        last_dt = getattr(u.project, "last_settled_at", None) if u.project else None
        next_dt = getattr(u.project, "next_settle_at", None) if u.project else None
        if (not next_dt) and u.project and u.project.allowance_interval:
            base = last_dt or _now()
            next_dt = _compute_next(base, u.project.allowance_interval)

        row["last_settlement_at"] = last_dt.isoformat() if last_dt else None
        row["next_settlement_at"] = next_dt.isoformat() if next_dt else None

        out.append(row)

    # ---- add synthetic rows per project (CommonPool, ProjectBudget) ----
    pq = select(Project)
    if project_id is not None:
        pq = pq.where(Project.id == project_id)
    pres = await session.execute(pq)
    projects = list(pres.scalars().all())

    for p in projects:
        # CommonPool row
        if p.has_common_pool:
            out.append({
                "display_id": f"CP-{p.id}",    # shown in UI
                "email": f"CommonPool[{p.id}]",
                "first_name": "CommonPool",
                "last_name": "",
                "project_id": p.id,
                "is_active": True,
                "credits": _dec(p.common_pool_balance),
            })
        # ProjectBudget row
        out.append({
            "display_id": f"PB-{p.id}",
            "email": f"ProjectBudget[{p.id}]",
            "first_name": "ProjectBudget",
            "last_name": "",
            "project_id": p.id,
            "is_active": True,
            "credits": _dec(p.credits),
        })

    # ---- search filter (email / names) ----
    if q:
        ql = q.lower()
        def _hit(r: dict) -> bool:
            return any(
                (r.get(k) or "").lower().find(ql) >= 0
                for k in ("email", "first_name", "last_name")
            )
        out = [r for r in out if _hit(r)]

    # ---- sorting (stable across numeric + synthetic ids) ----
    def _dec2(x):
        try:
            return Decimal(str(x))
        except Exception:
            return Decimal(0)

    sort = (sort or "id").lower()
    reverse = (order or "asc").lower() == "desc"

    def _key(r: dict):
        if sort == "credits":
            return (_dec2(r.get("credits")), r.get("email","").lower())
        if sort == "email":
            return (r.get("email","").lower(), )
        if sort == "project_id":
            # stringify to avoid int/str compare issues
            return (str(r.get("project_id") or ""), r.get("email","").lower())
        # default: id (numeric ids first, then synthetic by display_id)
        rid = r.get("id")
        if isinstance(rid, int):
            return (0, rid)
        return (1, r.get("display_id",""))

    out.sort(key=_key, reverse=reverse)
    return out



@router.post("/users")
async def create_user(
    email: str = Form(...),
    first_name: str = Form(""),
    last_name: str = Form(""),
    project_id: int | None = Form(None),
    session: AsyncSession = Depends(get_session),
):
    u = User(email=email, first_name=first_name, last_name=last_name, project_id=project_id)
    session.add(u)
    await session.flush()
    await session.commit()
    return {"id": u.id}

@router.post("/users/{user_id}/update")
async def update_user(
    user_id: int,
    email: str | None = Form(None),
    first_name: str | None = Form(None),
    last_name: str | None = Form(None),
    project_id: int | None = Form(None),
    is_active: bool | None = Form(None),
    session: AsyncSession = Depends(get_session),
):
    u = await session.get(User, user_id)
    if not u:
        raise HTTPException(404, f"User {user_id} not found.")
    if email is not None: u.email = email
    if first_name is not None: u.first_name = first_name
    if last_name is not None: u.last_name = last_name
    if project_id is not None:
        if project_id == "" or project_id == "null":
            u.project_id = None
        else:
            p = await session.get(Project, int(project_id))
            if not p:
                raise HTTPException(400, f"Project {project_id} does not exist.")
            u.project_id = p.id
    if is_active is not None:
        u.is_active = bool(int(is_active)) if isinstance(is_active, str) and is_active.isdigit() else bool(is_active)
    await session.commit()
    return {"ok": True}

@router.post("/users/delete")
async def delete_users(
    ids: str = Form(...),
    session: AsyncSession = Depends(get_session),
):
    id_list = _parse_ids(ids)
    if not id_list:
        return {"ok": True, "deleted": 0}
    await session.execute(delete(ApiKey).where(ApiKey.user_id.in_(id_list)))
    await session.execute(delete(ProjectShare).where(ProjectShare.user_id.in_(id_list)))
    await session.execute(delete(ProjectAllowance).where(ProjectAllowance.user_id.in_(id_list)))
    await session.execute(delete(UsageLog).where(UsageLog.user_id.in_(id_list)))
    await session.execute(delete(CreditLedger).where(CreditLedger.user_id.in_(id_list)))
    await session.execute(delete(User).where(User.id.in_(id_list)))
    await session.commit()
    return {"ok": True, "deleted": len(id_list)}

@router.post("/users/{user_id}/delete")
async def delete_user(user_id: int, session: AsyncSession = Depends(get_session)):
    await session.execute(delete(ApiKey).where(ApiKey.user_id == user_id))
    await session.execute(delete(ProjectShare).where(ProjectShare.user_id == user_id))
    await session.execute(delete(ProjectAllowance).where(ProjectAllowance.user_id == user_id))
    await session.execute(delete(UsageLog).where(UsageLog.user_id == user_id))
    await session.execute(delete(CreditLedger).where(CreditLedger.user_id == user_id))
    await session.execute(delete(User).where(User.id == user_id))
    await session.commit()
    return {"ok": True}

@router.post("/users/{user_id}/apikey")
async def create_or_rotate_key(user_id: int, session: AsyncSession = Depends(get_session)):
    u = await session.get(User, user_id)
    if not u:
        raise HTTPException(404, f"User {user_id} not found.")
    await session.execute(delete(ApiKey).where(ApiKey.user_id == user_id))
    public, h = generate_api_key(12)
    session.add(ApiKey(user_id=u.id, key_hash=h))
    await session.commit()
    return {"api_key": public}

@router.post("/users/{user_id}/credits")
async def set_credits(user_id: int, credits: float = Form(...), session: AsyncSession = Depends(get_session)):
    u = await session.get(User, user_id)
    if not u:
        raise HTTPException(404, "User not found")
    u.credits = Decimal(str(credits))
    session.add(u)
    await session.commit()
    return {"ok": True, "credits": str(u.credits)}

@router.post("/users/{user_id}/project")
async def assign_project(user_id: int, project_id: int | None = Form(None), session: AsyncSession = Depends(get_session)):
    u = await session.get(User, user_id)
    if not u:
        raise HTTPException(404, "User not found")
    if project_id in (None, "", "null"):
        u.project_id = None
    else:
        p = await session.get(Project, int(project_id))
        if not p:
            raise HTTPException(404, "Project not found")
        u.project_id = p.id
    session.add(u)
    await session.commit()
    return {"ok": True}

# -----------------------------------------------------------------------------
# PRICING
# -----------------------------------------------------------------------------

@router.get("/pricing")
async def list_pricing(session: AsyncSession = Depends(get_session)):
    q = await session.execute(select(ModelPricing))
    rows = q.scalars().all()
    return [
        {
            "id": p.id,
            "model": p.model,
            "provider": p.provider,
            "model_type": p.model_type.value if p.model_type else None,
            "p_input": _dec(p.price_per_input_token),
            "p_output": _dec(p.price_per_output_token),
            "p_char": _dec(p.price_per_character),
            "p_sec": _dec(p.price_per_second),
        }
        for p in rows
    ]

@router.post("/pricing")
async def upsert_pricing(
    model: str = Form(...),
    provider: str = Form(...),
    model_type: ModelType = Form(...),
    price_per_input_token: float = Form(0),
    price_per_output_token: float = Form(0),
    price_per_character: float = Form(0),
    price_per_second: float = Form(0),
    session: AsyncSession = Depends(get_session),
):
    q = await session.execute(select(ModelPricing).where(ModelPricing.model == model, ModelPricing.provider == provider))
    mp = q.scalar_one_or_none()
    if not mp:
        mp = ModelPricing(model=model, provider=provider, model_type=model_type)
    mp.price_per_input_token = Decimal(str(price_per_input_token))
    mp.price_per_output_token = Decimal(str(price_per_output_token))
    mp.price_per_character = Decimal(str(price_per_character))
    mp.price_per_second = Decimal(str(price_per_second))
    session.add(mp)
    await session.commit()
    return {"ok": True}

@router.post("/pricing/delete")
async def delete_pricing(
    ids: str = Form(...),
    session: AsyncSession = Depends(get_session),
):
    id_list = _parse_ids(ids)
    if not id_list:
        return {"ok": True, "deleted": 0}
    await session.execute(delete(ModelPricing).where(ModelPricing.id.in_(id_list)))
    await session.commit()
    return {"ok": True, "deleted": len(id_list)}

# -----------------------------------------------------------------------------
# PROVIDERS
# -----------------------------------------------------------------------------

@router.get("/providers")
async def list_providers(session: AsyncSession = Depends(get_session)):
    q = await session.execute(select(ProviderEndpoint))
    rows = q.scalars().all()
    return [
        {"id": p.id, "name": p.name, "base_url": p.base_url or "", "api_key": bool(p.api_key)}
        for p in rows
    ]

@router.post("/providers")
async def upsert_provider(
    name: str = Form(...),
    base_url: str | None = Form(None),
    api_key: str | None = Form(None),
    session: AsyncSession = Depends(get_session),
):
    q = await session.execute(select(ProviderEndpoint).where(ProviderEndpoint.name == name))
    pe = q.scalar_one_or_none()
    if not pe:
        pe = ProviderEndpoint(name=name)
    pe.base_url = base_url or pe.base_url
    if api_key is not None:
        pe.api_key = api_key or None
    session.add(pe)
    await session.commit()
    return {"ok": True}

@router.post("/providers/{provider_id}/apikey")
async def set_provider_key(
    provider_id: int,
    api_key: str = Form(""),
    session: AsyncSession = Depends(get_session),
):
    pe = await session.get(ProviderEndpoint, provider_id)
    if not pe:
        raise HTTPException(404, "Provider not found")
    pe.api_key = api_key or None
    session.add(pe)
    await session.commit()
    return {"ok": True}

@router.post("/providers/delete")
async def delete_providers(
    ids: str = Form(...),
    session: AsyncSession = Depends(get_session),
):
    id_list = _parse_ids(ids)
    if not id_list:
        return {"ok": True, "deleted": 0}
    await session.execute(delete(ProviderEndpoint).where(ProviderEndpoint.id.in_(id_list)))
    await session.commit()
    return {"ok": True, "deleted": len(id_list)}

@router.post("/providers/reset")
async def reset_providers(session: AsyncSession = Depends(get_session)):
    await session.execute(delete(ProviderEndpoint))
    await session.commit()
    return await bootstrap_providers(session)

# -----------------------------------------------------------------------------
# PROJECTS (+ CommonPool + settle)
# -----------------------------------------------------------------------------
@router.get("/projects")
async def list_projects(session: AsyncSession = Depends(get_session)):
    q = await session.execute(select(Project).options(selectinload(Project.users)))
    rows = q.scalars().unique().all()
    out = []
    now = _now()
    for p in rows:
        # ensure next is shown even if not yet stored
        last_dt = getattr(p, "last_settlement_at", None)
        next_dt = getattr(p, "next_settlement_at", None)
        if not next_dt and p.allowance_interval:
            base = last_dt or now
            next_dt = _compute_next(base, p.allowance_interval)

        out.append({
            "id": p.id,
            "name": p.name,
            "credits": _dec(p.credits),
            "split_strategy": p.split_strategy.value if p.split_strategy else SplitStrategy.EQUAL.value,
            "has_common_pool": bool(p.has_common_pool),
            "allowance_interval": p.allowance_interval.value if p.allowance_interval else None,
            "allowance_per_user": _dec(p.allowance_per_user),
            "common_pool_balance": _dec(p.common_pool_balance),
            "last_settlement_at": last_dt.isoformat() if last_dt else None,
            "next_settlement_at": next_dt.isoformat() if next_dt else None,
            "users": [u.id for u in (p.users or [])],
        })
    return out

@router.post("/projects")
async def upsert_project(
    name: str = Form(...),
    credits: float = Form(0),
    split_strategy: SplitStrategy = Form(SplitStrategy.EQUAL),
    has_common_pool: bool = Form(False),
    allowance_interval: AllowanceInterval | None = Form(None),
    allowance_per_user: float = Form(0),
    session: AsyncSession = Depends(get_session),
):
    q = await session.execute(select(Project).where(Project.name == name))
    p = q.scalar_one_or_none()
    if not p:
        p = Project(name=name)
    p.credits = Decimal(str(credits))
    p.split_strategy = split_strategy
    p.has_common_pool = has_common_pool
    p.allowance_interval = allowance_interval
    p.allowance_per_user = Decimal(str(allowance_per_user or 0))
    if hasattr(p, "next_settlement_at") and hasattr(p, "last_settlement_at"):
        now = _now()
        p.last_settlement_at = None
        p.next_settlement_at = _compute_next(now, p.allowance_interval)

    now = datetime.utcnow()
    if hasattr(p, "next_settle_at") and hasattr(p, "last_settled_at"):
      p.last_settled_at = None
      p.next_settle_at = _compute_next(now, p.allowance_interval)



    session.add(p)
    await session.commit()
    return {"ok": True, "id": p.id}

@router.post("/projects/{project_id}/delete")
async def delete_project(project_id: int, session: AsyncSession = Depends(get_session)):
    q = await session.execute(select(User).where(User.project_id == project_id))
    for u in q.scalars().all():
        u.project_id = None
        session.add(u)
    await session.execute(delete(ProjectAllowance).where(ProjectAllowance.project_id == project_id))
    await session.execute(delete(ProjectShare).where(ProjectShare.project_id == project_id))
    await session.execute(delete(Project).where(Project.id == project_id))
    await session.commit()
    return {"ok": True}

@router.post("/projects/delete")
async def delete_projects(ids: str = Form(...), session: AsyncSession = Depends(get_session)):
    id_list = _parse_ids(ids)
    deleted = 0
    for pid in id_list:
        await delete_project(pid, session)
        deleted += 1
    return {"ok": True, "deleted": deleted}

@router.post("/projects/{project_id}/shares")
async def set_share(
    project_id: int,
    user_id: int = Form(...),
    ratio: float = Form(...),
    session: AsyncSession = Depends(get_session),
):
    ps_q = await session.execute(
        select(ProjectShare).where(
            ProjectShare.project_id == project_id,
            ProjectShare.user_id == user_id,
        )
    )
    ps = ps_q.scalar_one_or_none()
    if not ps:
        ps = ProjectShare(project_id=project_id, user_id=user_id, share_ratio=Decimal(str(ratio)))
    else:
        ps.share_ratio = Decimal(str(ratio))
    session.add(ps)
    await session.commit()
    return {"ok": True}

@router.post("/projects/init")
async def init_project_with_users(
    name: str = Form(...),
    total_credits: float = Form(0),
    users_count: int = Form(...),
    prefix: str = Form("user"),
    rotate_or_create_keys: bool = Form(True),
    has_common_pool: bool = Form(False),
    allowance_interval: AllowanceInterval | None = Form(None),
    allowance_per_user: float = Form(0),
    randomize_all: bool = Form(False),
    rand_len: int = Form(6),
    session: AsyncSession = Depends(get_session),
):
    # upsert project
    q = await session.execute(select(Project).where(Project.name == name))
    p = q.scalar_one_or_none()
    if not p:
        p = Project(name=name)
        session.add(p)
        await session.flush()

    p.credits = Decimal(str(total_credits))
    p.has_common_pool = has_common_pool
    p.allowance_interval = allowance_interval
    p.allowance_per_user = Decimal(str(allowance_per_user or 0))

    # stamp initial schedule
    if hasattr(p, "last_settlement_at") and hasattr(p, "next_settlement_at"):
        p.last_settlement_at = None
        p.next_settlement_at = _compute_next(_now(), p.allowance_interval)

    await session.flush()

    created = []
    ratio = Decimal("1") / Decimal(users_count) if users_count > 0 else Decimal(0)

    # Create users
    for i in range(1, users_count + 1):
        uname = _rand_name(rand_len) if randomize_all else f"{prefix}-{i}-p{p.id}"
        email = f"{uname}@buddy.local"
        u = User(email=email, first_name="", last_name="", project_id=p.id, is_active=True)
        session.add(u)
        await session.flush()

        # keep equal shares recorded (useful if you switch to MANUAL later)
        session.add(ProjectShare(project_id=p.id, user_id=u.id, share_ratio=ratio))

        if rotate_or_create_keys:
            await session.execute(delete(ApiKey).where(ApiKey.user_id == u.id))
            public, h = generate_api_key(12)
            session.add(ApiKey(user_id=u.id, key_hash=h))
            created.append({"user_id": u.id, "email": u.email, "api_key": public})
        else:
            created.append({"user_id": u.id, "email": u.email, "api_key": None})

        if p.has_common_pool and p.allowance_interval:
            pa = ProjectAllowance(project_id=p.id, user_id=u.id, remaining=p.allowance_per_user)
            session.add(pa)

    # --- NEW: debit initial allowance from project budget (CommonPool projects) ---
    if p.has_common_pool and p.allowance_interval and p.allowance_per_user and users_count > 0:
        total_grant = Decimal(str(p.allowance_per_user)) * Decimal(users_count)
        p.credits = Decimal(p.credits or 0) - total_grant


    await session.commit()
    return {"ok": True, "project_id": p.id, "users": created}

@router.post("/projects/{project_id}/settle")
@router.post("/projects/{project_id}/settle_now")
async def settle_project_now(
    project_id: int,
    grant_per_user: float = Form(None),
    session: AsyncSession = Depends(get_session),
):
    p = await session.get(Project, project_id)
    if not p:
        raise HTTPException(404, "Not Found")

    uq = await session.execute(select(User).where(User.project_id == p.id))
    users = list(uq.scalars().all())
    n = len(users)

    # Requested grant per user
    requested = Decimal(str(grant_per_user)) if grant_per_user is not None else Decimal(str(p.allowance_per_user or 0))
    if n == 0:
        # still stamp timestamps even if no users
        now = _now()
        p.last_settlement_at = now if hasattr(p, "last_settlement_at") else getattr(p, "last_settlement_at", None)
        p.next_settlement_at = _compute_next(now, p.allowance_interval) if hasattr(p, "next_settlement_at") else getattr(p, "next_settlement_at", None)
        p.last_settled_at = now
        p.next_settle_at = _compute_next(now, p.allowance_interval)
        await session.commit()
        return {"ok": True, "users": 0, "swept": "0", "granted_per_user": "0", "requested_per_user": str(requested)}

    # ---- sweep phase ----
    swept_personal = Decimal(0)
    for u in users:
        b = Decimal(u.credits or 0)
        if b > 0:
            swept_personal += b
            u.credits = Decimal(0)

    swept_allowance = Decimal(0)
    allow_rows = {}
    if p.has_common_pool and p.allowance_interval:
        qa = await session.execute(select(ProjectAllowance).where(ProjectAllowance.project_id == p.id))
        allow_rows = {a.user_id: a for a in qa.scalars().all()}
        for u in users:
            a = allow_rows.get(u.id)
            if a and Decimal(a.remaining or 0) > 0:
                swept_allowance += Decimal(a.remaining or 0)

    p.common_pool_balance = Decimal(p.common_pool_balance or 0) + swept_personal + swept_allowance

    # ---- funding / pro-ration ----
    available = Decimal(p.credits or 0)
    needed_total = requested * Decimal(n)
    # what we can actually spend this period:
    allowed_total = min(available, needed_total)
    # equal per-user grant, rounded DOWN to 6 decimals to avoid overspend
    per_user = (allowed_total / Decimal(n)).quantize(Decimal("0.000001"), rounding=ROUND_DOWN)
    # re-align allowed_total to the rounded amount
    allowed_total = per_user * Decimal(n)

    partial = allowed_total < needed_total
    shortfall = (needed_total - allowed_total) if partial else Decimal(0)

    # ---- apply grants ----
    if p.has_common_pool and p.allowance_interval:
        # debit project for this period
        p.credits = available - allowed_total
        # re-seed allowance rows (not user credits)
        for u in users:
            a = allow_rows.get(u.id)
            if not a:
                a = ProjectAllowance(project_id=p.id, user_id=u.id, remaining=per_user)
                session.add(a)
            else:
                a.remaining = per_user
    else:
        # non-CommonPool: credit users directly
        p.credits = available - allowed_total
        for u in users:
            u.credits = Decimal(u.credits or 0) + per_user

    # ---- timestamps ----
    now = _now()
    if hasattr(p, "last_settlement_at"):
        p.last_settlement_at = now
    if hasattr(p, "next_settlement_at"):
        p.next_settlement_at = _compute_next(now, p.allowance_interval)
    p.last_settled_at = now
    p.next_settle_at = _compute_next(now, p.allowance_interval)

    await session.commit()
    return {
        "ok": True,
        "users": n,
        "swept": str(swept_personal + swept_allowance),
        "requested_per_user": str(requested),
        "granted_per_user": str(per_user),
        "partial": partial,
        "shortfall": str(shortfall),
        "project_credits_after": str(p.credits or 0),
        "warning": ("Insufficient project credits. Settled pro-rata; please top up the project." if partial else None),
        "last_settlement_at": p.last_settlement_at.isoformat() if getattr(p, "last_settlement_at", None) else None,
        "next_settlement_at": p.next_settlement_at.isoformat() if getattr(p, "next_settlement_at", None) else None,
    }


@router.get("/projects/{project_id}/export_keys")
async def export_project_keys(project_id: int, rotate: bool = True, session: AsyncSession = Depends(get_session)):
    q = await session.execute(select(User).where(User.project_id == project_id))
    users = q.scalars().all()
    rows = []
    for u in users:
        if rotate:
            await session.execute(delete(ApiKey).where(ApiKey.user_id == u.id))
            public, h = generate_api_key(12)
            session.add(ApiKey(user_id=u.id, key_hash=h))
            key = public
        else:
            qk = await session.execute(select(ApiKey).where(ApiKey.user_id == u.id))
            k = qk.scalar_one_or_none()
            if not k:
                public, h = generate_api_key(12)
                session.add(ApiKey(user_id=u.id, key_hash=h))
                key = public
            else:
                key = "EXISTING_KEY_(rotate=true to export plain)"
        rows.append([u.id, u.email or "", key])

    await session.commit()

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["user_id", "email", "api_key"])
    writer.writerows(rows)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.read()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="project_{project_id}_keys.csv"'},
    )

# -----------------------------------------------------------------------------
# ROUTES
# -----------------------------------------------------------------------------

@router.get("/routes")
async def list_routes(session: AsyncSession = Depends(get_session)):
    q = await session.execute(select(RoutePref).order_by(RoutePref.kind, RoutePref.priority, RoutePref.id))
    rows = q.scalars().all()
    return [
        dict(id=r.id, kind=r.kind.value, provider=r.provider, model=r.model, priority=r.priority, enabled=r.enabled)
        for r in rows
    ]

@router.post("/routes")
async def upsert_route(
    kind: RouteKind = Form(...),
    provider: str = Form(...),
    model: str = Form(...),
    priority: int = Form(1),
    enabled: bool = Form(True),
    session: AsyncSession = Depends(get_session),
):
    q = await session.execute(select(RoutePref).where(RoutePref.kind == kind, RoutePref.provider == provider))
    r = q.scalar_one_or_none()
    if not r:
        r = RoutePref(kind=kind, provider=provider, model=model, priority=priority, enabled=enabled)
    else:
        r.model = model
        r.priority = priority
        r.enabled = enabled
    session.add(r)
    await session.commit()
    return {"ok": True}

@router.post("/routes/delete")
async def delete_routes(ids: str = Form(...), session: AsyncSession = Depends(get_session)):
    id_list = _parse_ids(ids)
    if not id_list:
        return {"ok": True, "deleted": 0}
    await session.execute(delete(RoutePref).where(RoutePref.id.in_(id_list)))
    await session.commit()
    return {"ok": True, "deleted": len(id_list)}

@router.post("/routes/reset")
async def reset_routes(session: AsyncSession = Depends(get_session)):
    await session.execute(delete(RoutePref))
    await session.commit()
    return await bootstrap_routes(session)

# -----------------------------------------------------------------------------
# BOOTSTRAP
# -----------------------------------------------------------------------------

@router.post("/bootstrap/providers")
async def bootstrap_providers(session: AsyncSession = Depends(get_session)):
    defaults = [
        ("openai_compat", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")),
        ("groq",           os.getenv("GROQ_BASE_URL",   "https://api.groq.com/openai/v1")),
        ("gemini",         os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")),
        ("azure_openai",   os.getenv("AZURE_OPENAI_BASE_URL", "")),
        ("tts_openai",     os.getenv("TTS_BASE_URL", "https://api.openai.com/v1")),
        ("asr_openai",     os.getenv("ASR_BASE_URL", "https://api.openai.com/v1")),
    ]
    created = 0
    for name, base in defaults:
        q = await session.execute(select(ProviderEndpoint).where(ProviderEndpoint.name == name))
        pe = q.scalar_one_or_none()
        if not pe:
            pe = ProviderEndpoint(name=name, base_url=base or None, api_key=None)
            session.add(pe)
            created += 1
        else:
            if not pe.base_url and base:
                pe.base_url = base
    await session.commit()
    return {"ok": True, "created": created}

@router.post("/bootstrap/routes")
async def bootstrap_routes(session: AsyncSession = Depends(get_session)):
    wanted = [
        (RouteKind.LLM, "openai_compat", "gpt-4o-mini", 1),
        (RouteKind.LLM, "groq",           "llama-3.1-70b", 2),
        (RouteKind.LLM, "gemini",         "gemini-2.5-pro", 3),

        (RouteKind.VLM, "openai_compat", "gpt-4o", 1),
        (RouteKind.VLM, "gemini",        "gemini-2.5-pro", 2),

        (RouteKind.TTS, "tts_openai",    "tts-1", 1),
        (RouteKind.ASR, "asr_openai",    "whisper-1", 1),
    ]
    made, updated = 0, 0
    for kind, provider, model, prio in wanted:
        q = await session.execute(select(RoutePref).where(RoutePref.kind == kind, RoutePref.provider == provider))
        r = q.scalar_one_or_none()
        if not r:
            r = RoutePref(kind=kind, provider=provider, model=model, priority=prio, enabled=True)
            session.add(r)
            made += 1
        else:
            if not r.model:
                r.model = model; updated += 1
            if not r.priority:
                r.priority = prio; updated += 1
            if r.enabled is None:
                r.enabled = True; updated += 1
    await session.commit()
    return {"ok": True, "created": made, "updated": updated}

# -----------------------------------------------------------------------------
# BACKUP / RESTORE / RESET (SQLite only)
# -----------------------------------------------------------------------------

def _sqlite_snapshot_path() -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tmp.close()
    src = DATA_DIR / "buddy.db"
    with sqlite3.connect(src.as_posix()) as source, sqlite3.connect(tmp.name) as dest:
        source.backup(dest)
    return tmp.name

@router.get("/export/sqlite")
async def export_sqlite():
    if not IS_SQLITE:
        raise HTTPException(400, "SQLite export is only available when running in SQLite mode.")
    snap = _sqlite_snapshot_path()
    filename = os.path.basename(snap)
    return FileResponse(snap, filename=filename, media_type="application/octet-stream")

@router.post("/import/sqlite")
async def import_sqlite(
    file: UploadFile = File(...),
):
    if not IS_SQLITE:
        raise HTTPException(400, "SQLite import is only available when running in SQLite mode.")

    raw = await file.read()
    if len(raw) < 100 or not raw.startswith(b"SQLite format 3"):
        raise HTTPException(400, "Uploaded file is not a valid SQLite database.")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    with open(tmp.name, "wb") as f:
        f.write(raw)

    try:
        with sqlite3.connect(tmp.name) as conn:
            conn.execute("SELECT name FROM sqlite_master LIMIT 1").fetchall()
    except Exception:
        os.unlink(tmp.name)
        raise HTTPException(400, "The uploaded DB could not be opened by SQLite.")

    dst = DATA_DIR / "buddy.db"
    backup = DATA_DIR / f"buddy.backup.{int(_now().timestamp())}.db"
    try:
        if os.path.exists(dst):
            os.replace(dst.as_posix(), backup.as_posix())
        os.replace(tmp.name, dst.as_posix())
    except Exception as e:
        raise HTTPException(500, f"Failed to install uploaded DB: {e}")

    return {"ok": True, "replaced": str(dst), "backup": str(backup) if os.path.exists(backup) else None}

@router.post("/reset_all")
async def reset_all(session: AsyncSession = Depends(get_session)):
    # Children -> parents to avoid FK errors
    for model in (UsageLog, CreditLedger, ApiKey, ProjectAllowance, ProjectShare, User, Project, ProviderEndpoint, ModelPricing, RoutePref):
        await session.execute(delete(model))
    await session.commit()
    return {"ok": True}

# -----------------------------------------------------------------------------
# READ-ONLY LOGS + per-user series
# -----------------------------------------------------------------------------

@router.get("/usage")
async def list_usage(limit: int = 200, user_id: Optional[int] = None, session: AsyncSession = Depends(get_session)):
    stmt = select(UsageLog).order_by(UsageLog.id.desc()).limit(limit)
    if user_id:
        stmt = select(UsageLog).where(UsageLog.user_id == user_id).order_by(UsageLog.id.desc()).limit(limit)
    q = await session.execute(stmt)
    rows = q.scalars().all()
    return [
        dict(
            id=u.id,
            user_id=u.user_id,
            provider=u.provider,
            model=u.model,
            model_type=u.model_type.value if u.model_type else None,
            input=u.input_count,
            output=u.output_count,
            billed=_dec(u.billed_credits),
            created_at=u.created_at.isoformat() if u.created_at else None,
        )
        for u in rows
    ]

@router.get("/ledger")
async def list_ledger(limit: int = 200, session: AsyncSession = Depends(get_session)):
    q = await session.execute(select(CreditLedger).order_by(CreditLedger.id.desc()).limit(limit))
    rows = q.scalars().all()
    return [
        dict(
            id=l.id,
            user_id=l.user_id,
            delta=_dec(l.delta),
            reason=l.reason,
            created_at=l.created_at.isoformat() if l.created_at else None,
        )
        for l in rows
    ]

@router.get("/usage/series")
async def usage_series(
    user_id: int,
    days: int = 30,
    session: AsyncSession = Depends(get_session),
):
    q = await session.execute(
        select(UsageLog).where(UsageLog.user_id == user_id).order_by(UsageLog.id.asc())
    )
    rows = q.scalars().all()
    end = _now().date()
    start = end - timedelta(days=days-1)
    buckets: Dict[str, Decimal] = { (start + timedelta(days=i)).isoformat(): Decimal(0) for i in range(days) }
    for r in rows:
        if not r.created_at:
            continue
        d = r.created_at.date().isoformat()
        if d in buckets:
            buckets[d] += (r.billed_credits or Decimal(0))
    series = [{"date": k, "value": float(v)} for k, v in sorted(buckets.items())]
    return {"user_id": user_id, "days": days, "series": series}
