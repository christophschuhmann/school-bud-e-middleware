# billing.py
from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models import (
    User,
    Project,
    ProjectShare,
    ProjectAllowance,
    AllowanceInterval,
    ModelPricing,
    ModelType,
    UsageLog,
)


# ---------------------- Utilities ----------------------

def approx_tokens_from_text(text: str) -> int:
    """
    Very rough token estimate (safe for billing preview):
    ~ 4 chars per token, add small overhead.
    """
    if not text:
        return 0
    return max(1, int(len(text) / 4) + text.count(" "))


async def log_usage(
    session: AsyncSession,
    user: User,
    model: str,
    provider: str,
    model_type: ModelType,
    input_count: int,
    output_count: int,
    billed_credits: Decimal,
    request_meta: Dict[str, Any] | None = None,
    response_meta: Dict[str, Any] | None = None,
):
    ul = UsageLog(
        user_id=user.id,
        provider=provider,
        model=model,
        model_type=model_type,
        input_count=input_count or 0,
        output_count=output_count or 0,
        billed_credits=Decimal(str(billed_credits or 0)),
        request_meta=request_meta or {},
        response_meta=response_meta or {},
    )
    session.add(ul)


# ---------------------- CommonPool debit ----------------------

def _period_bounds(now: datetime, interval: AllowanceInterval) -> tuple[datetime, datetime]:
    if interval == AllowanceInterval.DAILY:
        start = datetime(now.year, now.month, now.day)
        end = start + timedelta(days=1)
    elif interval == AllowanceInterval.WEEKLY:
        start = datetime(now.year, now.month, now.day) - timedelta(days=now.weekday())
        end = start + timedelta(days=7)
    else:
        start = datetime(now.year, now.month, 1)
        if now.month == 12:
            end = datetime(now.year + 1, 1, 1)
        else:
            end = datetime(now.year, now.month + 1, 1)
    return start, end


async def _ensure_allowance(session: AsyncSession, user: User, project: Project) -> ProjectAllowance:
    now = datetime.utcnow()
    start, end = _period_bounds(now, project.allowance_interval)

    q = await session.execute(
        select(ProjectAllowance).where(
            ProjectAllowance.project_id == project.id,
            ProjectAllowance.user_id == user.id,
        )
    )
    pa = q.scalar_one_or_none()
    if not pa:
        pa = ProjectAllowance(
            project_id=project.id,
            user_id=user.id,
            remaining=project.allowance_per_user,
            period_start=start,
            period_end=end,
        )
        session.add(pa)
        await session.flush()
        return pa

    # Reset on boundary crossing; roll unused to common pool
    if not pa.period_start or not pa.period_end or now >= pa.period_end:
        unused = Decimal(pa.remaining or 0)
        project.common_pool_balance = Decimal(project.common_pool_balance or 0) + unused
        pa.remaining = Decimal(project.allowance_per_user or 0)
        pa.period_start, pa.period_end = start, end
        await session.flush()

    return pa


async def debit_credits(session: AsyncSession, user: User, cost: Decimal, model: str, provider: str) -> None:
    """
    Deduct 'cost' credits with the policy:

    CommonPool ON:
      1) per-user allowance.remaining
      2) project.common_pool_balance
      3) user.credits

    CommonPool OFF:
      1) user.credits
      2) project.credits

    Raises HTTPException(402) if funds are insufficient.
    """
    need = Decimal(cost or 0)
    if need <= 0:
        return

    project: Project | None = user.project

    if project and project.has_common_pool and project.allowance_interval:
        pa = await _ensure_allowance(session, user, project)

        # 1) allowance
        take = min(need, Decimal(pa.remaining or 0))
        if take > 0:
            pa.remaining = Decimal(pa.remaining or 0) - take
            project.credits = Decimal(project.credits or 0) - take
            need -= take

        # 2) common pool
        if need > 0:
            take = min(need, Decimal(project.common_pool_balance or 0))
            if take > 0:
                project.common_pool_balance = Decimal(project.common_pool_balance or 0) - take
                project.credits = Decimal(project.credits or 0) - take
                need -= take

        # 3) user's own credits
        if need > 0:
            take = min(need, Decimal(user.credits or 0))
            if take > 0:
                user.credits = Decimal(user.credits or 0) - take
                need -= take

        if need > 0:
            raise HTTPException(
                status_code=402,
                detail="Insufficient credits (allowance, common pool and user balance exhausted).",
            )
        return

    # ---- CommonPool OFF path ----
    take = min(need, Decimal(user.credits or 0))
    if take > 0:
        user.credits = Decimal(user.credits or 0) - take
        need -= take

    if need > 0 and project:
        take = min(need, Decimal(project.credits or 0))
        if take > 0:
            project.credits = Decimal(project.credits or 0) - take
            need -= take

    if need > 0:
        raise HTTPException(
            status_code=402,
            detail="Insufficient credits (user and project balances exhausted).",
        )


# ---------------------- Price charges ----------------------

async def _pricing(session: AsyncSession, model: str, provider: str) -> ModelPricing | None:
    q = await session.execute(
        select(ModelPricing).where(ModelPricing.model == model, ModelPricing.provider == provider)
    )
    return q.scalar_one_or_none()


async def charge_llm(
    session: AsyncSession,
    user: User,
    model: str,
    provider: str,
    input_tokens: int,
    output_tokens: int,
) -> Decimal:
    p = await _pricing(session, model, provider)
    pi = Decimal(p.price_per_input_token or 0) * Decimal(input_tokens or 0) if p else Decimal(0)
    po = Decimal(p.price_per_output_token or 0) * Decimal(output_tokens or 0) if p else Decimal(0)
    cost = pi + po
    await debit_credits(session, user, cost, model, provider)
    return cost


async def charge_tts(
    session: AsyncSession,
    user: User,
    model: str,
    provider: str,
    characters: int,
) -> Decimal:
    p = await _pricing(session, model, provider)
    cost = Decimal(p.price_per_character or 0) * Decimal(characters or 0) if p else Decimal(0)
    await debit_credits(session, user, cost, model, provider)
    return cost


async def charge_asr(
    session: AsyncSession,
    user: User,
    model: str,
    provider: str,
    seconds: int,
) -> Decimal:
    p = await _pricing(session, model, provider)
    cost = Decimal(p.price_per_second or 0) * Decimal(seconds or 0) if p else Decimal(0)
    await debit_credits(session, user, cost, model, provider)
    return cost
