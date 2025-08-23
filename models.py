# models.py
from __future__ import annotations
from sqlalchemy import DateTime
from datetime import datetime

import hashlib
import secrets
from datetime import datetime
from decimal import Decimal
from enum import Enum

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum as SAEnum,
    ForeignKey,
    Integer,
    JSON,
    Numeric,
    String,
    UniqueConstraint,
    Text,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

# --- Routing preferences (priority-based failover) ---------------------------
from sqlalchemy import UniqueConstraint
from enum import Enum

class RouteKind(str, Enum):
    LLM = "LLM"
    VLM = "VLM"
    TTS = "TTS"
    ASR = "ASR"
    OTHER = "OTHER"

class RoutePref(Base):
    __tablename__ = "route_prefs"
    id = Column(Integer, primary_key=True)
    # what type of model (LLM, VLM, TTS, ASR, …)
    kind = Column(SAEnum(RouteKind), nullable=False)
    # friendly provider name that must match ProviderEndpoint.name (e.g. "openai_compat", "gemini", "groq")
    provider = Column(String, nullable=False)
    # default model name to use on that provider (you can edit from Admin)
    model = Column(String, nullable=False)
    # priority: 1 is highest; ties are broken by id
    priority = Column(Integer, default=1, nullable=False)
    # enable/disable without deleting
    enabled = Column(Boolean, default=True, nullable=False)

    __table_args__ = (
        UniqueConstraint("kind", "provider", name="uq_kind_provider"),
    )


# -------------------------- Enums --------------------------

class ModelType(str, Enum):
    LLM = "LLM"
    VLM = "VLM"
    TTS = "TTS"
    ASR = "ASR"
    EMB = "EMB"


class SplitStrategy(str, Enum):
    EQUAL = "EQUAL"
    MANUAL = "MANUAL"
    NONE = "NONE"


class AllowanceInterval(str, Enum):
    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"


# -------------------------- Helpers --------------------------
# models.py
import os
import secrets
import hashlib

def generate_api_key(length: int = 12) -> tuple[str, str]:
    """
    Returns (public_token, key_hash).
    public_token is prefixed (default 'sbe-') and then 'length' chars from a safe alphabet.
    The full visible token (including prefix) is hashed.
    """
    prefix = os.getenv("UNIVERSAL_KEY_PREFIX", "sbe-")
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no O/0/I/1 to avoid confusion
    body = "".join(secrets.choice(alphabet) for _ in range(length))
    public = f"{prefix}{body}"
    key_hash = hashlib.sha256(public.encode("utf-8")).hexdigest()
    return public, key_hash



# -------------------------- Core Tables --------------------------

class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, index=True, nullable=False)
    credits = Column(Numeric(18, 6), default=0)
    split_strategy = Column(SAEnum(SplitStrategy), default=SplitStrategy.EQUAL)

    # CommonPool options
    has_common_pool = Column(Boolean, default=False)
    allowance_interval = Column(SAEnum(AllowanceInterval), nullable=True)
    allowance_per_user = Column(Numeric(18, 6), default=0)
    common_pool_balance = Column(Numeric(18, 6), default=0)

    last_settled_at = Column(DateTime(timezone=True), nullable=True)
    next_settle_at  = Column(DateTime(timezone=True), nullable=True)

    last_reset_at = Column(DateTime, default=datetime.utcnow)

    created_at = Column(DateTime, default=datetime.utcnow)

    users = relationship("User", back_populates="project", lazy="selectin")  # for eager loading


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String, unique=False, index=True, nullable=True)
    first_name = Column(String, default="")
    last_name = Column(String, default="")
    credits = Column(Numeric(18, 6), default=0)
    is_active = Column(Boolean, default=True)

    project_id = Column(Integer, ForeignKey("projects.id"), nullable=True, index=True)
    project = relationship("Project", back_populates="users", lazy="joined")

    created_at = Column(DateTime, default=datetime.utcnow)


class ApiKey(Base):
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    key_hash = Column(String(128), nullable=False)  # sha256
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", lazy="joined")


class ProviderEndpoint(Base):
    __tablename__ = "provider_endpoints"

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)          # e.g., openai_compat, gemini
    base_url = Column(String, nullable=True)
    api_key = Column(Text, nullable=True)                       # optional override


class ModelPricing(Base):
    __tablename__ = "model_pricing"

    id = Column(Integer, primary_key=True)
    model = Column(String, index=True, nullable=False)
    provider = Column(String, index=True, nullable=False)
    model_type = Column(SAEnum(ModelType), nullable=False)

    price_per_input_token = Column(Numeric(18, 8), default=0)
    price_per_output_token = Column(Numeric(18, 8), default=0)
    price_per_character = Column(Numeric(18, 8), default=0)
    price_per_second = Column(Numeric(18, 8), default=0)

    __table_args__ = (
        UniqueConstraint("model", "provider", name="uq_model_provider"),
    )


class ProjectShare(Base):
    __tablename__ = "project_shares"

    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    share_ratio = Column(Numeric(9, 6), default=0)  # 0..1

    project = relationship("Project", lazy="joined")
    user = relationship("User", lazy="joined")

    __table_args__ = (
        UniqueConstraint("project_id", "user_id", name="uq_project_user_share"),
    )


class ProjectAllowance(Base):
    """
    Tracks per-user allowance for CommonPool projects, per interval window.
    """
    __tablename__ = "project_allowances"

    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"), index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)

    remaining = Column(Numeric(18, 6), default=0)
    period_start = Column(DateTime, default=datetime.utcnow)
    period_end = Column(DateTime, nullable=True)

    project = relationship("Project", lazy="joined")
    user = relationship("User", lazy="joined")

    __table_args__ = (
        UniqueConstraint("project_id", "user_id", name="uq_project_user_allowance"),
    )


class UsageLog(Base):
    __tablename__ = "usage_log"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    provider = Column(String, nullable=False)
    model = Column(String, nullable=False)
    model_type = Column(SAEnum(ModelType), nullable=False)

    input_count = Column(Integer, default=0)
    output_count = Column(Integer, default=0)
    billed_credits = Column(Numeric(18, 6), default=0)

    request_meta = Column(JSON, default=dict)
    response_meta = Column(JSON, default=dict)

    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", lazy="joined")


class CreditLedger(Base):
    __tablename__ = "credit_ledger"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    delta = Column(Numeric(18, 6), nullable=False)  # +top-up / -spend
    reason = Column(String, default="")
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", lazy="joined")
