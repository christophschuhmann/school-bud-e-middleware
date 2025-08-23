# db.py
import os
import asyncio
import time
import sqlite3
from pathlib import Path
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker


from dotenv import load_dotenv
load_dotenv()

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import event

from models import Base

# ---------- Persistent data dir & DB path ----------

def _writable_dir(candidate: Path) -> Optional[Path]:
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        testfile = candidate / ".wtest"
        testfile.write_text("ok", encoding="utf-8")
        testfile.unlink(missing_ok=True)
        return candidate
    except Exception:
        return None

def get_data_dir() -> Path:
    # 1) env BUDDY_DATA_DIR
    env_dir = os.getenv("BUDDY_DATA_DIR", "").strip()
    if env_dir:
        p = _writable_dir(Path(env_dir))
        if p:
            return p

    # 2) /var/lib/buddy (if writable)
    p = _writable_dir(Path("/var/lib/buddy"))
    if p:
        return p

    # 3) ./data next to this file
    here = Path(__file__).resolve().parent
    p = _writable_dir(here / "data")
    if p:
        return p

    # 4) last resort: current working dir
    p = _writable_dir(Path.cwd() / "data")
    if p:
        return p

    # If all fail, raise
    raise RuntimeError("No writable data directory found. Set BUDDY_DATA_DIR to a writable path.")

DATA_DIR = get_data_dir()
SQLITE_FILE = Path(os.getenv("BUDDY_DB_PATH", "") or (DATA_DIR / "buddy.db")).resolve()
BACKUP_DIR = _writable_dir(DATA_DIR / "backup") or DATA_DIR

# ---------- Build database URL ----------

def _pick_db_url() -> str:
    raw = os.getenv("DATABASE_URL", "")
    # strip inline comments & whitespace
    val = raw.split("#", 1)[0].strip()
    if val:
        return val

    # Optional fallback to SQLite for rescue/dev
    if os.getenv("SQLITE_FALLBACK", "1") == "1":
        # ensure dir exists
        SQLITE_FILE.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite+aiosqlite:///{SQLITE_FILE.as_posix()}"

    raise RuntimeError(
        "DATABASE_URL is not set (or empty). "
        "Set Postgres URL (postgresql+asyncpg://user:pass@host:5432/db) or enable SQLITE_FALLBACK=1."
    )

DATABASE_URL = _pick_db_url()
IS_SQLITE = DATABASE_URL.startswith("sqlite+aiosqlite://")

# ---------- Engine & PRAGMAs (crash resilience) ----------

engine = create_async_engine(
    DATABASE_URL,
    future=True,
    echo=False,
    pool_pre_ping=True,
)

# Apply crash-safe PRAGMAs for SQLite (WAL journaling, etc.)
if IS_SQLITE:
    @event.listens_for(engine.sync_engine, "connect")
    def _set_sqlite_pragma(dbapi_con, _):
        cur = dbapi_con.cursor()
        # WAL journaling makes restarts safer and concurrent reads fast
        cur.execute("PRAGMA journal_mode=WAL;")
        # Normal sync + WAL is a good balance between durability and speed
        cur.execute("PRAGMA synchronous=NORMAL;")
        # Enforce FK constraints
        cur.execute("PRAGMA foreign_keys=ON;")
        # Keep temp in memory; small cache tweak
        cur.execute("PRAGMA temp_store=MEMORY;")
        cur.execute("PRAGMA cache_size=-2000;")  # ~2MB
        cur.close()

SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

async def get_session() -> AsyncSession:
    async with SessionLocal() as s:
        yield s
async def init_models():
    """
    Create tables and apply forward-compatible SQLite migrations so older DBs
    gain newly added columns/tables without crashing the app.
    """
    async with engine.begin() as conn:
        # 1) create any brand-new tables from models
        await conn.run_sync(Base.metadata.create_all)

        # 2) apply migrations for older SQLite DBs (add columns / tables / indexes)
        if IS_SQLITE:
            await _run_sqlite_migrations(conn)


async def _run_sqlite_migrations(conn):
    """
    SQLite in-place migrations:
      - Add missing columns to 'projects' and 'users'
      - Create 'project_allowances' table if missing
      - Create unique indexes to mimic UniqueConstraint for SQLite
      - Backfill sane defaults
    """
    def migrate(sync_conn):
        # Enable FK enforcement (SQLite)
        sync_conn.exec_driver_sql("PRAGMA foreign_keys = ON")

        def table_exists(name: str) -> bool:
            row = sync_conn.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE type='table' AND name = ?", (name,)
            ).fetchone()
            return row is not None

        def index_exists(name: str) -> bool:
            row = sync_conn.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE type='index' AND name = ?", (name,)
            ).fetchone()
            return row is not None

        def columns(table: str) -> set[str]:
            rows = sync_conn.exec_driver_sql(f"PRAGMA table_info({table})").fetchall()
            # PRAGMA table_info returns: cid, name, type, notnull, dflt_value, pk
            return {r[1] for r in rows}


        if table_exists("projects"):
                pcols = columns("projects")
                if "split_strategy" not in pcols:
                    sync_conn.exec_driver_sql("ALTER TABLE projects ADD COLUMN split_strategy VARCHAR")
                if "has_common_pool" not in pcols:
                    sync_conn.exec_driver_sql("ALTER TABLE projects ADD COLUMN has_common_pool BOOLEAN")
 

        # ---------- projects: add new columns if missing ----------
        if table_exists("projects"):
            pcols = columns("projects")

            if "has_common_pool" not in pcols:
                sync_conn.exec_driver_sql("ALTER TABLE projects ADD COLUMN has_common_pool BOOLEAN DEFAULT 0")
            if "allowance_interval" not in pcols:
                sync_conn.exec_driver_sql("ALTER TABLE projects ADD COLUMN allowance_interval TEXT")
            if "allowance_per_user" not in pcols:
                sync_conn.exec_driver_sql("ALTER TABLE projects ADD COLUMN allowance_per_user NUMERIC DEFAULT 0")
            if "common_pool_balance" not in pcols:
                sync_conn.exec_driver_sql("ALTER TABLE projects ADD COLUMN common_pool_balance NUMERIC DEFAULT 0")
            if "last_reset_at" not in pcols:
                sync_conn.exec_driver_sql("ALTER TABLE projects ADD COLUMN last_reset_at DATETIME")
            if "created_at" not in pcols:
                sync_conn.exec_driver_sql("ALTER TABLE projects ADD COLUMN created_at DATETIME")
                # backfill created_at
                sync_conn.exec_driver_sql("UPDATE projects SET created_at = COALESCE(created_at, CURRENT_TIMESTAMP)")


            if "last_settled_at" not in pcols:
                sync_conn.exec_driver_sql("ALTER TABLE projects ADD COLUMN last_settled_at DATETIME")

            if "next_settle_at" not in pcols:
                sync_conn.exec_driver_sql("ALTER TABLE projects ADD COLUMN next_settle_at DATETIME")

            # Optionaler Backfill aus alten Spalten-Namen (falls vorhanden)
            pcols = columns("projects")  # refresh
            if "last_settlement_at" in pcols and "last_settled_at" in pcols:
                sync_conn.exec_driver_sql(
                    "UPDATE projects SET last_settled_at = COALESCE(last_settled_at, last_settlement_at)"
                )
            if "next_settlement_at" in pcols and "next_settle_at" in pcols:
                sync_conn.exec_driver_sql(
                    "UPDATE projects SET next_settle_at = COALESCE(next_settle_at, next_settlement_at)"
                )

        # ---------- users: created_at column (for new models) ----------
        if table_exists("users"):
            ucols = columns("users")
            if "created_at" not in ucols:
                sync_conn.exec_driver_sql("ALTER TABLE users ADD COLUMN created_at DATETIME")
                sync_conn.exec_driver_sql("UPDATE users SET created_at = COALESCE(created_at, CURRENT_TIMESTAMP)")

        # ---------- project_shares: unique index (mimic UniqueConstraint) ----------
        if table_exists("project_shares") and not index_exists("uq_project_user_share"):
            sync_conn.exec_driver_sql(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_project_user_share "
                "ON project_shares (project_id, user_id)"
            )

        # ---------- project_allowances: create table if missing ----------
        if not table_exists("project_allowances"):
            sync_conn.exec_driver_sql("""
                CREATE TABLE project_allowances (
                  id INTEGER PRIMARY KEY,
                  project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                  remaining NUMERIC DEFAULT 0,
                  period_start DATETIME,
                  period_end DATETIME
                )
            """)
        # and its unique index
        if not index_exists("uq_project_user_allowance"):
            sync_conn.exec_driver_sql(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_project_user_allowance "
                "ON project_allowances (project_id, user_id)"
            )

    # run synchronously inside the open async connection
    await conn.run_sync(migrate)


# ---------- Periodic SQLite backups (safe, consistent) ----------

_backup_task: Optional[asyncio.Task] = None

def _sqlite_backup_once(src: Path, dst: Path):
    # Use sqlite3 backup API to copy a consistent snapshot (includes WAL)
    with sqlite3.connect(src.as_posix()) as source, sqlite3.connect(dst.as_posix()) as dest:
        source.backup(dest)
# --- put these in db.py, replacing the existing ones with the same names ---

async def _periodic_sqlite_backup(interval_sec: int = 600, keep: int = 10):
    from pathlib import Path
    import time, sqlite3, asyncio
    try:
        while True:
            try:
                ts = time.strftime("%Y%m%d-%H%M%S")
                dst = BACKUP_DIR / f"buddy-{ts}.db"
                with sqlite3.connect(SQLITE_FILE.as_posix()) as source, sqlite3.connect(dst.as_posix()) as dest:
                    source.backup(dest)
                files = sorted(BACKUP_DIR.glob("buddy-*.db"))
                if len(files) > keep:
                    for f in files[:-keep]:
                        try: f.unlink(missing_ok=True)
                        except Exception: pass
            except Exception as e:
                print(f"[backup] error: {e}")
            await asyncio.sleep(interval_sec)
    except asyncio.CancelledError:
        # graceful shutdown of the task
        return

def start_backup_task(interval_sec: int = 600, keep: int = 10):
    # start only once
    global _backup_task
    if not IS_SQLITE:
        return
    if _backup_task and not _backup_task.done():
        return
    import asyncio
    _backup_task = asyncio.create_task(_periodic_sqlite_backup(interval_sec, keep))

async def stop_backup_task():
    global _backup_task
    if _backup_task:
        _backup_task.cancel()
        try:
            await _backup_task
        except BaseException:
            # swallow CancelledError and any shutdown noise
            pass
        _backup_task = None
