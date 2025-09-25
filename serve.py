# -*- coding: utf-8 -*-
"""
serve.py

Entry point that:
  1) Starts the Buddy middleware (FastAPI + Admin UI).
  2) Optionally configures & launches a local "Vertex OpenAI proxy"
     that bridges to Google Vertex AI (Gemini) + Cloud Text-to-Speech (TTS).
  3) Persists configuration (.env / JSON) for smooth re-runs.
  4) Protects the Admin webpage with a password (bcrypt hash stored locally).
  5) (Simplified) Shows a short info guide about Vertex regions/models with
     a link to Google's official regional availability page, then asks the
     user to pick a default region and model. No auto-scanning is performed.

Why this version is simpler:
  - Instead of probing APIs that may not enumerate models for your project,
    we show an info panel with common endpoints and a direct link to
    Google's official availability page:
    https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations
  - You then select the region (e.g., europe-west4) and model
    (e.g., gemini-2.5-flash). This is reliable and fast.
"""

import os, sys, socket, pathlib, atexit, signal, subprocess, time, re, json, getpass
from typing import Dict, List, Tuple, Optional, Set

# --- Middleware imports (unchanged) ------------------------------------------
import uvicorn
from main import app
from admin import router as admin_router

app.include_router(admin_router)

# Paths for local files (.env, report, admin password config)
APP_DIR = pathlib.Path(__file__).resolve().parent
ENV_FILE = APP_DIR / ".env"
REPORT_FILE = APP_DIR / "vertex_inventory_report.txt"  # kept; not used by this flow

# ------------------------
# Middleware server config
# ------------------------
HOST = os.getenv("BUDDY_HOST", "0.0.0.0")
PORT = int(os.getenv("BUDDY_PORT", "8787"))
PID_FILE = pathlib.Path(os.getenv("BUDDY_PID", str(APP_DIR / "buddyserver.pid")))

# ------------------------
# Vertex proxy config keys
# ------------------------
V_KEYS = {
    "VERTEX_REGION":      None,  # e.g. "europe-west4"
    "VERTEX_MODEL_NAME":  "gemini-2.5-flash",
    "VERTEX_PROJECT_ID":  None,
    "VERTEX_SA_JSON":     None,  # absolute path to service account JSON
    # optional extras:
    "VERTEX_PROXY_HOST":  "127.0.0.1",
    "VERTEX_PROXY_PORT":  "8001",
}

# Candidate regions (for display only; user may choose others supported by Google)
EU_CANDIDATE_REGIONS = [
    "europe-west1", "europe-west2", "europe-west3", "europe-west4",
    "europe-west9", "europe-central2"
]
US_GLOBAL_REGIONS = [
    "us-central1", "us-east1", "us-east4", "us-west1", "us-west4",
    "global"
]

# === Admin Website password config ===========================================
import bcrypt
ADMIN_CFG_FILE = APP_DIR / "admin-website-config.json"

def load_admin_cfg() -> dict:
    if ADMIN_CFG_FILE.exists():
        try:
            return json.loads(ADMIN_CFG_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def save_admin_cfg(cfg: dict) -> None:
    ADMIN_CFG_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

def ensure_admin_password_interactive():
    """
    Ensure the admin password exists:
      - If not set, prompt for a new password (twice), store bcrypt hash.
      - If already set, ask whether to change it; if yes, prompt again.
    This runs before the server starts so the Admin UI is never exposed unguarded.
    """
    cfg = load_admin_cfg()
    if "password_bcrypt" not in cfg:
        print("\nAdmin website doesnt have a password yet.")
        print("Set a strong password for the admin console.\n")
        while True:
            pw1 = getpass.getpass("New admin password: ")
            pw2 = getpass.getpass("Repeat password     : ")
            if not pw1:
                print("Password cannot be empty.")
                continue
            if pw1 != pw2:
                print("Passwords do not match, try again.")
                continue
            break
        hashed = bcrypt.hashpw(pw1.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        cfg["password_bcrypt"] = hashed
        save_admin_cfg(cfg)
        print(f"Saved admin password hash to {ADMIN_CFG_FILE}")
    else:
        print("\nAn admin password is already configured.")
        ans = input("Change it? [y/N or c=change] : ").strip().lower()
        if ans in {"y", "yes", "c", "change"}:
            while True:
                pw1 = getpass.getpass("New admin password: ")
                pw2 = getpass.getpass("Repeat password     : ")
                if not pw1:
                    print("Password cannot be empty.")
                    continue
                if pw1 != pw2:
                    print("Passwords do not match, try again.")
                    continue
                break
            hashed = bcrypt.hashpw(pw1.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
            cfg["password_bcrypt"] = hashed
            save_admin_cfg(cfg)
            print("Admin password updated.")

# -------------
# Small helpers
# -------------
def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0

def write_pid():
    PID_FILE.write_text(str(os.getpid()), encoding="utf-8")

def remove_pid():
    try: PID_FILE.unlink()
    except FileNotFoundError: pass

def handle_exit(_sig=None, _frm=None):
    remove_pid()
    raise SystemExit(0)

def prompt_yes_no(msg: str, default: bool = False) -> bool:
    dv = "Y/n" if default else "y/N"
    ans = input(f"{msg} [{dv}]: ").strip().lower()
    if not ans:
        return default
    return ans in {"y", "yes"}

def prompt_nonempty(msg: str, default: Optional[str] = None) -> str:
    while True:
        raw = input(f"{msg}{' ['+default+']' if default else ''}: ").strip()
        if not raw and default:
            return default
        if raw:
            return raw
        print("Please enter a value.")

def load_env_file(path: pathlib.Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out

def save_env_updates(path: pathlib.Path, updates: Dict[str, str]) -> None:
    env = load_env_file(path)
    env.update({k: str(v) for k, v in updates.items() if v is not None})
    lines = [f"{k}={env[k]}" for k in sorted(env.keys())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def env_or_envfile(key: str) -> Optional[str]:
    if key in os.environ and os.environ[key].strip():
        return os.environ[key].strip()
    env = load_env_file(ENV_FILE)
    v = env.get(key)
    return v.strip() if v else None

def vertex_cfg_from_env() -> Dict[str, Optional[str]]:
    cfg = {}
    for k in V_KEYS:
        cfg[k] = env_or_envfile(k)
    if not cfg.get("VERTEX_MODEL_NAME"):
        cfg["VERTEX_MODEL_NAME"] = V_KEYS["VERTEX_MODEL_NAME"]
    if not cfg.get("VERTEX_PROXY_HOST"):
        cfg["VERTEX_PROXY_HOST"] = V_KEYS["VERTEX_PROXY_HOST"]
    if not cfg.get("VERTEX_PROXY_PORT"):
        cfg["VERTEX_PROXY_PORT"] = V_KEYS["VERTEX_PROXY_PORT"]
    return cfg

def validate_vertex_cfg(cfg: Dict[str, Optional[str]]) -> Tuple[bool, List[str]]:
    errs = []
    if not cfg.get("VERTEX_REGION"):
        errs.append("Missing VERTEX_REGION.")
    if not cfg.get("VERTEX_PROJECT_ID"):
        errs.append("Missing VERTEX_PROJECT_ID.")
    if not cfg.get("VERTEX_SA_JSON"):
        errs.append("Missing VERTEX_SA_JSON.")
    else:
        p = pathlib.Path(cfg["VERTEX_SA_JSON"])
        if not p.exists():
            errs.append(f"Service account JSON not found: {p}")
    ok = len(errs) == 0
    return ok, errs

# ---------------------------------------------------------------------------
# (Legacy) Scan helpers  kept for future use but NOT invoked in this version
# ---------------------------------------------------------------------------
def try_imports_for_scan():
    import importlib, subprocess
    def _ensure(pkg, mod=None):
        try:
            importlib.import_module(mod or pkg.replace("-", "_"))
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
    _ensure("google-cloud-aiplatform", "google.cloud.aiplatform")
    _ensure("google-cloud-texttospeech", "google.cloud.texttospeech")
    _ensure("vertexai", "vertexai")

def list_publisher_models_all(sa_json: str, project_id: str, regions: List[str]) -> Dict[str, List[str]]:
    # intentionally unused in this simplified flow
    return {r: [] for r in regions}

def fetch_gemini_models_via_developer_api() -> List[str]:
    # intentionally unused in this simplified flow
    return []

# --------------------------------------------------
# Interactive configuration flow for Vertex proxy
# --------------------------------------------------
def _print_vertex_region_info():
    """
    Print a short, friendly info panel:
      - Link to Google's official availability page
      - Commonly used endpoints
      - Privacy note for EU/German school contexts (non-legal, informational)
    """
    print("\n--- Vertex Regions & Models  Quick Info ---")
    print("You can look up which Gemini models are available in which region here:")
    print("  https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations")
    print("")
    print("Common endpoints you might use:")
    print("  EU (recommended for German schools / EU privacy):")
    print("    - europe-west4 (Netherlands)  ? popular; broad model coverage")
    print("    - europe-west1 (Belgium)")
    print("    - europe-west2 (London, UK)")
    print("    - europe-west3 (Frankfurt, DE)")
    print("    - europe-west9 (Paris, FR)")
    print("    - europe-central2 (Warsaw, PL)")
    print("  US:")
    print("    - us-central1 (Iowa)")
    print("    - us-east1 (S. Carolina) / us-east4 (N. Virginia)")
    print("    - us-west1 (Oregon) / us-west4 (Nevada)")
    print("")
    print("Privacy note (informational, not legal advice):")
    print("  - EU endpoints help keep data in the EU and are commonly chosen by schools in Germany.")
    print("  - By default, Google Vertex AI states that customer data is not used to train models.")
    print("    Always confirm your organizations policy & Google Cloud settings with your IT/DPO.")
    print("")

def configure_vertex_proxy_interactive(existing: Dict[str, Optional[str]]) -> Dict[str, str]:
    """
    Simplified wizard that:
      - Shows an info panel with the official link to region/model availability,
        a list of common endpoints, and a short privacy note.
      - Prompts for Service Account JSON + Project ID.
      - Prompts for default region and default model (typed by the user).
      - Saves choices into .env for next runs.
    """
    print("\n--- Vertex Proxy Setup (EU privacy compliant) ---")
    _print_vertex_region_info()

    # Ask for SA JSON & Project
    sa_json = prompt_nonempty("Path to Google Service Account JSON", existing.get("VERTEX_SA_JSON"))
    while not pathlib.Path(sa_json).exists():
        print("File not found. Please try again.")
        sa_json = prompt_nonempty("Path to Google Service Account JSON", None)

    project_id = prompt_nonempty("Your Google Cloud Project ID", existing.get("VERTEX_PROJECT_ID"))

    # No scanning; user selects region/model themselves (referencing the link/info above)
    default_region = existing.get("VERTEX_REGION") or "europe-west4"
    region = prompt_nonempty("Default Vertex region (e.g., europe-west4)", default_region)

    default_model = existing.get("VERTEX_MODEL_NAME") or "gemini-2.5-flash"
    model_name = prompt_nonempty("Default Gemini model (e.g., gemini-2.5-flash)", default_model)

    proxy_host = prompt_nonempty("Vertex proxy host", existing.get("VERTEX_PROXY_HOST") or "127.0.0.1")
    proxy_port = prompt_nonempty("Vertex proxy port", existing.get("VERTEX_PROXY_PORT") or "8001")

    updates = {
        "VERTEX_REGION": region,
        "VERTEX_MODEL_NAME": model_name,
        "VERTEX_PROJECT_ID": project_id,
        "VERTEX_SA_JSON": sa_json,
        "VERTEX_PROXY_HOST": proxy_host,
        "VERTEX_PROXY_PORT": proxy_port,
    }
    save_env_updates(ENV_FILE, updates)
    print(f"\nSaved Vertex proxy configuration to {ENV_FILE}")
    print("\nTip: If you later want to change the region/model, re-run this script and choose to modify settings.")
    return updates

def maybe_launch_vertex_proxy(cfg: Dict[str, str]) -> Optional[subprocess.Popen]:
    """
    Launch the local Vertex proxy via uvicorn:
        uvicorn vertex_openai_proxy:app --host HOST --port PORT
    If the port is already bound, we assume the proxy is running and skip launch.
    """
    host = cfg.get("VERTEX_PROXY_HOST") or "127.0.0.1"
    port = int(cfg.get("VERTEX_PROXY_PORT") or "8001")

    if is_port_in_use(port, host=host):
        print(f"[vertex-proxy] {host}:{port} already in use; assuming proxy is running.")
        return None

    # Export VERTEX_* vars so the proxy reads them on boot
    for k in V_KEYS:
        if k in cfg and cfg[k] is not None:
            os.environ[k] = str(cfg[k])

    cmd = [
        sys.executable, "-m", "uvicorn",
        "vertex_openai_proxy:app",
        "--host", host,
        "--port", str(port),
        "--log-level", "info",
    ]
    print(f"[vertex-proxy] starting: {' '.join(cmd)}")
    try:
        proc = subprocess.Popen(cmd, cwd=str(APP_DIR))
        time.sleep(1.0)
        if proc.poll() is None:
            print(f"[vertex-proxy] launched on http://{host}:{port}/")
            return proc
        else:
            print("[vertex-proxy] failed to start (process exited early).")
            return None
    except Exception as e:
        print(f"[vertex-proxy] launch error: {e}")
        return None

# -------- Public URL helpers (print both local & public Admin URLs) ----------
def _detect_public_ip(timeout: float = 2.0) -> Optional[str]:
    """
    Best-effort public IP detection:
      1) Try ipify.org over HTTPS via urllib (no extra deps).
      2) Fallback: outward-facing local IP via UDP socket to 8.8.8.8
         (may be private behind NAT; still useful on a LAN).
    """
    try:
        import urllib.request
        with urllib.request.urlopen("https://api.ipify.org", timeout=timeout) as r:
            ip = r.read().decode("utf-8").strip()
            if ip:
                return ip
    except Exception:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(timeout)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return None

def _print_admin_urls(host: str, port: int):
    """
    Print both local & best-effort public Admin URLs with instructions.
    """
    local_base = f"http://127.0.0.1:{port}"
    admin_candidates = ["/static/admin.html", "/admin", "/"]
    local_admins = [local_base + p for p in admin_candidates]

    public_ip = _detect_public_ip()
    public_admins = [f"http://{public_ip}:{port}{p}" for p in admin_candidates] if public_ip else []

    print("\n[Admin Console]")
    print(f"- Local (this machine): {local_admins[0]}  (try the next ones if that path 404s)")
    for alt in local_admins[1:]:
        print(f"                         {alt}")
    if public_admins:
        print(f"- Public (other device/browser or LAN):")
        print(f"    {public_admins[0]}  (alternates below if needed)")
        for alt in public_admins[1:]:
            print(f"    {alt}")
    else:
        print("- Public: Couldnt auto-detect a public IP. If behind NAT/firewall, expose or port-map,")
        print("          then use your public IP or domain with the same port.")

    print("\nHow to log in:")
    print("1) Open one of the Admin URLs above in your browser.")
    print("2) When prompted, enter the admin password you set earlier.")
    print("3) After login, go to Providers/Routes.")
    print("   Provider name for Vertex must be exactly: vertex")
    print("   Base URL for Vertex proxy: http://127.0.0.1:8001/v1 (or your chosen host/port)")

# -------------------------
# Main entry (interactive)
# -------------------------
if __name__ == "__main__":
    # Prevent double-starts
    if PID_FILE.exists():
        try:
            old = int(PID_FILE.read_text().strip())
            if old > 0:
                try: os.kill(old, 0)
                except OSError: pass
                else:
                    print(f"[serve] instance already running (pid {old}); exiting.")
                    raise SystemExit(0)
        except Exception:
            pass

    if is_port_in_use(PORT, host="127.0.0.1"):
        print(f"[serve] port {PORT} already in use; not starting a second instance.")
        raise SystemExit(1)

    # Protect Admin UI before starting anything
    ensure_admin_password_interactive()

    # 1) Vertex proxy config
    vcfg = vertex_cfg_from_env()
    ok, errs = validate_vertex_cfg(vcfg)

    if not ok:
        print("\nVertex proxy is not fully configured yet.")
        for e in errs:
            print(" -", e)
        print("\nYou can configure and start the Vertex proxy now (EU-compliant endpoints),")
        print("or skip it and start only the classic middleware.\n")
        if prompt_yes_no("Configure Vertex proxy now?", default=True):
            vcfg = configure_vertex_proxy_interactive(vcfg)
            ok, errs = validate_vertex_cfg(vcfg)
            if not ok:
                print("\nConfiguration still incomplete:")
                for e in errs: print(" -", e)
                print("Starting classic middleware without Vertex proxy.\n")
                vproc = None
            else:
                vproc = maybe_launch_vertex_proxy(vcfg)
        else:
            vproc = None
    else:
        # Already configured  show summary and offer to change.
        print("\nCurrent Vertex proxy configuration:")
        print(f"  Region      : {vcfg['VERTEX_REGION']}")
        print(f"  Model       : {vcfg['VERTEX_MODEL_NAME']}")
        print(f"  Project ID  : {vcfg['VERTEX_PROJECT_ID']}")
        print(f"  SA JSON     : {vcfg['VERTEX_SA_JSON']}")
        print(f"  Proxy URL   : http://{vcfg.get('VERTEX_PROXY_HOST','127.0.0.1')}:{vcfg.get('VERTEX_PROXY_PORT','8001')}/")
        if prompt_yes_no("Change these settings?", default=False):
            vcfg = configure_vertex_proxy_interactive(vcfg)
            ok, errs = validate_vertex_cfg(vcfg)
            if not ok:
                print("\nConfiguration incomplete; starting classic middleware without Vertex proxy.")
                vproc = None
            else:
                vproc = maybe_launch_vertex_proxy(vcfg)
        else:
            vproc = maybe_launch_vertex_proxy(vcfg)

    # 2) Reminder + both Admin URLs
    if ok:
        print("\n[Reminder] In Buddy Admin ? Providers:")
        print("  - name your provider exactly: vertex  (lowercase)")
        print("  - base_url:  http://127.0.0.1:8001/v1   (or your chosen host/port)")
        print("  - api_key:   any non-empty string (middleware requires a value)")
        print("In Routes, add a TTS route with provider=vertex, priority=1, enabled=true.")
        print("Model can be: auto | de-DE-Chirp3-HD-Achernar | en-US-Wavenet-D | etc.\n")
    _print_admin_urls(HOST, PORT)

    # 3) Start middleware
    write_pid()
    atexit.register(remove_pid)
    signal.signal(signal.SIGTERM, handle_exit)
    signal.signal(signal.SIGINT, handle_exit)

    print(f"[serve] starting middleware on http://{HOST}:{PORT}/")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
