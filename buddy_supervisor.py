# buddy_supervisor.py
import os, sys, time, socket, subprocess, signal, pathlib, urllib.request, urllib.error

APP_DIR = pathlib.Path(__file__).resolve().parent
SERVE = str(APP_DIR / "serve.py")
HOST = os.getenv("BUDDY_HOST", "0.0.0.0")
PORT = int(os.getenv("BUDDY_PORT", "8787"))
PIDFILE = APP_DIR / "buddy_supervisor_child.pid"
HEALTH_URL = f"http://127.0.0.1:{PORT}/healthz"

BACKOFF_START = 2
BACKOFF_MAX = 30

def port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0

def health_ok(timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False

def write_pid(pid: int):
    PIDFILE.write_text(str(pid), encoding="utf-8")

def read_pid() -> int | None:
    if PIDFILE.exists():
        try: return int(PIDFILE.read_text().strip())
        except Exception: return None
    return None

def kill_child():
    pid = read_pid()
    if not pid: return
    try:
        os.kill(pid, signal.SIGTERM)
        for _ in range(40):
            time.sleep(0.1)
            os.kill(pid, 0)
        # if still alive after ~4s, use SIGKILL
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except Exception:
        pass
    try: PIDFILE.unlink()
    except FileNotFoundError: pass

def start_child() -> subprocess.Popen:
    env = os.environ.copy()
    # make sure we use the same interpreter (your venv)
    cmd = [sys.executable, SERVE]
    proc = subprocess.Popen(cmd, cwd=str(APP_DIR))
    write_pid(proc.pid)
    return proc

def main():
    backoff = BACKOFF_START
    child: subprocess.Popen | None = None

    def handle_sigterm(_sig, _frm):
        kill_child()
        sys.exit(0)
    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)

    while True:
        # If healthy and port is serving, just sleep a bit
        if port_in_use(PORT) and health_ok():
            backoff = BACKOFF_START
            time.sleep(5)
            continue

        # If we had started a child previously, check if it died
        pid = read_pid()
        if pid:
            # process may be ours but dead; clean pidfile
            try:
                os.kill(pid, 0)
                # process exists but health failed → restart it
                kill_child()
            except ProcessLookupError:
                try: PIDFILE.unlink()
                except FileNotFoundError: pass

        # If some OTHER process is on the port, wait (don’t start a second instance)
        if port_in_use(PORT) and not health_ok():
            time.sleep(5)
            continue

        # Start (or restart) our child
        child = start_child()
        # give it a few seconds to bind and report healthy
        for _ in range(20):
            time.sleep(0.5)
            if health_ok():
                backoff = BACKOFF_START
                break
        else:
            # not healthy → kill and backoff
            kill_child()
            time.sleep(backoff)
            backoff = min(BACKOFF_MAX, backoff * 2)

if __name__ == "__main__":
    main()
