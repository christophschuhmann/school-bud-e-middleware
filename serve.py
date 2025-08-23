# serve.py
import os, socket, pathlib, atexit, signal
import uvicorn
from main import app
from admin import router as admin_router

app.include_router(admin_router)

HOST = os.getenv("BUDDY_HOST", "0.0.0.0")
PORT = int(os.getenv("BUDDY_PORT", "8787"))
APP_DIR = pathlib.Path(__file__).resolve().parent
PID_FILE = pathlib.Path(os.getenv("BUDDY_PID", str(APP_DIR / "buddyserver.pid")))

def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0

def write_pid():
    PID_FILE.write_text(str(os.getpid()), encoding="utf-8")

def remove_pid():
    try: PID_FILE.unlink()
    except FileNotFoundError: pass

def handle_exit(_sig=None, _frm=None):
    remove_pid()
    raise SystemExit(0)

if __name__ == "__main__":
    # prevent duplicates via pidfile and port probe
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

    if is_port_in_use(PORT):
        print(f"[serve] port {PORT} already in use; not starting a second instance.")
        raise SystemExit(1)

    write_pid()
    atexit.register(remove_pid)
    signal.signal(signal.SIGTERM, handle_exit)
    signal.signal(signal.SIGINT, handle_exit)

    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
