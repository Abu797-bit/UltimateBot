# phone_control.py — control your trading bot from Telegram on Render
# Commands: /startbot  /stopbot  /status  /tail  /whoami  /cmd
# Requirements: pyTelegramBotAPI, requests (+ whatever ultimate_bot_v3p1.py needs)
#
# On Render:
#   - Service type: Web Service (Free plan)
#   - Build command:  pip install -r requirements.txt
#   - Start command:  python phone_control.py
#   - Environment variables:
#       TELEGRAM_TOKEN        = <your bot token from BotFather>
#       TELEGRAM_ALLOWED_CHAT = <comma-separated chat IDs, e.g. "123456789,987654321">

import os
import sys
import threading
import time
import subprocess
import re
import signal
import atexit
import importlib.util
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer

import telebot
from telebot import apihelper as tg_api
from telebot.apihelper import ApiTelegramException

# ========= BOT PROCESS CONFIG (Render / Linux friendly) =========

# Directory where this file lives (your repo root on Render)
HERE = Path(__file__).resolve().parent

# Workdir for the trading bot process (we assume ultimate_bot_v3p1.py is in the same repo)
WORKDIR = HERE  # keep as Path; convert to str when passing to subprocess

# Command to launch your trading bot from WORKDIR
BOT_CMD = [
    "python", "-u", "ultimate_bot_v3p1.py",
    "--live-csv",
    # use current working dir (WORKDIR) on Render
    "--data-dir", ".",
    "--prefer-csv", "*_oanda_M5.csv",
    "--log-level", "INFO",
]

LOG_PATH = str(WORKDIR / "bot.log")
LOCK_PATH = WORKDIR / "phone_control.lock"

# Stream everything from the child process into Telegram
LINE_FILTER = re.compile(r".*")

PROC = None
PROC_LOCK = threading.Lock()
STOP_EVENT = threading.Event()

# ---------- Secrets loader ----------


def _import_accesstoken_by_path(p: Path):
    if not p.exists():
        return None
    spec = importlib.util.spec_from_file_location("accesstoken", str(p))
    if not spec or not spec.loader:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def _parse_allowed_ids(val) -> set[int]:
    out: set[int] = set()
    if val is None:
        return out
    if isinstance(val, (set, list, tuple)):
        for x in val:
            try:
                out.add(int(x))
            except Exception:
                pass
        return out
    if isinstance(val, int):
        return {int(val)}
    if isinstance(val, str):
        for part in val.replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            try:
                out.add(int(part))
            except Exception:
                pass
    return out


def _load_secrets():
    """
    Load TELEGRAM_TOKEN and ALLOWED_CHAT_IDS from:
      1) accesstoken.py (next to this file or in WORKDIR), or
      2) env vars TELEGRAM_TOKEN and TELEGRAM_ALLOWED_CHAT
    """
    token = ""
    allowed: set[int] = set()
    src = "unknown"

    # 1) Try accesstoken.py (optional, mainly for local dev)
    for candidate in [HERE / "accesstoken.py", WORKDIR / "accesstoken.py"]:
        try:
            mod = _import_accesstoken_by_path(candidate)
            if mod:
                t = str(getattr(mod, "TELEGRAM_TOKEN", "") or "").strip()
                ids = getattr(mod, "ALLOWED_CHAT_IDS", set())
                if t:
                    token, src = t, str(candidate)
                allowed |= _parse_allowed_ids(ids)
                if token:
                    break
        except Exception as e:
            print(f"Note: couldn't import {candidate}: {e}", file=sys.stderr)

    # 2) Fallback to environment
    if not token:
        token = os.environ.get("TELEGRAM_TOKEN", "").strip()
        if token:
            src = "env TELEGRAM_TOKEN"
    allowed |= _parse_allowed_ids(os.environ.get("TELEGRAM_ALLOWED_CHAT", ""))

    import re as _re
    if not token or not _re.fullmatch(r"\d{6,}:[A-Za-z0-9_-]{30,}", token):
        print(
            "ERROR: TELEGRAM_TOKEN invalid or missing. "
            "Set it as env TELEGRAM_TOKEN (or in accesstoken.py).",
            file=sys.stderr,
        )
        sys.exit(1)
    if not allowed:
        print(
            "ERROR: No ALLOWED_CHAT_IDS configured. "
            "Set TELEGRAM_ALLOWED_CHAT env (e.g. 123456789,987654321) "
            "or ALLOWED_CHAT_IDS in accesstoken.py.",
            file=sys.stderr,
        )
        sys.exit(1)
    masked = token[:8] + "..." if len(token) >= 8 else "***"
    print(
        f"[SECRETS] Loaded token from {src} = {masked}  ALLOWED_CHAT_IDS={sorted(allowed)}"
    )
    return token, allowed


TELEGRAM_TOKEN, ALLOWED_CHAT_IDS = _load_secrets()
bot = telebot.TeleBot(TELEGRAM_TOKEN, parse_mode="Markdown")

# ---------- Single instance lock ----------


def _acquire_lock() -> None:
    try:
        fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        print(f"[LOCK] Acquired {LOCK_PATH}")
    except FileExistsError:
        print(
            f"ERROR: Another controller instance seems running (lock exists: {LOCK_PATH}). "
            f"Delete it if stale.",
            file=sys.stderr,
        )
        sys.exit(1)


def _release_lock():
    try:
        if LOCK_PATH.exists():
            LOCK_PATH.unlink()
            print(f"[LOCK] Released {LOCK_PATH}")
    except Exception:
        pass


atexit.register(_release_lock)

# ---------- Helpers ----------


def _authorized(message) -> bool:
    if message.chat.id not in ALLOWED_CHAT_IDS:
        bot.reply_to(message, "Not authorized for this bot.")
        return False
    return True


def _send_async(chat_id: int, text: str):
    try:
        # Telegram hard limit ~4096 chars; keep a safety margin
        if len(text) <= 4000:
            bot.send_message(chat_id, text)
        else:
            # chunk large text
            for i in range(0, len(text), 4000):
                bot.send_message(chat_id, text[i:i + 4000])
    except Exception:
        # swallow send errors to avoid crashing
        pass


def _stream_output(chat_id: int, proc: subprocess.Popen):
    try:
        for line in iter(proc.stdout.readline, ""):
            if not line:
                break
            line = line.rstrip("\n")
            if LINE_FILTER.search(line):
                _send_async(chat_id, f"```text\n{line}\n```")
            if STOP_EVENT.is_set():
                break
    except Exception as e:
        _send_async(chat_id, f"stream error: {e}")
    finally:
        code = proc.poll()
        _send_async(chat_id, f"`[child exited] code={code}`")

# ---------- Bot process control ----------


def start_bot(chat_id: int) -> str:
    global PROC
    with PROC_LOCK:
        if PROC and PROC.poll() is None:
            return "Bot is already running."

        STOP_EVENT.clear()

        # fresh log each run
        try:
            if os.path.isfile(LOG_PATH):
                os.remove(LOG_PATH)
        except Exception:
            pass

        try:
            if os.name == "nt":
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                creationflags = 0

            env = dict(os.environ)
            env["PYTHONUNBUFFERED"] = "1"

            PROC = subprocess.Popen(
                BOT_CMD,
                cwd=str(WORKDIR),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True,
                creationflags=creationflags,
            )
            threading.Thread(
                target=_stream_output, args=(chat_id, PROC), daemon=True
            ).start()
            return f"Started bot (PID {PROC.pid})."
        except Exception as e:
            return f"Failed to start: {e}"


def stop_bot() -> str:
    global PROC
    with PROC_LOCK:
        if not PROC or PROC.poll() is not None:
            return "Bot is not running."
        try:
            STOP_EVENT.set()
            if os.name == "nt":
                try:
                    PROC.send_signal(signal.CTRL_BREAK_EVENT)
                except Exception:
                    pass
                # wait briefly
                for _ in range(25):
                    if PROC.poll() is not None:
                        break
                    time.sleep(0.2)
                if PROC.poll() is None:
                    subprocess.run(
                        ["taskkill", "/PID", str(PROC.pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
            else:
                PROC.terminate()
                try:
                    PROC.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    PROC.kill()
            return "Stopped bot."
        finally:
            PROC = None


def status_bot() -> str:
    with PROC_LOCK:
        if PROC and PROC.poll() is None:
            return f"Running (PID {PROC.pid})."
        return "Not running."


def tail_log(n: int = 60) -> str:
    try:
        p = Path(LOG_PATH)
        if not p.exists():
            return "No log file yet."
        # fast tail without loading whole file if large
        try:
            with p.open("rb") as fh:
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
                chunk = 64 * 1024
                data = b""
                pos = size
                while pos > 0 and data.count(b"\n") <= n + 1:
                    pos = max(0, pos - chunk)
                    fh.seek(pos)
                    data = fh.read(size - pos) + data
                text = data.decode("utf-8", errors="ignore")
        except Exception:
            text = p.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()
        tail = "\n".join(lines[-n:])
        return "```text\n" + tail[-3800:] + "\n```"
    except Exception as e:
        return f"Tail failed: {e}"

# ---------- Telegram handlers ----------


@bot.message_handler(commands=["start", "help"])
def _help(message):
    if not _authorized(message):
        return
    bot.reply_to(
        message,
        "Commands:\n"
        "• /startbot – start the bot\n"
        "• /stopbot – stop the bot\n"
        "• /status – show if running\n"
        "• /tail – last 60 log lines\n"
        "• /whoami – show your chat id\n"
        "• /cmd – show WORKDIR, CMD, and log path\n",
    )


@bot.message_handler(commands=["whoami"])
def _whoami(message):
    bot.reply_to(message, f"Your chat id: `{message.chat.id}`")


@bot.message_handler(commands=["cmd"])
def _cmd(message):
    if not _authorized(message):
        return
    info = (
        f"WORKDIR: `{WORKDIR}`\n"
        f"CMD: `{' '.join(BOT_CMD)}`\n"
        f"LOG_PATH: `{LOG_PATH}`\n"
    )
    bot.reply_to(message, info)


@bot.message_handler(commands=["startbot"])
def _startbot(message):
    if not _authorized(message):
        return
    bot.reply_to(message, start_bot(message.chat.id))


@bot.message_handler(commands=["stopbot"])
def _stopbot(message):
    if not _authorized(message):
        return
    bot.reply_to(message, stop_bot())


@bot.message_handler(commands=["status"])
def _status(message):
    if not _authorized(message):
        return
    bot.reply_to(message, status_bot())


@bot.message_handler(commands=["tail"])
def _tail(message):
    if not _authorized(message):
        return
    bot.reply_to(message, tail_log(60))

# ---------- HTTP server for Render Web Service ----------


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Simple health endpoint for Render
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"OK\n")

    def log_message(self, format, *args):
        # Silence default HTTP request logging
        return


def _start_http_server():
    port = int(os.environ.get("PORT", "10000"))
    server = HTTPServer(("0.0.0.0", port), _HealthHandler)
    print(f"[HTTP] Listening on 0.0.0.0:{port}")
    server.serve_forever()

# ---------- Polling loop ----------


def _ensure_no_webhook():
    try:
        info = tg_api.get_webhook_info(TELEGRAM_TOKEN)
        print(
            f"[TG] Webhook info: pending={getattr(info, 'pending_update_count', 'n/a')} "
            f"url={getattr(info, 'url', '')}"
        )
    except Exception as e:
        print(f"[TG] getWebhookInfo warn: {e}")
    try:
        tg_api.delete_webhook(TELEGRAM_TOKEN, drop_pending_updates=True)
        print("[TG] deleteWebhook(drop_pending_updates=True) called.")
    except Exception as e:
        print(f"[TG] deleteWebhook warn: {e}")


def _poll_with_409_retry():
    attempts = 0
    while True:
        try:
            bot.infinity_polling(skip_pending=True, timeout=60)
            break
        except ApiTelegramException as e:
            if e.error_code == 409:
                attempts += 1
                print(
                    "[TG] 409 Conflict: another getUpdates is active for this token."
                )
                _ensure_no_webhook()
                if attempts >= 3:
                    print(
                        "\n*** ACTION REQUIRED ***\n"
                        "Another process is polling this bot token.\n"
                        "1) Stop any other python/PM2/servers using this token.\n"
                        "2) Or /revoke the token with BotFather and update TELEGRAM_TOKEN.\n"
                    )
                    raise
                time.sleep(5)
                continue
            raise
        except Exception as e:
            print(f"[TG] polling error: {e}; retrying in 5s...")
            time.sleep(5)


if __name__ == "__main__":
    # Ensure WORKDIR exists (mainly for local use; on Render it's your repo directory)
    WORKDIR.mkdir(parents=True, exist_ok=True)
    print(f"[PATH] WORKDIR={WORKDIR}")
    print(f"[CMD]  {' '.join(BOT_CMD)}")
    print(f"[LOG]  {LOG_PATH}")

    # Check token quickly
    try:
        me = bot.get_me()
        print(f"[TG] getMe ok: @{me.username} id={me.id}")
    except Exception as e:
        print(f"[TG] getMe failed: {e}")

    _ensure_no_webhook()
    _acquire_lock()

    # Start HTTP server in background for Render Web Service
    http_thread = threading.Thread(target=_start_http_server, daemon=True)
    http_thread.start()

    try:
        _poll_with_409_retry()
    finally:
        _release_lock()
