"""Small web server: the caption display, the operator control page, and a live event stream."""

from __future__ import annotations

import io
import json
import mimetypes
import queue
import secrets
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

WEB_DIR = Path(__file__).resolve().parent / "web"
LOCAL = {"127.0.0.1", "::1", "::ffff:127.0.0.1"}


def lan_ip() -> str | None:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.254.254.254", 1))  # no packet is sent
        ip = s.getsockname()[0]
        return None if ip.startswith("127.") else ip
    except OSError:
        return None
    finally:
        s.close()


def qr_svg(text: str) -> bytes | None:
    try:
        import qrcode
        import qrcode.image.svg
    except ImportError:
        return None
    img = qrcode.make(text, image_factory=qrcode.image.svg.SvgPathFillImage, box_size=10, border=2)
    buf = io.BytesIO()
    img.save(buf)
    return buf.getvalue()


class CaptionServer:
    def __init__(self, hub, pipeline, host: str, port: int, config_path: str | None = None):
        self.hub = hub
        self.pipeline = pipeline
        self.config_path = config_path
        self.host = host
        self.port = port
        self.key = secrets.token_urlsafe(6)
        self.httpd = ThreadingHTTPServer((host, port), self._handler())
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]

    def urls(self) -> dict:
        local = f"http://127.0.0.1:{self.port}"
        out = {"display": f"{local}/", "control": f"{local}/control"}
        ip = lan_ip() if self.host in ("0.0.0.0", "::") else None
        if ip:
            out["lan"] = f"http://{ip}:{self.port}/"
            out["lan_control"] = f"http://{ip}:{self.port}/control?key={self.key}"
        return out

    def start(self):
        threading.Thread(target=self.httpd.serve_forever, name="http", daemon=True).start()

    def stop(self):
        self.httpd.shutdown()

    # ---------------------------------------------------------------- request handling

    def _handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt, *args):  # keep the console for captions
                pass

            def _send(self, code: int, body: bytes, ctype: str = "application/json"):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _json(self, obj, code: int = 200):
                self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

            def _allowed(self, query) -> bool:
                if self.client_address[0] in LOCAL:
                    return True
                key = (query.get("key") or [""])[0] or self.headers.get("X-Control-Key", "")
                return secrets.compare_digest(key, server.key)

            # -- GET ---------------------------------------------------------

            def do_GET(self):
                url = urlparse(self.path)
                query = parse_qs(url.query)
                path = url.path
                if path == "/events":
                    return self._events(status="status" in query and self._allowed(query))
                if path == "/api/state":
                    return self._json(server.hub.snapshot())
                if path == "/api/info":
                    return self._json({"urls": server.urls(), "control": self._allowed(query)})
                if path == "/api/devices":
                    if not self._allowed(query):
                        return self._json({"error": "not allowed"}, 403)
                    return self._json(self._devices())
                if path == "/qr.svg":
                    target = server.urls().get("lan") or server.urls()["display"]
                    svg = qr_svg(target)
                    if svg is None:
                        return self._send(404, b"pip install qrcode", "text/plain")
                    return self._send(200, svg, "image/svg+xml")
                if path in ("/", "/display"):
                    return self._file("display.html")
                if path == "/control":
                    return self._file("control.html")
                name = path.lstrip("/")
                if "/" not in name and (WEB_DIR / name).is_file():
                    return self._file(name)
                self._send(404, b"not found", "text/plain")

            def _devices(self) -> dict:
                from .audio import MicSource, list_input_devices

                src = server.pipeline.source
                current = getattr(src, "device", None) if isinstance(src, MicSource) else None
                try:
                    devs = [d for d in list_input_devices() if "WDM-KS" not in d["hostapi"]]
                except BaseException as e:
                    return {"devices": [], "current": None, "error": str(e), "is_mic": False}
                return {
                    "devices": devs,
                    "current": current,
                    "current_name": getattr(src, "name", ""),
                    "is_mic": isinstance(src, MicSource),
                }

            def _file(self, name: str):
                p = WEB_DIR / name
                ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
                if ctype.startswith("text/") or ctype.endswith("javascript"):
                    ctype += "; charset=utf-8"
                self._send(200, p.read_bytes(), ctype)

            def _events(self, status: bool):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()
                q = server.hub.subscribe(status=status)
                try:
                    self.wfile.write(b"retry: 1000\n\n")
                    while server.pipeline.running:
                        try:
                            ev = q.get(timeout=10)
                            data = json.dumps(ev, ensure_ascii=False)
                            self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                        except queue.Empty:
                            self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                    pass
                finally:
                    server.hub.unsubscribe(q)
                    self.close_connection = True

            # -- POST --------------------------------------------------------

            def do_POST(self):
                url = urlparse(self.path)
                query = parse_qs(url.query)
                length = int(self.headers.get("Content-Length") or 0)
                try:
                    body = json.loads(self.rfile.read(length) or b"{}") if length else {}
                except json.JSONDecodeError:
                    return self._json({"error": "bad json"}, 400)
                if not self._allowed(query):
                    return self._json({"error": "control is only allowed from this computer, or with ?key="}, 403)
                hub, pipe = server.hub, server.pipeline
                path = url.path
                if path == "/api/mode":
                    mode = body.get("mode")
                    if mode not in ("auto", "en", "zh"):
                        return self._json({"error": "mode must be auto, en or zh"}, 400)
                    if mode in ("en", "zh"):
                        pipe.current_lang = mode
                    return self._json(hub.update_settings(mode=mode))
                if path == "/api/pause":
                    return self._json(hub.update_settings(paused=bool(body.get("paused", True))))
                if path == "/api/clear":
                    hub.clear()
                    return self._json({"ok": True})
                if path == "/api/remove":
                    return self._json({"ok": hub.remove(int(body.get("id", -1)))})
                if path == "/api/message":
                    return self._json(hub.update_settings(message=str(body.get("text", ""))[:300]))
                if path == "/api/device":
                    from .audio import list_input_devices

                    try:
                        idx = int(body.get("index"))
                        dev = next(d for d in list_input_devices() if d["index"] == idx)
                    except (TypeError, ValueError, StopIteration):
                        return self._json({"error": "unknown device"}, 400)
                    try:
                        name = pipe.switch_device(idx)
                    except Exception as e:
                        return self._json({"error": f"couldn't open {dev['name']}: {e}"}, 500)
                    saved = False
                    if server.config_path:
                        from .config import save_setting

                        saved = save_setting(server.config_path, "audio", "device", dev["name"]) and save_setting(
                            server.config_path, "audio", "device_api", dev["hostapi"]
                        )
                    print(f"[audio] switched to: {name} ({dev['hostapi']})" + (" — saved in config.toml" if saved else ""))
                    return self._json({"ok": True, "name": name, "saved": saved, **self._devices()})
                self._json({"error": "unknown endpoint"}, 404)

        return Handler
