"""The local web server: Python's ``http.server`` on 127.0.0.1, a small JSON API and three files.

Local only, by construction:

* it binds 127.0.0.1 and refuses any other address;
* requests must name this server in ``Host`` (127.0.0.1 or localhost at its port), which stops
  a web page elsewhere from reaching it through a rebound DNS name;
* a ``POST`` must be ``application/json`` (a cross-site form cannot send that without the
  browser asking first), its ``Origin``, when sent, must be this server, and its body is at most
  ``MAX_BODY`` bytes;
* the page loads nothing from outside (vanilla HTML, CSS and JS served from ``static/``, with a
  Content-Security-Policy of ``'self'``), and the server makes no outgoing requests.

API (JSON in and out): ``GET /api/state``, ``GET /api/options``; ``POST /api/new`` (settings),
``/api/act`` ({token, action, amount?}), ``/api/next``, ``/api/read`` ({on}), ``/api/end``.
Bad input is 400 with ``{"error": ...}``; a stale decision or a request that does not fit the
session's state is 409; a forbidden origin or host is 403.
"""

from __future__ import annotations

import json
import threading
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from poker_lab.play import pushfold
from poker_lab.play.channel import StaleDecision
from poker_lab.play.data import DataDir
from poker_lab.play.exploit_slot import DEFAULT, ENGINES
from poker_lab.play.session import Conflict, Session, Settings

HOST = "127.0.0.1"
MAX_BODY = 4096
STATIC = Path(__file__).with_name("static")
FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
}
CSP = (
    "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
    "img-src 'self' data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)


class BadRequest(ValueError):
    def __init__(self, message: str, status: HTTPStatus = HTTPStatus.BAD_REQUEST) -> None:
        super().__init__(message)
        self.status = status


class PlayServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, port: int, data: DataDir, *, host: str = HOST, **session_kw: Any) -> None:
        if host != HOST:
            raise ValueError("the play server binds 127.0.0.1 only")
        super().__init__((host, port), Handler)
        self.data = data
        self.session_kw = session_kw
        self.session: Session | None = None
        self.session_lock = threading.Lock()
        p = self.server_address[1]
        self.hosts = {f"127.0.0.1:{p}", f"localhost:{p}"}
        self.origins = {f"http://{h}" for h in self.hosts}

    def end_session(self) -> dict[str, Any] | None:
        with self.session_lock:
            return self.session.end() if self.session is not None else None


class Handler(BaseHTTPRequestHandler):
    server: PlayServer
    server_version = "PokerLabPlay"
    sys_version = ""

    def log_message(self, format: str, *args: Any) -> None:
        pass  # keep the console for the URL and errors

    # ------------------------------------------------------------------ responses
    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", CSP)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, obj: Any) -> None:
        self._send(status, json.dumps(obj).encode("utf-8"), "application/json")

    def _error(self, status: int, message: str) -> None:
        self._json(status, {"error": message})

    def _check_host(self) -> None:
        if self.headers.get("Host") not in self.server.hosts:
            raise BadRequest("unknown host", HTTPStatus.FORBIDDEN)

    # ------------------------------------------------------------------ GET
    def do_GET(self) -> None:
        try:
            self._check_host()
        except BadRequest as e:
            return self._error(e.status, str(e))
        path = self.path.split("?", 1)[0]
        if path in FILES:
            name, ctype = FILES[path]
            return self._send(HTTPStatus.OK, (STATIC / name).read_bytes(), ctype)
        if path == "/api/state":
            s = self.server.session
            return self._json(HTTPStatus.OK, {"session": None} if s is None else s.state())
        if path == "/api/options":
            return self._json(
                HTTPStatus.OK,
                {
                    "stack": [pushfold.MIN_STACK, pushfold.MAX_STACK],
                    "classes": pushfold.classes(),
                    "engines": [
                        {"name": e.name, "kind": e.kind, "description": e.description}
                        for e in ENGINES.values()
                    ],
                    "default_engine": DEFAULT,
                    "data_dir": str(self.server.data.root),
                },
            )
        self._error(HTTPStatus.NOT_FOUND, "not found")

    # ------------------------------------------------------------------ POST
    def _body(self) -> Any:
        self._check_host()
        origin = self.headers.get("Origin")
        if origin is not None and origin not in self.server.origins:
            raise BadRequest("cross-origin request", HTTPStatus.FORBIDDEN)
        if self.headers.get("Sec-Fetch-Site") not in (None, "same-origin", "none"):
            raise BadRequest("cross-site request", HTTPStatus.FORBIDDEN)
        ctype = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
        if ctype != "application/json":
            raise BadRequest(
                "Content-Type must be application/json", HTTPStatus.UNSUPPORTED_MEDIA_TYPE
            )
        length = self.headers.get("Content-Length")
        if length is None or not length.isdigit():
            raise BadRequest("Content-Length required", HTTPStatus.LENGTH_REQUIRED)
        n = int(length)
        if n > MAX_BODY:
            raise BadRequest("body too large", HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
        raw = self.rfile.read(n)
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
            raise BadRequest("body is not valid JSON") from None

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        try:
            body = self._body()
            result = self._route(path, body)
        except BadRequest as e:
            if e.status == HTTPStatus.REQUEST_ENTITY_TOO_LARGE:
                self.close_connection = True
            return self._error(e.status, str(e))
        except (StaleDecision, Conflict) as e:
            return self._error(HTTPStatus.CONFLICT, str(e))
        except ValueError as e:  # includes IllegalActionError
            return self._error(HTTPStatus.BAD_REQUEST, str(e))
        except Exception as e:  # a bug: answer instead of dropping the connection
            traceback.print_exc()
            return self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(e).__name__}: {e}")
        self._json(HTTPStatus.OK, result)

    def _route(self, path: str, body: Any) -> Any:
        srv = self.server
        if path == "/api/new":
            settings = Settings.parse(body)
            with srv.session_lock:
                if srv.session is not None:
                    srv.session.end()
                srv.session = Session(settings, srv.data, **srv.session_kw)
                return srv.session.state()
        if path not in ("/api/act", "/api/next", "/api/read", "/api/end"):
            raise BadRequest("not found", HTTPStatus.NOT_FOUND)
        session = srv.session
        if session is None:
            raise Conflict("no session: start one first")
        if path == "/api/act":
            session.submit(body)
        elif path == "/api/read":
            if not isinstance(body, dict) or set(body) != {"on"} or type(body["on"]) is not bool:
                raise BadRequest('expected {"on": true|false}')
            session.set_read(body["on"])
        else:
            if body != {}:
                raise BadRequest("expected an empty JSON object")
            if path == "/api/next":
                session.next_hand()
            else:
                with srv.session_lock:
                    session.end()
        return session.state()


def make_server(port: int, data: DataDir, **session_kw: Any) -> PlayServer:
    return PlayServer(port, data, **session_kw)
