#!/usr/bin/env python3
"""OCI GenAI local proxy -- OpenAI-compatible endpoint backed by OCI GenAI.

Starts a threaded local server that accepts standard OpenAI API calls
(including streaming) and forwards them to OCI GenAI using OCI
authentication from ~/.oci/config.

Prerequisites:
    pip install -r requirements.txt
    # Configure ~/.oci/config with your OCI credentials

Usage:
    python proxy.py                          # starts on port 9999
    OCI_PROXY_PORT=8888 python proxy.py      # custom port

Then configure Pythia with:
    backend: oci-genai
    oci_genai:
      base_url: "http://localhost:9999/v1"
      model: "xai.grok-4"
"""

import json
import os
import sys
from hmac import compare_digest
from http.server import HTTPServer, BaseHTTPRequestHandler
from ipaddress import ip_address
from socketserver import ThreadingMixIn

from oci_client import create_oci_client

PROXY_PORT = int(os.getenv("OCI_PROXY_PORT", "9999"))
PROXY_HOST = os.getenv("OCI_PROXY_HOST", "127.0.0.1")
PROXY_API_KEY = os.getenv("OCI_PROXY_API_KEY", "oci-genai")
PROXY_ALLOWED_ORIGIN = os.getenv("OCI_PROXY_ALLOWED_ORIGIN", "http://localhost:8900")


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle each request in a separate thread."""

    daemon_threads = True


class OCIProxyHandler(BaseHTTPRequestHandler):
    client = None

    # ── CORS ────────────────────────────────────────────────────
    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", PROXY_ALLOWED_ORIGIN)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers", "Content-Type, Authorization"
        )

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    # ── POST /v1/chat/completions ───────────────────────────────
    def do_POST(self):
        if "/chat/completions" not in self.path:
            return self._json(404, {"error": {"message": "Not found"}})
        if not self._authorized():
            return self._json(401, {"error": {"message": "Unauthorized"}})

        content_length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(content_length))
        except json.JSONDecodeError:
            return self._json(400, {"error": {"message": "Invalid JSON"}})
        stream = body.get("stream", False)

        try:
            if stream:
                self._handle_stream(body)
            else:
                response = self.client.chat.completions.create(**body)
                self._json(200, response.model_dump())
        except Exception as exc:
            self._json(
                500,
                {"error": {"message": str(exc), "type": "oci_genai_error"}},
            )

    def _handle_stream(self, body):
        """Forward a streaming chat-completion as Server-Sent Events."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self._cors_headers()
        self.end_headers()
        try:
            for chunk in self.client.chat.completions.create(**body):
                data = json.dumps(chunk.model_dump())
                self.wfile.write(f"data: {data}\n\n".encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except Exception as exc:
            err = json.dumps(
                {"error": {"message": str(exc), "type": "oci_genai_error"}}
            )
            self.wfile.write(f"data: {err}\n\n".encode())
            self.wfile.flush()

    # ── GET endpoints ───────────────────────────────────────────
    def do_GET(self):
        if "/models" in self.path:
            self._json(200, {"object": "list", "data": []})
        elif "/health" in self.path:
            self._json(200, {"status": "ok"})
        else:
            self._json(404, {"error": {"message": "Not found"}})

    # ── Helpers ─────────────────────────────────────────────────
    def _authorized(self):
        expected = f"Bearer {PROXY_API_KEY}"
        provided = self.headers.get("Authorization", "")
        return compare_digest(provided, expected)

    def _json(self, code, data):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self._cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, fmt, *args):  # noqa: ARG002
        sys.stderr.write(f"[oci-proxy] {args[0]}\n")


# ── Main ────────────────────────────────────────────────────────
def main():
    if not os.getenv("OCI_COMPARTMENT_ID"):
        print("ERROR: OCI_COMPARTMENT_ID environment variable is required.")
        print("Set it to your OCI compartment OCID.")
        sys.exit(1)
    try:
        host_ip = ip_address(PROXY_HOST)
    except ValueError:
        host_ip = None
    if (host_ip is None or not host_ip.is_loopback) and PROXY_API_KEY == "oci-genai":  # pragma: allowlist secret
        print("ERROR: Set OCI_PROXY_API_KEY before binding the proxy to a non-loopback host.")
        sys.exit(1)

    client = create_oci_client()
    OCIProxyHandler.client = client

    server = ThreadedHTTPServer((PROXY_HOST, PROXY_PORT), OCIProxyHandler)
    print(f"OCI GenAI proxy listening on http://{PROXY_HOST}:{PROXY_PORT}/v1")
    print(f"  Region:      {os.getenv('OCI_REGION', 'us-chicago-1')}")
    print(f"  Profile:     {os.getenv('OCI_PROFILE', 'DEFAULT')}")
    print(f"  Compartment: {os.getenv('OCI_COMPARTMENT_ID', '')[:50]}...")
    print(f"  CORS origin: {PROXY_ALLOWED_ORIGIN}")
    print()
    print("Configure Pythia with (in pythia.yaml):")
    print("  backend: oci-genai")
    print("  oci_genai:")
    print(f'    base_url: "http://localhost:{PROXY_PORT}/v1"')
    print('    model: "xai.grok-4"')
    print()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
