"""Service manager for Pythia infrastructure components."""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Service health status."""
    STOPPED = auto()
    STARTING = auto()
    RUNNING = auto()
    ERROR = auto()


@dataclass
class ServiceInfo:
    """Service status information."""
    name: str
    status: ServiceStatus
    message: str = ""


class ServiceManager:
    """Manages lifecycle of Pythia services: API server, Oracle DB, SearXNG."""

    def __init__(
        self,
        config_path: str = "pythia.yaml",
        host: str = "0.0.0.0",
        port: int = 8900,
        docker_compose_path: str | None = None,
    ) -> None:
        self.config_path = config_path
        self.host = host
        self.port = port
        self.docker_compose_path = docker_compose_path or self._find_docker_compose()

        self._api_process: asyncio.subprocess.Process | None = None
        self._status_callbacks: list[Callable[[dict[str, ServiceInfo]], None]] = []
        self._health_check_task: asyncio.Task | None = None
        self._running = False
        self._owns_api = False
        self._owns_docker = False

    def _find_docker_compose(self) -> str:
        """Find docker-compose.yml relative to this module."""
        possible_paths = [
            Path(__file__).parent.parent.parent.parent / "docker-compose.yml",
            Path.cwd() / "docker-compose.yml",
            Path(__file__).parent / "docker-compose.yml",
        ]
        for path in possible_paths:
            if path.exists():
                return str(path)
        raise FileNotFoundError(
            "docker-compose.yml not found. Please run from the project root directory."
        )

    def register_status_callback(self, callback: Callable[[dict[str, ServiceInfo]], None]) -> None:
        """Register callback for status updates."""
        self._status_callbacks.append(callback)

    def _notify_status(self, statuses: dict[str, ServiceInfo]) -> None:
        """Notify all callbacks of status change."""
        for callback in self._status_callbacks:
            try:
                callback(statuses)
            except Exception as e:
                logger.exception(f"Status callback failed: {e}")

    async def start_all(self) -> None:
        """Start all services: Oracle DB, SearXNG (Docker), then API server."""
        self._running = True

        oracle_was_ready = await self._check_oracle_ready()
        searxng_was_ready = await self._check_searxng_ready()
        docker_was_ready = oracle_was_ready and searxng_was_ready
        self._owns_docker = not oracle_was_ready and not searxng_was_ready

        if docker_was_ready:
            self._notify_status({
                "oracle": ServiceInfo("Oracle DB", ServiceStatus.RUNNING, "Ready"),
                "searxng": ServiceInfo("SearXNG", ServiceStatus.RUNNING, "Ready"),
                "api": ServiceInfo("API Server", ServiceStatus.STOPPED, "Waiting for infrastructure..."),
            })
        else:
            await self._start_docker_services()

        await self._wait_for_docker_services()

        api_was_ready = await self._check_api_server_ready()
        self._owns_api = not api_was_ready
        if api_was_ready:
            self._notify_status({
                "oracle": ServiceInfo("Oracle DB", ServiceStatus.RUNNING, "Ready"),
                "searxng": ServiceInfo("SearXNG", ServiceStatus.RUNNING, "Ready"),
                "api": ServiceInfo("API Server", ServiceStatus.RUNNING, f"Running on port {self.port}"),
            })
        else:
            await self._start_api_server()

        self._health_check_task = asyncio.create_task(self._health_check_loop())

    async def stop_all(self) -> None:
        """Stop all services gracefully."""
        self._running = False

        # Stop health check loop
        if self._health_check_task:
            self._health_check_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._health_check_task

        if self._owns_api:
            await self._stop_api_server()

        if self._owns_docker:
            await self._stop_docker_services()

        self._owns_api = False
        self._owns_docker = False

    async def _start_docker_services(self) -> None:
        """Start Oracle DB and SearXNG via docker compose."""
        self._notify_status({
            "oracle": ServiceInfo("Oracle DB", ServiceStatus.STARTING, "Starting container..."),
            "searxng": ServiceInfo("SearXNG", ServiceStatus.STARTING, "Starting container..."),
            "api": ServiceInfo("API Server", ServiceStatus.STOPPED, "Waiting for infrastructure..."),
        })

        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "compose", "-f", self.docker_compose_path, "up", "-d",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                raise RuntimeError(f"docker compose up failed: {stderr.decode()}")

            self._notify_status({
                "oracle": ServiceInfo("Oracle DB", ServiceStatus.STARTING, "Container started, waiting for ready..."),
                "searxng": ServiceInfo("SearXNG", ServiceStatus.STARTING, "Container started, waiting for ready..."),
                "api": ServiceInfo("API Server", ServiceStatus.STOPPED, "Waiting for infrastructure..."),
            })

        except FileNotFoundError:
            self._notify_status({
                "oracle": ServiceInfo("Oracle DB", ServiceStatus.ERROR, "Docker not found"),
                "searxng": ServiceInfo("SearXNG", ServiceStatus.ERROR, "Docker not found"),
                "api": ServiceInfo("API Server", ServiceStatus.STOPPED, "Waiting for infrastructure..."),
            })
        except Exception as e:
            self._notify_status({
                "oracle": ServiceInfo("Oracle DB", ServiceStatus.ERROR, str(e)),
                "searxng": ServiceInfo("SearXNG", ServiceStatus.ERROR, str(e)),
                "api": ServiceInfo("API Server", ServiceStatus.STOPPED, "Waiting for infrastructure..."),
            })

    async def _stop_docker_services(self) -> None:
        """Stop Docker containers."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "compose", "-f", self.docker_compose_path, "down",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
        except Exception as e:
            logger.warning(f"Failed to stop Docker containers: {e}")

    async def _wait_for_docker_services(self, timeout: float = 120.0) -> None:
        """Wait for Oracle and SearXNG to be ready."""
        start = asyncio.get_running_loop().time()
        oracle_ready = False
        searxng_ready = False

        while asyncio.get_running_loop().time() - start < timeout:
            if not oracle_ready:
                oracle_ready = await self._check_oracle_ready()
                if oracle_ready:
                    self._notify_status({
                        "oracle": ServiceInfo("Oracle DB", ServiceStatus.RUNNING, "Ready"),
                        "searxng": ServiceInfo("SearXNG", ServiceStatus.STARTING, "Waiting for ready..."),
                        "api": ServiceInfo("API Server", ServiceStatus.STOPPED, "Waiting for infrastructure..."),
                    })

            if not searxng_ready:
                searxng_ready = await self._check_searxng_ready()
                if searxng_ready:
                    self._notify_status({
                        "oracle": ServiceInfo("Oracle DB", ServiceStatus.RUNNING, "Ready"),
                        "searxng": ServiceInfo("SearXNG", ServiceStatus.RUNNING, "Ready"),
                        "api": ServiceInfo("API Server", ServiceStatus.STARTING, "Starting..."),
                    })

            if oracle_ready and searxng_ready:
                return

            await asyncio.sleep(2.0)

        raise TimeoutError(
            f"Docker services did not start within {timeout}s. "
            f"Oracle: {'ready' if oracle_ready else 'not ready'}, "
            f"SearXNG: {'ready' if searxng_ready else 'not ready'}"
        )

    async def _check_oracle_ready(self) -> bool:
        """Check if Oracle DB is ready by testing TCP connection."""
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection("localhost", 1523),
                timeout=2.0
            )
            writer.close()
            await writer.wait_closed()
            return True
        except Exception:
            return False

    async def _check_searxng_ready(self) -> bool:
        """Check if SearXNG is ready by making HTTP request."""
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get("http://localhost:8889/healthz")
                return resp.status_code == 200
        except Exception:
            return False

    async def _check_api_server_ready(self) -> bool:
        """Check if the API server is already serving health requests."""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"http://localhost:{self.port}/health", timeout=2.0)
                return resp.status_code == 200
        except Exception:
            return False

    async def _start_api_server(self) -> None:
        """Start the API server as an async subprocess."""
        self._notify_status({
            "oracle": ServiceInfo("Oracle DB", ServiceStatus.RUNNING, "Ready"),
            "searxng": ServiceInfo("SearXNG", ServiceStatus.RUNNING, "Ready"),
            "api": ServiceInfo("API Server", ServiceStatus.STARTING, "Starting..."),
        })

        try:
            project_root = Path(self.docker_compose_path).resolve().parent
            resolved_config = Path(self.config_path).expanduser().resolve()

            env = os.environ.copy()
            env["PYTHIA_CONFIG"] = str(resolved_config)

            self._api_process = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "pythia", "serve",
                "--host", self.host,
                "--port", str(self.port),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(project_root),
                env=env,
            )

            await self._wait_for_api_server()

        except Exception as e:
            self._notify_status({
                "oracle": ServiceInfo("Oracle DB", ServiceStatus.RUNNING, "Ready"),
                "searxng": ServiceInfo("SearXNG", ServiceStatus.RUNNING, "Ready"),
                "api": ServiceInfo("API Server", ServiceStatus.ERROR, str(e)),
            })

    async def _stop_api_server(self) -> None:
        """Stop the API server."""
        if self._api_process:
            self._api_process.terminate()
            try:
                await asyncio.wait_for(self._api_process.wait(), timeout=5.0)
            except TimeoutError:
                self._api_process.kill()
                await self._api_process.wait()
            self._api_process = None

    async def _wait_for_api_server(self, timeout: float = 30.0) -> None:
        """Wait for API server to be ready."""
        start = asyncio.get_running_loop().time()
        while asyncio.get_running_loop().time() - start < timeout:
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(f"http://localhost:{self.port}/health", timeout=2.0)
                    if resp.status_code == 200:
                        self._notify_status({
                            "oracle": ServiceInfo("Oracle DB", ServiceStatus.RUNNING, "Ready"),
                            "searxng": ServiceInfo("SearXNG", ServiceStatus.RUNNING, "Ready"),
                            "api": ServiceInfo("API Server", ServiceStatus.RUNNING, f"Running on port {self.port}"),
                        })
                        return
            except Exception:
                pass

            if self._api_process and self._api_process.returncode is not None:
                returncode = self._api_process.returncode
                stderr_text = ""
                if self._api_process.stderr:
                    try:
                        stderr_bytes = await self._api_process.stderr.read()
                        stderr_text = stderr_bytes.decode(errors="replace")[-500:]
                    except Exception:
                        pass
                if returncode == 0:
                    raise RuntimeError("API server exited unexpectedly (success)")
                raise RuntimeError(
                    f"API server crashed (exit code {returncode})"
                    + (f": {stderr_text}" if stderr_text else "")
                )

            await asyncio.sleep(0.5)

        raise TimeoutError("API server did not start within timeout")

    async def _health_check_loop(self) -> None:
        """Periodically check health of all services."""
        while self._running:
            try:
                statuses = await self._check_all_health()
                self._notify_status(statuses)
            except Exception as e:
                logger.exception(f"Health check failed: {e}")
                self._notify_status({
                    "api": ServiceInfo("API Server", ServiceStatus.ERROR, f"Health check error: {e}"),
                    "oracle": ServiceInfo("Oracle DB", ServiceStatus.ERROR, "Health check failed"),
                    "searxng": ServiceInfo("SearXNG", ServiceStatus.ERROR, "Health check failed"),
                })
            await asyncio.sleep(2.0)

    async def _check_all_health(self) -> dict[str, ServiceInfo]:
        """Check health of all services."""
        statuses: dict[str, ServiceInfo] = {}

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"http://localhost:{self.port}/health", timeout=2.0)
                if resp.status_code == 200:
                    data = resp.json()
                    statuses["api"] = ServiceInfo("API Server", ServiceStatus.RUNNING, f"Running on port {self.port}")
                    statuses["oracle"] = ServiceInfo(
                        "Oracle DB",
                        ServiceStatus.RUNNING if data.get("oracle") else ServiceStatus.ERROR,
                        "Connected" if data.get("oracle") else "Connection failed"
                    )
                    statuses["searxng"] = ServiceInfo(
                        "SearXNG",
                        ServiceStatus.RUNNING if data.get("searxng") else ServiceStatus.ERROR,
                        f"Connected ({data.get('cache_size', 0)} cached)" if data.get("searxng") else "Connection failed"
                    )
                else:
                    statuses["api"] = ServiceInfo("API Server", ServiceStatus.ERROR, f"Status: {resp.status_code}")
        except Exception as e:
            statuses["api"] = ServiceInfo("API Server", ServiceStatus.ERROR, str(e))
            statuses["oracle"] = ServiceInfo("Oracle DB", ServiceStatus.ERROR, "Health check failed")
            statuses["searxng"] = ServiceInfo("SearXNG", ServiceStatus.ERROR, "Health check failed")

        return statuses
