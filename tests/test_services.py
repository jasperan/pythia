"""Tests for service manager — unit tests with mocked infrastructure."""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from pythia.services import ServiceManager, ServiceStatus, ServiceInfo


# --- Data classes ---

def test_service_status_values():
    assert ServiceStatus.STOPPED.name == "STOPPED"
    assert ServiceStatus.STARTING.name == "STARTING"
    assert ServiceStatus.RUNNING.name == "RUNNING"
    assert ServiceStatus.ERROR.name == "ERROR"


def test_service_info_defaults():
    info = ServiceInfo(name="Test", status=ServiceStatus.RUNNING)
    assert info.name == "Test"
    assert info.status == ServiceStatus.RUNNING
    assert info.message == ""


def test_service_info_with_message():
    info = ServiceInfo(name="API", status=ServiceStatus.ERROR, message="Connection refused")
    assert info.message == "Connection refused"


# --- ServiceManager init ---

def test_service_manager_init():
    with patch.object(ServiceManager, "_find_docker_compose", return_value="/path/docker-compose.yml"):
        mgr = ServiceManager(config_path="test.yaml", host="127.0.0.1", port=9000)
        assert mgr.config_path == "test.yaml"
        assert mgr.host == "127.0.0.1"
        assert mgr.port == 9000
        assert mgr.docker_compose_path == "/path/docker-compose.yml"
        assert mgr._running is False
        assert mgr._api_process is None


def test_service_manager_explicit_docker_path():
    mgr = ServiceManager(docker_compose_path="/custom/docker-compose.yml")
    assert mgr.docker_compose_path == "/custom/docker-compose.yml"


def test_find_docker_compose_not_found():
    with patch("pathlib.Path.exists", return_value=False):
        with pytest.raises(FileNotFoundError, match="docker-compose.yml not found"):
            ServiceManager()


# --- Status callbacks ---

def test_register_and_notify_callbacks():
    with patch.object(ServiceManager, "_find_docker_compose", return_value="/path"):
        mgr = ServiceManager()

    received = []
    mgr.register_status_callback(lambda s: received.append(s))

    statuses = {"api": ServiceInfo("API", ServiceStatus.RUNNING)}
    mgr._notify_status(statuses)

    assert len(received) == 1
    assert received[0]["api"].status == ServiceStatus.RUNNING


def test_notify_callback_error_doesnt_crash():
    with patch.object(ServiceManager, "_find_docker_compose", return_value="/path"):
        mgr = ServiceManager()

    def bad_callback(s):
        raise RuntimeError("boom")

    mgr.register_status_callback(bad_callback)
    # Should not raise
    mgr._notify_status({"api": ServiceInfo("API", ServiceStatus.RUNNING)})


# --- Health checks ---

@pytest.mark.asyncio
async def test_check_oracle_ready_success():
    with patch.object(ServiceManager, "_find_docker_compose", return_value="/path"):
        mgr = ServiceManager()

    mock_writer = AsyncMock()
    mock_writer.close = MagicMock()
    mock_writer.wait_closed = AsyncMock()

    with patch("asyncio.open_connection", return_value=(AsyncMock(), mock_writer)):
        assert await mgr._check_oracle_ready() is True


@pytest.mark.asyncio
async def test_check_oracle_ready_failure():
    with patch.object(ServiceManager, "_find_docker_compose", return_value="/path"):
        mgr = ServiceManager()

    with patch("asyncio.open_connection", side_effect=ConnectionRefusedError()):
        assert await mgr._check_oracle_ready() is False


@pytest.mark.asyncio
async def test_check_searxng_ready_success():
    with patch.object(ServiceManager, "_find_docker_compose", return_value="/path"):
        mgr = ServiceManager()

    mock_resp = MagicMock()
    mock_resp.status_code = 200

    with patch("httpx.AsyncClient") as MockClient:
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_resp)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_http

        assert await mgr._check_searxng_ready() is True


@pytest.mark.asyncio
async def test_check_searxng_ready_failure():
    with patch.object(ServiceManager, "_find_docker_compose", return_value="/path"):
        mgr = ServiceManager()

    with patch("httpx.AsyncClient") as MockClient:
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(side_effect=ConnectionError("refused"))
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_http

        assert await mgr._check_searxng_ready() is False


@pytest.mark.asyncio
async def test_check_all_health_success():
    with patch.object(ServiceManager, "_find_docker_compose", return_value="/path"):
        mgr = ServiceManager(port=8900)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"oracle": True, "searxng": True, "cache_size": 10}

    with patch("httpx.AsyncClient") as MockClient:
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_resp)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_http

        statuses = await mgr._check_all_health()
        assert statuses["api"].status == ServiceStatus.RUNNING
        assert statuses["oracle"].status == ServiceStatus.RUNNING
        assert statuses["searxng"].status == ServiceStatus.RUNNING


@pytest.mark.asyncio
async def test_check_all_health_api_down():
    with patch.object(ServiceManager, "_find_docker_compose", return_value="/path"):
        mgr = ServiceManager()

    with patch("httpx.AsyncClient") as MockClient:
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(side_effect=ConnectionError("refused"))
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_http

        statuses = await mgr._check_all_health()
        assert statuses["api"].status == ServiceStatus.ERROR
        assert statuses["oracle"].status == ServiceStatus.ERROR


@pytest.mark.asyncio
async def test_check_all_health_oracle_down():
    with patch.object(ServiceManager, "_find_docker_compose", return_value="/path"):
        mgr = ServiceManager()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"oracle": False, "searxng": True, "cache_size": 0}

    with patch("httpx.AsyncClient") as MockClient:
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_resp)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_http

        statuses = await mgr._check_all_health()
        assert statuses["api"].status == ServiceStatus.RUNNING
        assert statuses["oracle"].status == ServiceStatus.ERROR
        assert statuses["searxng"].status == ServiceStatus.RUNNING


# --- Stop when nothing running ---

@pytest.mark.asyncio
async def test_stop_api_server_noop_when_no_process():
    with patch.object(ServiceManager, "_find_docker_compose", return_value="/path"):
        mgr = ServiceManager()
    mgr._api_process = None
    await mgr._stop_api_server()  # should not raise


@pytest.mark.asyncio
async def test_stop_api_server_terminates_process():
    with patch.object(ServiceManager, "_find_docker_compose", return_value="/path"):
        mgr = ServiceManager()

    mock_proc = AsyncMock()
    mock_proc.terminate = MagicMock()
    mock_proc.wait = AsyncMock(return_value=0)
    mgr._api_process = mock_proc

    await mgr._stop_api_server()
    mock_proc.terminate.assert_called_once()
    assert mgr._api_process is None
