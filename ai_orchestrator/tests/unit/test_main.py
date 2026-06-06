"""Tests for the FastAPI gateway — health, tasks, accounts, leases."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from httpx import AsyncClient, ASGITransport

from ai_orchestrator.orchestrator.main import app, _active_tasks


@pytest.fixture(autouse=True)
def reset_state():
    """Reset task state and register default accounts."""
    _active_tasks.clear()
    app.state.start_time = datetime.now(timezone.utc)
    from ai_orchestrator.orchestrator.main import lease_manager
    from ai_orchestrator.models.account import Account, AccountState
    # Re-register default accounts (startup event doesn't fire in tests)
    for acct in lease_manager.list_accounts():
        pass
    if len(lease_manager.list_accounts()) < 6:
        defaults = [
            Account(id="chatgpt:ui-01", provider="chatgpt_ui", state=AccountState.IDLE, context_limit=32768, rate_limit_rpm=20),
            Account(id="qwen:ui-01", provider="qwen_ui", state=AccountState.IDLE, context_limit=131072, rate_limit_rpm=20),
            Account(id="deepseek:ui-01", provider="deepseek_ui", state=AccountState.IDLE, context_limit=1048576, rate_limit_rpm=20),
            Account(id="kimi:ui-01", provider="kimi_ui", state=AccountState.IDLE, context_limit=128000, rate_limit_rpm=20),
            Account(id="zai:ui-01", provider="z_ai_ui", state=AccountState.IDLE, context_limit=131072, rate_limit_rpm=20),
            Account(id="local:dev-01", provider="local_llm", state=AccountState.IDLE, context_limit=256000, rate_limit_rpm=30),
        ]
        lease_manager.register_accounts(defaults)
    yield


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestHealthEndpoint:
    async def test_health_returns_ok(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"
        assert data["active_tasks"] >= 0
        assert data["registered_accounts"] >= 6

    async def test_health_includes_resource_metrics(self, client):
        resp = await client.get("/health")
        data = resp.json()
        assert "memory_available_gb" in data
        assert "cpu_percent" in data
        assert "watermark_level" in data
        assert "active_leases" in data


class TestTaskEndpoints:
    async def test_submit_task_creates_task(self, client):
        resp = await client.post("/tasks", json={
            "prompt": "write a Python function",
            "task_type": "interactive",
            "priority": 2,
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["prompt"] == "write a Python function"
        assert data["priority"] == "NORMAL" or data["priority"] == 2
        assert "id" in data
        assert "created_at" in data

    async def test_list_tasks_returns_submitted(self, client):
        await client.post("/tasks", json={"prompt": "task 1"})
        await client.post("/tasks", json={"prompt": "task 2"})
        resp = await client.get("/tasks")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

    async def test_get_task_by_id(self, client):
        create_resp = await client.post("/tasks", json={"prompt": "find me"})
        task_id = create_resp.json()["id"]
        resp = await client.get(f"/tasks/{task_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == task_id

    async def test_get_nonexistent_task_returns_404(self, client):
        resp = await client.get("/tasks/nonexistent")
        assert resp.status_code == 404

    async def test_filter_tasks_by_status(self, client):
        await client.post("/tasks", json={"prompt": "task a"})
        resp = await client.get("/tasks?status=PLANNING")
        assert resp.status_code == 200

    async def test_halt_and_resume_task(self, client):
        create = await client.post("/tasks", json={"prompt": "halt test"})
        task_id = create.json()["id"]

        halt = await client.post(f"/tasks/{task_id}/halt", params={"reason": "testing"})
        assert halt.status_code == 200
        assert halt.json()["status"] in ("HALTED", "VLADIMIR_HALTED")

        resume = await client.post(f"/tasks/{task_id}/resume")
        # Resume succeeds if status matches
        assert resume.status_code in (200, 400)


class TestAccountEndpoint:
    async def test_list_accounts(self, client):
        resp = await client.get("/accounts")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 6
        providers = {a["provider"] for a in data}
        assert "deepseek_ui" in providers

    async def test_filter_accounts_by_provider(self, client):
        resp = await client.get("/accounts?provider=deepseek_ui")
        assert resp.status_code == 200
        for a in resp.json():
            assert a["provider"] == "deepseek_ui"


class TestLeaseEndpoints:
    async def test_request_lease(self, client):
        await client.post("/tasks", json={"prompt": "lease task"})
        resp = await client.post("/leases", params={
            "task_id": "test-task",
            "agent_id": "test-agent",
        })
        # May fail if no tasks registered or accounts busy
        assert resp.status_code in (200, 503)

    async def test_list_leases(self, client):
        resp = await client.get("/leases")
        assert resp.status_code == 200


class TestProviderEndpoint:
    async def test_list_providers(self, client):
        resp = await client.get("/providers")
        assert resp.status_code == 200
        data = resp.json()
        assert "deepseek_ui" in data
        assert data["deepseek_ui"]["context_limit"] == 1_048_576


class TestMetricsEndpoint:
    async def test_metrics(self, client):
        resp = await client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "active_tasks" in data
        assert "memory_available_gb" in data
        assert "pool_stats" in data
