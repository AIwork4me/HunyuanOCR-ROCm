# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""CPU unit tests for the circuit-breaking endpoint pool (Phase 3.1).

Uses a fake clock and fake health check — no network, no GPU.
"""

from __future__ import annotations

import pytest

from hunyuan_ocr.endpoint_pool import EndpointPool, NoHealthyEndpoint


class FakeClock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


def _pool(check_map, **kw):
    """check_map: {base_url: bool} controlling each endpoint's health response."""
    eps = [(f"p{u.split(':')[-1]}", u) for u in check_map]
    return EndpointPool(
        eps,
        check=lambda url: check_map[url],
        failure_threshold=kw.get("failure_threshold", 3),
        cooldown=kw.get("cooldown", 30.0),
        clock=kw.get("clock", FakeClock()),
    )


def test_initial_probe_records_health():
    pool = _pool({"http://h:1/v1": True, "http://h:2/v1": False})
    pool.probe_initial()
    snap = {s["base_url"]: s for s in pool.snapshot()}
    assert snap["http://h:1/v1"]["initial_health"] == "healthy"
    assert snap["http://h:1/v1"]["state"] == "closed"
    assert snap["http://h:2/v1"]["initial_health"] == "unhealthy"
    assert snap["http://h:2/v1"]["state"] == "open"


def test_acquire_round_robin_among_healthy():
    pool = _pool({"http://h:1/v1": True, "http://h:2/v1": True})
    pool.probe_initial()
    picks = [pool.acquire().base_url for _ in range(4)]
    # both healthy endpoints are used (round-robin)
    assert set(picks) == {"http://h:1/v1", "http://h:2/v1"}


def test_circuit_opens_after_threshold():
    pool = _pool({"http://h:1/v1": True}, failure_threshold=3)
    pool.probe_initial()
    ep = pool.acquire().base_url
    for _ in range(3):
        pool.report(ep, False)
    snap = pool.snapshot()[0]
    assert snap["state"] == "open"
    assert snap["circuit_open_count"] == 1
    with pytest.raises(NoHealthyEndpoint):
        pool.acquire()


def test_all_unhealthy_from_start_fast_fails():
    pool = _pool({"http://h:1/v1": False, "http://h:2/v1": False})
    pool.probe_initial()
    assert not pool.has_healthy()
    with pytest.raises(NoHealthyEndpoint):
        pool.acquire()


def test_half_open_recovery_after_cooldown():
    clk = FakeClock()
    pool = _pool({"http://h:1/v1": True}, failure_threshold=2, cooldown=30.0, clock=clk)
    pool.probe_initial()
    ep = pool.acquire().base_url
    pool.report(ep, False)
    pool.report(ep, False)  # circuit opens
    assert pool.snapshot()[0]["state"] == "open"
    with pytest.raises(NoHealthyEndpoint):
        pool.acquire()
    clk.t += 31  # cooldown elapses -> half_open
    assert pool.has_healthy()
    ep2 = pool.acquire().base_url
    assert ep2 == ep
    pool.report(ep2, True)  # success closes the circuit
    assert pool.snapshot()[0]["state"] == "closed"


def test_snapshot_counts():
    pool = _pool({"http://h:1/v1": True})
    pool.probe_initial()
    pool.report(pool.acquire().base_url, True)
    pool.report(pool.acquire().base_url, False)
    s = pool.snapshot()[0]
    assert s["requests"] == 2 and s["successes"] == 1 and s["failures"] == 1


def test_empty_or_duplicate_endpoints_rejected():
    with pytest.raises(ValueError):
        EndpointPool([], check=lambda u: True)
    with pytest.raises(ValueError):
        EndpointPool([("a", "http://h:1/v1"), ("b", "http://h:1/v1")], check=lambda u: True)
