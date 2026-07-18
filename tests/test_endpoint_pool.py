# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""CPU unit tests for the circuit-breaking endpoint pool (Phase 3.1).

Uses a fake clock and fake health check — no network, no GPU.
"""

from __future__ import annotations

import threading

import pytest

from hunyuan_ocr.endpoint_pool import AllProbesInFlight, EndpointPool, NoHealthyEndpoint


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


def test_half_open_single_probe_one_at_a_time():
    """While one endpoint is half-open, only ONE acquire may take it; the rest
    are refused (AllProbesInFlight) until report() resolves the probe."""
    clk = FakeClock()
    pool = _pool({"http://h:1/v1": True}, failure_threshold=2, cooldown=30.0, clock=clk)
    pool.probe_initial()
    ep = pool.acquire().base_url
    pool.report(ep, False)
    pool.report(ep, False)  # circuit opens
    clk.t += 31  # -> half_open
    # first acquire reserves the half-open endpoint for a single probe
    got = pool.acquire().base_url
    assert got == ep
    snap = pool.snapshot()[0]
    assert snap["state"] == "half_open" and snap["half_open_in_flight"] is True
    # a second acquire while the probe is outstanding must be refused
    with pytest.raises(AllProbesInFlight):
        pool.acquire()
    # resolving the probe (success) releases the reservation and closes
    pool.report(ep, True)
    snap = pool.snapshot()[0]
    assert snap["state"] == "closed" and snap["half_open_in_flight"] is False


def test_half_open_failed_probe_releases_reservation():
    clk = FakeClock()
    pool = _pool({"http://h:1/v1": True}, failure_threshold=2, cooldown=30.0, clock=clk)
    pool.probe_initial()
    ep = pool.acquire().base_url
    pool.report(ep, False)
    pool.report(ep, False)
    clk.t += 31
    pool.acquire()  # reserves half-open probe
    assert pool.snapshot()[0]["half_open_in_flight"] is True
    pool.report(ep, False)  # probe fails
    snap = pool.snapshot()[0]
    # reservation cleared even on failure; circuit re-opens and will cool down
    assert snap["state"] == "open"
    assert snap["half_open_in_flight"] is False


def test_concurrent_half_open_probe_not_double_acquired():
    """10 threads racing a single half-open endpoint: exactly ONE acquires it,
    the other 9 raise AllProbesInFlight. Proves the in-flight reservation holds
    under real concurrency (not just sequential calls)."""
    import time

    clk = FakeClock()
    pool = _pool({"http://h:1/v1": True}, failure_threshold=2, cooldown=30.0, clock=clk)
    pool.probe_initial()
    ep = pool.acquire().base_url
    pool.report(ep, False)
    pool.report(ep, False)  # open
    clk.t += 31  # -> half_open

    n_threads = 10
    start = threading.Barrier(n_threads)
    winner_release = threading.Event()  # winner holds the probe in-flight until set
    outcomes: list[tuple[str, str | None]] = []
    out_lock = threading.Lock()

    def racer():
        try:
            start.wait()
            ep_got = pool.acquire().base_url
            with out_lock:
                outcomes.append(("got", ep_got))
            # Hold the probe outstanding so stragglers observe it in-flight.
            winner_release.wait(timeout=5)
            pool.report(ep_got, True)
        except AllProbesInFlight:
            with out_lock:
                outcomes.append(("probe_in_flight", None))
        except NoHealthyEndpoint:
            with out_lock:
                outcomes.append(("no_healthy", None))

    threads = [threading.Thread(target=racer) for _ in range(n_threads)]
    for t in threads:
        t.start()

    # Wait until 9 threads have refused (i.e. observed the in-flight probe),
    # then release the single winner so the suite never deadlocks.
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        with out_lock:
            refusals = sum(1 for o in outcomes if o[0] in ("probe_in_flight", "no_healthy"))
        if refusals >= n_threads - 1:
            break
        time.sleep(0.005)
    winner_release.set()
    for t in threads:
        t.join(timeout=5)

    got = [o for o in outcomes if o[0] == "got"]
    refused = [o for o in outcomes if o[0] in ("probe_in_flight", "no_healthy")]
    assert len(outcomes) == n_threads
    assert len(got) == 1, f"expected exactly 1 winner, got {len(got)}: {outcomes}"
    assert len(refused) == n_threads - 1
    # the losers must be told it was a probe-in-flight, not a hard outage
    assert all(o[0] == "probe_in_flight" for o in refused)
