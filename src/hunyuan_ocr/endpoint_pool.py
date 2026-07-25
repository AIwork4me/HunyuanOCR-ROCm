# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""Thread-safe OpenAI-compatible endpoint pool with circuit breaking.

A "round-robin over ports" is not health management: a crashed server keeps
receiving requests until every page has retried against it. This pool probes each
endpoint up front, routes requests only to healthy endpoints, opens a circuit
after ``failure_threshold`` consecutive failures, cools down, half-open-probes,
and fast-fails (rather than manufacturing hundreds of doomed requests) when no
endpoint is healthy.

No network framework: the health check and the clock are injected so the whole
state machine is unit-testable with fakes on a CPU.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


class NoHealthyEndpoint(RuntimeError):
    """Raised when every endpoint's circuit is open and none are half-open."""


class AllProbesInFlight(NoHealthyEndpoint):
    """Raised when the only usable endpoints are half-open and already probing.

    A subclass of :class:`NoHealthyEndpoint` so existing ``except NoHealthyEndpoint``
    handlers still catch it. Callers that want to distinguish "permanently out"
    from "momentarily saturated while a probe resolves" can catch this type
    specifically and back off briefly instead of hard-failing the page.
    """


@dataclass
class EndpointStats:
    alias: str
    base_url: str
    initial_health: str | None = None  # "healthy" | "unhealthy" | None (unprobed)
    requests: int = 0
    successes: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    circuit_open_count: int = 0
    state: str = "closed"  # "closed" | "open" | "half_open"
    opened_at: float | None = None
    # True while a single half-open probe for this endpoint is outstanding.
    # Enforces the documented "one probe at a time" contract: once a thread has
    # acquired a half-open endpoint, no other thread may acquire it again until
    # report() resolves the probe (success -> closed, failure -> open).
    half_open_in_flight: bool = False

    def as_dict(self) -> dict:
        return {
            "alias": self.alias,
            "base_url": self.base_url,
            "initial_health": self.initial_health,
            "requests": self.requests,
            "successes": self.successes,
            "failures": self.failures,
            "consecutive_failures": self.consecutive_failures,
            "circuit_open_count": self.circuit_open_count,
            "state": self.state,
            "half_open_in_flight": self.half_open_in_flight,
        }


@dataclass
class EndpointPool:
    """Circuit-breaking round-robin pool over OpenAI-compatible endpoints.

    Parameters
    ----------
    endpoints:
        list of (alias, base_url). Aliases need not be unique; base_urls must be.
    check:
        ``callable(base_url) -> bool`` — return True if healthy. Injected so tests
        can drive the state machine without a network.
    failure_threshold:
        consecutive failures on one endpoint before its circuit opens.
    cooldown:
        seconds an open circuit waits before allowing a single half-open probe.
    clock:
        monotonic clock callable (injectable for deterministic tests).
    """

    endpoints: list[tuple[str, str]]
    check: object
    failure_threshold: int = 3
    cooldown: float = 30.0
    clock: object = field(default=time.monotonic)
    _stats: dict[str, EndpointStats] = field(default_factory=dict)
    _rr: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        if not self.endpoints:
            raise ValueError("EndpointPool requires at least one endpoint")
        if len({u for _, u in self.endpoints}) != len(self.endpoints):
            raise ValueError("endpoint base_urls must be unique")
        for alias, url in self.endpoints:
            self._stats[url] = EndpointStats(alias=alias, base_url=url)

    def _now(self) -> float:
        return float(self.clock())  # type: ignore[operator]

    def probe_initial(self) -> None:
        """Probe every endpoint once and record its initial health."""
        for url in list(self._stats):
            healthy = bool(self.check(url))
            st = self._stats[url]
            st.initial_health = "healthy" if healthy else "unhealthy"
            if not healthy:
                st.state = "open"
                st.opened_at = self._now()
                st.circuit_open_count = 1

    def _maybe_half_open(self, st: EndpointStats) -> None:
        """Promote an open circuit to half_open once its cooldown has elapsed."""
        if st.state == "open" and st.opened_at is not None and self._now() - st.opened_at >= self.cooldown:
            st.state = "half_open"

    def acquire(self) -> EndpointStats:
        """Return a usable endpoint, round-robin among them.

        A *usable* endpoint is either ``closed`` (fully healthy, any number of
        concurrent requests) or ``half_open`` with **no probe in flight**. The
        first thread to acquire a half-open endpoint marks it in-flight, so no
        other thread can acquire it again until :meth:`report` resolves the probe
        — enforcing the documented "one half-open probe at a time" contract.

        Raises :class:`NoHealthyEndpoint` if every circuit is open and none is
        half-open, or :class:`AllProbesInFlight` (a subclass) if usable endpoints
        exist but every half-open one already has a probe outstanding. Either way
        the driver stops dispatching instead of queuing predictable failures.
        """
        with self._lock:
            candidates = list(self._stats.values())
            for st in candidates:
                self._maybe_half_open(st)
            usable = [
                s for s in candidates if s.state == "closed" or (s.state == "half_open" and not s.half_open_in_flight)
            ]
            if not usable:
                # Distinguish "nothing half-open at all" from "half-open probes
                # are resolving" so callers can back off rather than hard-fail.
                probing = [s for s in candidates if s.state == "half_open" and s.half_open_in_flight]
                if probing:
                    raise AllProbesInFlight(
                        "no routable endpoint: every half-open circuit already has a probe "
                        f"in flight ({len(probing)}). Back off briefly and retry. "
                        f"Endpoints: {[s.as_dict() for s in candidates]}"
                    )
                raise NoHealthyEndpoint(
                    f"no healthy endpoint available; all circuits open. Endpoints: {[s.as_dict() for s in candidates]}"
                )
            # round-robin among usable by index into the full ordered list
            order = [s for s in candidates if s in usable]
            pick = order[self._rr % len(order)]
            self._rr += 1
            if pick.state == "half_open":
                # Reserve this endpoint for a single probe until report() resolves.
                pick.half_open_in_flight = True
            pick.requests += 1
            return pick

    def report(self, base_url: str, ok: bool) -> None:
        """Record the outcome of a request and update circuit state.

        Always clears ``half_open_in_flight`` when resolving a half-open probe,
        whether the probe succeeded (-> closed) or failed (-> open), so the
        endpoint never gets stuck reserved if the probing request errors.
        """
        with self._lock:
            st = self._stats[base_url]
            if ok:
                st.successes += 1
                st.consecutive_failures = 0
                if st.state == "half_open":
                    st.state = "closed"
                    st.opened_at = None
                    st.half_open_in_flight = False
            else:
                st.failures += 1
                st.consecutive_failures += 1
                if st.state == "half_open" or st.consecutive_failures >= self.failure_threshold:
                    if st.state != "open":
                        st.circuit_open_count += 1
                    st.state = "open"
                    st.opened_at = self._now()
                    # A failed probe concludes the half-open trial: release the
                    # reservation (the endpoint is now open and will cool down).
                    st.half_open_in_flight = False

    def has_healthy(self) -> bool:
        with self._lock:
            for st in self._stats.values():
                self._maybe_half_open(st)
                if st.state == "closed" or (st.state == "half_open" and not st.half_open_in_flight):
                    return True
            return False

    def snapshot(self) -> list[dict]:
        """Point-in-time list of per-endpoint stats for the run manifest."""
        with self._lock:
            return [s.as_dict() for s in self._stats.values()]
