from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class ProviderStats:
    requests: int = 0
    successes: int = 0
    failures: int = 0
    fallbacks: int = 0
    last_error: Optional[str] = None
    last_error_ts: Optional[int] = None
    last_latency_ms: Optional[int] = None


@dataclass
class EngineStats:
    providers: Dict[str, ProviderStats] = field(default_factory=dict)
    last_update_ts: Dict[str, Dict[str, int]] = field(default_factory=dict)  # timeframe -> symbol -> ts

    def provider(self, name: str) -> ProviderStats:
        if name not in self.providers:
            self.providers[name] = ProviderStats()
        return self.providers[name]

    def record_request(self, provider: str) -> None:
        self.provider(provider).requests += 1

    def record_success(self, provider: str, latency_ms: Optional[int] = None) -> None:
        p = self.provider(provider)
        p.successes += 1
        if latency_ms is not None:
            p.last_latency_ms = int(latency_ms)

    def record_failure(self, provider: str, error: str) -> None:
        p = self.provider(provider)
        p.failures += 1
        p.last_error = str(error)
        p.last_error_ts = int(time.time())

    def record_fallback(self, from_provider: str, to_provider: str) -> None:
        self.provider(from_provider).fallbacks += 1
        self.provider(to_provider).fallbacks += 1

    def record_last_update(self, timeframe: str, symbol: str, ts: int) -> None:
        if timeframe not in self.last_update_ts:
            self.last_update_ts[timeframe] = {}
        self.last_update_ts[timeframe][symbol] = int(ts)


ENGINE_STATS = EngineStats()

