from __future__ import annotations

import os
import sqlite3
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class SQLiteStoreConfig:
    path: str


class SQLiteCandleStore:
    """
    Simple durable candle store for Step 2B.

    Schema enforces uniqueness on (symbol, timeframe, timestamp) for deterministic idempotency.
    """

    def __init__(self, config: Optional[SQLiteStoreConfig] = None):
        db_path = (config.path if config else None) or os.getenv("TL_CANDLE_DB_PATH") or "tradelens_candles.sqlite3"
        self.path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS candles (
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    PRIMARY KEY (symbol, timeframe, timestamp)
                );
                """
            )
            self._conn.commit()

    def put_many(self, symbol: str, timeframe: str, candles: List[Dict]) -> int:
        if not candles:
            return 0
        rows = [
            (
                symbol,
                timeframe,
                int(c["timestamp"]),
                float(c["open"]),
                float(c["high"]),
                float(c["low"]),
                float(c["close"]),
                float(c["volume"]),
            )
            for c in candles
        ]
        with self._lock:
            cur = self._conn.executemany(
                """
                INSERT OR IGNORE INTO candles
                (symbol, timeframe, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                rows,
            )
            self._conn.commit()
            return cur.rowcount if cur.rowcount is not None else 0

    def put_one(self, symbol: str, timeframe: str, candle: Dict) -> bool:
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT OR IGNORE INTO candles
                (symbol, timeframe, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    symbol,
                    timeframe,
                    int(candle["timestamp"]),
                    float(candle["open"]),
                    float(candle["high"]),
                    float(candle["low"]),
                    float(candle["close"]),
                    float(candle["volume"]),
                ),
            )
            self._conn.commit()
            return cur.rowcount == 1

    def get_recent(self, symbol: str, timeframe: str, limit: int) -> List[Dict]:
        if limit <= 0:
            return []
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT timestamp, open, high, low, close, volume
                FROM candles
                WHERE symbol = ? AND timeframe = ?
                ORDER BY timestamp DESC
                LIMIT ?;
                """,
                (symbol, timeframe, int(limit)),
            )
            rows = cur.fetchall()
        rows.reverse()
        return [
            {
                "symbol": symbol,
                "timestamp": int(ts),
                "open": float(o),
                "high": float(h),
                "low": float(l),
                "close": float(c),
                "volume": float(v),
            }
            for (ts, o, h, l, c, v) in rows
        ]

    def get_latest_timestamp(self, symbol: str, timeframe: str) -> Optional[int]:
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT MAX(timestamp)
                FROM candles
                WHERE symbol = ? AND timeframe = ?;
                """,
                (symbol, timeframe),
            )
            (val,) = cur.fetchone()
        return int(val) if val is not None else None

