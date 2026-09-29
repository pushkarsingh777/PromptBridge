"""
memory.py — Conversation memory using SQLite

Two levels:
  Level 1: In-session memory (list in RAM, instant)
  Level 2: Cross-session memory (SQLite, persists forever)

Usage:
  memory = Memory(session_id="user_123")
  memory.add("user", "what is chain of thought?")
  memory.add("assistant", "Chain of thought is...")
  context = memory.get_context()   # last 3 turns as string
"""

import sqlite3
from datetime import datetime
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH       = "./data/memory.db"
MAX_TURNS     = 3     # how many past turns to inject into prompt
MAX_SESSIONS  = 5     # how many past sessions to load on reconnect

class Memory:
    def __init__(self, session_id: str = "default"):
        self.session_id      = session_id
        self.session_history = []    # Level 1: in-session (RAM)
        Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()              # Level 2: cross-session (SQLite)
        self._load_session()         # load current session from DB

    # ── DB setup ──────────────────────────────────────────────────────────────
    def _init_db(self):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role       TEXT NOT NULL,     -- 'user' or 'assistant'
                    content    TEXT NOT NULL,
                    ts         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_session
                ON conversations(session_id, ts)
            """)
            conn.commit()

    # ── Load session from DB ──────────────────────────────────────────────────
    def _load_session(self):
        """Load this session's history from DB into RAM."""
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute("""
                SELECT role, content FROM conversations
                WHERE session_id = ?
                ORDER BY ts ASC
            """, (self.session_id,)).fetchall()

        self.session_history = [
            {"role": row[0], "content": row[1]}
            for row in rows
        ]

    # ── Add a message ─────────────────────────────────────────────────────────
    def add(self, role: str, content: str):
        """Save a message to RAM + SQLite."""
        msg = {"role": role, "content": content}

        # Level 1: in-session RAM
        self.session_history.append(msg)

        # Level 2: SQLite persistence
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO conversations (session_id, role, content)
                VALUES (?, ?, ?)
            """, (self.session_id, role, content))
            conn.commit()

    # ── Get context for prompt ────────────────────────────────────────────────
    def get_context(self, max_turns: int = MAX_TURNS) -> str:
        """
        Returns last N turns formatted as a string
        to inject into the LLM prompt.
        """
        # Take last max_turns*2 messages (each turn = 1 user + 1 assistant)
        recent = self.session_history[-(max_turns * 2):]

        if not recent:
            return ""

        lines = []
        for msg in recent:
            role    = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"][:300]   # truncate long messages
            lines.append(f"{role}: {content}")

        return "\n".join(lines)

    # ── Get full history ──────────────────────────────────────────────────────
    def get_history(self) -> list[dict]:
        """Returns full session history as list of dicts."""
        return self.session_history.copy()

    # ── Clear session ─────────────────────────────────────────────────────────
    def clear_session(self):
        """Clear this session from RAM and DB."""
        self.session_history = []
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "DELETE FROM conversations WHERE session_id = ?",
                (self.session_id,)
            )
            conn.commit()

    # ── List all sessions ─────────────────────────────────────────────────────
    def list_sessions(self) -> list[dict]:
        """List all sessions with message count."""
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute("""
                SELECT session_id, COUNT(*) as msg_count, MAX(ts) as last_seen
                FROM conversations
                GROUP BY session_id
                ORDER BY last_seen DESC
                LIMIT 20
            """).fetchall()
        return [
            {"session_id": r[0], "messages": r[1], "last_seen": r[2]}
            for r in rows
        ]

    @property
    def turn_count(self) -> int:
        return len(self.session_history) // 2