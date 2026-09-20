import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class ConversationStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        connection = self._connect()
        try:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, user_query TEXT NOT NULL,
                    response TEXT NOT NULL, created_at TEXT NOT NULL
                )"""
            )
        finally:
            connection.close()

    def _connect(self):
        return sqlite3.connect(self.path)

    def recent(self, session_id: str, limit: int = 3) -> list[dict]:
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT user_query, response FROM conversations WHERE session_id = ? "
                "ORDER BY id DESC LIMIT ?", (session_id, limit)
            ).fetchall()
        finally:
            connection.close()
        return [{"query": query, "response": response} for query, response in reversed(rows)]

    def save(self, session_id: str, query: str, response: str) -> None:
        connection = self._connect()
        try:
            connection.execute(
                "INSERT INTO conversations (session_id, user_query, response, created_at) VALUES (?, ?, ?, ?)",
                (session_id, query, response, datetime.now(timezone.utc).isoformat()),
            )
            connection.commit()
        finally:
            connection.close()
