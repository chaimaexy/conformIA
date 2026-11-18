"""Analyses et conversations de chaque utilisateur, et quotas.

Chaque ligne a un propriétaire (`owner_id`, l'identifiant Supabase Auth). Les routes de l'API ne
rendent une analyse ou une conversation qu'à son propriétaire ; pour les autres, elle n'existe
pas (404, pas 403 : on ne révèle pas qu'un identifiant est pris).
"""

import uuid
from datetime import UTC, datetime, timedelta

from app.rag.memory import MAX_MESSAGES
from app.storage.db import Database

TITLE_CHARS = 80
EVENT_STATUS = {"debut": "en_cours", "questions": "questions", "rapport": "termine"}


def now() -> datetime:
    return datetime.now(UTC)


def title(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= TITLE_CHARS else text[: TITLE_CHARS - 1].rstrip() + "…"


class AnalysisRecords:
    """Une ligne par analyse : propriétaire, statut, et copie du rapport une fois fini."""

    def __init__(self, db: Database):
        self.db = db

    def create(self, analysis_id: str, owner_id: str, titre: str, source: str) -> None:
        at = now()
        self.db.execute(
            "insert into analyses (id, owner_id, titre, source, statut, created_at, updated_at)"
            " values (%s, %s, %s, %s, 'en_attente', %s, %s)",
            (analysis_id, owner_id, title(titre), source, at, at),
        )

    def get(self, analysis_id: str, owner_id: str) -> dict | None:
        rows = self.db.execute(
            "select * from analyses where id = %s and owner_id = %s", (analysis_id, owner_id)
        )
        if not rows:
            return None
        row = rows[0]
        row["rapport"] = self.db.from_json(row["rapport"])
        return row

    def of_owner(self, owner_id: str, limit: int = 50) -> list[dict]:
        return self.db.execute(
            "select id, titre, source, statut, niveau, created_at, updated_at from analyses"
            " where owner_id = %s order by created_at desc limit %s",
            (owner_id, limit),
        )

    def set_status(self, analysis_id: str, statut: str) -> None:
        self.db.execute(
            "update analyses set statut = %s, updated_at = %s where id = %s",
            (statut, now(), analysis_id),
        )

    def on_event(self, analysis_id: str, event: dict) -> None:
        """Suit les événements de l'analyse (exécuté par le worker)."""
        kind = event.get("type")
        if kind == "rapport":
            rapport = event["rapport"]
            self.db.execute(
                "update analyses set statut = 'termine', niveau = %s, rapport = %s,"
                " markdown = %s, updated_at = %s where id = %s",
                (
                    rapport["classification"]["niveau"],
                    self.db.to_json(rapport),
                    event.get("markdown"),
                    now(),
                    analysis_id,
                ),
            )
        elif kind == "erreur":
            self.set_status(analysis_id, "erreur")
        elif kind in EVENT_STATUS:
            self.set_status(analysis_id, EVENT_STATUS[kind])

    def delete(self, analysis_id: str, owner_id: str) -> bool:
        rows = self.db.execute(
            "delete from analyses where id = %s and owner_id = %s returning id",
            (analysis_id, owner_id),
        )
        return bool(rows)

    def count_since(self, owner_id: str, since: datetime) -> tuple[int, str | None]:
        """Analyses lancées depuis `since`, et date de la plus ancienne d'entre elles."""
        row = self.db.execute(
            "select count(*) as n, min(created_at) as first from analyses"
            " where owner_id = %s and created_at >= %s",
            (owner_id, since),
        )[0]
        return row["n"], row["first"]


class Conversations:
    """Historique du chat par utilisateur. get / add : l'interface de la mémoire du ChatService
    (session_id = identifiant de la conversation)."""

    def __init__(self, db: Database, max_messages: int = MAX_MESSAGES):
        self.db, self.max_messages = db, max_messages

    def create(self, owner_id: str, first_message: str) -> str:
        conversation_id, at = uuid.uuid4().hex, now()
        self.db.execute(
            "insert into conversations (id, owner_id, titre, created_at, updated_at)"
            " values (%s, %s, %s, %s, %s)",
            (conversation_id, owner_id, title(first_message), at, at),
        )
        return conversation_id

    def owned(self, conversation_id: str, owner_id: str) -> bool:
        return bool(
            self.db.execute(
                "select 1 as ok from conversations where id = %s and owner_id = %s",
                (conversation_id, owner_id),
            )
        )

    def of_owner(self, owner_id: str, limit: int = 50) -> list[dict]:
        return self.db.execute(
            "select id, titre, created_at, updated_at from conversations"
            " where owner_id = %s order by updated_at desc limit %s",
            (owner_id, limit),
        )

    def messages(self, conversation_id: str) -> list[dict]:
        rows = self.db.execute(
            "select role, content, sources, created_at from messages"
            " where conversation_id = %s order by id",
            (conversation_id,),
        )
        for row in rows:
            row["sources"] = self.db.from_json(row["sources"])
        return rows

    def get(self, session_id: str) -> list[dict[str, str]]:
        """Derniers messages, pour le routeur (« et pour les PME ? »)."""
        rows = self.db.execute(
            "select role, content from messages where conversation_id = %s"
            " order by id desc limit %s",
            (session_id, self.max_messages),
        )
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def add(self, session_id: str, question: str, answer: str, sources=None) -> None:
        at = now()
        self.db.execute(
            "insert into messages (conversation_id, role, content, sources, created_at)"
            " values (%s, 'user', %s, null, %s), (%s, 'assistant', %s, %s, %s)",
            (session_id, question, at, session_id, answer, self.db.to_json(sources), at),
        )
        self.db.execute("update conversations set updated_at = %s where id = %s", (at, session_id))

    def delete(self, conversation_id: str, owner_id: str) -> bool:
        rows = self.db.execute(
            "delete from conversations where id = %s and owner_id = %s returning id",
            (conversation_id, owner_id),
        )
        return bool(rows)

    def questions_since(self, owner_id: str, since: datetime) -> tuple[int, str | None]:
        row = self.db.execute(
            "select count(*) as n, min(m.created_at) as first from messages m"
            " join conversations c on c.id = m.conversation_id"
            " where c.owner_id = %s and m.role = 'user' and m.created_at >= %s",
            (owner_id, since),
        )[0]
        return row["n"], row["first"]


class QuotaExceeded(Exception):
    def __init__(self, message: str, retry_at: datetime):
        super().__init__(message)
        self.message, self.retry_at = message, retry_at

    def detail(self) -> dict:
        # l'interface affiche l'heure dans le fuseau de l'utilisateur
        return {"message": self.message, "prochaine": self.retry_at.isoformat()}

    def retry_after_s(self) -> int:
        return max(int((self.retry_at - now()).total_seconds()) + 1, 1)


def check_quota(count_since, owner_id: str, limit: int, window: timedelta, what: str) -> None:
    """Refuse au-delà de `limit` sur une fenêtre glissante : la place se libère quand la plus
    ancienne demande de la fenêtre en sort."""
    count, first = count_since(owner_id, now() - window)
    if count < limit:
        return
    retry_at = datetime.fromisoformat(first) + window if first else now() + window
    raise QuotaExceeded(f"Limite atteinte : {limit} {what}.", retry_at)
