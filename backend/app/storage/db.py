"""Base de données de l'application : Postgres (Supabase) si DATABASE_URL, sinon SQLite.

- Postgres : les analyses et conversations des utilisateurs (schéma de la migration
  supabase/migrations/20261116000000_comptes_analyses_conversations.sql) et l'état de LangGraph
  (schéma `langgraph`). Une seule réserve de connexions par processus, partagée par les deux.
- SQLite (poste de développement, tests) : mêmes tables dans data/cache/app.sqlite, créées ici.

Les requêtes sont écrites une fois, avec des paramètres `%s` ; les dates sont passées par le code
(UTC) et rendues en texte ISO, le JSON est rendu en objets Python, quelle que soit la base.

Connexion à Supabase : le « session pooler » (port 5432), et non la connexion directe, qui ne
répond qu'en IPv6 (inaccessible depuis Docker et la plupart des hébergeurs).
"""

import json
import sqlite3
import threading
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from app.core.config import DATA_DIR, get_settings

SQLITE_PATH = DATA_DIR / "cache" / "app.sqlite"
# l'état de LangGraph d'abord : ses tables (checkpoints...) y sont créées ; les tables de
# l'application, absentes de `langgraph`, sont trouvées dans `public`
SEARCH_PATH = "langgraph, public"
POOL_MAX = 4  # le pooler gratuit de Supabase limite les connexions : API et worker en ont 4

SQLITE_SCHEMA = """
create table if not exists analyses (
    id text primary key,
    owner_id text not null,
    titre text not null,
    source text not null,
    statut text not null default 'en_attente',
    niveau text,
    rapport text,
    markdown text,
    created_at text not null,
    updated_at text not null
);
create index if not exists analyses_owner_created_idx on analyses (owner_id, created_at desc);
create table if not exists conversations (
    id text primary key,
    owner_id text not null,
    titre text not null,
    created_at text not null,
    updated_at text not null
);
create index if not exists conversations_owner_updated_idx
    on conversations (owner_id, updated_at desc);
create table if not exists messages (
    id integer primary key autoincrement,
    conversation_id text not null references conversations (id) on delete cascade,
    role text not null,
    content text not null,
    sources text,
    created_at text not null
);
create index if not exists messages_conversation_idx on messages (conversation_id, id);
"""


def _iso(value):
    return value.isoformat() if isinstance(value, datetime) else value


class Database:
    """Exécute une requête et rend des dictionnaires, sur Postgres ou SQLite."""

    def __init__(self, url: str | None = None, sqlite_path: Path = SQLITE_PATH, pool=None):
        self.postgres = bool(url or pool)
        if self.postgres:
            self.pool = pool or make_pool(url)
        else:
            sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(sqlite_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("pragma foreign_keys = on")
            self._conn.execute("pragma journal_mode = wal")  # API et worker lisent ensemble
            self._conn.executescript(SQLITE_SCHEMA)
            self._lock = threading.Lock()

    def execute(self, sql: str, params: tuple = ()) -> list[dict]:
        if self.postgres:
            with self.pool.connection() as conn:
                cursor = conn.execute(sql, params)
                rows = cursor.fetchall() if cursor.description else []
            return [{k: _iso(v) for k, v in row.items()} for row in rows]
        params = tuple(_iso(p) for p in params)
        with self._lock, self._conn:  # une transaction par requête
            cursor = self._conn.execute(sql.replace("%s", "?"), params)
            return [dict(row) for row in cursor.fetchall()]

    def to_json(self, value):
        """Valeur JSON à écrire (colonne jsonb ou texte)."""
        if value is None:
            return None
        if self.postgres:
            from psycopg.types.json import Jsonb

            return Jsonb(value)
        return json.dumps(value, ensure_ascii=False)

    def from_json(self, value):
        return json.loads(value) if isinstance(value, str) else value


def make_pool(url: str):
    """Réserve de connexions Postgres, réglée pour LangGraph (autocommit, lignes en dict)."""
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    def configure(conn) -> None:
        conn.execute(f"set search_path to {SEARCH_PATH}")

    return ConnectionPool(
        url,
        min_size=1,
        max_size=POOL_MAX,
        # réglages de PostgresSaver.from_conn_string : autocommit (ses migrations créent des
        # index CONCURRENTLY, impossible dans une transaction), lignes rendues en dict
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        configure=configure,
        open=True,
    )


@lru_cache
def get_database() -> Database:
    return Database(get_settings().database_url or None)
