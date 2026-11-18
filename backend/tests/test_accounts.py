"""Phase 7b : sessions Supabase, analyses et conversations par utilisateur, quotas, effacement.

Les tests Postgres (en fin de fichier) tournent si TEST_DATABASE_URL désigne une base jetable
(par exemple : docker run -d -p 54329:5432 -e POSTGRES_PASSWORD=test postgres:17-alpine).
"""

import json
import os
import time
from datetime import timedelta
from pathlib import Path

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.core.auth import TokenVerifier
from app.storage.records import (
    AnalysisRecords,
    Conversations,
    QuotaExceeded,
    check_quota,
    now,
    title,
)

ISSUER = "https://projet.supabase.co/auth/v1"
KEY = ec.generate_private_key(ec.SECP256R1())  # clé de test : signe comme Supabase (ES256)
OTHER_KEY = ec.generate_private_key(ec.SECP256R1())
VERIFY = TokenVerifier(ISSUER, lambda token: KEY.public_key())


def token(key=KEY, **claims) -> str:
    payload = {
        "sub": "3f1c0d2e-0000-4000-8000-000000000001",
        "aud": "authenticated",
        "iss": ISSUER,
        "exp": int(time.time()) + 3600,
        "role": "authenticated",
        **claims,
    }
    return jwt.encode(payload, key, algorithm="ES256", headers={"kid": "cle-test"})


# ----------------------------------------------------------------------------- jetons
def test_a_valid_supabase_token_gives_the_user():
    user = VERIFY(token())
    assert user.id == "3f1c0d2e-0000-4000-8000-000000000001" and not user.anonyme
    assert VERIFY(token(is_anonymous=True)).anonyme  # « Essayer sans compte »


@pytest.mark.parametrize(
    "bad",
    [
        {"key": OTHER_KEY},  # signé par une autre clé
        {"exp": int(time.time()) - 10},  # expiré
        {"aud": "anon"},  # pas une session d'utilisateur
        {"iss": "https://autre-projet.supabase.co/auth/v1"},  # autre projet Supabase
    ],
)
def test_invalid_tokens_are_refused(bad):
    with pytest.raises(HTTPException) as refused:
        VERIFY(token(**bad))
    assert refused.value.status_code == 401


def test_a_token_signed_with_a_shared_secret_is_refused():
    forged = jwt.encode(
        {"sub": "x", "aud": "authenticated", "iss": ISSUER, "exp": int(time.time()) + 60},
        "un secret partagé assez long pour HS256, 32 octets",
        algorithm="HS256",
    )
    with pytest.raises(HTTPException):
        VERIFY(forged)


def test_requests_without_a_session_are_refused():
    from app.main import app

    client = TestClient(app)
    assert client.get("/analyses").status_code == 401
    assert client.post("/chat", json={"message": "Bonjour"}).status_code == 401
    assert client.get("/conversations").status_code == 401
    assert client.get("/health").status_code == 200  # publique


def test_auth_can_be_disabled_only_on_a_local_machine(monkeypatch):
    from app.core import auth
    from app.core.config import Settings

    local = Settings(_env_file=None, auth_disabled=True, environment="local")
    monkeypatch.setattr(auth, "get_settings", lambda: local)
    assert auth.current_user(None).id == auth.LOCAL_USER_ID
    production = Settings(_env_file=None, auth_disabled=True, environment="production")
    monkeypatch.setattr(auth, "get_settings", lambda: production)
    with pytest.raises(RuntimeError):
        auth.current_user(None)


# ----------------------------------------------------------------------------- base
def test_titles_are_short_and_on_one_line():
    assert title("Un  outil\nde tri") == "Un outil de tri"
    assert len(title("mot " * 100)) == 80 and title("mot " * 100).endswith("…")


def test_analysis_records_follow_the_events(db):
    records = AnalysisRecords(db)
    records.create("a1", "alice", "Tri de CV", "description")
    assert records.get("a1", "bob") is None  # pas à lui
    records.on_event("a1", {"type": "debut"})
    assert records.get("a1", "alice")["statut"] == "en_cours"
    records.on_event("a1", {"type": "questions", "questions": []})
    assert records.get("a1", "alice")["statut"] == "questions"
    rapport = {"classification": {"niveau": "haut_risque"}, "obligations": []}
    records.on_event("a1", {"type": "rapport", "rapport": rapport, "markdown": "# Rapport"})
    row = records.get("a1", "alice")
    assert (row["statut"], row["niveau"], row["rapport"]) == ("termine", "haut_risque", rapport)
    assert [r["id"] for r in records.of_owner("alice")] == ["a1"] and records.of_owner("bob") == []
    assert not records.delete("a1", "bob") and records.delete("a1", "alice")
    assert records.get("a1", "alice") is None


def test_conversations_keep_messages_and_sources(db):
    conversations = Conversations(db, max_messages=2)
    cid = conversations.create("alice", "Quand l'AI Act s'applique-t-il ?")
    conversations.add(
        cid, "Quand ?", "Le 2 août 2026 [1].", sources=[{"n": 1, "titre": "Art. 113"}]
    )
    conversations.add(cid, "Et pour les PME ?", "Pareil [1].", sources=[])
    assert conversations.owned(cid, "alice") and not conversations.owned(cid, "bob")
    assert conversations.get(cid) == [  # mémoire du routeur : les 2 derniers messages
        {"role": "user", "content": "Et pour les PME ?"},
        {"role": "assistant", "content": "Pareil [1]."},
    ]
    messages = conversations.messages(cid)
    assert [m["role"] for m in messages] == ["user", "assistant"] * 2
    assert messages[1]["sources"] == [{"n": 1, "titre": "Art. 113"}]
    assert conversations.of_owner("alice")[0]["titre"] == "Quand l'AI Act s'applique-t-il ?"
    assert conversations.questions_since("alice", now() - timedelta(hours=1))[0] == 2
    assert conversations.delete(cid, "alice") and conversations.messages(cid) == []  # en cascade


def test_quota_says_when_a_place_frees_up(db):
    records = AnalysisRecords(db)
    for n in range(3):
        records.create(f"a{n}", "alice", "x", "description")
    check_quota(records.count_since, "alice", 4, timedelta(days=1), "analyses par 24 h")
    with pytest.raises(QuotaExceeded) as exceeded:
        check_quota(records.count_since, "alice", 3, timedelta(days=1), "analyses par 24 h")
    first = records.get("a0", "alice")["created_at"]
    assert exceeded.value.detail()["prochaine"] > first  # 24 h après la plus ancienne
    assert 86000 < exceeded.value.retry_after_s() <= 86401
    check_quota(records.count_since, "bob", 3, timedelta(days=1), "analyses par 24 h")


# ----------------------------------------------------------------------------- API
@pytest.fixture
def api(monkeypatch, tmp_path, db, as_user):
    from test_agent import FakeSources, fake_ask

    from app.agents.graph import AnalysisService
    from app.agents.queue import local_queue
    from app.api.routes import analyses
    from app.main import app

    service = AnalysisService(FakeSources(), fake_ask, db_path=tmp_path / "a.sqlite")
    records = AnalysisRecords(db)
    queue = local_queue(service, records)
    monkeypatch.setattr(analyses, "get_service", lambda: service)
    monkeypatch.setattr(analyses, "get_records", lambda: records)
    monkeypatch.setattr(analyses, "get_queue", lambda: queue)
    monkeypatch.setattr(analyses, "POLL_S", 0.01)
    client = TestClient(app)
    client.login, client.service = as_user, service
    return client


def flux(client, ticket) -> list[dict]:
    response = client.get(ticket["flux"])
    return [json.loads(line[6:]) for line in response.text.split("\n\n") if line]


DESCRIPTION = (
    "Nous sommes une PME de 80 salariés. Nous utilisons un logiciel acheté qui trie "
    "automatiquement les CV reçus et classe les candidats pour nos recrutements."
)


def test_an_analysis_exists_only_for_its_author(api):
    ticket = api.post("/analyses", json={"description": DESCRIPTION}).json()
    analysis_id = ticket["analyse_id"]
    questions = flux(api, ticket)[-1]
    assert questions["type"] == "questions"
    mine = api.get("/analyses").json()
    assert [(a["id"], a["statut"], a["source"]) for a in mine] == [
        (analysis_id, "questions", "description")
    ]
    assert api.get(f"/analyses/{analysis_id}").json()["titre"].startswith("Nous sommes une PME")

    api.login("utilisateur-b")
    assert api.get("/analyses").json() == []
    assert api.get(f"/analyses/{analysis_id}").status_code == 404
    assert api.get(ticket["flux"]).status_code == 404
    answers = {"reponses": {q["champ"]: "non" for q in questions["questions"]}}
    assert api.post(f"/analyses/{analysis_id}/reponses", json=answers).status_code == 404
    assert api.delete(f"/analyses/{analysis_id}").status_code == 404

    api.login("utilisateur-a")
    final = flux(api, api.post(f"/analyses/{analysis_id}/reponses", json=answers).json())[-1]
    assert final["type"] == "rapport"
    row = api.get("/analyses").json()[0]
    assert (row["statut"], row["niveau"]) == ("termine", "haut_risque")


def test_the_state_gives_where_to_read_the_stream_again(api):
    """Une page rechargée relit le flux à partir de l'état : jamais d'anciennes questions."""
    ticket = api.post("/analyses", json={"description": DESCRIPTION}).json()
    analysis_id = ticket["analyse_id"]
    started = flux(api, ticket)
    state = api.get(f"/analyses/{analysis_id}").json()
    assert [e["type"] for e in flux(api, state)] == ["questions"]  # la question en attente
    answers = {"reponses": {q["champ"]: "non" for q in started[-1]["questions"]}}
    flux(api, api.post(f"/analyses/{analysis_id}/reponses", json=answers).json())
    state = api.get(f"/analyses/{analysis_id}").json()
    assert [e["type"] for e in flux(api, state)] == ["rapport"]  # pas les questions répondues


def test_deleting_an_analysis_erases_its_state(api):
    ticket = api.post("/analyses", json={"description": DESCRIPTION}).json()
    analysis_id = ticket["analyse_id"]
    flux(api, ticket)
    assert api.service.get(analysis_id) is not None
    assert api.delete(f"/analyses/{analysis_id}").status_code == 204
    assert api.service.get(analysis_id) is None  # profil, réponses : effacés de LangGraph
    assert api.get(f"/analyses/{analysis_id}").status_code == 404
    assert api.get("/analyses").json() == []


def test_analyses_are_limited_per_user(api, monkeypatch):
    from app.api.routes import analyses
    from app.core.config import AppConfig, QuotasConfig

    config = AppConfig(quotas=QuotasConfig(analyses_par_jour=1))
    monkeypatch.setattr(analyses, "get_app_config", lambda: config)
    flux(api, api.post("/analyses", json={"description": DESCRIPTION}).json())
    refused = api.post("/analyses", json={"description": DESCRIPTION})
    assert refused.status_code == 429 and int(refused.headers["Retry-After"]) > 86000
    assert refused.json()["detail"]["message"] == "Limite atteinte : 1 analyses par 24 h."
    api.login("utilisateur-b")  # le quota est par utilisateur
    assert api.post("/analyses", json={"description": DESCRIPTION}).status_code == 202


@pytest.fixture
def chat_api(monkeypatch, db, as_user):
    from test_rag import make_service

    from app.api.routes import chat
    from app.main import app
    from app.rag import service as service_module

    service, _ = make_service(monkeypatch, 0.8, "Applicable dès le 2 août 2026 [1].")
    conversations = Conversations(db)
    service.memory = conversations
    monkeypatch.setattr(chat, "get_service", lambda: service)
    monkeypatch.setattr(chat, "get_conversations", lambda: conversations)
    monkeypatch.setattr(
        service_module, "stream_text", lambda m, model=None, effort=None: iter(["Le 2 août [1]."])
    )
    client = TestClient(app)
    client.login = as_user
    return client


def test_chat_history_belongs_to_its_author(chat_api):
    first = chat_api.post("/chat", json={"message": "Quand l'AI Act s'applique-t-il ?"}).json()
    cid = first["session_id"]
    chat_api.post("/chat/stream", json={"message": "Et pour les PME ?", "session_id": cid})
    listed = chat_api.get("/conversations").json()
    assert [(c["id"], c["titre"]) for c in listed] == [(cid, "Quand l'AI Act s'applique-t-il ?")]
    messages = chat_api.get(f"/conversations/{cid}").json()["messages"]
    assert [m["content"] for m in messages][::2] == [
        "Quand l'AI Act s'applique-t-il ?",
        "Et pour les PME ?",
    ]
    assert messages[1]["sources"][0]["n"] == 1 and "text" not in messages[1]["sources"][0]

    chat_api.login("utilisateur-b")
    assert chat_api.get("/conversations").json() == []
    assert chat_api.get(f"/conversations/{cid}").status_code == 404
    taken = chat_api.post("/chat", json={"message": "Et alors ?", "session_id": cid})
    assert taken.status_code == 404  # on ne poursuit pas la conversation d'un autre
    assert chat_api.delete(f"/conversations/{cid}").status_code == 404

    chat_api.login("utilisateur-a")
    assert chat_api.delete(f"/conversations/{cid}").status_code == 204
    assert chat_api.get("/conversations").json() == []


def test_questions_are_limited_per_user(chat_api, monkeypatch):
    from app.api.routes import chat
    from app.core.config import AppConfig, QuotasConfig

    monkeypatch.setattr(
        chat, "get_app_config", lambda: AppConfig(quotas=QuotasConfig(questions_par_heure=2))
    )
    for _ in range(2):
        assert chat_api.post("/chat", json={"message": "Quand ?"}).status_code == 200
    refused = chat_api.post("/chat", json={"message": "Quand ?"})
    assert (
        refused.status_code == 429
        and "2 questions par heure" in refused.json()["detail"]["message"]
    )
    chat_api.login("utilisateur-b")
    assert chat_api.post("/chat", json={"message": "Quand ?"}).status_code == 200


# ----------------------------------------------------------------------------- Postgres
PG_URL = os.environ.get("TEST_DATABASE_URL")
MIGRATIONS = Path(__file__).resolve().parents[2] / "supabase" / "migrations"
# celles des phases 7b et 7c (les précédentes demandent pgvector, absent d'un Postgres nu)
PHASE_7_MIGRATIONS = [
    MIGRATIONS / "20261116000000_comptes_analyses_conversations.sql",
    MIGRATIONS / "20261118000000_effacer_etat_langgraph.sql",
]
# ce que Supabase fournit et qu'une base Postgres nue n'a pas
SUPABASE_STUB = """
create schema auth;
create table auth.users (id uuid primary key);
create function auth.uid() returns uuid language sql stable as $$ select null::uuid $$;
create role anon nologin;
create role authenticated nologin;
"""
ALICE = "3f1c0d2e-0000-4000-8000-00000000000a"
BOB = "3f1c0d2e-0000-4000-8000-00000000000b"
postgres_only = pytest.mark.skipif(not PG_URL, reason="TEST_DATABASE_URL absent")


@pytest.fixture(scope="module")
def pg():
    import psycopg

    from app.storage.db import Database

    with psycopg.connect(PG_URL, autocommit=True) as conn:  # base remise à neuf
        conn.execute("drop schema if exists public, langgraph, auth cascade")
        conn.execute("create schema public")
        for role in ("anon", "authenticated"):
            conn.execute(f"drop role if exists {role}")
        conn.execute(SUPABASE_STUB)
        for migration in PHASE_7_MIGRATIONS:
            conn.execute(migration.read_text(encoding="utf-8"))
        conn.execute("insert into auth.users (id) values (%s), (%s)", (ALICE, BOB))
    database = Database(PG_URL)
    yield database
    database.pool.close()


@postgres_only
def test_records_and_conversations_on_postgres(pg):
    records, conversations = AnalysisRecords(pg), Conversations(pg)
    records.create("pg1", ALICE, "Tri de CV", "document")
    rapport = {"classification": {"niveau": "minimal"}, "obligations": []}
    records.on_event("pg1", {"type": "rapport", "rapport": rapport, "markdown": "# R"})
    row = records.get("pg1", ALICE)
    assert row["rapport"] == rapport and row["created_at"].endswith("+00:00")
    assert records.get("pg1", BOB) is None
    assert records.count_since(ALICE, now() - timedelta(days=1))[0] == 1
    cid = conversations.create(ALICE, "Quand ?")
    conversations.add(cid, "Quand ?", "Le 2 août [1].", sources=[{"n": 1}])
    assert conversations.messages(cid)[1]["sources"] == [{"n": 1}]
    assert conversations.questions_since(ALICE, now() - timedelta(hours=1))[0] == 1
    with pytest.raises(Exception, match="foreign key"):  # propriétaire inconnu de Supabase Auth
        records.create("pg2", "3f1c0d2e-0000-4000-8000-0000000000ff", "x", "description")


@postgres_only
def test_deleting_a_user_deletes_their_data(pg):
    records = AnalysisRecords(pg)
    records.create("pg3", BOB, "x", "description")
    pg.execute("delete from auth.users where id = %s", (BOB,))
    assert records.get("pg3", BOB) is None


@postgres_only
def test_langgraph_state_lives_in_its_own_schema(pg, tmp_path):
    from test_agent import CV, FakeSources, fake_ask

    from app.agents.graph import AnalysisService, postgres_checkpointer

    service = AnalysisService(FakeSources(), fake_ask, checkpointer=postgres_checkpointer(pg.pool))
    first = list(service.start(CV))[-1]
    analysis_id = first["analyse_id"]
    answers = {q["champ"]: "non" for q in first["questions"]}
    assert list(service.resume(analysis_id, answers))[-1]["type"] == "rapport"
    assert service.get(analysis_id)["statut"] == "termine"
    tables = pg.execute(
        "select table_schema, table_name from information_schema.tables"
        " where table_name like 'checkpoint%%' order by table_name"
    )
    assert {t["table_schema"] for t in tables} == {"langgraph"}
    service.delete(analysis_id)
    assert service.get(analysis_id) is None


@postgres_only
def test_deleting_a_user_also_erases_the_langgraph_state(pg):
    """Pas de clé étrangère vers `langgraph` : c'est le déclencheur qui efface l'état."""
    from test_agent import CV, FakeSources, fake_ask

    from app.agents.graph import AnalysisService, postgres_checkpointer

    carol = "3f1c0d2e-0000-4000-8000-00000000000c"
    pg.execute("insert into auth.users (id) values (%s)", (carol,))
    service = AnalysisService(FakeSources(), fake_ask, checkpointer=postgres_checkpointer(pg.pool))
    analysis_id = list(service.start(CV))[-1]["analyse_id"]  # profil et questions sauvegardés
    AnalysisRecords(pg).create(analysis_id, carol, "Tri de CV", "description")

    def state_rows() -> int:
        return sum(
            pg.execute(f"select count(*) as n from {table} where thread_id = %s", (analysis_id,))[
                0
            ]["n"]
            for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes")
        )

    assert state_rows() > 0
    pg.execute("delete from auth.users where id = %s", (carol,))
    assert state_rows() == 0 and service.get(analysis_id) is None


@postgres_only
def test_public_api_roles_cannot_write_or_read_other_rows(pg):
    grants = pg.execute(
        "select grantee, privilege_type from information_schema.role_table_grants"
        " where table_schema = 'public' and table_name = 'analyses'"
        " and grantee in ('anon', 'authenticated')"
    )
    assert {g["privilege_type"] for g in grants if g["grantee"] == "authenticated"} <= {
        "SELECT",
        "REFERENCES",
        "TRIGGER",
    }
    assert not [g for g in grants if g["grantee"] == "anon"]
    schema = pg.execute(
        "select has_schema_privilege('authenticated', 'langgraph', 'usage') as usage"
    )[0]
    assert schema["usage"] is False
