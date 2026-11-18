"""Fixtures communes : base SQLite jetable et utilisateur connecté pour les tests de l'API."""

import pytest


@pytest.fixture
def db(tmp_path):
    from app.storage.db import Database

    return Database(sqlite_path=tmp_path / "app.sqlite")


@pytest.fixture
def as_user():
    """as_user("a") : les requêtes suivantes sont faites par l'utilisateur « a » (sans jeton :
    la vérification des jetons a ses propres tests, dans test_accounts.py)."""
    from app.core.auth import User, current_user
    from app.main import app

    def login(user_id: str) -> None:
        app.dependency_overrides[current_user] = lambda: User(id=user_id)

    login("utilisateur-a")
    yield login
    app.dependency_overrides.pop(current_user, None)
