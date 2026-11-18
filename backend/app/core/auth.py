"""Utilisateur de la requête : jeton de session Supabase Auth, vérifié par le backend.

L'interface envoie `Authorization: Bearer <jeton>` (le jeton d'accès de la session Supabase,
y compris pour une session anonyme « Essayer sans compte »). Le backend vérifie :
- la signature, avec les clés publiques du projet (JWKS, ES256) : pas de secret partagé ;
- l'émetteur (`https://<projet>.supabase.co/auth/v1`), l'audience (`authenticated`) et la date
  d'expiration.
Les clés publiques sont gardées 10 minutes, comme le cache de Supabase : une clé révoquée
cesse d'être acceptée peu après.

AUTH_DISABLED=true (poste de développement seulement, avec SQLite) : pas de jeton, un
utilisateur local fixe. Refusé hors de ENVIRONMENT=local.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import get_settings

ALGORITHMS = ["ES256", "RS256"]  # clés asymétriques de Supabase (ES256 pour ce projet)
AUDIENCE = "authenticated"
JWKS_CACHE_S = 600
JWKS_TIMEOUT_S = 10
LOCAL_USER_ID = "00000000-0000-0000-0000-000000000000"


@dataclass(frozen=True)
class User:
    id: str
    anonyme: bool = False


class TokenVerifier:
    """Vérifie un jeton ; `signing_key(token)` rend la clé publique qui l'a signé."""

    def __init__(self, issuer: str, signing_key):
        self.issuer, self.signing_key = issuer, signing_key

    def __call__(self, token: str) -> User:
        try:
            claims = jwt.decode(
                token,
                self.signing_key(token),
                algorithms=ALGORITHMS,
                audience=AUDIENCE,
                issuer=self.issuer,
                options={"require": ["exp", "sub", "iss", "aud"]},
            )
        except jwt.PyJWTError as exc:
            raise HTTPException(401, "Session invalide ou expirée : reconnectez-vous.") from exc
        return User(id=claims["sub"], anonyme=bool(claims.get("is_anonymous")))


@lru_cache
def get_verifier() -> TokenVerifier:
    settings = get_settings()
    if not settings.supabase_url:
        raise RuntimeError("SUPABASE_URL manquant : impossible de vérifier les sessions")
    base = settings.supabase_url.rstrip("/")
    jwks = jwt.PyJWKClient(
        settings.supabase_jwks_url or f"{base}/auth/v1/.well-known/jwks.json",
        lifespan=JWKS_CACHE_S,
        timeout=JWKS_TIMEOUT_S,  # Supabase injoignable : refus rapide plutôt que requête bloquée
    )
    return TokenVerifier(f"{base}/auth/v1", lambda token: jwks.get_signing_key_from_jwt(token).key)


bearer = HTTPBearer(auto_error=False)


def current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> User:
    settings = get_settings()
    if settings.auth_disabled:
        if settings.environment != "local":
            raise RuntimeError("AUTH_DISABLED n'est permis qu'en environnement local")
        return User(id=LOCAL_USER_ID)
    if credentials is None:
        raise HTTPException(401, "Connexion requise.", headers={"WWW-Authenticate": "Bearer"})
    return get_verifier()(credentials.credentials)


CurrentUser = Annotated[User, Depends(current_user)]
