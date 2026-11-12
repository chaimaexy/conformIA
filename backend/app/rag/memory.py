"""Mémoire de conversation : les derniers échanges de chaque session.

En mémoire du processus pour l'instant ; l'interface (get / add) restera la même quand
l'historique sera rattaché à l'utilisateur dans Supabase (phase 7, avec l'authentification).
"""

import threading
from collections import defaultdict, deque

MAX_MESSAGES = 6  # 3 échanges : assez pour résoudre « et pour les PME ? »


class ConversationStore:
    def __init__(self, max_messages: int = MAX_MESSAGES):
        self._sessions: dict[str, deque] = defaultdict(lambda: deque(maxlen=max_messages))
        self._lock = threading.Lock()

    def get(self, session_id: str) -> list[dict[str, str]]:
        with self._lock:
            return list(self._sessions.get(session_id, ()))

    def add(self, session_id: str, question: str, answer: str) -> None:
        with self._lock:
            self._sessions[session_id].append({"role": "user", "content": question})
            self._sessions[session_id].append({"role": "assistant", "content": answer})


def format_history(history: list[dict[str, str]], max_chars: int = 400) -> str:
    """Historique lisible pour le routeur ; les longues réponses sont tronquées."""
    if not history:
        return "(aucun)"
    names = {"user": "Utilisateur", "assistant": "Assistant"}
    lines = []
    for message in history:
        content = message["content"]
        if len(content) > max_chars:
            content = content[:max_chars] + "..."
        lines.append(f"{names[message['role']]} : {content}")
    return "\n".join(lines)
