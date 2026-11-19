"""En-têtes des flux Server-Sent Events (chat, analyses).

Derrière un proxy (hébergeur), une réponse peut être mise en mémoire tampon puis envoyée d'un
bloc : le chat n'apparaîtrait plus au fil de l'eau. `X-Accel-Buffering: no` le demande aux proxys
de type nginx, `Cache-Control: no-cache` aux caches.
"""

SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
