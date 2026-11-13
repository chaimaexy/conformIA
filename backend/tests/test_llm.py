"""Limiteur de débit des appels LLM (horloge simulée : aucun appel réseau, aucune attente)."""

from app.llm.json_chat import RateLimiter, estimate_tokens


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_two_requests_per_minute_are_spaced():
    clock = FakeClock()
    limiter = RateLimiter(rpm=2, tpm=40000, clock=clock, sleep=clock.sleep)
    starts = []
    for _ in range(3):
        limiter.wait(100)
        starts.append(round(clock.now, 1))
    assert starts == [0.0, 30.0, 60.0]


def test_token_budget_limits_large_prompts():
    clock = FakeClock()
    limiter = RateLimiter(rpm=50, tpm=20000, clock=clock, sleep=clock.sleep)
    starts = []
    for _ in range(4):
        limiter.wait(8000)  # 2 appels tiennent dans 20 000 tokens/min, le 3e attend
        starts.append(round(clock.now, 1))
    assert starts[:2] == [0.0, 1.2] and starts[2] >= 60.0 and starts[3] >= 61.2


def test_estimate_tokens_counts_prompt_and_answer():
    body = {"messages": [{"role": "user", "content": "x" * 3000}], "max_tokens": 500}
    assert estimate_tokens(body) == 1500


def test_text_of_ignores_reasoning_blocks():
    from app.llm.json_chat import text_of

    content = [
        {"type": "thinking", "thinking": [{"type": "text", "text": "je réfléchis"}]},
        {"type": "text", "text": '{"ok": true}'},
    ]
    assert text_of(content) == '{"ok": true}'
    assert text_of("simple") == "simple" and text_of(None) == ""


def test_reasoning_effort_is_sent_and_separates_the_cache(monkeypatch, tmp_path):
    from app.llm import json_chat

    bodies = []

    def fake_complete(body):
        bodies.append(body)
        return f"réponse {body.get('reasoning_effort')}"

    monkeypatch.setattr(json_chat, "_complete", fake_complete)
    monkeypatch.setattr(json_chat, "_cache", json_chat.LLMCache(tmp_path / "llm.sqlite"))
    messages = [{"role": "user", "content": "q"}]
    assert json_chat.chat_text(messages, model="m") == "réponse None"
    assert json_chat.chat_text(messages, model="m", effort="low") == "réponse low"
    assert json_chat.chat_text(messages, model="m", effort="low") == "réponse low"  # cache
    assert "reasoning_effort" not in bodies[0] and bodies[1]["reasoning_effort"] == "low"
    assert len(bodies) == 2
