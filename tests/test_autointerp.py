import json

import pytest

from qi.llm.backboard import BackboardClient
from qi.sae.autointerp import CATEGORIES, Example, categorize, enrichment_table, explain_feature


class _Resp:
    def __init__(self, status, payload):
        self.status_code, self._p = status, payload
        self.text = json.dumps(payload)

    def json(self):
        return self._p


class _Session:
    """Fake requests session: first call 503, then 200; records the request body."""

    def __init__(self):
        self.bodies, self.n = [], 0

    def post(self, url, json=None, headers=None, timeout=None):
        self.bodies.append((url, json, headers))
        self.n += 1
        if self.n == 1:
            return _Resp(503, {"error": "busy"})
        return _Resp(200, {"content": "Fires on the final digit of multi-digit numbers in arithmetic.", "thread_id": "t1"})


def test_backboard_client_retries_and_formats_request(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    s = _Session()
    c = BackboardClient(api_key="k", model_name="claude-x", session=s)
    out = c.complete("hello", system="be terse")
    assert "final digit" in out and c.calls == 2
    url, body, headers = s.bodies[-1]
    assert url.endswith("/threads/messages") and headers["X-API-Key"] == "k"
    assert body["memory"] == "off" and body["llm_provider"] == "anthropic" and body["model_name"] == "claude-x"
    assert body["content"].startswith("be terse") and body["content"].endswith("hello")


class _Stub:
    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def complete(self, prompt, system=None):
        self.prompts.append((system, prompt))
        return self.reply


def test_explain_and_categorize():
    ex = Example(tokens=["The", " sum", " is", " 4", "2", "."], activations=[0, 0, 0, 1.5, 9.0, 0])
    stub = _Stub("Fires on digits inside numbers.")
    assert "digits" in explain_feature(stub, [ex])
    assert "<<2>>" in stub.prompts[-1][1]
    assert categorize(_Stub("Numeral_Arithmetic"), "x") == "numeral_arithmetic"
    assert categorize(_Stub("code"), "x") == "code"
    assert categorize(_Stub("something unexpected"), "x") == "other"


def test_enrichment_table():
    top = ["numeral_arithmetic"] * 30 + ["code"] * 10 + ["other"] * 60
    ctl = ["numeral_arithmetic"] * 5 + ["code"] * 10 + ["other"] * 85
    t = enrichment_table(top, ctl)
    assert set(t) == set(CATEGORIES)
    assert t["numeral_arithmetic"]["enrichment"] == pytest.approx(6.0)
    assert t["code"]["enrichment"] == pytest.approx(1.0)
