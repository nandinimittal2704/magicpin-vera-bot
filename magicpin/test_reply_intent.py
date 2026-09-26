"""
test_reply_intent.py — Pytest regression test suite for /v1/reply intent branching and pending_action tracking

Tests:
4a. Question ("what is competition") -> gets an answer, no execute CTA, no action mode copy.
4b. "yes" following competitor/win-back nudge -> response mentions win-back offer, NOT Google Business listing.
4c. "yes" with no pending_action -> asks for clarification, does NOT execute anything.
4d. Auto-reply sent twice in a row -> 1st gets polite follow-up, 2nd gets graceful "end".
4e. "Stop messaging me" -> "end" + suppression_key set; subsequent /v1/tick does NOT emit a new nudge.
4f. Deferral ("busy, talk later") -> "wait" action, pending_action is preserved; resuming with "yes" later resolves to original offer.
"""

import pytest
from fastapi.testclient import TestClient
from bot import app, CONTEXT_STORE, CONVERSATION_STORE, FIRED_SUPPRESSIONS

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_teardown():
    """Wipes state before each test."""
    client.post("/v1/teardown")
    # Seed category & merchant
    client.post("/v1/context", json={
        "scope": "category", "context_id": "dentists", "version": 1,
        "payload": {"slug": "dentists", "voice": {"tone": "peer_clinical", "taboos": ["cure", "guaranteed"]}, "peer_stats": {"avg_ctr": 0.030}}
    })
    client.post("/v1/context", json={
        "scope": "merchant", "context_id": "m_001", "version": 1,
        "payload": {
            "merchant_id": "m_001", "category_slug": "dentists",
            "identity": {"name": "Dr. Meera Dental Clinic", "locality": "Lajpat Nagar"}
        }
    })
    yield
    client.post("/v1/teardown")


def test_4a_question_does_not_trigger_action_mode():
    """Question ('what is competition') -> gets an answer, no execute CTA."""
    # Push competitor trigger
    client.post("/v1/context", json={
        "scope": "trigger", "context_id": "trg_comp", "version": 1,
        "payload": {
            "id": "trg_comp", "kind": "competitor_opened", "merchant_id": "m_001",
            "payload": {"headline": "Competitor win-back offer"}, "suppression_key": "supp_comp_1"
        }
    })
    # Run tick to emit nudge
    client.post("/v1/tick", json={"now": "2026-04-26T10:00:00Z", "available_triggers": ["trg_comp"]})

    conv_id = "conv_m_001_trg_comp"
    res = client.post("/v1/reply", json={
        "conversation_id": conv_id,
        "merchant_id": "m_001",
        "message": "what is competition",
        "turn_number": 2
    })
    assert res.status_code == 200
    data = res.json()

    assert data["action"] == "send"
    assert data.get("cta") in ["none", "open_ended"]
    body = data.get("body", "")
    assert "Starting this now" not in body
    assert "applied" not in body.lower()
    assert "scheduled" not in body.lower()
    assert "prepared" not in body.lower()
    assert "Should I go ahead and execute this now" not in body


def test_4b_yes_resolves_to_specific_pending_action():
    """'yes' following competitor/win-back nudge -> response mentions win-back offer, NOT Google Business listing."""
    client.post("/v1/context", json={
        "scope": "trigger", "context_id": "trg_winback", "version": 1,
        "payload": {
            "id": "trg_winback", "kind": "competitor_opened", "merchant_id": "m_001",
            "payload": {"headline": "Competitor win-back campaign"}, "suppression_key": "supp_winback_1"
        }
    })
    client.post("/v1/tick", json={"now": "2026-04-26T10:00:00Z", "available_triggers": ["trg_winback"]})

    conv_id = "conv_m_001_trg_winback"
    res = client.post("/v1/reply", json={
        "conversation_id": conv_id,
        "merchant_id": "m_001",
        "message": "yes",
        "turn_number": 2
    })
    assert res.status_code == 200
    data = res.json()

    assert data["action"] == "send"
    body = data.get("body", "")
    assert "Competitor win-back campaign" in body or "competitor" in body.lower()
    assert "Google Business listing optimization update" not in body


def test_4c_yes_with_no_pending_action_asks_for_clarification():
    """'yes' with no pending_action -> asks for clarification."""
    res = client.post("/v1/reply", json={
        "conversation_id": "conv_no_pending",
        "merchant_id": "m_001",
        "message": "yes",
        "turn_number": 1
    })
    assert res.status_code == 200
    data = res.json()

    assert data["action"] == "send"
    assert data.get("cta") == "open_ended"
    body = data.get("body", "")
    assert "clarify" in body.lower()
    assert "Starting this now" not in body


def test_4d_auto_reply_escalation():
    """Auto-reply sent twice in a row -> 1st gets polite follow-up, 2nd gets graceful 'end'."""
    conv_id = "conv_auto_test"
    canned_msg = "Thank you for contacting us! Our team will respond shortly."

    # 1st auto-reply
    res1 = client.post("/v1/reply", json={
        "conversation_id": conv_id, "merchant_id": "m_001", "message": canned_msg, "turn_number": 1
    }).json()
    assert res1["action"] == "send"

    # 2nd auto-reply
    res2 = client.post("/v1/reply", json={
        "conversation_id": conv_id, "merchant_id": "m_001", "message": canned_msg, "turn_number": 2
    }).json()
    assert res2["action"] == "end"


def test_4e_decline_sets_suppression_and_prevents_future_nudges():
    """'Stop messaging me' -> 'end' + suppression_key set; subsequent /v1/tick does NOT emit nudge."""
    client.post("/v1/context", json={
        "scope": "trigger", "context_id": "trg_stop_test", "version": 1,
        "payload": {
            "id": "trg_stop_test", "kind": "perf_dip", "merchant_id": "m_001",
            "payload": {"headline": "Profile calls dip"}, "suppression_key": "supp_stop_99"
        }
    })
    client.post("/v1/tick", json={"now": "2026-04-26T10:00:00Z", "available_triggers": ["trg_stop_test"]})

    conv_id = "conv_m_001_trg_stop_test"
    res = client.post("/v1/reply", json={
        "conversation_id": conv_id, "merchant_id": "m_001", "message": "Stop messaging me. This is useless spam.", "turn_number": 2
    }).json()

    assert res["action"] == "end"

    # Verify subsequent tick for the same trigger does NOT emit new action
    tick_res = client.post("/v1/tick", json={"now": "2026-04-26T10:05:00Z", "available_triggers": ["trg_stop_test"]}).json()
    assert len(tick_res["actions"]) == 0


def test_4f_deferral_preserves_pending_action():
    """Deferral ('busy, talk later') -> 'wait' action, pending_action is preserved; resuming with 'yes' resolves to original offer."""
    client.post("/v1/context", json={
        "scope": "trigger", "context_id": "trg_defer", "version": 1,
        "payload": {
            "id": "trg_defer", "kind": "festival_upcoming", "merchant_id": "m_001",
            "payload": {"headline": "Diwali Combo Package"}, "suppression_key": "supp_defer_1"
        }
    })
    client.post("/v1/tick", json={"now": "2026-04-26T10:00:00Z", "available_triggers": ["trg_defer"]})

    conv_id = "conv_m_001_trg_defer"
    
    # 1. Merchant defers
    res1 = client.post("/v1/reply", json={
        "conversation_id": conv_id, "merchant_id": "m_001", "message": "Busy right now, talk later.", "turn_number": 2
    }).json()
    assert res1["action"] == "wait"
    assert res1["wait_seconds"] == 1800

    # 2. Merchant resumes later with "yes"
    res2 = client.post("/v1/reply", json={
        "conversation_id": conv_id, "merchant_id": "m_001", "message": "yes", "turn_number": 3
    }).json()

    assert res2["action"] == "send"
    body = res2.get("body", "")
    assert "Diwali Combo Package" in body
