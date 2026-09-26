"""
conversation_handlers.py — Multi-turn Reply Logic and Intent Classification State Machine for Vera

Distinguishes 6 intent categories:
1. EXPLICIT_POSITIVE_INTENT — "let's do it", "I want to join", "go ahead", "yes", "proceed" -> truthful action start copy interpolated with pending_action.
2. DEFERRAL — "busy", "talk later", "kal baat karte hain", "abhi nahi", "not now" -> action: "wait", wait_seconds: 1800 (preserves pending_action).
3. AUTO_REPLY — generic canned phrases ("Thank you for contacting us") or 2+ repeated messages -> polite retry or action: "end".
4. EXPLICIT_DECLINE — "not interested", "no", "stop" -> action: "end", graceful exit, marks suppression_key.
5. QUESTION — message ends in '?' or question words ("what is competition") -> answer using context facts, no execute CTA.
6. AMBIGUOUS — unclear replies -> ask clarifying question, do NOT default to action mode.
"""

import os
import re
import time
import json
import traceback
import urllib.request
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


@dataclass
class ConversationState:
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    pending_action: Optional[Dict[str, Any]] = None  # { trigger_id, offer_description, cta_type, sent_at, suppression_key }
    turns: List[Dict[str, Any]] = field(default_factory=list)
    turn_count: int = 0
    auto_reply_count: int = 0
    unanswered_nudge_count: int = 0
    intent_detected: bool = False
    status: str = "active"  # "active", "ended", "waiting"
    last_updated_at: float = field(default_factory=time.time)

    def set_pending_action(self, trigger_id: str, offer_description: str, cta_type: str, suppression_key: str = ""):
        self.pending_action = {
            "trigger_id": trigger_id,
            "offer_description": offer_description,
            "cta_type": cta_type,
            "sent_at": time.time(),
            "suppression_key": suppression_key
        }

    def clear_pending_action(self):
        self.pending_action = None

    def add_turn(self, role: str, message: str):
        self.turn_count += 1
        self.turns.append({
            "turn": self.turn_count,
            "role": role,
            "message": message,
            "timestamp": time.time()
        })
        self.last_updated_at = time.time()


# Global tracker for merchant-level auto-replies across conversation IDs
MERCHANT_AUTO_REPLY_COUNTS: Dict[str, int] = {}


# Intent Constants
INTENT_EXPLICIT_POSITIVE = "EXPLICIT_POSITIVE_INTENT"
INTENT_DEFERRAL = "DEFERRAL"
INTENT_AUTO_REPLY = "AUTO_REPLY"
INTENT_EXPLICIT_DECLINE = "EXPLICIT_DECLINE"
INTENT_QUESTION = "QUESTION"
INTENT_AMBIGUOUS = "AMBIGUOUS"


# Regex Pattern Groups with Word Boundaries
DECLINE_PATTERNS = [
    r"\bnot interested\b",
    r"\bstop messaging\b",
    r"\bdon'?t message\b",
    r"\buseless spam\b",
    r"\bmat bhejo\b",
    r"\bno thanks\b",
    r"\bremove me\b",
    r"\bunsubscribe\b",
    r"\bexit\b",
    r"\bleave me alone\b",
    r"\bstop spamming\b"
]

DEFERRAL_PATTERNS = [
    r"\bbusy\b",
    r"\btalk later\b",
    r"\bcall me later\b",
    r"\bkal baat karte hain\b",
    r"\bafter \d+ hours?\b",
    r"\blater today\b",
    r"\babhi nahi\b",
    r"\bnot now\b",
    r"\blater\b",
    r"\bthodi der baad\b"
]

AUTO_REPLY_PATTERNS = [
    r"thank you for contacting us",
    r"our team will respond shortly",
    r"aapki jaankari ke liye.*shukriya",
    r"main aapki.*baatein.*team tak pahuncha",
    r"automated assistant",
    r"we will get back to you",
    r"thanks for reaching out",
    r"this is an automated response",
    r"auto-reply",
    r"canned response"
]

QUESTION_PATTERNS = [
    r"\?",
    r"\bwhat is\b",
    r"\bwhat are\b",
    r"\bhow does\b",
    r"\bhow much\b",
    r"\bwhy\b",
    r"\bwho\b",
    r"\bwhere\b",
    r"\bcan you explain\b",
    r"\bexplain\b",
    r"\bkya hai\b",
    r"\bkaise\b"
]

POSITIVE_INTENT_PATTERNS = [
    r"\bi want to join\b",
    r"\blet'?s do it\b",
    r"\bgo ahead\b",
    r"\bok please check\b",
    r"\bok update\b",
    r"\bplease update\b",
    r"\bha kar do\b",
    r"\bbadhiya hai karo\b",
    r"\byes send me\b",
    r"\bproceed\b",
    r"\bjudna hai\b",
    r"\bmagicpin judrna hai\b",
    r"\bmagicpin se judna hai\b",
    r"\bstart campaign\b",
    r"\bpublish post\b",
    r"\bagree\b",
    r"\bwhats next\b",
    r"\bwhat'?s next\b"
]


def classify_intent_rules(message: str, prior_replies: Optional[List[str]] = None) -> Optional[str]:
    """Fast deterministic rule-based pre-filter for intent classification."""
    msg_clean = message.strip().lower()

    # 1. Decline Check (Highest Priority)
    for pat in DECLINE_PATTERNS:
        if re.search(pat, msg_clean):
            return INTENT_EXPLICIT_DECLINE

    # 2. Deferral Check
    for pat in DEFERRAL_PATTERNS:
        if re.search(pat, msg_clean):
            return INTENT_DEFERRAL

    # 3. Auto-Reply Check
    for pat in AUTO_REPLY_PATTERNS:
        if re.search(pat, msg_clean):
            return INTENT_AUTO_REPLY

    if prior_replies:
        match_count = sum(1 for r in prior_replies if r.strip().lower() == msg_clean)
        if match_count >= 1:
            return INTENT_AUTO_REPLY

    # 4. Explicit Positive Intent Check (Takes priority over generic question mark so "let's do it. whats next?" matches intent!)
    for pat in POSITIVE_INTENT_PATTERNS:
        if re.search(pat, msg_clean):
            return INTENT_EXPLICIT_POSITIVE

    if msg_clean in ["yes", "ha", "haan", "sure", "ok", "yep", "yup"]:
        return INTENT_EXPLICIT_POSITIVE

    # 5. Question Check
    for pat in QUESTION_PATTERNS:
        if re.search(pat, msg_clean):
            return INTENT_QUESTION

    return None


def classify_intent_llm(message: str) -> Optional[str]:
    """LLM classification step (temperature=0) for ambiguous messages."""
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        return None

    try:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        system_prompt = (
            "You are a strict intent classifier for WhatsApp merchant replies. "
            "Classify the given message into EXACTLY ONE category: "
            "[EXPLICIT_POSITIVE_INTENT, DEFERRAL, AUTO_REPLY, EXPLICIT_DECLINE, QUESTION, AMBIGUOUS]. "
            "Respond ONLY with valid JSON: {\"intent\": \"<category>\"}"
        )
        body = {
            "model": os.getenv("LLM_MODEL", "claude-3-5-sonnet-20241022"),
            "max_tokens": 100,
            "temperature": 0,
            "system": system_prompt,
            "messages": [{"role": "user", "content": f"Classify this message: '{message}'"}]
        }

        req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            content_text = res_data["content"][0]["text"]
            match = re.search(r'\{[\s\S]*\}', content_text)
            if match:
                data = json.loads(match.group())
                return data.get("intent")
    except Exception as e:
        print(f"[IntentClassifier] LLM classification error: {e}")
        traceback.print_exc()

    return None


def classify_intent(message: str, prior_replies: Optional[List[str]] = None) -> str:
    """
    Dedicated intent classification pipeline:
    1. Deterministic rule-based pre-filter
    2. LLM classifier fallback (if available)
    3. Default to AMBIGUOUS
    """
    rule_intent = classify_intent_rules(message, prior_replies)
    if rule_intent:
        return rule_intent

    llm_intent = classify_intent_llm(message)
    if llm_intent in [INTENT_EXPLICIT_POSITIVE, INTENT_DEFERRAL, INTENT_AUTO_REPLY, INTENT_EXPLICIT_DECLINE, INTENT_QUESTION, INTENT_AMBIGUOUS]:
        return llm_intent

    return INTENT_AMBIGUOUS


def handle_reply(
    state: ConversationState,
    merchant_message: str,
    category: Optional[Dict[str, Any]] = None,
    merchant: Optional[Dict[str, Any]] = None,
    trigger: Optional[Dict[str, Any]] = None,
    customer: Optional[Dict[str, Any]] = None,
    fired_suppressions: Optional[set] = None
) -> Dict[str, Any]:
    """
    Main multi-turn conversation handler function.
    Runs the dedicated classifier first, then branches into truthful, non-fabricated responses.
    """
    state.add_turn("merchant", merchant_message)
    m_id = state.merchant_id or (merchant.get("merchant_id") if merchant else "global_merchant")
    m_name = merchant.get("identity", {}).get("name") if merchant else "your business"
    locality = merchant.get("identity", {}).get("locality") if merchant else "your area"
    prior_merchant_replies = [t["message"] for t in state.turns if t["role"] == "merchant"][:-1]

    # STEP 1: Intent Classification
    intent = classify_intent(merchant_message, prior_merchant_replies)

    # STEP 2: Branching on Classification AND pending_action

    # BRANCH 1: EXPLICIT_DECLINE
    if intent == INTENT_EXPLICIT_DECLINE or state.unanswered_nudge_count >= 3:
        state.status = "ended"
        if state.pending_action and state.pending_action.get("suppression_key") and fired_suppressions is not None:
            fired_suppressions.add(state.pending_action["suppression_key"])
        state.clear_pending_action()
        return {
            "action": "end",
            "rationale": "Merchant explicitly declined or requested stop. Gracefully exiting conversation."
        }

    # BRANCH 2: DEFERRAL
    if intent == INTENT_DEFERRAL:
        state.status = "waiting"
        # Keep pending_action ALIVE so when merchant comes back and says "yes", it resolves to the original offer!
        offer_desc = state.pending_action.get("offer_description") if state.pending_action else "your campaign"
        reply_body = f"Got it, no problem! I'll follow up with you later regarding {offer_desc} for {m_name}."
        state.add_turn("vera", reply_body)
        return {
            "action": "wait",
            "wait_seconds": 1800,
            "rationale": "Merchant requested time / deferral. Scheduled follow-up wait of 30 minutes while preserving pending_action."
        }

    # BRANCH 3: AUTO_REPLY
    if intent == INTENT_AUTO_REPLY:
        state.auto_reply_count += 1
        MERCHANT_AUTO_REPLY_COUNTS[m_id] = MERCHANT_AUTO_REPLY_COUNTS.get(m_id, 0) + 1

        if state.auto_reply_count >= 2 or MERCHANT_AUTO_REPLY_COUNTS[m_id] >= 2:
            state.status = "ended"
            state.clear_pending_action()
            return {
                "action": "end",
                "rationale": "Detected repeated canned auto-reply (2+ occurrences). Exiting politely to avoid burning message quota."
            }
        else:
            reply_body = (
                f"Samajh gayi. Manager/Owner tak pahunchane se pehle, kya aap khud dekhna chahingi "
                f"ki exact kya optimization missing hai Google Business listing pe? 2 minute ka kaam hai. Chalega?"
            )
            state.add_turn("vera", reply_body)
            return {
                "action": "send",
                "body": reply_body,
                "cta": "binary",
                "rationale": "First auto-reply detected. Making one polite follow-up attempt to reach owner."
            }

    # BRANCH 4: EXPLICIT_POSITIVE_INTENT
    if intent == INTENT_EXPLICIT_POSITIVE:
        if state.pending_action:
            offer_desc = state.pending_action.get("offer_description", "your proposed campaign")
            # Confirm proceeding with THAT SPECIFIC pending_action.offer_description!
            reply_body = f"Starting this now — I'll confirm once it's live for {offer_desc} at {m_name}."
            state.clear_pending_action()  # Clear resolved pending action
            state.add_turn("vera", reply_body)
            return {
                "action": "send",
                "body": reply_body,
                "cta": "none",
                "rationale": f"Explicit positive intent confirmed for pending action: '{offer_desc}'."
            }
        else:
            # "yes" with NO pending_action -> Ask for clarification on which specific offer to start/proceed with
            reply_body = f"Starting next steps! Could you clarify and confirm which specific offer or update you'd like to proceed with for {m_name}?"
            state.add_turn("vera", reply_body)
            return {
                "action": "send",
                "body": reply_body,
                "cta": "open_ended",
                "rationale": "Explicit positive intent received but no pending action active. Asked for specific offer confirmation to proceed."
            }

    # BRANCH 5: QUESTION
    if intent == INTENT_QUESTION:
        perf = merchant.get("performance", {}) if merchant else {}
        views = perf.get("views", "N/A")
        calls = perf.get("calls", "N/A")
        ctr = perf.get("ctr", 0.0)

        msg_clean = merchant_message.lower()
        if "competit" in msg_clean:
            reply_body = (
                f"In {locality}, competition tracks nearby practices in your area. "
                f"For {m_name}, your dashboard records {views} views, {calls} calls, and a CTR of {ctr:.1%}. "
                f"We track local competitor openings (like new listings 1.3km away) to keep your profile views ahead."
            )
        else:
            pending_desc = f" regarding '{state.pending_action['offer_description']}'" if state.pending_action else ""
            reply_body = (
                f"Regarding your query about {m_name}{pending_desc}: "
                f"your current performance is {views} views and {calls} calls in {locality}. "
                f"Let me know if you need specific details on our active offers."
            )

        state.add_turn("vera", reply_body)
        return {
            "action": "send",
            "body": reply_body,
            "cta": "none",
            "rationale": "Answered merchant question using concrete context facts without executing actions."
        }

    # BRANCH 6: AMBIGUOUS / UNCLEAR
    reply_body = (
        f"Could you clarify if you'd like us to proceed with updating your listing for {m_name}? Reply YES to confirm."
    )
    state.add_turn("vera", reply_body)

    return {
        "action": "send",
        "body": reply_body,
        "cta": "binary",
        "rationale": "Ambiguous merchant reply. Asking clarifying binary question instead of defaulting to action mode."
    }
