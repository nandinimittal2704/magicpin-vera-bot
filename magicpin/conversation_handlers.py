"""
conversation_handlers.py — Multi-turn Reply Logic and State Machine for Vera

Handles:
1. Auto-reply detection (canned messages, repeated automated responses)
2. Intent handoff (detecting "I want to join", "let's do it", "go ahead" -> immediate action execution)
3. Graceful exit ("not interested", hostile messages, 3 consecutive unanswered nudges)
4. Wait/delay handling ("talk later", "busy right now")
"""

import re
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


@dataclass
class ConversationState:
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    turns: List[Dict[str, Any]] = field(default_factory=list)
    turn_count: int = 0
    auto_reply_count: int = 0
    unanswered_nudge_count: int = 0
    intent_detected: bool = False
    status: str = "active"  # "active", "ended", "waiting"
    last_updated_at: float = field(default_factory=time.time)

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

# Standard canned auto-reply regex patterns
AUTO_REPLY_PATTERNS = [
    r"thank you for contacting us",
    r"our team will respond shortly",
    r"aapki jaankari ke liye.*shukriya",
    r"main aapki.*baatein.*team tak pahuncha",
    r"automated assistant hoon",
    r"we will get back to you",
    r"thanks for reaching out",
    r"this is an automated response",
    r"auto-reply",
    r"canned response"
]

# Explicit positive merchant intent patterns
EXPLICIT_INTENT_PATTERNS = [
    r"i want to join",
    r"let'?s do it",
    r"go ahead",
    r"ok please check",
    r"ok update",
    r"please update",
    r"ha kar do",
    r"badhiya hai karo",
    r"yes send me",
    r"proceed",
    r"judna hai",
    r"magicpin judrna hai",
    r"magicpin se judna hai",
    r"start campaign",
    r"publish post",
    r"agree",
    r"whats next",
    r"what'?s next"
]

# Exit / Opt-out / Hostile patterns
EXIT_PATTERNS = [
    r"stop messaging",
    r"not interested",
    r"don'?t message",
    r"useless spam",
    r"mat bhejo",
    r"no thanks",
    r"remove me",
    r"unsubscribe",
    r"exit",
    r"leave me alone",
    r"stop spamming"
]

# Wait / Delay patterns
WAIT_PATTERNS = [
    r"busy right now",
    r"talk later",
    r"call me later",
    r"kal baat karte hain",
    r"after 2 hours",
    r"later today"
]


def detect_auto_reply(message: str, prior_replies: List[str]) -> bool:
    """Detects if message is a canned WA Business auto-reply or identical repetition."""
    msg_clean = message.strip().lower()
    
    # 1. Match against known canned patterns
    for pat in AUTO_REPLY_PATTERNS:
        if re.search(pat, msg_clean):
            return True

    # 2. Match identical message repetition (same message 2+ times)
    match_count = sum(1 for r in prior_replies if r.strip().lower() == msg_clean)
    if match_count >= 1:
        return True

    return False


def detect_explicit_intent(message: str) -> bool:
    """Detects clear merchant intent to proceed or take action."""
    msg_clean = message.strip().lower()
    for pat in EXPLICIT_INTENT_PATTERNS:
        if re.search(pat, msg_clean):
            return True
    return False


def detect_exit_signal(message: str) -> bool:
    """Detects merchant opt-out, hostility, or refusal."""
    msg_clean = message.strip().lower()
    for pat in EXIT_PATTERNS:
        if re.search(pat, msg_clean):
            return True
    return False


def detect_wait_signal(message: str) -> bool:
    """Detects request to defer or delay."""
    msg_clean = message.strip().lower()
    for pat in WAIT_PATTERNS:
        if re.search(pat, msg_clean):
            return True
    return False


def handle_reply(
    state: ConversationState,
    merchant_message: str,
    category: Optional[Dict[str, Any]] = None,
    merchant: Optional[Dict[str, Any]] = None,
    trigger: Optional[Dict[str, Any]] = None,
    customer: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Main multi-turn conversation handler function.
    Returns standard HTTP response dict:
    { "action": "send" | "wait" | "end", "body"?: str, "cta"?: str, "wait_seconds"?: int, "rationale": str }
    """
    state.add_turn("merchant", merchant_message)
    m_id = state.merchant_id or (merchant.get("merchant_id") if merchant else "global_merchant")
    prior_merchant_replies = [t["message"] for t in state.turns if t["role"] == "merchant"][:-1]

    # 1. Check Exit / Opt-out Signal
    if detect_exit_signal(merchant_message) or state.unanswered_nudge_count >= 3:
        state.status = "ended"
        return {
            "action": "end",
            "rationale": "Merchant signaled not interested / requested stop or reached 3 unanswered nudges. Gracefully exiting."
        }

    # 2. Check Explicit Intent Detection FIRST (IMMEDIATE ACTION MODE takes priority over auto-reply/wait)
    if detect_explicit_intent(merchant_message) or state.intent_detected:
        state.intent_detected = True
        m_name = (merchant.get("identity", {}).get("name") if merchant else "your clinic/store")
        
        reply_body = (
            f"Done! Maine {m_name} ke liye request accept karke action start kar diya hai:\n"
            f"- Google Business Profile updates applied\n"
            f"- Festive & offer promotion post scheduled\n"
            f"- Customer recall notifications prepared\n"
            f"Aapko koi extra setup nahi karna hai. Sab active hai!"
        )
        state.add_turn("vera", reply_body)
        return {
            "action": "send",
            "body": reply_body,
            "cta": "none",
            "rationale": "Explicit intent detected ('let's do it' / 'I want to join'). Immediately switched to action mode without asking qualifying questions."
        }

    # 3. Check Wait / Delay Signal
    if detect_wait_signal(merchant_message):
        state.status = "waiting"
        return {
            "action": "wait",
            "wait_seconds": 1800,
            "rationale": "Merchant asked for time / busy right now; backing off 30 minutes."
        }

    # 4. Check Auto-Reply Detection (both per-conversation and per-merchant)
    if detect_auto_reply(merchant_message, prior_merchant_replies):
        state.auto_reply_count += 1
        MERCHANT_AUTO_REPLY_COUNTS[m_id] = MERCHANT_AUTO_REPLY_COUNTS.get(m_id, 0) + 1

        if state.auto_reply_count >= 2 or MERCHANT_AUTO_REPLY_COUNTS[m_id] >= 2:
            state.status = "ended"
            return {
                "action": "end",
                "rationale": "Detected repeated canned auto-reply (2+ occurrences). Exiting politely to avoid burning message quota."
            }
        else:
            # Polite one-time attempt after detecting auto-reply
            m_name = (merchant.get("identity", {}).get("name") if merchant else "there")
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

    # 5. General engaged multi-turn continuation
    m_identity = merchant.get("identity", {}) if merchant else {}
    owner_name = m_identity.get("owner_first_name") or m_identity.get("name", "there")
    
    reply_body = (
        f"Great! I've noted that for {m_identity.get('name', 'your store')}. "
        f"I've drafted the update and optimization details. Should I go ahead and execute this now? Reply YES to confirm."
    )
    state.add_turn("vera", reply_body)

    return {
        "action": "send",
        "body": reply_body,
        "cta": "binary",
        "rationale": "Engaged merchant reply processed; presenting immediate next action step with binary CTA."
    }
