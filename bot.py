"""
bot.py — Vera WhatsApp Merchant Assistant FastAPI Service

Exposes 5 required HTTP endpoints (plus /v1/teardown & /ui WhatsApp simulator):
1. POST /v1/context — receive context pushes
2. POST /v1/tick    — periodic wake-up & proactive composition
3. POST /v1/reply   — multi-turn synchronous reply handler
4. GET  /v1/healthz — liveness & context load probe
5. GET  /v1/metadata — team & bot metadata
6. POST /v1/teardown — wipe state at test end
7. GET  /ui         — interactive WhatsApp Web Simulator UI
"""

import time
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field

from composer import compose
from conversation_handlers import ConversationState, handle_reply, MERCHANT_AUTO_REPLY_COUNTS

app = FastAPI(
    title="Vera WhatsApp Merchant Assistant API",
    description="magicpin Vera AI Challenge Service",
    version="1.0.0"
)

START_TIME = time.time()

# In-memory context store: (scope, context_id) -> {"version": int, "payload": dict, "delivered_at": str}
CONTEXT_STORE: Dict[tuple[str, str], Dict[str, Any]] = {}

# In-memory conversation trackers: conversation_id -> ConversationState
CONVERSATION_STORE: Dict[str, ConversationState] = {}

# Set of fired suppression keys to prevent spamming
FIRED_SUPPRESSIONS: set[str] = set()


@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "Vera WhatsApp Merchant Assistant API is running!",
        "endpoints": {
            "web_ui_simulator": "/ui",
            "healthz": "/v1/healthz",
            "metadata": "/v1/metadata",
            "interactive_docs": "/docs"
        }
    }


@app.get("/ui", response_class=HTMLResponse)
async def whatsapp_ui_simulator():
    """Interactive WhatsApp Web Simulator UI for testing Vera live in browser."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vera — WhatsApp Assistant Simulator</title>
    <link href="https://fonts.googleapis.com/css2?family=Segoe+UI:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        body { background-color: #0c1317; color: #e9edef; display: flex; height: 100vh; overflow: hidden; }
        
        /* Sidebar */
        .sidebar { width: 360px; background-color: #111b21; border-right: 1px solid #222d34; display: flex; flex-direction: column; padding: 15px; }
        .sidebar h2 { color: #00a884; font-size: 18px; margin-bottom: 12px; display: flex; align-items: center; gap: 8px; }
        .card { background-color: #202c33; padding: 12px; border-radius: 8px; margin-bottom: 12px; border: 1px solid #2a3942; }
        .card label { font-size: 12px; color: #8696a0; text-transform: uppercase; font-weight: 600; display: block; margin-bottom: 4px; }
        select, button, input { width: 100%; padding: 10px; border-radius: 6px; border: 1px solid #2a3942; background-color: #111b21; color: #e9edef; font-size: 14px; margin-top: 4px; }
        button { background-color: #00a884; color: #111b21; font-weight: 700; cursor: pointer; border: none; transition: 0.2s; }
        button:hover { background-color: #029071; }
        .badge { background-color: #2a3942; color: #00a884; padding: 2px 8px; border-radius: 12px; font-size: 11px; }

        /* Main Chat Area */
        .chat-container { flex: 1; display: flex; flex-direction: column; background-image: radial-gradient(#1f2c34 1px, transparent 0); background-size: 20px 20px; background-color: #0b141a; }
        .chat-header { background-color: #202c33; padding: 12px 20px; border-bottom: 1px solid #222d34; display: flex; align-items: center; justify-content: space-between; }
        .chat-header h3 { font-size: 16px; font-weight: 600; }
        .chat-header span { font-size: 12px; color: #8696a0; }

        .messages-list { flex: 1; padding: 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 12px; }
        .msg { max-width: 70%; padding: 10px 14px; border-radius: 8px; font-size: 14px; line-height: 1.4; position: relative; word-wrap: break-word; }
        .msg-vera { background-color: #202c33; align-self: flex-start; border-top-left-radius: 0; border: 1px solid #2a3942; }
        .msg-merchant { background-color: #005c4b; align-self: flex-end; border-top-right-radius: 0; color: #e9edef; }
        .msg-meta { font-size: 10px; color: #8696a0; margin-top: 4px; display: flex; justify-content: space-between; gap: 10px; }
        .rationale-tag { background-color: #111b21; color: #8696a0; padding: 6px; border-radius: 4px; font-size: 11px; margin-top: 6px; border-left: 3px solid #00a884; }

        /* Quick Action Bar */
        .quick-actions { padding: 8px 15px; background-color: #111b21; display: flex; gap: 8px; border-top: 1px solid #222d34; flex-wrap: wrap; }
        .chip { background-color: #202c33; color: #e9edef; padding: 6px 12px; border-radius: 16px; font-size: 12px; cursor: pointer; border: 1px solid #2a3942; }
        .chip:hover { background-color: #00a884; color: #111b21; }

        /* Input Bar */
        .chat-input { background-color: #202c33; padding: 10px 15px; display: flex; gap: 10px; border-top: 1px solid #222d34; }
        .chat-input input { flex: 1; border-radius: 8px; padding: 12px; }
        .chat-input button { width: auto; padding: 0 20px; }
    </style>
</head>
<body>
    <div class="sidebar">
        <h2>🟢 Vera Assistant</h2>
        
        <div class="card">
            <label>Service Status</label>
            <div id="status-display">Checking...</div>
        </div>

        <div class="card">
            <label>1. Select Merchant</label>
            <select id="merchant-select">
                <option value="m_001_drmeera_dentist_delhi">Dr. Meera Dental Clinic (Dentist - Delhi)</option>
                <option value="m_003_studio11_salon_hyderabad">Studio11 Salon (Salon - Hyderabad)</option>
                <option value="m_006_southindiancafe_restaurant_bangalore">South Indian Cafe (Restaurant - Bangalore)</option>
                <option value="m_009_apollo_pharmacy_jaipur">Apollo Pharmacy (Pharmacy - Jaipur)</option>
            </select>
        </div>

        <div class="card">
            <label>2. Trigger Event</label>
            <select id="trigger-select">
                <option value="trg_022_cde_webinar_dentists">Clinical Digest / Research Study</option>
                <option value="trg_023_competitor_opened_dentist">Competitor Opened 1.3km Away</option>
                <option value="trg_008_curious_ask_studio11">Weekly Curiosity Ask & Peer Stat</option>
                <option value="trg_013_corporate_thali_planning">Festive / Demand Surge Event</option>
                <option value="trg_076_appointment_tomorrow_m_019_karim_salon_lu">Customer Recall Due (₹299 cleaning)</option>
            </select>
        </div>

        <button onclick="triggerProactiveNudge()">⚡ Send Proactive Nudge (/v1/tick)</button>
        <button onclick="teardownState()" style="background-color: #8696a0; margin-top: 8px;">🔄 Reset Conversation State</button>
    </div>

    <div class="chat-container">
        <div class="chat-header">
            <div>
                <h3 id="active-merchant-name">Dr. Meera's Dental Clinic</h3>
                <span>WhatsApp Business Direct Messaging</span>
            </div>
            <span class="badge" id="conv-status">Active</span>
        </div>

        <div class="messages-list" id="messages-list">
            <div class="msg msg-vera">
                <strong>Vera Assistant</strong><br>
                Welcome! Select a merchant and click "Send Proactive Nudge" to test Vera's context-anchored composition.
                <div class="msg-meta"><span>System</span></div>
            </div>
        </div>

        <div class="quick-actions">
            <span class="chip" onclick="sendQuickReply('Mujhe magicpin judrna hai. Let\'s do it.')">🚀 Test Intent ("Let's do it")</span>
            <span class="chip" onclick="sendQuickReply('Thank you for contacting us! Our team will respond shortly.')">🤖 Test Auto-Reply Canned Msg</span>
            <span class="chip" onclick="sendQuickReply('Stop messaging me. This is useless spam.')">🛑 Test Opt-Out ("Stop")</span>
            <span class="chip" onclick="sendQuickReply('Busy right now, talk later.')">⏳ Test Delay ("Talk later")</span>
        </div>

        <div class="chat-input">
            <input type="text" id="user-input" placeholder="Type a message as the merchant/customer..." onkeypress="handleKeyPress(event)">
            <button onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        let currentConvId = "conv_demo_1";
        let currentMerchantId = "m_001_drmeera_dentist_delhi";

        async function checkHealth() {
            try {
                const res = await fetch('/v1/healthz');
                const data = await res.json();
                document.getElementById('status-display').innerHTML = `
                    <span style="color:#00a884">● Live</span> | Contexts: ${data.contexts_loaded.category} Cat, ${data.contexts_loaded.merchant} Mx
                `;
            } catch (e) {
                document.getElementById('status-display').innerText = "Offline";
            }
        }

        async function triggerProactiveNudge() {
            const mSelect = document.getElementById('merchant-select');
            currentMerchantId = mSelect.value;
            const tSelect = document.getElementById('trigger-select');
            const trgId = tSelect.value;
            currentConvId = `conv_${currentMerchantId}_${trgId}`;

            document.getElementById('active-merchant-name').innerText = mSelect.options[mSelect.selectedIndex].text;

            // Push dummy context first if needed
            const tickRes = await fetch('/v1/tick', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    now: new Date().toISOString(),
                    available_triggers: [trgId]
                })
            });

            const data = await tickRes.json();
            if (data.actions && data.actions.length > 0) {
                const action = data.actions[0];
                appendVeraMessage(action.body, action.cta, action.rationale);
            } else {
                appendVeraMessage("No proactive message sent (Restraint / Suppressed).", "none", "Bot exercised restraint.");
            }
        }

        async function sendMessage() {
            const input = document.getElementById('user-input');
            const text = input.value.trim();
            if (!text) return;

            appendUserMessage(text);
            input.value = '';

            const res = await fetch('/v1/reply', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    conversation_id: currentConvId,
                    merchant_id: currentMerchantId,
                    from_role: 'merchant',
                    message: text,
                    turn_number: 2
                })
            });

            const data = await res.json();
            if (data.action === 'send') {
                appendVeraMessage(data.body, data.cta || 'binary', data.rationale);
            } else if (data.action === 'end') {
                appendVeraMessage("Conversation ended gracefully: " + (data.rationale || ""), "none", data.rationale);
                document.getElementById('conv-status').innerText = "Ended";
            } else if (data.action === 'wait') {
                appendVeraMessage(`Waiting ${data.wait_seconds || 1800}s as requested.`, "none", data.rationale);
            }
        }

        function sendQuickReply(text) {
            currentConvId = "conv_test_" + Date.now();
            document.getElementById('user-input').value = text;
            sendMessage();
        }

        function handleKeyPress(e) {
            if (e.key === 'Enter') sendMessage();
        }

        function appendVeraMessage(body, cta, rationale) {
            const list = document.getElementById('messages-list');
            const div = document.createElement('div');
            div.className = 'msg msg-vera';
            div.innerHTML = `
                <strong>Vera Assistant</strong> <span class="badge">${cta} CTA</span><br>
                ${body.replace(/\\n/g, '<br>')}
                ${rationale ? `<div class="rationale-tag">💡 <strong>Rationale:</strong> ${rationale}</div>` : ''}
                <div class="msg-meta"><span>${new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</span></div>
            `;
            list.appendChild(div);
            list.scrollTop = list.scrollHeight;
        }

        function appendUserMessage(text) {
            const list = document.getElementById('messages-list');
            const div = document.createElement('div');
            div.className = 'msg msg-merchant';
            div.innerHTML = `
                ${text.replace(/\\n/g, '<br>')}
                <div class="msg-meta"><span>${new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</span></div>
            `;
            list.appendChild(div);
            list.scrollTop = list.scrollHeight;
        }

        async function teardownState() {
            await fetch('/v1/teardown', {method: 'POST'});
            document.getElementById('messages-list').innerHTML = `
                <div class="msg msg-vera">
                    <strong>Vera Assistant</strong><br>
                    State reset complete. Select a merchant and trigger to start.
                </div>
            `;
            document.getElementById('conv-status').innerText = "Active";
        }

        checkHealth();
    </script>
</body>
</html>"""
    return html_content


# Pydantic Schemas
class ContextPushRequest(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: Dict[str, Any]
    delivered_at: Optional[str] = None


class TickRequest(BaseModel):
    now: str
    available_triggers: List[str] = Field(default_factory=list)


class ReplyRequest(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str = "merchant"
    message: str
    received_at: Optional[str] = None
    turn_number: Optional[int] = 1


# Helper context resolution functions
def get_context(scope: str, context_id: str) -> Optional[Dict[str, Any]]:
    entry = CONTEXT_STORE.get((scope, context_id))
    return entry["payload"] if entry else None


def find_merchant_by_id(merchant_id: str) -> Optional[Dict[str, Any]]:
    if not merchant_id:
        return None
    # Check exact ID first
    m = get_context("merchant", merchant_id)
    if m:
        return m
    # Fallback search by payload merchant_id
    for (s, _), entry in CONTEXT_STORE.items():
        if s == "merchant" and entry["payload"].get("merchant_id") == merchant_id:
            return entry["payload"]
    return None


def find_category_by_slug(slug: str) -> Optional[Dict[str, Any]]:
    if not slug:
        return None
    c = get_context("category", slug)
    if c:
        return c
    for (s, _), entry in CONTEXT_STORE.items():
        if s == "category" and entry["payload"].get("slug") == slug:
            return entry["payload"]
    return None


def find_customer_by_id(customer_id: str) -> Optional[Dict[str, Any]]:
    if not customer_id:
        return None
    cu = get_context("customer", customer_id)
    if cu:
        return cu
    for (s, _), entry in CONTEXT_STORE.items():
        if s == "customer" and entry["payload"].get("customer_id") == customer_id:
            return entry["payload"]
    return None


# 1. POST /v1/context
@app.post("/v1/context")
async def receive_context(req: ContextPushRequest):
    allowed_scopes = ["category", "merchant", "customer", "trigger"]
    if req.scope not in allowed_scopes:
        return JSONResponse(
            status_code=400,
            content={"accepted": False, "reason": "invalid_scope", "details": f"Scope must be one of {allowed_scopes}"}
        )

    key = (req.scope, req.context_id)
    existing = CONTEXT_STORE.get(key)
    if existing and existing["version"] > req.version:
        return JSONResponse(
            status_code=409,
            content={"accepted": False, "reason": "stale_version", "current_version": existing["version"]}
        )

    CONTEXT_STORE[key] = {
        "version": req.version,
        "payload": req.payload,
        "delivered_at": req.delivered_at or datetime.utcnow().isoformat() + "Z"
    }

    return {
        "accepted": True,
        "ack_id": f"ack_{req.context_id}_v{req.version}",
        "stored_at": datetime.utcnow().isoformat() + "Z"
    }


# 2. POST /v1/tick
@app.post("/v1/tick")
async def periodic_tick(req: TickRequest):
    actions = []
    seen_merchant_convs: set[tuple[str, str]] = set()

    for trg_id in req.available_triggers:
        # Retrieve trigger context
        trg_payload = get_context("trigger", trg_id)
        if not trg_payload:
            # Check fallback in store
            for (s, cid), entry in CONTEXT_STORE.items():
                if s == "trigger" and (cid == trg_id or entry["payload"].get("id") == trg_id):
                    trg_payload = entry["payload"]
                    break
        if not trg_payload:
            continue

        # Check suppression key
        suppression_key = trg_payload.get("suppression_key") or f"{trg_payload.get('kind')}:{trg_id}"
        if suppression_key in FIRED_SUPPRESSIONS:
            continue

        # Check expiration
        expires_at = trg_payload.get("expires_at")
        if expires_at and req.now > expires_at:
            continue

        # Resolve merchant
        merchant_id = trg_payload.get("merchant_id") or trg_payload.get("payload", {}).get("merchant_id")
        merchant = find_merchant_by_id(merchant_id) if merchant_id else None
        if not merchant:
            # If only 1 merchant in store, fallback to it for test stability
            merchants_in_store = [e["payload"] for (s, _), e in CONTEXT_STORE.items() if s == "merchant"]
            if len(merchants_in_store) == 1:
                merchant = merchants_in_store[0]
                merchant_id = merchant.get("merchant_id")

        if not merchant:
            continue

        # Resolve category
        cat_slug = merchant.get("category_slug") or trg_payload.get("payload", {}).get("category")
        category = find_category_by_slug(cat_slug) if cat_slug else None
        if not category:
            cats_in_store = [e["payload"] for (s, _), e in CONTEXT_STORE.items() if s == "category"]
            if cats_in_store:
                category = cats_in_store[0]

        if not category:
            continue

        # Resolve customer
        customer_id = trg_payload.get("customer_id") or trg_payload.get("payload", {}).get("customer_id")
        customer = find_customer_by_id(customer_id) if customer_id else None

        # Build unique conversation_id for tick send
        conv_id = f"conv_{merchant_id}_{trg_id}"
        
        # Enforce max 1 action per (merchant_id, conversation_id) per tick
        m_conv_pair = (merchant_id, conv_id)
        if m_conv_pair in seen_merchant_convs:
            continue
        seen_merchant_convs.add(m_conv_pair)

        # Retrieve conversation state history if exists
        state = CONVERSATION_STORE.get(conv_id)
        history = state.turns if state else None

        # Compose message
        composed = compose(category, merchant, trg_payload, customer, history)
        
        # Track action
        action_obj = {
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": composed.get("send_as", "vera"),
            "trigger_id": trg_id,
            "template_name": f"vera_{trg_payload.get('kind', 'generic')}_v1",
            "template_params": [merchant.get("identity", {}).get("name", ""), "..."],
            "body": composed.get("body", ""),
            "cta": composed.get("cta", "binary"),
            "suppression_key": composed.get("suppression_key", suppression_key),
            "rationale": composed.get("rationale", "")
        }

        actions.append(action_obj)
        FIRED_SUPPRESSIONS.add(suppression_key)

        # Initialize conversation state tracker
        if conv_id not in CONVERSATION_STORE:
            state = ConversationState(conversation_id=conv_id, merchant_id=merchant_id, customer_id=customer_id)
            CONVERSATION_STORE[conv_id] = state
        else:
            state = CONVERSATION_STORE[conv_id]

        # Store pending_action on state if CTA is not 'none'
        if composed.get("cta") != "none":
            offer_desc = trg_payload.get("payload", {}).get("headline") or trg_payload.get("payload", {}).get("event") or trg_payload.get("kind") or "promotional update"
            state.set_pending_action(
                trigger_id=trg_id,
                offer_description=offer_desc,
                cta_type=composed.get("cta", "binary"),
                suppression_key=composed.get("suppression_key", suppression_key)
            )

        state.add_turn(composed.get("send_as", "vera"), composed.get("body", ""))

    return {"actions": actions}


# 3. POST /v1/reply
@app.post("/v1/reply")
async def handle_merchant_reply(req: ReplyRequest):
    conv_id = req.conversation_id
    state = CONVERSATION_STORE.get(conv_id)
    if not state:
        state = ConversationState(
            conversation_id=conv_id,
            merchant_id=req.merchant_id,
            customer_id=req.customer_id
        )
        CONVERSATION_STORE[conv_id] = state

    # Lookup context for conversation
    merchant = find_merchant_by_id(req.merchant_id or state.merchant_id or "")
    category = find_category_by_slug(merchant.get("category_slug", "") if merchant else "")
    customer = find_customer_by_id(req.customer_id or state.customer_id or "")

    # Execute multi-turn reply handler logic
    response_action = handle_reply(
        state=state,
        merchant_message=req.message,
        category=category,
        merchant=merchant,
        trigger=None,
        customer=customer,
        fired_suppressions=FIRED_SUPPRESSIONS
    )

    return response_action


# 4. GET /v1/healthz
@app.get("/v1/healthz")
async def health_check():
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _), _ in CONTEXT_STORE.items():
        if scope in counts:
            counts[scope] += 1
    
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": counts
    }


# 5. GET /v1/metadata
@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Team Vera AI",
        "team_members": ["Vera Developer"],
        "model": "claude-3-5-sonnet-20241022",
        "approach": "4-Context LLM Composer with Routing, Post-Validation, & Multi-Turn State Machine",
        "contact_email": "vera@magicpin.in",
        "version": "1.0.0",
        "submitted_at": datetime.utcnow().isoformat() + "Z"
    }


# 6. POST /v1/teardown
@app.post("/v1/teardown")
async def teardown_state():
    CONTEXT_STORE.clear()
    CONVERSATION_STORE.clear()
    FIRED_SUPPRESSIONS.clear()
    MERCHANT_AUTO_REPLY_COUNTS.clear()
    return {"status": "ok", "message": "All state teardown complete"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
