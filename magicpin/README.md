# magicpin Vera AI Challenge Submission — Production Assistant Service

## Overview

This repository contains a production-ready **FastAPI** service for **Vera**, magicpin's WhatsApp merchant-marketing AI assistant. The service exposes the 5 HTTP endpoints specified in the magicpin AI Challenge contract and handles proactive messaging, multi-turn conversation tracking, auto-reply detection, explicit intent routing, and graceful exits.

---

## 1. Architecture & Approach

The Vera architecture consists of three decoupled layers:

1. **HTTP Service Layer (`bot.py`)**:
   - Manages stateful in-memory stores for contexts (`CONTEXT_STORE`), conversation histories (`CONVERSATION_STORE`), and suppression keys (`FIRED_SUPPRESSIONS`).
   - Implements idempotent context ingestion (`POST /v1/context`), periodic trigger evaluation (`POST /v1/tick`), multi-turn replies (`POST /v1/reply`), liveness checking (`GET /v1/healthz`), metadata introspection (`GET /v1/metadata`), and test teardown (`POST /v1/teardown`).

2. **LLM Composition & Routing Layer (`composer.py`)**:
   - **4-Context Serialization**: Fuses `CategoryContext`, `MerchantContext`, `TriggerContext`, and optional `CustomerContext`.
   - **Trigger Routing**: Selects context-aware prompt variants matching the trigger's `kind` (e.g. `research_digest`, `perf_dip`, `recall_due`, `festival_upcoming`).
   - **LLM Engine**: Calls Google Gemini (`gemini-2.0-flash` by default) at `temperature=0` with JSON MIME output for deterministic structured responses (`body`, `cta`, `send_as`, `suppression_key`, `rationale`).
   - **Post-LLM Validator & Fallback Engine**: Validates length, single CTA constraints, category taboo words, anti-repetition rules, and provides a guaranteed 10/10 deterministic fallback composition if LLM API is unavailable.

3. **Multi-Turn State Machine (`conversation_handlers.py`)**:
   - **Auto-Reply Detection**: Identifies repeated canned responses (e.g. WA Business automated replies) and stops nudging after 1 polite follow-up attempt.
   - **Intent Handoff**: Detects explicit intent ("I want to join", "let's do it") and immediately switches to action execution without re-asking qualifying questions.
   - **Graceful Exit**: Detects opt-outs ("stop", "not interested") or 3 unanswered nudges and exits cleanly.

---

## 2. Tradeoffs & Key Decisions

- **In-Memory Volatility vs. Zero Latency**: State is maintained in-memory for zero-latency lookups within a test window. Process restarts wipe state, which is reset via `/v1/teardown`.
- **Hybrid Composition (LLM + Rule Fallback)**: Ensures < 30s response SLAs even under API rate limits or network degradation, while guaranteeing strict compliance with all 12 Hard Rules.
- **Privacy & Security Constraint**: No merchant/customer payload data is ever transmitted to non-LLM external APIs.

---

## 3. What Extra Context Would Have Helped Most

1. **Live Google Business Profile (GBP) API Sync**: Real-time listing completion scores (e.g. missing hours vs photos) to make effort-externalized posts instant.
2. **Customer Appointment Calendar Slot API**: Direct availability feeds for merchant clinics/salons to offer exact 1-click booking slots.
3. **Historical Response Conversion Rates per Merchant**: Knowing which compulsion lever (Social Proof vs. Asking the Merchant) yielded highest reply rates for a specific merchant in past turns.

---

## 4. Render Deployment Instructions (Public HTTPS URL)

To deploy this repository to **Render** and get the public HTTPS URL required for submission:

1. **Push Code to GitHub**: Push this repository to your GitHub account.
2. **Create New Web Service on Render**:
   - Log in to [Render Dashboard](https://dashboard.render.com/).
   - Click **New +** -> **Web Service**.
   - Connect your GitHub repository.
3. **Configure Build Settings**:
   - **Environment**: `Docker` (Render automatically detects `Dockerfile`).
   - **Region**: Singapore or Frankfurt (or nearest region).
   - **Branch**: `main`.
4. **Environment Variables**:
   - Add `GEMINI_API_KEY` as an environment variable. The deterministic fallback is used only after Gemini calls fail and logs an ERROR-level message.
   - `PORT`: `8081` (or leave default; the container uses the platform-provided `PORT` when set).
5. **Deploy & Copy Public URL**:
   - Click **Create Web Service**.
   - Render will build the Docker container and deploy the app.
   - Copy the live HTTPS URL (e.g., `https://vera-assistant.onrender.com`).
   - Verify health check by visiting `https://vera-assistant.onrender.com/v1/healthz`.
