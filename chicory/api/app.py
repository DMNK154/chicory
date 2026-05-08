"""FastAPI REST API for the Chicory web demo."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

from chicory.api.sessions import (
    MAX_FILE_SIZE_BYTES,
    NETWORK_DIR,
    Session,
    SessionManager,
)
from chicory.api.user_db import UserDB
from chicory.ingest.ingestor import ingest_file
from chicory.orchestrator.tool_handlers import dispatch_tool_call

logger = logging.getLogger(__name__)

_session_manager: SessionManager | None = None
_user_db: UserDB | None = None

# Load .env early so CHICORY_PASSWORD is available at import time
try:
    from dotenv import load_dotenv as _early_dotenv
    _early_dotenv()
except ImportError:
    pass

# Shared password from env (required)
CHICORY_PASSWORD = os.environ.get("CHICORY_PASSWORD", "")

# Paths that don't require auth
_PUBLIC_PATHS = frozenset({"/", "/api/auth", "/static"})


class PasswordAuthMiddleware(BaseHTTPMiddleware):
    """Gate all /api routes behind a shared password cookie."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        # Allow public paths and static files
        if path in _PUBLIC_PATHS or path.startswith("/static"):
            return await call_next(request)
        # If no password configured, allow everything
        if not CHICORY_PASSWORD:
            return await call_next(request)
        # Check cookie (strip quotes — RFC 6265 may quote special chars like @)
        token = request.cookies.get("chicory_auth", "").strip('"')
        if token != CHICORY_PASSWORD:
            return JSONResponse(status_code=401, content={"detail": "Not authenticated"})
        return await call_next(request)


# ── Lifespan ─────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup: eagerly load embedding model, start session cleanup.
    Shutdown: close all sessions."""
    global _session_manager, _user_db

    # Load .env file so Stripe keys etc. are available via os.environ
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # Prevent sentence-transformers HF Hub HTTP requests (deadlock prevention).
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    # Eagerly load embedding model in main thread to avoid torch/asyncio
    # deadlock. Uses a throwaway in-memory DB — the model weights are cached
    # globally by sentence-transformers so all future sessions benefit.
    try:
        from chicory.config import load_config as _load_config
        from chicory.db.engine import DatabaseEngine
        from chicory.db.schema import apply_schema
        from chicory.layer1.embedding_engine import EmbeddingEngine

        warmup_config = _load_config(db_path=Path(":memory:"))
        warmup_db = DatabaseEngine(warmup_config)
        warmup_db.connect()
        apply_schema(warmup_db)
        warmup_engine = EmbeddingEngine(warmup_config, warmup_db)

        t0 = time.time()
        warmup_engine._load_model()
        logger.info("Embedding model loaded in %.1fs", time.time() - t0)

        t1 = time.time()
        warmup_engine.search_similar(warmup_engine.embed("warmup"), top_k=1)
        logger.info("FAISS warmup in %.1fs", time.time() - t1)

        warmup_db.close()
    except Exception:
        logger.exception("Eager embedding load failed (sessions will lazy-load)")

    _user_db = UserDB()
    _user_db.connect()

    _session_manager = SessionManager()
    _session_manager.start_cleanup_loop()
    logger.info("Chicory API started, session cleanup running")

    yield

    _session_manager.stop()
    _user_db.close()
    logger.info("Chicory API shut down, all sessions closed")


# ── App ──────────────────────────────────────────────────────────────


app = FastAPI(
    title="Chicory API",
    description="REST API for the Chicory memory system web demo",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(PasswordAuthMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_STATIC_DIR = Path(__file__).parent / "static"


@app.get("/")
async def root():
    """Serve the demo frontend."""
    return FileResponse(_STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


# ── Auth ──────────────────────────────────────────────────────────


class AuthRequest(BaseModel):
    password: str


@app.post("/api/auth")
async def authenticate(request: AuthRequest):
    """Validate the shared password and set an auth cookie."""
    if not CHICORY_PASSWORD:
        # No password configured — open access
        return {"status": "ok"}
    if request.password != CHICORY_PASSWORD:
        raise HTTPException(status_code=403, detail="Wrong password")
    response = JSONResponse({"status": "ok"})
    response.set_cookie(
        key="chicory_auth",
        value=CHICORY_PASSWORD,
        httponly=True,
        secure=True,
        samesite="strict",
        max_age=60 * 60 * 24 * 30,  # 30 days
    )
    return response


# ── Helpers ──────────────────────────────────────────────────────────


def _get_session(session_id: str) -> Session:
    if _session_manager is None:
        raise HTTPException(status_code=503, detail="Server not ready")
    session = _session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    return session


def _cleanup_messages(messages: list[dict[str, Any]]) -> None:
    """Remove trailing messages back to the last user text message.
    Mirrors ChatSession._cleanup_orphaned_tool_use."""
    while messages:
        last = messages[-1]
        if last["role"] == "user" and isinstance(last["content"], str):
            messages.pop()
            break
        messages.pop()


# ── Request / Response models ────────────────────────────────────────


class SessionResponse(BaseModel):
    session_id: str
    created_at: str
    expires_at: str


class UploadResponse(BaseModel):
    files_ingested: int
    memories_created: int
    errors: list[str]


class ChatRequest(BaseModel):
    message: str


class ToolCallRecord(BaseModel):
    name: str
    input: dict[str, Any]
    result: dict[str, Any]


class ChatResponse(BaseModel):
    response: str
    tool_calls: list[ToolCallRecord]


class LoginRequest(BaseModel):
    email: str


class LoginResponse(BaseModel):
    email: str
    file_credits: int
    message_credits: int
    is_subscriber: bool = False
    new_session_id: str | None = None


class CreditCheckoutRequest(BaseModel):
    file_credits: int = 0
    message_credits: int = 0


class StatusResponse(BaseModel):
    session_id: str
    memory_count: int
    tag_list: list[str]
    message_count: int
    files_remaining: int
    chats_remaining: int
    created_at: str
    expires_at: str
    logged_in: bool = False
    email: str | None = None
    file_credits: int = 0
    message_credits: int = 0
    is_subscriber: bool = False


# ── Endpoints ────────────────────────────────────────────────────────


@app.post("/api/sessions", response_model=SessionResponse)
async def create_session(tier: str | None = None) -> SessionResponse:
    """Create a new ephemeral session. Pass ?tier=free to test the free tier."""
    if _session_manager is None:
        raise HTTPException(status_code=503, detail="Server not ready")
    session = _session_manager.create_session()
    if tier and tier in ("free", "pro"):
        session.tier = tier
    return SessionResponse(
        session_id=session.session_id,
        created_at=session.created_at.isoformat(),
        expires_at=session.expires_at.isoformat(),
    )


@app.delete("/api/sessions/{session_id}")
async def destroy_session(session_id: str) -> dict[str, str]:
    """Destroy a session and wipe its data."""
    if _session_manager is None:
        raise HTTPException(status_code=503, detail="Server not ready")
    if not _session_manager.destroy_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "destroyed"}


@app.post("/api/sessions/{session_id}/login", response_model=LoginResponse)
async def login(session_id: str, request: LoginRequest) -> LoginResponse:
    """Link an email to this session, loading any purchased credits.

    If the user is an active subscriber with a saved network and the current
    session is empty, swap to a restored subscriber session.
    """
    if _user_db is None or _session_manager is None:
        raise HTTPException(status_code=503, detail="Server not ready")
    session = _get_session(session_id)
    email = request.email.strip().lower()

    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Invalid email")

    user = _user_db.get_or_create_user(email)
    is_sub = _user_db.is_subscriber(email)
    new_session_id: str | None = None

    # If subscriber with a saved network and current session is empty,
    # swap to a restored subscriber session.
    persistent_db = NETWORK_DIR / str(user["id"]) / "chicory.db"
    if is_sub and persistent_db.exists() and session.files_uploaded == 0:
        _session_manager.destroy_session_silent(session_id)
        session = _session_manager.create_session_for_subscriber(user["id"])
        new_session_id = session.session_id

    session.email = email
    session.file_credits = user["file_credits"]
    session.message_credits = user["message_credits"]
    session.is_subscriber = is_sub
    session.user_id = user["id"]

    return LoginResponse(
        email=email,
        file_credits=user["file_credits"],
        message_credits=user["message_credits"],
        is_subscriber=is_sub,
        new_session_id=new_session_id,
    )


@app.post("/api/sessions/{session_id}/credit-checkout")
async def credit_checkout(
    session_id: str, request: CreditCheckoutRequest, http_request: Request,
) -> dict[str, str]:
    """Create a Stripe Checkout session for purchasing credits."""
    import stripe

    stripe.api_key = os.environ.get("STRIPE_SECRET_KEY")
    if not stripe.api_key:
        raise HTTPException(status_code=503, detail="Payments not configured")

    session = _get_session(session_id)
    if not session.email:
        raise HTTPException(status_code=400, detail="Must be logged in to purchase credits")

    if request.file_credits <= 0 and request.message_credits <= 0:
        raise HTTPException(status_code=400, detail="Must select at least one credit type")

    line_items = []
    if request.file_credits > 0:
        price_id = os.environ.get("STRIPE_FILE_CREDIT_PRICE_ID")
        if not price_id:
            raise HTTPException(status_code=503, detail="File credit price not configured")
        line_items.append({"price": price_id, "quantity": request.file_credits})

    if request.message_credits > 0:
        price_id = os.environ.get("STRIPE_MESSAGE_CREDIT_PRICE_ID")
        if not price_id:
            raise HTTPException(status_code=503, detail="Message credit price not configured")
        line_items.append({"price": price_id, "quantity": request.message_credits})

    base_url = str(http_request.base_url).rstrip("/")

    checkout_session = stripe.checkout.Session.create(
        mode="payment",
        customer_email=session.email,
        line_items=line_items,
        success_url=f"{base_url}/?purchased=1&session_id={session_id}",
        cancel_url=f"{base_url}/?cancelled=1&session_id={session_id}",
        metadata={
            "chicory_session_id": session_id,
            "chicory_email": session.email,
            "chicory_file_credits": str(request.file_credits),
            "chicory_message_credits": str(request.message_credits),
        },
    )
    return {"checkout_url": checkout_session.url}


@app.post("/api/sessions/{session_id}/subscribe-checkout")
async def subscribe_checkout(
    session_id: str, http_request: Request,
) -> dict[str, str]:
    """Create a Stripe Checkout session for a monthly subscription."""
    import stripe

    stripe.api_key = os.environ.get("STRIPE_SECRET_KEY")
    if not stripe.api_key:
        raise HTTPException(status_code=503, detail="Payments not configured")

    price_id = os.environ.get("STRIPE_SUBSCRIPTION_PRICE_ID")
    if not price_id:
        raise HTTPException(status_code=503, detail="Subscription price not configured")

    session = _get_session(session_id)
    if not session.email:
        raise HTTPException(status_code=400, detail="Must be logged in to subscribe")

    base_url = str(http_request.base_url).rstrip("/")

    checkout_session = stripe.checkout.Session.create(
        mode="subscription",
        customer_email=session.email,
        line_items=[{"price": price_id, "quantity": 1}],
        success_url=f"{base_url}/?subscribed=1&session_id={session_id}",
        cancel_url=f"{base_url}/?cancelled=1&session_id={session_id}",
        metadata={
            "chicory_session_id": session_id,
            "chicory_email": session.email,
        },
    )
    return {"checkout_url": checkout_session.url}


@app.post("/api/stripe-webhook")
async def stripe_webhook(request: Request) -> dict[str, str]:
    """Handle Stripe webhook events for credit purchases."""
    import stripe

    stripe.api_key = os.environ.get("STRIPE_SECRET_KEY")
    webhook_secret = os.environ.get("STRIPE_WEBHOOK_SECRET")

    payload = await request.body()
    sig = request.headers.get("stripe-signature")

    if webhook_secret and sig:
        try:
            event = stripe.Webhook.construct_event(payload, sig, webhook_secret)
        except (ValueError, stripe.error.SignatureVerificationError):
            raise HTTPException(status_code=400, detail="Invalid webhook signature")
    else:
        import json as _json
        event = _json.loads(payload)

    event_type = event.get("type", "")
    obj = event.get("data", {}).get("object", {})

    if event_type == "checkout.session.completed":
        _handle_checkout_completed(obj)
    elif event_type == "customer.subscription.updated":
        _handle_subscription_updated(event, obj)
    elif event_type == "customer.subscription.deleted":
        _handle_subscription_deleted(event, obj)

    return {"status": "ok"}


def _handle_checkout_completed(cs: dict) -> None:
    """Process a completed checkout — credits or subscription."""
    metadata = cs.get("metadata", {})
    email = metadata.get("chicory_email")
    stripe_sid = cs.get("id", "")
    mode = cs.get("mode", "payment")

    if not email or not _user_db:
        return

    if mode == "subscription":
        # Subscription checkout — activate subscription
        customer_id = cs.get("customer", "")
        subscription_id = cs.get("subscription", "")
        _user_db.set_subscription(
            email=email,
            status="active",
            stripe_customer_id=customer_id,
            stripe_subscription_id=subscription_id,
            expires_at=None,
        )
        # Update in-memory session
        chicory_sid = metadata.get("chicory_session_id")
        if chicory_sid and _session_manager:
            session = _session_manager.get_session(chicory_sid)
            if session and session.email == email:
                user = _user_db.get_or_create_user(email)
                session.is_subscriber = True
                session.user_id = user["id"]
                session._persistent_path = NETWORK_DIR / str(user["id"])
        logger.info("Subscription activated for %s", email)
    else:
        # Credit purchase
        file_qty = int(metadata.get("chicory_file_credits", "0"))
        msg_qty = int(metadata.get("chicory_message_credits", "0"))

        if file_qty > 0:
            _user_db.add_credits(email, "file", file_qty, stripe_sid + ":file")
        if msg_qty > 0:
            _user_db.add_credits(email, "message", msg_qty, stripe_sid + ":msg")

        # Refresh in-memory session balances if still alive
        chicory_sid = metadata.get("chicory_session_id")
        if chicory_sid and _session_manager:
            session = _session_manager.get_session(chicory_sid)
            if session and session.email == email:
                balances = _user_db.get_balances(email)
                session.file_credits = balances["file_credits"]
                session.message_credits = balances["message_credits"]

        logger.info("Credits added for %s: %d file, %d message", email, file_qty, msg_qty)


def _handle_subscription_updated(event: dict, sub: dict) -> None:
    """Handle subscription status changes (renewals, payment failures, cancellations)."""
    if not _user_db:
        return

    stripe_event_id = event.get("id", "")
    customer_id = sub.get("customer", "")
    user = _user_db.find_user_by_stripe_customer(customer_id)
    if not user:
        logger.warning("Subscription update for unknown customer: %s", customer_id)
        return

    # Idempotency check
    if not _user_db.record_subscription_event(stripe_event_id, "subscription.updated", user["id"]):
        return

    stripe_status = sub.get("status", "")
    status_map = {
        "active": "active",
        "past_due": "past_due",
        "canceled": "canceled",
        "unpaid": "expired",
        "incomplete_expired": "expired",
    }
    our_status = status_map.get(stripe_status, "canceled")

    period_end = sub.get("current_period_end")
    expires_at = None
    if period_end:
        from datetime import datetime, timezone
        expires_at = datetime.fromtimestamp(period_end, tz=timezone.utc).isoformat()

    _user_db.set_subscription(
        email=user["email"],
        status=our_status,
        stripe_customer_id=customer_id,
        stripe_subscription_id=sub.get("id", ""),
        expires_at=expires_at,
    )
    logger.info("Subscription updated for %s: %s", user["email"], our_status)


def _handle_subscription_deleted(event: dict, sub: dict) -> None:
    """Handle subscription cancellation/expiry."""
    if not _user_db:
        return

    stripe_event_id = event.get("id", "")
    customer_id = sub.get("customer", "")
    user = _user_db.find_user_by_stripe_customer(customer_id)
    if not user:
        logger.warning("Subscription delete for unknown customer: %s", customer_id)
        return

    if not _user_db.record_subscription_event(stripe_event_id, "subscription.deleted", user["id"]):
        return

    _user_db.set_subscription(
        email=user["email"],
        status="expired",
        stripe_customer_id=customer_id,
        stripe_subscription_id=sub.get("id", ""),
        expires_at=None,
    )
    logger.info("Subscription expired for %s", user["email"])


@app.post("/api/sessions/{session_id}/upload", response_model=UploadResponse)
async def upload_files(session_id: str, files: list[UploadFile]) -> UploadResponse:
    """Upload and ingest files into the session's memory store."""
    session = _get_session(session_id)

    if session.files_limited:
        raise HTTPException(
            status_code=429,
            detail="No file credits remaining. Purchase more to continue uploading.",
        )
    if session.max_files > 0:
        if session.email:
            remaining = (session.max_files + session.file_credits) - session.files_uploaded
        else:
            remaining = session.max_files - session.files_uploaded
    else:
        remaining = len(files)

    files_ingested = 0
    memories_created = 0
    errors: list[str] = []

    for upload in files[:remaining]:
        if not upload.filename:
            continue

        # Read and check file size
        content = await upload.read()
        if len(content) > MAX_FILE_SIZE_BYTES:
            errors.append(
                f"{upload.filename}: exceeds {MAX_FILE_SIZE_BYTES // (1024*1024)}MB limit"
            )
            continue

        # Save uploaded file to session's upload directory
        dest = session.upload_dir / upload.filename
        try:
            dest.write_bytes(content)
        except Exception as e:
            errors.append(f"Failed to save {upload.filename}: {e}")
            continue

        # Ingest via existing function (run in thread to avoid blocking event loop)
        try:
            count = await asyncio.to_thread(
                ingest_file,
                session.orchestrator,
                dest,
                base_dir=session.upload_dir,
                llm_client=session.llm_client,
            )
            if count > 0:
                files_ingested += 1
                memories_created += count
                session.files_uploaded += 1
                # Decrement purchased credits once past the free tier
                if session.email and _user_db and session.files_uploaded > session.max_files:
                    _user_db.use_credit(session.email, "file", session.session_id)
                    session.file_credits = max(0, session.file_credits - 1)
        except Exception as e:
            errors.append(f"Failed to ingest {upload.filename}: {e}")

    if len(files) > remaining:
        errors.append(f"{len(files) - remaining} file(s) skipped — limit is {session.max_files}")

    # Prime layers 2-3: run a retrieval so trends/phase space populate
    if memories_created > 0:
        o = session.orchestrator
        tags = o.tag_manager.list_active_names()
        semantic = [t for t in tags if len(t) > 1
                    and not t.startswith(("day-", "month-", "minute-"))]
        if semantic:
            try:
                await asyncio.to_thread(
                    o.handle_retrieve_memories,
                    query=" ".join(semantic[:5]),
                    method="hybrid",
                    top_k=5,
                )
            except Exception:
                pass

    return UploadResponse(
        files_ingested=files_ingested,
        memories_created=memories_created,
        errors=errors,
    )


@app.post("/api/sessions/{session_id}/chat", response_model=ChatResponse)
def chat(session_id: str, request: ChatRequest) -> ChatResponse:
    """Send a message and get a response with tool call transparency."""
    session = _get_session(session_id)

    if session.chats_limited:
        raise HTTPException(
            status_code=429,
            detail="No message credits remaining. Purchase more to continue chatting.",
        )

    session.chat_turns += 1
    # Decrement purchased credits once past the free tier
    if session.email and _user_db and session.chat_turns > session.max_chats:
        _user_db.use_credit(session.email, "message", session.session_id)
        session.message_credits = max(0, session.message_credits - 1)

    # Update system prompt with tags relevant to this query
    active_tags = session.orchestrator.get_relevant_tags(request.message)
    session.llm_client.update_active_tags(active_tags)

    session.messages.append({"role": "user", "content": request.message})

    tool_call_records: list[ToolCallRecord] = []
    response_text = ""

    # Tool-use loop (mirrors ChatSession._process_message)
    max_iterations = 10
    for _ in range(max_iterations):
        try:
            llm_response = session.llm_client.chat(session.messages)
        except Exception as e:
            _cleanup_messages(session.messages)
            raise HTTPException(status_code=502, detail=f"LLM error: {e}")

        assistant_content = llm_response.content
        has_tool_use = any(b.type == "tool_use" for b in assistant_content)

        if not has_tool_use:
            # Final text response
            session.messages.append({
                "role": "assistant",
                "content": assistant_content,
            })
            text_parts = [
                b.text for b in assistant_content
                if hasattr(b, "text") and b.text.strip()
            ]
            response_text = "\n\n".join(text_parts)

            if not response_text and llm_response.stop_reason == "max_tokens":
                session.messages.append({
                    "role": "user",
                    "content": "Please continue your response.",
                })
                continue

            break

        # Handle tool calls
        session.messages.append({
            "role": "assistant",
            "content": assistant_content,
        })

        tool_results = []
        for block in assistant_content:
            if block.type == "tool_use":
                try:
                    result = dispatch_tool_call(
                        session.orchestrator, block.name, block.input,
                    )
                except Exception as e:
                    result = {"error": str(e)}

                tool_call_records.append(ToolCallRecord(
                    name=block.name,
                    input=block.input,
                    result=result,
                ))
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result, default=str),
                })

        session.messages.append({"role": "user", "content": tool_results})
    else:
        response_text = "[Max tool call iterations reached]"

    if not response_text:
        response_text = "[No response text generated]"

    return ChatResponse(
        response=response_text,
        tool_calls=tool_call_records,
    )


@app.get("/api/sessions/{session_id}/network")
def get_network(session_id: str) -> dict[str, Any]:
    """Get visualization data: phase space, lattice, synchronicities, trends."""
    session = _get_session(session_id)
    o = session.orchestrator

    phase_space = o.handle_get_phase_space()
    trends = o.handle_get_trends()
    synchronicities = o.handle_get_synchronicities(limit=50)
    lattice = o.handle_get_lattice_resonances()
    meta_patterns = o.handle_get_meta_patterns()

    # Build network graph from tag co-occurrences
    ps = phase_space.get("phase_space", {})
    trend_lookup = {t["tag"]: t for t in trends.get("trends", [])}
    tag_lookup: dict[str, dict[str, Any]] = {}
    id_to_name: dict[int, str] = {}

    for quadrant, items in ps.items():
        for item in items:
            name = item["tag"]
            trend = trend_lookup.get(name, {})
            tag_obj = o.tag_manager.get_by_name(name)
            tag_id = tag_obj.id if tag_obj else -1
            tag_lookup[name] = {
                "tag": name,
                "temperature": item["temperature"],
                "retrieval_freq": item["retrieval_freq"],
                "quadrant": quadrant,
                "event_count": trend.get("event_count", 0),
            }
            if tag_id >= 0:
                id_to_name[tag_id] = name

    co_occurrences = o.tag_manager.get_all_co_occurrences(min_count=1)
    edges = []
    for tag_a_id, tag_b_id, count in co_occurrences:
        if tag_a_id in id_to_name and tag_b_id in id_to_name:
            edges.append({
                "source": id_to_name[tag_a_id],
                "target": id_to_name[tag_b_id],
                "weight": count,
            })

    return {
        "phase_space": ps,
        "trends": trends.get("trends", []),
        "synchronicities": synchronicities,
        "lattice": lattice,
        "meta_patterns": meta_patterns,
        "network": {
            "nodes": list(tag_lookup.values()),
            "edges": edges,
        },
    }


@app.get("/api/sessions/{session_id}/status", response_model=StatusResponse)
def get_status(session_id: str) -> StatusResponse:
    """Get session info: memory count, tag list, session age."""
    session = _get_session(session_id)
    o = session.orchestrator

    if session.max_files == 0:
        files_rem = -1
    elif session.email:
        files_rem = (session.max_files + session.file_credits) - session.files_uploaded
    else:
        files_rem = session.max_files - session.files_uploaded

    if session.max_chats == 0:
        chats_rem = -1
    elif session.email:
        chats_rem = (session.max_chats + session.message_credits) - session.chat_turns
    else:
        chats_rem = session.max_chats - session.chat_turns

    return StatusResponse(
        session_id=session.session_id,
        memory_count=o.memory_store.count(),
        tag_list=o.tag_manager.list_active_names(),
        message_count=len(session.messages),
        files_remaining=files_rem,
        chats_remaining=chats_rem,
        created_at=session.created_at.isoformat(),
        expires_at=session.expires_at.isoformat(),
        logged_in=session.email is not None,
        email=session.email,
        file_credits=session.file_credits,
        message_credits=session.message_credits,
        is_subscriber=session.is_subscriber,
    )


# ── Entry point ──────────────────────────────────────────────────────


def main() -> None:
    """Run the Chicory API server."""
    import uvicorn

    uvicorn.run(
        "chicory.api.app:app",
        host=os.environ.get("CHICORY_HOST", "0.0.0.0"),
        port=int(os.environ.get("CHICORY_PORT", "8000")),
        log_level="info",
    )


if __name__ == "__main__":
    main()
