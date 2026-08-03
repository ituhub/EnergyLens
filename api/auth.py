"""
EnergyLens — Authentication & User Registration.

Firebase Auth token verification (server-side) + Firestore user storage.
Same GCP project as MarketLens, separate Firestore collection.

Setup:
  1. Enable Firebase Auth (Email/Password) in Firebase Console
  2. GOOGLE_APPLICATION_CREDENTIALS or default GCP credentials on Cloud Run
  3. Set ENERGYLENS_ADMIN_EMAILS env var (comma-separated) for admin users

Usage in FastAPI:
    from api.auth import require_auth, require_admin, get_current_user

    @app.get("/api/protected")
    async def protected(user = Depends(require_auth)):
        return {"uid": user["uid"], "email": user["email"]}

    @app.get("/api/admin/users")
    async def admin_users(user = Depends(require_admin)):
        return get_all_users()
"""

import os
import logging
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from typing import Optional

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from api.logging_config import set_request_context, get_logger

logger = get_logger("energylens.auth")

# Lazy-init Firebase to avoid import errors during testing
_firebase_app = None
_firestore_client = None

ADMIN_EMAILS = set(
    e.strip() for e in os.getenv("ENERGYLENS_ADMIN_EMAILS", "").split(",") if e.strip()
)

USERS_COLLECTION = "energylens_users"

security = HTTPBearer(auto_error=False)


# ---------------------------------------------------------------------------
# Firebase / Firestore initialization
# ---------------------------------------------------------------------------

def _init_firebase():
    """Initialize Firebase Admin SDK (once)."""
    global _firebase_app
    if _firebase_app is not None:
        return

    import firebase_admin
    from firebase_admin import credentials

    try:
        _firebase_app = firebase_admin.get_app()
    except ValueError:
        # No app initialized yet — use default credentials (works on Cloud Run)
        cred = credentials.ApplicationDefault()
        _firebase_app = firebase_admin.initialize_app(cred)

    logger.info("Firebase Admin SDK initialized", extra={"event": "firebase_init"})


def _get_firestore():
    """Get Firestore client (lazy singleton)."""
    global _firestore_client
    if _firestore_client is None:
        _init_firebase()
        from google.cloud import firestore
        _firestore_client = firestore.Client()
    return _firestore_client


# ---------------------------------------------------------------------------
# Token verification
# ---------------------------------------------------------------------------

def verify_firebase_token(id_token: str) -> dict:
    """
    Verify a Firebase Auth ID token and return the decoded claims.
    Returns dict with uid, email, name, etc.
    """
    _init_firebase()
    from firebase_admin import auth

    try:
        decoded = auth.verify_id_token(id_token)
        return decoded
    except auth.ExpiredIdTokenError:
        raise HTTPException(status_code=401, detail="Token expired — please sign in again")
    except auth.InvalidIdTokenError:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    except Exception as e:
        logger.error(f"Token verification failed: {e}", extra={"event": "auth_error"})
        raise HTTPException(status_code=401, detail="Authentication failed")


# ---------------------------------------------------------------------------
# Firestore user management
# ---------------------------------------------------------------------------

def register_or_update_user(decoded_token: dict) -> dict:
    """
    Create or update user document in Firestore on login.
    Returns the full user document.

    Firestore structure:
        energylens_users/{uid}
            email: str
            name: str | null
            role: "admin" | "user"
            registered_at: datetime
            last_login: datetime
            login_count: int
            prediction_count: int
            is_active: bool
    """
    db = _get_firestore()
    uid = decoded_token["uid"]
    email = decoded_token.get("email", "")
    name = decoded_token.get("name", "")

    doc_ref = db.collection(USERS_COLLECTION).document(uid)
    doc = doc_ref.get()
    now = datetime.now(timezone.utc)

    # Determine role
    role = "admin" if email in ADMIN_EMAILS else "user"

    if doc.exists:
        # Returning user — update last_login and bump count
        from google.cloud.firestore_v1 import Increment
        doc_ref.update({
            "last_login": now,
            "login_count": Increment(1),
            "role": role,  # Re-evaluate on each login in case admin list changed
        })
        user_data = doc.to_dict()
        user_data["last_login"] = now
        user_data["role"] = role
        logger.info(
            f"User login: {email} (returning)",
            extra={"event": "user_login", "email": email, "role": role},
        )
    else:
        # New user — create document
        user_data = {
            "uid": uid,
            "email": email,
            "name": name,
            "role": role,
            "registered_at": now,
            "last_login": now,
            "login_count": 1,
            "prediction_count": 0,
            "last_prediction_at": None,
            "is_active": True,
        }
        doc_ref.set(user_data)
        logger.info(
            f"New user registered: {email}",
            extra={"event": "user_register", "email": email, "role": role},
        )

    return user_data


def increment_prediction_count(uid: str):
    """Bump prediction_count after a forecast run."""
    try:
        db = _get_firestore()
        from google.cloud.firestore_v1 import Increment
        db.collection(USERS_COLLECTION).document(uid).update({
            "prediction_count": Increment(1),
            "last_prediction_at": datetime.now(timezone.utc),
        })
    except Exception as e:
        logger.warning(f"Failed to increment prediction count: {e}")


def get_all_users() -> list[dict]:
    """Get all registered users (admin only)."""
    db = _get_firestore()
    docs = db.collection(USERS_COLLECTION).stream()
    users = []
    for doc in docs:
        data = doc.to_dict()
        # Convert datetime objects to ISO strings for JSON serialization
        for key in ("registered_at", "last_login", "last_prediction_at"):
            if data.get(key) and hasattr(data[key], "isoformat"):
                data[key] = data[key].isoformat()
        users.append(data)
    return sorted(users, key=lambda u: u.get("last_login", ""), reverse=True)


def get_user_by_uid(uid: str) -> Optional[dict]:
    """Get a single user document."""
    db = _get_firestore()
    doc = db.collection(USERS_COLLECTION).document(uid).get()
    if doc.exists:
        return doc.to_dict()
    return None


def toggle_user_active(uid: str, is_active: bool) -> bool:
    """Enable or disable a user account (admin action)."""
    db = _get_firestore()
    doc_ref = db.collection(USERS_COLLECTION).document(uid)
    if not doc_ref.get().exists:
        return False
    doc_ref.update({"is_active": is_active})
    logger.info(
        f"User {'activated' if is_active else 'deactivated'}: {uid}",
        extra={"event": "user_toggle", "role": "admin"},
    )
    return True


# ---------------------------------------------------------------------------
# FastAPI dependency injection
# ---------------------------------------------------------------------------

async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[dict]:
    """
    Extract and verify the Firebase token from the Authorization header.
    Returns None if no token provided (for public endpoints).
    """
    if credentials is None:
        return None

    decoded = verify_firebase_token(credentials.credentials)
    user_data = register_or_update_user(decoded)

    # Set logging context for this request
    request_id = str(uuid.uuid4())[:8]
    set_request_context(request_id, decoded["uid"])

    # Block deactivated users
    if not user_data.get("is_active", True):
        raise HTTPException(status_code=403, detail="Account is deactivated")

    return {
        "uid": decoded["uid"],
        "email": decoded.get("email", ""),
        "name": decoded.get("name", ""),
        "role": user_data.get("role", "user"),
    }


async def require_auth(user=Depends(get_current_user)) -> dict:
    """Dependency that requires a valid Firebase token."""
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


async def require_admin(user=Depends(require_auth)) -> dict:
    """Dependency that requires admin role."""
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user
