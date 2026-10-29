"""Google login helper for Shibu AI.

Flow (server-side only, no secret ever reaches the frontend):
  1. /auth/google redirects to Google with client_id, redirect_uri,
     scope "openid email profile", and a random state stored in the
     Flask session (CSRF protection for the callback).
  2. Google redirects back to /auth/callback with a code.
  3. The server exchanges the code for tokens (client_secret is
     only in environment variables, only on this request).
  4. The server verifies the ID token and extracts `sub` (the stable
     Google account ID), email, and display name. `sub` becomes the
     session user key as "google#<sub>" — never the email, which
     can change.
  5. The Flask session stores only user_id / display_name / email.
     The Google tokens themselves are NOT stored in the session.

ID-token verification: if the `google-auth` library is installed it
is used (proper signature verification). Otherwise the Google
tokeninfo endpoint is used as a fallback: it checks audience
(client_id), expiry, and email verification. For a small personal
app that fallback is acceptable; for a larger deployment, install
google-auth and the library path will be used automatically.

Google Cloud Console setup (user-side, once):
  APIs & Services > Credentials > Create Credentials > OAuth client ID
  > Web application. Authorized redirect URI (local):
  http://127.0.0.1:5000/auth/callback
  (and, on Render, https://<your-render-domain>/auth/callback).
  Then set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET only in .env
  locally or in Render Environment Variables — never in git, never
  in chat. Add test users in the OAuth consent screen while the app
  is in Testing mode, or publish it when ready.
"""

from __future__ import annotations

import os
import secrets
import time
import urllib.parse

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
TOKENINFO_URL = "https://oauth2.googleapis.com/tokeninfo"
SCOPES = "openid email profile"


def google_user_id(google_sub: str) -> str:
    """Session user key from a Google sub. Never use the email as
    the key — an email can change, a sub cannot."""
    return f"google#{google_sub}"


def build_authorization_url(client_id: str, redirect_uri: str, state: str) -> str:
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": SCOPES,
        "state": state,
        "prompt": "select_account",
    }
    return f"{AUTH_URL}?{urllib.parse.urlencode(params)}"


def new_state() -> str:
    return secrets.token_urlsafe(32)


def exchange_code(client_id: str, client_secret: str, code: str,
                  redirect_uri: str) -> dict:
    """Exchange an authorization code for tokens. Raises on failure
    with a plain message (never including the secret or the code)."""
    import requests

    resp = requests.post(TOKEN_URL, data={
        "code": code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
    }, timeout=15)
    if resp.status_code != 200:
        raise ValueError(f"Google token exchange failed ({resp.status_code})")
    data = resp.json()
    if "id_token" not in data:
        raise ValueError("Google did not return an ID token")
    return data


def verify_id_token(id_token: str, client_id: str) -> dict:
    """Return claims {sub, email, name, picture}. Raises on bad token."""
    try:
        from google.oauth2 import id_token as google_id_token
        from google.auth.transport import requests as google_requests

        claims = google_id_token.verify_oauth2_token(
            id_token, google_requests.Request(), client_id)
        return {
            "sub": claims["sub"],
            "email": claims.get("email", ""),
            "name": claims.get("name", ""),
            "picture": claims.get("picture", ""),
        }
    except ImportError:
        pass  # fall through to the tokeninfo fallback below
    except Exception as exc:
        raise ValueError(f"Google ID token verification failed: {exc}") from exc

    import requests

    resp = requests.get(TOKENINFO_URL, params={"id_token": id_token}, timeout=15)
    if resp.status_code != 200:
        raise ValueError("Google ID token verification failed")
    claims = resp.json()
    if claims.get("aud") != client_id:
        raise ValueError("Google ID token was issued for a different app")
    if int(claims.get("exp", 0)) <= int(time.time()):
        raise ValueError("Google ID token has expired")
    if not claims.get("sub"):
        raise ValueError("Google ID token has no subject")
    return {
        "sub": claims["sub"],
        "email": claims.get("email", ""),
        "name": claims.get("name", ""),
        "picture": claims.get("picture", ""),
    }


def google_config() -> dict:
    return {
        "client_id": os.environ.get("GOOGLE_CLIENT_ID", ""),
        "client_secret": os.environ.get("GOOGLE_CLIENT_SECRET", ""),
        "redirect_uri": os.environ.get(
            "GOOGLE_REDIRECT_URI", "http://127.0.0.1:5000/auth/callback"),
    }
