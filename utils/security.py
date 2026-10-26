"""Input validation and safe file handling."""
import os
import re

MAX_MESSAGE_LENGTH = 2000
MAX_PDF_TEXT_LENGTH = 20000


def validate_message(message):
    """Return (ok, cleaned_or_error)."""
    if message is None:
        return False, "Message is required."
    if not isinstance(message, str):
        return False, "Message must be text."
    cleaned = message.strip()
    if not cleaned:
        return False, "Message cannot be empty."
    if len(cleaned) > MAX_MESSAGE_LENGTH:
        return False, f"Message is too long (max {MAX_MESSAGE_LENGTH} characters)."
    return True, cleaned


def sanitize_filename(name: str) -> str:
    """Make a user-supplied title safe to use as a file name."""
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    name = name.strip("._") or "document"
    return name[:80]


def safe_join(base_dir: str, filename: str) -> str:
    """Join a file name to a directory, rejecting path traversal."""
    clean = sanitize_filename(filename)
    path = os.path.abspath(os.path.join(base_dir, clean))
    base = os.path.abspath(base_dir)
    if not path.startswith(base + os.sep):
        raise ValueError("Unsafe file path.")
    return path
