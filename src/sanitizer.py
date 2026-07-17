"""
sanitizer.py

Rule-based text sanitization filters for AudioShield middleware.
Redacts sensitive content (PII, credentials, URLs, shell commands, code blocks)
from LLM responses before they are returned to the user.
"""

import re

# ── Existing Patterns ─────────────────────────────────────────────────────────

IP_REGEX = re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b')
URL_REGEX = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+')
API_KEY_REGEX = re.compile(
    r'\b(?:api_key|apikey|secret|password|passwd|token|credential|auth_key)\b[^\w]*(?:[a-zA-Z0-9_\-]{16,64})',
    re.IGNORECASE
)
CODE_BLOCK_REGEX = re.compile(r'```[\s\S]*?```')

SHELL_COMMANDS = [
    r'\bsudo\b', r'\brm\s+-rf\b', r'\bchmod\b', r'\bchown\b', r'\bcurl\b',
    r'\bwget\b', r'\bbash\b', r'\bsh\b', r'\bpython\s+-c\b', r'\bpip\s+install\b'
]
SHELL_REGEX = re.compile(r'(?:' + '|'.join(SHELL_COMMANDS) + r')', re.IGNORECASE)

# ── PII Patterns ──────────────────────────────────────────────────────────────

EMAIL_REGEX = re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b')
PHONE_REGEX = re.compile(
    r'(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
)
SSN_REGEX = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')

# ── Credential / Token Patterns ──────────────────────────────────────────────

AWS_KEY_REGEX = re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b')
JWT_REGEX = re.compile(r'\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_\-+/=]{10,}\b')

# ── File Path Patterns ───────────────────────────────────────────────────────

UNIX_PATH_REGEX = re.compile(r'(?<!\w)/(?:etc|var|tmp|home|root|usr|opt|srv|dev|proc|sys)/[^\s<>"\']+')
WINDOWS_PATH_REGEX = re.compile(r'[A-Za-z]:\\(?:[^\s<>"\'\\]+\\)*[^\s<>"\'\\]*')


# ── Filter Functions ─────────────────────────────────────────────────────────

def redact_secrets(text: str) -> str:
    """Redact IP addresses, API keys/tokens, AWS keys, and JWT tokens."""
    text = IP_REGEX.sub("[REDACTED_IP]", text)
    text = API_KEY_REGEX.sub("[REDACTED_SECRET]", text)
    text = AWS_KEY_REGEX.sub("[REDACTED_AWS_KEY]", text)
    text = JWT_REGEX.sub("[REDACTED_JWT]", text)
    return text


def redact_pii(text: str) -> str:
    """Redact personally identifiable information (emails, phones, SSNs)."""
    text = EMAIL_REGEX.sub("[REDACTED_EMAIL]", text)
    text = SSN_REGEX.sub("[REDACTED_SSN]", text)
    text = PHONE_REGEX.sub("[REDACTED_PHONE]", text)
    return text


def remove_urls(text: str) -> str:
    """Redact URLs."""
    return URL_REGEX.sub("[REDACTED_URL]", text)


def remove_code_blocks(text: str) -> str:
    """Remove markdown code blocks."""
    return CODE_BLOCK_REGEX.sub("[REDACTED_CODE_BLOCK]", text)


def remove_shell_commands(text: str) -> str:
    """Remove or redact lines containing shell commands."""
    lines = text.splitlines()
    filtered_lines = []
    for line in lines:
        if SHELL_REGEX.search(line) or line.strip().startswith('$') or line.strip().startswith('#'):
            filtered_lines.append("[REDACTED_COMMAND]")
        else:
            filtered_lines.append(line)
    return "\n".join(filtered_lines)


def redact_file_paths(text: str) -> str:
    """Redact Unix and Windows absolute file paths."""
    text = UNIX_PATH_REGEX.sub("[REDACTED_PATH]", text)
    text = WINDOWS_PATH_REGEX.sub("[REDACTED_PATH]", text)
    return text


def sanitize_all(text: str) -> str:
    """Apply all local sanitization filters in the recommended order."""
    if not text:
        return text
    text = remove_code_blocks(text)
    text = remove_shell_commands(text)
    text = remove_urls(text)
    text = redact_secrets(text)
    text = redact_pii(text)
    text = redact_file_paths(text)
    return text
