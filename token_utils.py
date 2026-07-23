import jwt
import os
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.getenv("jwt_secret_key")

if not SECRET_KEY:
    raise RuntimeError(
        "jwt_secret_key is not configured. "
        "Set it in your environment or .env file before starting the API."
    )
    
ALGORITHM = "HS256"
TOKEN_EXPIRE_MINUTES = 60
MAX_REFRESH_AGE_HOURS = 24


def create_chat_token(
    doc_id: str,
    refresh_expires_at: int | None = None,
):
    now = datetime.now(timezone.utc)

    if refresh_expires_at is None:
        refresh_expires_at = int(
            (now + timedelta(hours=MAX_REFRESH_AGE_HOURS)).timestamp()
        )

    payload = {
        "doc_id": doc_id,
        "iat": now,
        "exp": now + timedelta(minutes=TOKEN_EXPIRE_MINUTES),
        "refresh_expires_at": refresh_expires_at,
    }

    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def verify_chat_token(token: str):
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])


def decode_expired_token(token: str):
    return jwt.decode(
        token,
        SECRET_KEY,
        algorithms=[ALGORITHM],
        options={"verify_exp": False},
    )
