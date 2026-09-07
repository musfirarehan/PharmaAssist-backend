import hashlib
import hmac
import os
import secrets
import sqlite3
from pathlib import Path


DATABASE_PATH = Path(os.getenv("AUTH_DATABASE_PATH", "app/data/pharmaassist.db"))
ITERATIONS = 310_000


def _connection():
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_auth_database():
    with _connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('patient', 'pharmacist')),
                license_number TEXT NOT NULL DEFAULT '',
                license_authority TEXT NOT NULL DEFAULT '',
                verification_status TEXT NOT NULL DEFAULT 'verified',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token_hash TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                expires_at TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sender_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                recipient_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                body TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS medication_schedules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                medicine TEXT NOT NULL,
                dose TEXT NOT NULL,
                frequency TEXT NOT NULL,
                times_json TEXT NOT NULL,
                duration TEXT NOT NULL,
                food_instructions TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL CHECK (status IN ('pending', 'verified', 'rejected')),
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS medication_doses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                schedule_id INTEGER NOT NULL REFERENCES medication_schedules(id) ON DELETE CASCADE,
                dose_date TEXT NOT NULL,
                scheduled_time TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('upcoming', 'due', 'taken', 'snoozed', 'skipped', 'missed')),
                actual_time TEXT,
                snoozed_until TEXT,
                UNIQUE(schedule_id, dose_date, scheduled_time)
            );
            """
        )
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(medication_doses)").fetchall()
        }
        if "snoozed_until" not in columns:
            connection.execute("ALTER TABLE medication_doses ADD COLUMN snoozed_until TEXT")


def _password_hash(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ITERATIONS).hex()


def _public_user(row):
    return {
        "id": row["id"],
        "name": row["name"],
        "email": row["email"],
        "username": row["username"],
        "role": row["role"],
        "verification_status": row["verification_status"],
    }


def register_user(payload):
    initialize_auth_database()
    if payload.role == "pharmacist" and not payload.license_number.strip():
        raise ValueError("A pharmacist license number is required.")

    salt = secrets.token_bytes(16)
    verification_status = "pending" if payload.role == "pharmacist" else "verified"
    try:
        with _connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO users
                (name, email, username, password_hash, password_salt, role,
                 license_number, license_authority, verification_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload.name.strip(),
                    payload.email.strip().lower(),
                    payload.username.strip(),
                    _password_hash(payload.password, salt),
                    salt.hex(),
                    payload.role,
                    payload.license_number.strip(),
                    payload.license_authority.strip(),
                    verification_status,
                ),
            )
            user_id = cursor.lastrowid
            row = connection.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    except sqlite3.IntegrityError as exc:
        field = "email" if "email" in str(exc) else "username"
        raise ValueError(f"That {field} is already registered.") from exc

    return _create_session(row)


def login_user(payload):
    initialize_auth_database()
    with _connection() as connection:
        row = connection.execute(
            "SELECT * FROM users WHERE username = ? OR email = ?",
            (payload.username.strip(), payload.username.strip().lower()),
        ).fetchone()

    if not row or not hmac.compare_digest(
        _password_hash(payload.password, bytes.fromhex(row["password_salt"])),
        row["password_hash"],
    ):
        raise ValueError("Invalid username or password.")

    return _create_session(row)


def _create_session(row):
    token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    with _connection() as connection:
        connection.execute(
            "INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (?, ?, datetime('now', '+7 days'))",
            (token_hash, row["id"]),
        )
    return {"token": token, "user": _public_user(row)}


def get_user_from_token(token: str):
    if not token:
        return None
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    initialize_auth_database()
    with _connection() as connection:
        row = connection.execute(
            """
            SELECT users.* FROM sessions
            JOIN users ON users.id = sessions.user_id
            WHERE sessions.token_hash = ? AND sessions.expires_at > CURRENT_TIMESTAMP
            """,
            (token_hash,),
        ).fetchone()
    return _public_user(row) if row else None