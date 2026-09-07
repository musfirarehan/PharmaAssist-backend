from app.services.auth_service import _connection, initialize_auth_database


def _user_payload(row):
    return {
        "id": row["id"],
        "name": row["name"],
        "email": row["email"],
        "username": row["username"],
        "role": row["role"],
    }


def list_contacts(user_id: int, role: str):
    initialize_auth_database()
    opposite_role = "patient" if role == "pharmacist" else "pharmacist"
    with _connection() as connection:
        rows = connection.execute(
            "SELECT id, name, email, username, role FROM users WHERE role = ? ORDER BY name",
            (opposite_role,),
        ).fetchall()
    return [_user_payload(row) for row in rows]


def get_messages(user_id: int, contact_id: int):
    initialize_auth_database()
    with _connection() as connection:
        contact = connection.execute(
            "SELECT id, name, email, username, role FROM users WHERE id = ?",
            (contact_id,),
        ).fetchone()
        if not contact or contact["id"] == user_id:
            raise ValueError("Contact not found.")
        rows = connection.execute(
            """
            SELECT id, sender_id, recipient_id, body, created_at
            FROM messages
            WHERE (sender_id = ? AND recipient_id = ?)
               OR (sender_id = ? AND recipient_id = ?)
            ORDER BY id
            """,
            (user_id, contact_id, contact_id, user_id),
        ).fetchall()
    return {"contact": _user_payload(contact), "messages": [dict(row) for row in rows]}


def send_message(sender_id: int, recipient_id: int, body: str):
    initialize_auth_database()
    with _connection() as connection:
        sender = connection.execute("SELECT role FROM users WHERE id = ?", (sender_id,)).fetchone()
        recipient = connection.execute(
            "SELECT id, name, email, username, role FROM users WHERE id = ?",
            (recipient_id,),
        ).fetchone()
        if not sender or not recipient or sender_id == recipient_id or sender["role"] == recipient["role"]:
            raise ValueError("Messages can only be sent between a patient and a pharmacist.")
        cursor = connection.execute(
            "INSERT INTO messages (sender_id, recipient_id, body) VALUES (?, ?, ?)",
            (sender_id, recipient_id, body.strip()),
        )
        row = connection.execute(
            "SELECT id, sender_id, recipient_id, body, created_at FROM messages WHERE id = ?",
            (cursor.lastrowid,),
        ).fetchone()
    return dict(row)