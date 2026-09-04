import json
import re
from datetime import datetime, timedelta

from app.services.auth_service import _connection, initialize_auth_database


DEFAULT_TIMES = {
    "once": ["08:00 AM"],
    "twice": ["08:00 AM", "08:00 PM"],
    "three": ["08:00 AM", "02:00 PM", "08:00 PM"],
    "four": ["08:00 AM", "12:00 PM", "04:00 PM", "08:00 PM"],
}
GRACE_MINUTES = 60


def _frequency_times(frequency: str):
    value = (frequency or "").lower()
    if "four" in value or "qid" in value:
        return DEFAULT_TIMES["four"]
    if "three" in value or "tds" in value:
        return DEFAULT_TIMES["three"]
    if "twice" in value or "bd" in value:
        return DEFAULT_TIMES["twice"]
    if "once" in value or "od" in value or "daily" in value:
        return DEFAULT_TIMES["once"]
    return []


def _duration_days(duration: str):
    match = re.search(r"(\d+)\s*day", duration or "", re.IGNORECASE)
    return int(match.group(1)) if match else None


def _parse_time(value: str):
    for format_string in ("%I:%M %p", "%H:%M"):
        try:
            return datetime.strptime(value.strip(), format_string).time()
        except ValueError:
            continue
    return None


def _refresh_statuses(connection, user_id: int):
    now = datetime.now()
    rows = connection.execute(
        "SELECT doses.id, doses.dose_date, doses.scheduled_time, doses.status, doses.snoozed_until FROM medication_doses doses JOIN medication_schedules schedules ON schedules.id = doses.schedule_id WHERE schedules.user_id = ? AND doses.status IN ('upcoming', 'due', 'snoozed')",
        (user_id,),
    ).fetchall()
    for row in rows:
        scheduled = _parse_time(row["scheduled_time"])
        if not scheduled:
            continue
        due_at = datetime.combine(datetime.strptime(row["dose_date"], "%Y-%m-%d").date(), scheduled)
        if row["snoozed_until"]:
            due_at = datetime.fromisoformat(row["snoozed_until"])
        if now >= due_at + timedelta(minutes=GRACE_MINUTES):
            status = "missed"
        elif now >= due_at:
            status = "due"
        else:
            status = "snoozed" if row["status"] == "snoozed" else "upcoming"
        connection.execute("UPDATE medication_doses SET status = ? WHERE id = ?", (status, row["id"]))


def _dose_payload(row):
    return {
        "id": row["id"], "schedule_id": row["schedule_id"], "medicine": row["medicine"],
        "dose": row["dose"], "scheduled_time": row["scheduled_time"],
        "food_instructions": row["food_instructions"], "status": row["dose_status"],
        "actual_time": row["actual_time"], "snoozed_until": row["snoozed_until"],
        "frequency": row["frequency"], "duration": row["duration"],
    }


def generate_reminders(medicines):
    """Preserve the original non-persistent schedule preview endpoint."""
    return {"reminders": [{
        "medicine": medicine.get("name", ""), "dosage": medicine.get("dosage", ""),
        "times": medicine.get("scheduled_times") or _frequency_times(medicine.get("frequency", "")),
        "duration": medicine.get("duration", ""), "food_instructions": medicine.get("instructions", ""),
    } for medicine in medicines]}


def create_schedule(user_id: int, medicines: list[dict], role: str):
    initialize_auth_database()
    created = []
    with _connection() as connection:
        for medicine in medicines:
            frequency = medicine.get("frequency", "")
            times = medicine.get("scheduled_times") or _frequency_times(frequency)
            duration = medicine.get("duration", "")
            days = _duration_days(duration)
            complete = bool(medicine.get("name") and medicine.get("dosage") and frequency and times and days)
            status = "verified" if role == "pharmacist" and complete and float(medicine.get("confidence", 100)) >= 70 else "pending"
            cursor = connection.execute(
                "INSERT INTO medication_schedules (user_id, medicine, dose, frequency, times_json, duration, food_instructions, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (user_id, medicine["name"], medicine.get("dosage", ""), frequency, json.dumps(times), duration, medicine.get("food_instructions") or medicine.get("instructions", ""), status),
            )
            schedule_id = cursor.lastrowid
            created.append({"id": schedule_id, "medicine": medicine["name"], "status": status})
            if status == "verified":
                for day_offset in range(days):
                    dose_date = (datetime.now().date() + timedelta(days=day_offset)).isoformat()
                    for scheduled_time in times:
                        connection.execute("INSERT OR IGNORE INTO medication_doses (schedule_id, dose_date, scheduled_time, status) VALUES (?, ?, ?, 'upcoming')", (schedule_id, dose_date, scheduled_time))
    return {"schedules": created, "requires_verification": any(item["status"] == "pending" for item in created)}


def get_today_doses(user_id: int):
    initialize_auth_database()
    with _connection() as connection:
        _refresh_statuses(connection, user_id)
        rows = connection.execute("SELECT doses.*, schedules.medicine, schedules.dose, schedules.frequency, schedules.duration, schedules.food_instructions, doses.status AS dose_status FROM medication_doses doses JOIN medication_schedules schedules ON schedules.id = doses.schedule_id WHERE schedules.user_id = ? AND schedules.status = 'verified' AND doses.dose_date = date('now') ORDER BY doses.scheduled_time", (user_id,)).fetchall()
        return [_dose_payload(row) for row in rows]


def update_dose(user_id: int, dose_id: int, action: str):
    initialize_auth_database()
    with _connection() as connection:
        _refresh_statuses(connection, user_id)
        row = connection.execute("SELECT doses.*, schedules.medicine, schedules.dose, schedules.frequency, schedules.duration, schedules.food_instructions, doses.status AS dose_status FROM medication_doses doses JOIN medication_schedules schedules ON schedules.id = doses.schedule_id WHERE doses.id = ? AND schedules.user_id = ?", (dose_id, user_id)).fetchone()
        if not row:
            raise ValueError("Dose not found.")
        if action == "taken":
            connection.execute("UPDATE medication_doses SET status = 'taken', actual_time = ? WHERE id = ?", (datetime.now().isoformat(timespec="seconds"), dose_id))
        elif action == "skip":
            connection.execute("UPDATE medication_doses SET status = 'skipped' WHERE id = ?", (dose_id,))
        else:
            connection.execute("UPDATE medication_doses SET status = 'snoozed', snoozed_until = ? WHERE id = ?", ((datetime.now() + timedelta(minutes=10)).isoformat(timespec="seconds"), dose_id))
        updated = connection.execute("SELECT doses.*, schedules.medicine, schedules.dose, schedules.frequency, schedules.duration, schedules.food_instructions, doses.status AS dose_status FROM medication_doses doses JOIN medication_schedules schedules ON schedules.id = doses.schedule_id WHERE doses.id = ?", (dose_id,)).fetchone()
        return _dose_payload(updated)


def get_adherence(user_id: int):
    initialize_auth_database()
    with _connection() as connection:
        row = connection.execute("SELECT COUNT(*) AS total, SUM(CASE WHEN doses.status = 'taken' THEN 1 ELSE 0 END) AS taken, SUM(CASE WHEN doses.status = 'missed' THEN 1 ELSE 0 END) AS missed FROM medication_doses doses JOIN medication_schedules schedules ON schedules.id = doses.schedule_id WHERE schedules.user_id = ? AND doses.dose_date >= date('now', '-6 days') AND doses.dose_date <= date('now')", (user_id,)).fetchone()
    total, taken = row["total"] or 0, row["taken"] or 0
    return {"total": total, "taken": taken, "missed": row["missed"] or 0, "percent": round(taken / total * 100) if total else 0}


def get_pending_schedules():
    initialize_auth_database()
    with _connection() as connection:
        rows = connection.execute("SELECT * FROM medication_schedules WHERE status = 'pending' ORDER BY created_at DESC").fetchall()
    return [dict(row) | {"times": json.loads(row["times_json"])} for row in rows]


def verify_schedule(schedule_id: int, approved: bool, edits: dict | None = None):
    initialize_auth_database()
    with _connection() as connection:
        row = connection.execute("SELECT * FROM medication_schedules WHERE id = ?", (schedule_id,)).fetchone()
        if not row:
            raise ValueError("Schedule not found.")
        values = edits or {}
        status = "verified" if approved else "rejected"
        medicine, dose = values.get("medicine", row["medicine"]), values.get("dose", row["dose"])
        frequency, times = values.get("frequency", row["frequency"]), values.get("times", json.loads(row["times_json"]))
        duration, food = values.get("duration", row["duration"]), values.get("food_instructions", row["food_instructions"])
        connection.execute("UPDATE medication_schedules SET medicine = ?, dose = ?, frequency = ?, times_json = ?, duration = ?, food_instructions = ?, status = ? WHERE id = ?", (medicine, dose, frequency, json.dumps(times), duration, food, status, schedule_id))
        if approved:
            for day_offset in range(_duration_days(duration) or 1):
                dose_date = (datetime.now().date() + timedelta(days=day_offset)).isoformat()
                for scheduled_time in times:
                    connection.execute("INSERT OR IGNORE INTO medication_doses (schedule_id, dose_date, scheduled_time, status) VALUES (?, ?, ?, 'upcoming')", (schedule_id, dose_date, scheduled_time))
    return {"id": schedule_id, "status": status}