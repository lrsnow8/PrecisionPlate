import os
import sqlite3
from datetime import date, timedelta

DB_PATH = "db/precision_plate.db"


def get_db_connection() -> sqlite3.Connection:
    """Return a sqlite3 connection, creating db/ and schema if needed."""
    os.makedirs("db", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    _create_schema(conn)
    return conn


def _create_schema(conn: sqlite3.Connection) -> None:
    """Create all tables if they do not already exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT,
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS goals (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER REFERENCES users(id),
            calories    REAL,
            protein_g   REAL,
            carbs_g     REAL,
            fat_g       REAL,
            updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS meals (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER REFERENCES users(id),
            logged_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
            description TEXT,
            source      TEXT
        );

        CREATE TABLE IF NOT EXISTS meal_items (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            meal_id     INTEGER REFERENCES meals(id),
            food_name   TEXT,
            calories    REAL,
            protein_g   REAL,
            carbs_g     REAL,
            fat_g       REAL
        );
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# User bootstrap
# ---------------------------------------------------------------------------


def bootstrap_user() -> str:
    """
    Return the user id as a string.
    Inserts a default user on first run; always returns "1".
    """
    conn = get_db_connection()
    try:
        row = conn.execute("SELECT id FROM users LIMIT 1").fetchone()
        if row is None:
            conn.execute("INSERT INTO users (name) VALUES (?)", ("default",))
            conn.commit()
            return "1"
        return str(row["id"])
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Goals
# ---------------------------------------------------------------------------


def set_goal(calories: float, protein_g: float, carbs_g: float, fat_g: float) -> None:
    """Insert a new goal row for the user (most recent row = active goal)."""
    conn = get_db_connection()
    try:
        conn.execute(
            """
            INSERT INTO goals (user_id, calories, protein_g, carbs_g, fat_g)
            VALUES (?, ?, ?, ?, ?)
            """,
            (1, calories, protein_g, carbs_g, fat_g),
        )
        conn.commit()
    finally:
        conn.close()


def get_goal() -> dict | None:
    """Return the active goal dict for the user, or None if not set."""
    conn = get_db_connection()
    try:
        row = conn.execute(
            """
            SELECT calories, protein_g, carbs_g, fat_g
            FROM goals
            WHERE user_id = 1
            ORDER BY updated_at DESC
            LIMIT 1
            """,
        ).fetchone()
        if row is None:
            return None
        return dict(row)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Meals
# ---------------------------------------------------------------------------


def log_meal(description: str, source: str) -> int:
    """
    Insert a meal session row and return the new meal id.
    source should be 'text' or 'photo'.
    """
    conn = get_db_connection()
    try:
        cursor = conn.execute(
            """
            INSERT INTO meals (user_id, description, source)
            VALUES (?, ?, ?)
            """,
            (1, description, source),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def log_meal_items(meal_id: int, items: list[dict]) -> None:
    """
    Insert individual food items for a meal.
    Each item dict must have keys: food_name, calories, protein_g, carbs_g, fat_g.
    """
    conn = get_db_connection()
    try:
        conn.executemany(
            """
            INSERT INTO meal_items (meal_id, food_name, calories, protein_g, carbs_g, fat_g)
            VALUES (:meal_id, :food_name, :calories, :protein_g, :carbs_g, :fat_g)
            """,
            [
                {
                    "meal_id": meal_id,
                    "food_name": item.get("food_name", ""),
                    "calories": item.get("calories", 0.0),
                    "protein_g": item.get("protein_g", 0.0),
                    "carbs_g": item.get("carbs_g", 0.0),
                    "fat_g": item.get("fat_g", 0.0),
                }
                for item in items
            ],
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Daily summary
# ---------------------------------------------------------------------------


def get_daily_summary() -> dict:
    """
    Return the current date's macro totals and the active goal.

    Returns:
        {
            "calories": float,
            "protein_g": float,
            "carbs_g": float,
            "fat_g": float,
            "goal": {
                "calories": float | None,
                "protein_g": float | None,
                "carbs_g": float | None,
                "fat_g": float | None,
            }
        }
    """
    today = date.today().isoformat()  # 'YYYY-MM-DD'
    conn = get_db_connection()
    try:
        # Today's totals
        row = conn.execute(
            """
            SELECT
                COALESCE(SUM(mi.calories),  0) AS calories,
                COALESCE(SUM(mi.protein_g), 0) AS protein_g,
                COALESCE(SUM(mi.carbs_g),   0) AS carbs_g,
                COALESCE(SUM(mi.fat_g),     0) AS fat_g
            FROM meal_items mi
            JOIN meals m ON m.id = mi.meal_id
            WHERE m.user_id = 1
              AND DATE(m.logged_at) = ?
            """,
            (today,),
        ).fetchone()

        totals = {
            "calories": row["calories"],
            "protein_g": row["protein_g"],
            "carbs_g": row["carbs_g"],
            "fat_g": row["fat_g"],
        }

        # Active goal
        goal_row = conn.execute(
            """
            SELECT calories, protein_g, carbs_g, fat_g
            FROM goals
            WHERE user_id = 1
            ORDER BY updated_at DESC
            LIMIT 1
            """,
        ).fetchone()

        if goal_row is None:
            goal = {"calories": None, "protein_g": None, "carbs_g": None, "fat_g": None}
        else:
            goal = dict(goal_row)

        totals["goal"] = goal
        return totals
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Historical report
# ---------------------------------------------------------------------------


def get_historical_report(period: str) -> list[dict]:
    """
    Return daily aggregated macro totals for the given period.
    period: "week" (last 7 days) or "month" (last 30 days).

    Returns a list of dicts, one per day that has meals logged:
        [{"date": "YYYY-MM-DD", "calories": float, "protein_g": float,
          "carbs_g": float, "fat_g": float}, ...]
    """
    if period == "week":
        days = 7
    elif period == "month":
        days = 30
    else:
        raise ValueError(f"period must be 'week' or 'month', got {period!r}")

    since = (date.today() - timedelta(days=days)).isoformat()

    conn = get_db_connection()
    try:
        rows = conn.execute(
            """
            SELECT
                DATE(m.logged_at)           AS day,
                SUM(mi.calories)            AS calories,
                SUM(mi.protein_g)           AS protein_g,
                SUM(mi.carbs_g)             AS carbs_g,
                SUM(mi.fat_g)               AS fat_g
            FROM meal_items mi
            JOIN meals m ON m.id = mi.meal_id
            WHERE m.user_id = 1
              AND DATE(m.logged_at) >= ?
            GROUP BY day
            ORDER BY day ASC
            """,
            (since,),
        ).fetchall()

        return [
            {
                "date": row["day"],
                "calories": row["calories"],
                "protein_g": row["protein_g"],
                "carbs_g": row["carbs_g"],
                "fat_g": row["fat_g"],
            }
            for row in rows
        ]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    user_id = bootstrap_user()
    print(user_id)
