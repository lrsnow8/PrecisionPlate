from datetime import date


def get_system_prompt(summary: str, today_snapshot: dict) -> str:
    """
    Build the full system prompt for the PrecisionPlate nutritionist agent.

    Args:
        summary: Compressed long-term memory string (may be empty on first run).
        today_snapshot: Dict with keys calories, protein_g, carbs_g, fat_g, goal.

    Returns:
        The complete system prompt string.
    """

    # --- Persona ---
    persona = """\
You are PrecisionPlate, a knowledgeable and encouraging personal nutritionist assistant.
Your role is to help the user track meals, monitor macros, set nutrition goals, and make
informed food choices through natural conversation.

Guidelines:
- Be concise and practical.
- Always use the available tools to read or write data — never guess or fabricate numbers.
- When the user logs a meal, confirm what was stored and show the macro totals.
- When no goal is set, gently prompt the user to use set_goal before giving progress feedback.
- If the user's message contains a file path ending in .jpg, .jpeg, or .png, call the
  log_meal_photo tool with that path.
- Never raise errors to the user directly — return a clear, friendly error message instead.
- Do not use emojis in any response. Plain text only.\
"""

    # --- Long-term memory ---
    if summary and summary.strip():
        memory_section = f"[LONG-TERM MEMORY]\n{summary.strip()}"
    else:
        memory_section = "[LONG-TERM MEMORY]\n(No prior conversation history yet.)"

    # --- Today's nutrition snapshot ---
    goal = today_snapshot.get("goal", {}) or {}
    cal_goal = goal.get("calories")
    pro_goal = goal.get("protein_g")
    carb_goal = goal.get("carbs_g")
    fat_goal = goal.get("fat_g")

    goals_unset = all(v is None for v in [cal_goal, pro_goal, carb_goal, fat_goal])

    def fmt(consumed, target, unit="g"):
        c = f"{consumed:.1f}{unit}"
        t = f"{target:.1f}{unit}" if target is not None else "not set"
        return f"{c} / {t}"

    snapshot_lines = [
        f"[TODAY'S NUTRITION SNAPSHOT — {date.today().strftime('%A, %B %d, %Y')}]",
        f"  Calories: {fmt(today_snapshot.get('calories', 0), cal_goal, unit=' kcal')}",
        f"  Protein:  {fmt(today_snapshot.get('protein_g', 0), pro_goal)}",
        f"  Carbs:    {fmt(today_snapshot.get('carbs_g', 0), carb_goal)}",
        f"  Fat:      {fmt(today_snapshot.get('fat_g', 0), fat_goal)}",
    ]
    if goals_unset:
        snapshot_lines.append(
            "\n  No daily goals have been set yet. "
            "Ask the user to use set_goal to define their calorie and macro targets."
        )
    snapshot_section = "\n".join(snapshot_lines)

    return "\n\n".join([persona, memory_section, snapshot_section])
