from langchain_core.tools import tool

from db.database import get_daily_summary as _get_daily_summary


@tool
def get_daily_summary() -> str:
    """
    Return today's calorie and macro totals compared to the active goal.

    Returns:
        A formatted string showing today's intake vs. goal.
    """
    try:
        data = _get_daily_summary()
        goal = data.get("goal", {})

        def fmt(value, target, unit="g"):
            v = f"{value:.1f}{unit}"
            t = f"{target:.1f}{unit}" if target is not None else "not set"
            return f"{v} / {t}"

        return (
            f"Today's nutrition summary:\n"
            f"  Calories: {fmt(data['calories'], goal.get('calories'), unit=' kcal')}\n"
            f"  Protein:  {fmt(data['protein_g'], goal.get('protein_g'))}\n"
            f"  Carbs:    {fmt(data['carbs_g'],   goal.get('carbs_g'))}\n"
            f"  Fat:      {fmt(data['fat_g'],     goal.get('fat_g'))}"
        )

    except Exception as e:
        return f"Error retrieving daily summary: {e}"
