from langchain_core.tools import tool

from db.database import set_goal as _set_goal, get_goal as _get_goal


@tool
def set_goal(user_id: str, calories: float, protein_g: float,
             carbs_g: float, fat_g: float) -> str:
    """
    Set or update the user's daily macro and calorie goals.

    Args:
        user_id: The user's ID string.
        calories: Daily calorie target.
        protein_g: Daily protein target in grams.
        carbs_g: Daily carbohydrate target in grams.
        fat_g: Daily fat target in grams.

    Returns:
        A confirmation string showing the new goals.
    """
    try:
        _set_goal(user_id, calories, protein_g, carbs_g, fat_g)
        return (
            f"Goals updated:\n"
            f"  Calories: {calories:.0f} kcal\n"
            f"  Protein:  {protein_g:.1f}g\n"
            f"  Carbs:    {carbs_g:.1f}g\n"
            f"  Fat:      {fat_g:.1f}g"
        )
    except Exception as e:
        return f"Error setting goal: {e}"


@tool
def get_goal(user_id: str) -> str:
    """
    Retrieve the user's current daily macro and calorie goals.

    Args:
        user_id: The user's ID string.

    Returns:
        A formatted string showing the active goals, or a message if none are set.
    """
    try:
        goal = _get_goal(user_id)
        if goal is None:
            return "No goals have been set yet. Use set_goal to define your daily targets."
        return (
            f"Current goals:\n"
            f"  Calories: {goal['calories']:.0f} kcal\n"
            f"  Protein:  {goal['protein_g']:.1f}g\n"
            f"  Carbs:    {goal['carbs_g']:.1f}g\n"
            f"  Fat:      {goal['fat_g']:.1f}g"
        )
    except Exception as e:
        return f"Error retrieving goal: {e}"
