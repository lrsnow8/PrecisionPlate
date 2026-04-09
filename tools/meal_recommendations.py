from langchain_core.tools import tool

from db.database import get_daily_summary as _get_daily_summary
from rag.retriever import query as _rag_query


@tool
def get_meal_recommendation(user_id: str) -> str:
    """
    Suggest a next meal based on the user's remaining macro budget for today
    combined with evidence-based dietary guidance from the nutrition knowledge base.

    Args:
        user_id: The user's ID string.

    Returns:
        A meal suggestion string tailored to remaining macros and nutritional guidance.
    """
    try:
        summary = _get_daily_summary(user_id)
        goal = summary.get("goal", {})

        # Compute remaining macros
        def remaining(consumed, target):
            if target is None:
                return None
            return max(0.0, target - consumed)

        rem_calories  = remaining(summary["calories"],  goal.get("calories"))
        rem_protein   = remaining(summary["protein_g"], goal.get("protein_g"))
        rem_carbs     = remaining(summary["carbs_g"],   goal.get("carbs_g"))
        rem_fat       = remaining(summary["fat_g"],     goal.get("fat_g"))

        # Build a context-rich RAG query based on remaining needs
        if rem_protein is not None and rem_protein > 20:
            rag_topic = "high protein meal ideas to meet daily protein goals"
        elif rem_calories is not None and rem_calories < 300:
            rag_topic = "low calorie light snack ideas"
        elif rem_carbs is not None and rem_carbs > 50:
            rag_topic = "healthy carbohydrate food sources whole grains"
        else:
            rag_topic = "balanced meal recommendations macros fiber nutrients"

        rag_context = _rag_query(rag_topic, top_k=3)

        # Format remaining budget section
        def fmt_remaining(value, unit="g"):
            if value is None:
                return "goal not set"
            return f"{value:.1f}{unit} remaining"

        budget_lines = (
            f"  Calories: {fmt_remaining(rem_calories, unit=' kcal')}\n"
            f"  Protein:  {fmt_remaining(rem_protein)}\n"
            f"  Carbs:    {fmt_remaining(rem_carbs)}\n"
            f"  Fat:      {fmt_remaining(rem_fat)}"
        )

        return (
            f"Remaining macro budget for today:\n{budget_lines}\n\n"
            f"Nutrition guidance:\n{rag_context}\n\n"
            f"Based on your remaining budget and the above guidance, "
            f"consider a meal that fills your most under-met macros while "
            f"staying within your calorie target."
        )

    except Exception as e:
        return f"Error generating meal recommendation: {e}"
