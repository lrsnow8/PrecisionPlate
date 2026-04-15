import os
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from db.database import log_meal, log_meal_items

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatAnthropic(
            model="claude-sonnet-4-6",
            api_key=os.environ.get("CLAUDE_API_KEY"),
            max_tokens=512,
        )
    return _llm


_PARSE_PROMPT = """
You are a nutrition parser. The user will describe a meal in natural language.
Extract the food items and estimate their macros.
Return ONLY a JSON array with this exact structure — no markdown, no explanation:
[
  {{
    "food_name": "item name",
    "calories": <number>,
    "protein_g": <number>,
    "carbs_g": <number>,
    "fat_g": <number>
  }}
]
Meal description: {description}
"""


@tool
def log_meal_text(description: str) -> str:
    """
    Log a meal from a natural language description.
    Parses the description using Claude to extract food items and macros,
    then stores them in the database.

    Args:
        description: Natural language description of the meal.

    Returns:
        A confirmation string with the logged macro totals.
    """
    try:
        llm = _get_llm()
        prompt = _PARSE_PROMPT.format(description=description)
        response = llm.invoke([HumanMessage(content=prompt)])

        import json
        try:
            items = json.loads(response.content)
        except json.JSONDecodeError:
            # Try to extract JSON array from the response
            content = response.content
            start = content.find("[")
            end = content.rfind("]") + 1
            if start != -1 and end > start:
                items = json.loads(content[start:end])
            else:
                return "Error: could not parse macro data from meal description. Please try rephrasing."

        meal_id = log_meal(description, source="text")
        log_meal_items(meal_id, items)

        total_calories = sum(i.get("calories", 0) for i in items)
        total_protein  = sum(i.get("protein_g", 0) for i in items)
        total_carbs    = sum(i.get("carbs_g", 0) for i in items)
        total_fat      = sum(i.get("fat_g", 0) for i in items)

        lines = [f"Meal logged: {description}"]
        for item in items:
            lines.append(
                f"  - {item.get('food_name', 'Unknown')}: "
                f"{item.get('calories', 0):.0f} kcal | "
                f"P {item.get('protein_g', 0):.1f}g | "
                f"C {item.get('carbs_g', 0):.1f}g | "
                f"F {item.get('fat_g', 0):.1f}g"
            )
        lines.append(
            f"Total: {total_calories:.0f} kcal | "
            f"P {total_protein:.1f}g | C {total_carbs:.1f}g | F {total_fat:.1f}g"
        )
        return "\n".join(lines)

    except Exception as e:
        return f"Error logging meal: {e}"
