import json
from langchain_core.tools import tool

from db.database import log_meal, log_meal_items
from image_to_macro.image_to_macro import describe_image

VISION_PROMPT = """
Analyze the food in this image and return ONLY a JSON object with this exact structure:
{
  "description": "brief meal description",
  "calories": <number>,
  "protein_g": <number>,
  "carbs_g": <number>,
  "fat_g": <number>
}
Do not include any other text, markdown, or explanation outside the JSON object.
"""


@tool
def log_meal_photo(image_path: str) -> str:
    """
    Log a meal from a photo by extracting macro data via vision analysis.
    Calls describe_image() on the provided image path, parses the JSON response,
    and stores the meal in the database.

    Args:
        image_path: Absolute or relative path to a .jpg, .jpeg, or .png image file.

    Returns:
        A confirmation string with the logged macro totals, or an error string.
    """
    try:
        response = describe_image(image_path, VISION_PROMPT)

        try:
            data = json.loads(response.content)
        except json.JSONDecodeError:
            return "Error: could not parse macro data from image. Please try a clearer photo."

        description = data.get("description", "Photo meal")
        item = {
            "food_name": description,
            "calories":  data.get("calories",  0),
            "protein_g": data.get("protein_g", 0),
            "carbs_g":   data.get("carbs_g",   0),
            "fat_g":     data.get("fat_g",     0),
        }

        meal_id = log_meal(description, source="photo")
        log_meal_items(meal_id, [item])

        return (
            f"Meal logged from photo: {description}\n"
            f"  Calories: {item['calories']:.0f} kcal | "
            f"P {item['protein_g']:.1f}g | "
            f"C {item['carbs_g']:.1f}g | "
            f"F {item['fat_g']:.1f}g"
        )

    except json.JSONDecodeError:
        return "Error: could not parse macro data from image. Please try a clearer photo."
    except Exception as e:
        return f"Error logging meal from photo: {e}"
