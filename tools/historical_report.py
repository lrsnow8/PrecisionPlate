from langchain_core.tools import tool

from db.database import get_historical_report as _get_historical_report


@tool
def get_historical_report(period: str) -> str:
    """
    Return an aggregated nutrition report for the past week or month.

    Args:
        period: Either "week" (last 7 days) or "month" (last 30 days).

    Returns:
        A formatted daily breakdown of calories and macros for the given period.
    """
    try:
        if period not in ("week", "month"):
            return "Error: period must be 'week' or 'month'."

        rows = _get_historical_report(period)

        if not rows:
            return f"No meals logged in the past {period}."

        label = "7 days" if period == "week" else "30 days"
        lines = [f"Nutrition report — past {label}:\n"]
        total_cal = total_pro = total_carb = total_fat = 0.0

        for row in rows:
            cal  = row["calories"]  or 0.0
            pro  = row["protein_g"] or 0.0
            carb = row["carbs_g"]   or 0.0
            fat  = row["fat_g"]     or 0.0
            lines.append(
                f"  {row['date']}: "
                f"{cal:.0f} kcal | P {pro:.1f}g | C {carb:.1f}g | F {fat:.1f}g"
            )
            total_cal  += cal
            total_pro  += pro
            total_carb += carb
            total_fat  += fat

        n = len(rows)
        lines.append(
            f"\nAverage over {n} day(s): "
            f"{total_cal/n:.0f} kcal | "
            f"P {total_pro/n:.1f}g | "
            f"C {total_carb/n:.1f}g | "
            f"F {total_fat/n:.1f}g"
        )
        return "\n".join(lines)

    except Exception as e:
        return f"Error generating historical report: {e}"
