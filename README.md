# PrecisionPlate

PrecisionPlate is a conversational nutrition assistant that runs in your terminal. Talk to it like a personal nutritionist — log meals in plain English or from a photo, set daily macro goals, check your progress, and get meal recommendations. All your data and conversation history are stored locally.

---

## How it works

PrecisionPlate is built as a single LangGraph agent created to be used with claude. It maintains memory across sessions, tracks your meals in a local SQLite database, and answers nutrition questions using a built-in knowledge base (ChromaDB + sentence-transformers). No data leaves your machine except the LLM API calls to Anthropic.

---

## Setup

### 1. Install dependencies

```
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 2. Set your Anthropic API key

```
Do this however you want but be safe
```

Add this to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.) to make it permanent.

### 3. Ingest the nutrition knowledge base

This only needs to be run once (or whenever you add new documents to `rag/docs/`):

```
python rag/ingest.py
```

### 4. Run the app

```
python main.py
```

---

## Sample session

```
PrecisionPlate — Your personal nutrition assistant
Type your message and press Enter. Type exit or quit to stop.

You: hi, can you set my daily goals to 2200 calories, 180g protein, 220g carbs, 70g fat

PrecisionPlate:
Goals updated:
  Calories: 2200 kcal
  Protein:  180.0g
  Carbs:    220.0g
  Fat:      70.0g

You: I just had breakfast — two scrambled eggs, two slices of whole wheat toast, and a banana

PrecisionPlate:
Meal logged: two scrambled eggs, two slices of whole wheat toast, and a banana
  - Scrambled eggs (x2): 182 kcal | P 12.6g | C 1.4g | F 13.4g
  - Whole wheat toast (x2): 138 kcal | P 5.8g | C 24.0g | F 2.0g
  - Banana: 105 kcal | P 1.3g | C 27.0g | F 0.4g
Total: 425 kcal | P 19.7g | C 52.4g | F 15.8g

You: how am I doing today?

PrecisionPlate:
Today's nutrition summary:
  Calories: 425.0 kcal / 2200.0 kcal
  Protein:  19.7g / 180.0g
  Carbs:    52.4g / 220.0g
  Fat:      15.8g / 70.0g

You are off to a reasonable start on carbs and fat, but protein is well behind pace
for the day. Your next meal should prioritize a high-protein source to close that gap.

You: what should I eat for lunch to hit my protein goal?

PrecisionPlate:
With 160g of protein still to go and around 1775 kcal remaining, a high-protein lunch
would make a significant dent. Consider:

- Grilled chicken breast (200g): roughly 60g protein, 330 kcal
- 1 cup cooked lentils: 18g protein, 230 kcal
- Greek yogurt (200g): 20g protein, 130 kcal

A grilled chicken and lentil bowl with a side of Greek yogurt would give you nearly
100g of protein in one meal while staying well within your calorie budget.

You: how much fiber should I be eating?

PrecisionPlate:
The recommended adequate intake for fiber is 25g per day for adult women and 38g per
day for adult men. Most people fall well short of this. Good sources include oats,
beans, lentils, vegetables, and whole grains. Increasing fiber intake gradually and
drinking plenty of water helps avoid digestive discomfort.

You: show me my week

PrecisionPlate:
Nutrition report — past 7 days:

  2026-04-08: 425 kcal | P 19.7g | C 52.4g | F 15.8g

Average over 1 day(s): 425 kcal | P 19.7g | C 52.4g | F 15.8g

You: exit
Goodbye!
```

---

## Photo meal logging

Send a message containing the path to a `.jpg`, `.jpeg`, or `.png` image and PrecisionPlate will analyze the food in the photo and log it automatically:

```
You: log this meal for me /home/user/pictures/dinner.jpg
```

---

## What you can ask

| Task | Example prompt |
|---|---|
| Set goals | "Set my goal to 1800 calories, 150g protein, 180g carbs, 60g fat" |
| Log a meal | "I had a chicken Caesar salad and a Diet Coke for lunch" |
| Log from photo | "What did I eat? /path/to/meal.jpg" |
| Check today | "How am I doing on my macros today?" |
| Get a recommendation | "What should I eat for dinner?" |
| Look up nutrition | "How much omega-3 should I get per day?" |
| Weekly report | "Show me my nutrition for the past week" |
| Monthly report | "Give me a summary of my eating this month" |

---

## Project structure

```
PrecisionPlate/
├── main.py                        # Entry point — CLI loop
├── requirements.txt
├── agent/
│   ├── graph.py                   # LangGraph StateGraph
│   ├── state.py                   # NutritionState TypedDict
│   └── prompts.py                 # System prompt builder
├── tools/                         # All 7 LangChain tools
├── rag/
│   ├── ingest.py                  # One-time ingestion script
│   ├── retriever.py               # ChromaDB query function
│   └── docs/                      # Nutrition knowledge base files
├── db/
│   └── database.py                # SQLite schema + CRUD helpers
└── image_to_macro/
    └── image_to_macro.py          # Vision-based macro extraction
```

---

## Notes

- Conversation history persists across sessions automatically via the LangGraph checkpointer.
- The SQLite database (`db/precision_plate.db`) and ChromaDB store (`rag/chroma_db/`) are local and gitignored.
- To add more nutrition knowledge, drop `.txt` files into `rag/docs/` and re-run `python rag/ingest.py`.
- The app is single-user by design. All data is stored under user ID `1`.
