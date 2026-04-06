import os
import base64
import mimetypes
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

ANTHROPIC_API_KEY = os.environ.get("CLAUDE_API_KEY")

llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    api_key=ANTHROPIC_API_KEY,
    max_tokens=200,
)


def chat(user_message, system_prompt):
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]
    response = llm.invoke(messages)
    return response


def describe_image(image_path, prompt):
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type not in ("image/jpeg", "image/png", "image/gif", "image/webp"):
        mime_type = "image/jpeg"

    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    message = HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
            },
            {
                "type": "text",
                "text": prompt,
            },
        ]
    )

    response = llm.invoke([message])
    return response


foodPrompt = """
Please estimate the calories and breakdown the macros of the food on the plate in the image. 
Provide the user a breakdown of the caloric breakdown of the major macro groups. 
provide insight into the overall healthyness of the meal
"""


if __name__ == "__main__":
    # prompt = "give me chicken parm recipes"
    # print(f"User: {prompt}")
    # answer = chat(prompt)
    # print(f"Claude: {answer}")

    # print()

    image_path = "food2.jpg"
    print(f"image: {image_path}")
    description = describe_image(image_path, foodPrompt)
    print(f"claude: {description}")
