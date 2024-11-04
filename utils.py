from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
for env_filename in [".env", ".env.secret"]:
    load_dotenv(env_filename, override=True)


def generate_gpt(
    messages: list[dict] = [],
    model: str = "gpt-4o-mini",
    temperature: float = 0.1,
) -> str:
    """Generate answer using OpenAI's GPT model."""

    response = OpenAI().chat.completions.create(
        messages=messages,
        model=model,
        temperature=temperature,
    )
    return response.choices[0].message.content
