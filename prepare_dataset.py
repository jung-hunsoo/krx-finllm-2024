import os
import pandas as pd
import random
from datasets import load_dataset, Dataset
from dotenv import load_dotenv
from litellm import completion, batch_completion


# Load environment variables
for env_filename in [".env", ".env.prod", ".env.secret"]:
    load_dotenv(env_filename, override=True)


def get_random_section(long_string, length=1_000):
    if len(long_string) <= length:
        return long_string

    start_index = random.randint(0, len(long_string) - length)

    return long_string[start_index : start_index + length]


def main():
    books_ds = load_dataset("alvanlii/finance-textbooks")["train"]

    texts = []
    for bt in books_ds["book_text"][:2]:
        for i in range(2):
            texts.append(get_random_section(bt, 100))

    qrys = []
    for text in texts:
        messages = [
            {
                "content": "Your job is creating multi-hop reasoning questions in fluent Korean. You will be given a part of a text. Make a question based on it. The question should require multiple steps of reasoning related to the text. Return the question only without any other text.",
                "role": "system",
            },
            {"content": text, "role": "user"},
        ]
        qrys.append(messages)

    responses = batch_completion(model="gpt-4o-mini", messages=qrys)
    resps = [i.choices[0].message.content for i in responses]
    total_prompt_tokens_for_q = sum([r.usage.prompt_tokens for r in responses])
    total_completion_tokens_for_q = sum([r.usage.completion_tokens for r in responses])
    result_df = pd.DataFrame({"sampled_text": texts, "question": resps})

    qrys = []
    for t in resps:
        messages = [
            {
                "content": "You are a skilled financial expert in Korea. Make a response for the question. DO NOT introduce yourself.",
                "role": "system",
            },
            {"content": t, "role": "user"},
        ]
        qrys.append(messages)

    # Generate answers
    responses = batch_completion(model="gpt-4o-mini", messages=qrys)
    resps = [i.choices[0].message.content for i in responses]
    result_df["response"] = resps
    total_prompt_tokens_for_a = sum([r.usage.prompt_tokens for r in responses])
    total_completion_tokens_for_a = sum([r.usage.completion_tokens for r in responses])

    print("total prompt tokens:", total_prompt_tokens_for_q + total_prompt_tokens_for_a)
    print(
        "prompt token costs: $",
        round(
            (total_prompt_tokens_for_q + total_prompt_tokens_for_a) / 1_000_000 * 0.150,
            6,
        ),
    )
    print(
        "total completion tokens:",
        total_completion_tokens_for_q + total_completion_tokens_for_a,
    )
    print(
        "completion token costs: $",
        round(
            (total_completion_tokens_for_q + total_completion_tokens_for_a)
            / 1_000_000
            * 0.600,
            6,
        ),
    )

    os.makedirs("output", exist_ok=True)
    result_df.to_csv("output/generated.csv")
    # Upload to HF
    result_ds = Dataset.from_pandas(result_df)
    # result_ds.push_to_hub("hf/dataset", token=os.getenv("HF_TOKEN"))
    print(result_ds)


if __name__ == "__main__":
    main()
