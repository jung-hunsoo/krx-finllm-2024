# Generate synthetic QnAs for the given prompt.

import argparse
import os
import polars as pl
import random

from dotenv import load_dotenv
from glob import glob
from openai import OpenAI
from pydantic import BaseModel

from utils import generate_gpt

# Load environment variables
for env_filename in [".env", ".env.secret"]:
    load_dotenv(env_filename, override=True)


class QnA(BaseModel):
    question: str
    answer: str


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Synthetic Q&A set generator")

    parser.add_argument(
        "-m",
        "--model",
        choices=["gpt-4o", "gpt-4o-mini"],
        type=str,
        default="gpt-4o-mini",
        help="Model name",
    )
    # 국내기업, 재무회계, 주가예측, 금융에이전트, 금융시장
    parser.add_argument(
        "-t",
        "--task",
        choices=["company", "finance", "stock", "agent", "market", "other"],
        type=str,
        default="other",
        help="Task type",
    )

    args, _ = parser.parse_known_args()
    return args


def get_random_sentences(content: str, chunk_size: int = 3) -> str:
    content = content.replace("\n", " ")
    content = content.replace("\t", " ")
    chunks = content.split(". ")
    if len(chunks) <= chunk_size:
        return content

    start_index = random.randint(0, len(chunks) - chunk_size)
    return " ".join(chunks[start_index : start_index + chunk_size])


def main():
    args = parse_args()
    client = OpenAI()
    os.makedirs(os.path.join("data", "generated"), exist_ok=True)
    result_df = pl.DataFrame(schema={"question": pl.String, "answer": pl.String})

    match args.task:
        case "stock":
            # 샘플에서 inc-N은 주가의 N일 전 종가 - 현재 종가로 보임
            # 10일 이상 데이터가 없으므로, inc-5만 활용 가능하도록 window size를 5로 설정
            pass
        case _:
            csv_filepaths: list[str] = glob("data/parsed/*.csv")
            for csv_filepath in csv_filepaths:
                csv_df = pl.read_csv(csv_filepath)

                cnt = 0
                for csv_sr in csv_df.iter_rows():
                    cnt += 1
                    last_questionable_part: str = ""
                    last_generated_question: QnA["question"] = ""
                    for i in range(4):
                        try:
                            questionable_part = get_random_sentences(
                                csv_sr[0], chunk_size=4 * (i + 1)
                            )
                            if questionable_part == last_questionable_part:
                                continue

                            last_questionable_part = questionable_part
                            prompt = f"다음의 내용으로 객관식 단답형 시험 문제와 정답을 작성하자.\n\n{questionable_part}"
                            messages = [
                                {
                                    "role": "system",
                                    "content": "너는 사실 기반의 금융, 재무, 회계, 경제, 투자 등에 대해 잘 알고 있는 교수이다.",
                                },
                                {"role": "user", "content": prompt},
                            ]
                            completion = client.beta.chat.completions.parse(
                                model=args.model,
                                messages=messages,
                                response_format=QnA,
                                temperature=0.1,
                            )
                            qna_generated: QnA = completion.choices[0].message.parsed
                            if qna_generated.question == last_generated_question:
                                continue

                            last_generated_question = qna_generated.question

                            result_df.vstack(
                                pl.DataFrame(qna_generated.model_dump()), in_place=True
                            )
                        except:
                            pass
                    if cnt % 200 == 0:
                        print(f"Processed {cnt} contents...")
                        result_df.write_csv(f"data/generated/{args.task}.csv")

    # Save final output
    result_df.write_csv(f"data/generated/{args.task}.csv")
    print(f"Done! {result_df.shape[0]} QnAs generated.")


if __name__ == "__main__":
    main()
