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
    cnt = 0

    match args.task:
        case "agent":
            prompt = f"""다음의 예시를 참고하여, 주어진 Pandas 데이터프레임으로 Python Pandas 코딩 테스트용 단답형 시험 문제와 정답을 한글로 작성하자. 단, 시험 문제는 겹치지 않도록 한다.

<예시>
주어진 데이터프레임을 보고 질문에 요구에 알맞는 코드를 선택하시요.
### df.head()
|    | Symbol    | Series   | Date        |   Prev Close |   Open Price |   High Price |   Low Price |   Last Price |   Close Price |   Average Price |   Total Traded Quantity |    Turnover |   No. of Trades |   Deliverable Qty |   % Dly Qt to Traded Qty |
|---:|:----------|:---------|:------------|-------------:|-------------:|-------------:|------------:|-------------:|--------------:|----------------:|------------------------:|------------:|----------------:|------------------:|-------------------------:|
|  0 | GODREJIND | EQ       | 15-May-2017 |       564.6  |       581    |       584    |      568.5  |       578.9  |        578.55 |          578.09 |                  797171 | 4.60836e+08 |           21649 |            360927 |                    45.28 |
|  1 | GODREJIND | EQ       | 16-May-2017 |       578.55 |       581.45 |       589    |      572.25 |       583.8  |        584.8  |          583.6  |                  500223 | 2.9193e+08  |           17204 |            210364 |                    42.05 |
|  2 | GODREJIND | EQ       | 17-May-2017 |       584.8  |       583    |       594    |      576.85 |       584.9  |        588.6  |          588.74 |                  504155 | 2.96815e+08 |            8567 |            261667 |                    51.9  |
|  3 | GODREJIND | EQ       | 18-May-2017 |       588.6  |       582    |       588.85 |      571.2  |       572.25 |        574.6  |          580.9  |                  223583 | 1.29879e+08 |            7144 |             99785 |                    44.63 |
|  4 | GODREJIND | EQ       | 19-May-2017 |       574.6  |       581    |       585.8  |      567.55 |       579.85 |        578    |          577.31 |                  245436 | 1.41692e+08 |            4969 |             68041 |                    27.72 |

### 질문:
"종가" 열의 평균 값을 계산합니다.

### 선택지:
a) ```python
df['Close Price'].mean()
```
b) ```python
df['Close_Price'].mean()
```
c) ```python
df['Total Traded Quantity'].median()
```
d) ```python
sum(df['Close Price']) / len(df['Close_Price'])
```
"""
            messages = [
                {
                    "role": "system",
                    "content": "너는 대학교에서 파이썬 프로그래밍과 데이터 사이언티스트를 가르치는 교수이다.",
                },
                {"role": "user", "content": prompt},
            ]
            completion = client.beta.chat.completions.parse(
                model="gpt-4o",  # mini는 성능이 낮음
                messages=messages,
                response_format=QnA,
                temperature=0.9,
                n=100,
            )
            for c in completion.choices:
                qna_generated: QnA = c.message.parsed
                result_df.vstack(
                    pl.DataFrame(qna_generated.model_dump()), in_place=True
                )
                cnt += 1

            result_df.write_csv(f"data/generated/{args.task}.csv")
        case "stock":
            # 샘플에서 inc-N은 주가의 N일 전 종가 - 현재 종가로 보임
            # 10일 이상 데이터가 없으므로, inc-5만 활용 가능하도록 window size를 5로 설정
            pass
        case "finance":
            # https://huggingface.co/datasets/alvanlii/finance-textbooks
            raw_df = pl.read_parquet(
                "hf://datasets/alvanlii/finance-textbooks/data/train-00000-of-00001.parquet"
            )
            for book_text in raw_df["book_text"]:
                # 길이에 따라 chunk_size를 다르게 설정
                for i in range(5):
                    chunk_size = 5 * (i + 1)
                    # text 길이에 따라 생성 문제의 수를 다르게 설정
                    len_book_text = len(book_text)
                    cnt_generatable_questions = max(len_book_text // 10, 100)
                    for _ in range(cnt_generatable_questions):
                        try:
                            questionable_part = get_random_sentences(
                                book_text, chunk_size=chunk_size
                            )
                            prompt = f"다음의 내용으로 객관식 단답형 시험 문제와 정답을 한글로 작성하자.\n\n{questionable_part}"
                            messages = [
                                {
                                    "role": "system",
                                    "content": "너는 사실 기반의 금융, 재무, 회계, 경제, 투자 등에 대해 잘 알고 있는 한국에서 학생을 가르치고 있는 교수이다.",
                                },
                                {"role": "user", "content": prompt},
                            ]
                            completion = client.beta.chat.completions.parse(
                                model=args.model,
                                messages=messages,
                                response_format=QnA,
                                temperature=0.0,
                            )
                            qna_generated: QnA = completion.choices[0].message.parsed
                            result_df.vstack(
                                pl.DataFrame(qna_generated.model_dump()), in_place=True
                            )
                            cnt += 1
                        except:
                            pass
                        # 중간 저장
                        if cnt % 200 == 0:
                            print(f"Processed {cnt} contents...")
                            result_df.write_csv(f"data/generated/{args.task}.csv")
        case _:
            csv_filepaths: list[str] = glob("data/parsed/*.csv")
            for csv_filepath in csv_filepaths:
                csv_df = pl.read_csv(csv_filepath)

                for csv_sr in csv_df.iter_rows():
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
                                    "content": "너는 사실 기반의 금융, 재무, 회계, 경제, 투자 등에 대해 잘 알고 있는 한국에서 학생을 가르치고 있는 교수이다.",
                                },
                                {"role": "user", "content": prompt},
                            ]
                            completion = client.beta.chat.completions.parse(
                                model=args.model,
                                messages=messages,
                                response_format=QnA,
                                temperature=0.0,
                            )
                            qna_generated: QnA = completion.choices[0].message.parsed
                            if qna_generated.question == last_generated_question:
                                continue

                            last_generated_question = qna_generated.question

                            result_df.vstack(
                                pl.DataFrame(qna_generated.model_dump()), in_place=True
                            )
                            cnt += 1
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
