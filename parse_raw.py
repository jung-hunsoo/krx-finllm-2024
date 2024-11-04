# Parse the raw data files from the data folder and save them as a csv file containing `content` column.

import argparse
import json
import os
import polars as pl
from glob import glob

from utils import generate_gpt


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Raw data parser (for 2024 KRX Financial sLLM Challenge)"
    )
    parser.add_argument(
        "-d",
        "--dataset-id",
        choices=["aihub_90", "aihub_97", "aihub_71782"],
        type=str,
        help="Dataset ID",
    )
    parser.add_argument(
        "-f",
        "--filter",
        help="Whether to filter the dataset leveraging GPT",
        action="store_true",
    )
    parser.add_argument(
        "-r",
        "--rebuild",
        help="Whether to rebuild the dataset from the raw data",
        action="store_true",
    )
    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    raw_root = "data/raw"

    os.makedirs(os.path.join("data", "parsing"), exist_ok=True)
    os.makedirs(os.path.join("data", "parsed"), exist_ok=True)
    parsing_filepath = os.path.join("data", "parsing", f"{args.dataset_id}.csv")
    parsed_filepath = os.path.join("data", "parsed", f"{args.dataset_id}.csv")

    if os.path.exists(parsing_filepath) and not args.rebuild:
        print(f"* Resume parsing the dataset from {parsing_filepath}...")
        parsing_df = pl.read_csv(parsing_filepath)
    else:
        print("* Parsing the raw data files...")
        contents: list[str] = []
        match args.dataset_id:
            case "aihub_90":
                # 논문자료 요약 (https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=90)
                raw_filepaths: list[str] = glob(
                    f"{raw_root}/aihub_90/**/*.json", recursive=True
                )
                for raw_filepath in raw_filepaths:
                    with open(raw_filepath) as f:
                        json_data = json.load(f)[0]["data"]
                        len_articles: int = len(json_data)
                        for i in range(len_articles):
                            if json_data[i]["ipc"] != "복합학":
                                continue
                            # 주의! original_text가 아닌 orginal_text임
                            document: str = ""
                            json_item: dict = json_data[i]
                            for j in range(len(json_item["summary_entire"])):
                                document += json_item["summary_entire"][j][
                                    "orginal_text"
                                ]
                            for j in range(len(json_item["summary_section"])):
                                document += json_item["summary_section"][j][
                                    "orginal_text"
                                ]
                            contents.append(document.strip())

            case "aihub_97":
                # 문서요약 텍스트 (https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=97)
                raw_filepaths: list[str] = glob(
                    f"{raw_root}/aihub_97/**/*.json", recursive=True
                )
                for raw_filepath in raw_filepaths:
                    with open(raw_filepath) as f:
                        json_data = json.load(f)

                        for json_item in json_data:
                            len_sentences: int = len(json_item["text"])
                            document: str = ""
                            for i in range(len_sentences):
                                document += json_item["text"][i]["sentence"]

                            contents.append(document)
            case "aihub_71782":
                # 금융 분야 다국어 병렬 말뭉치 데이터 (https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71782)
                raw_filepaths: list[str] = glob(
                    f"{raw_root}/aihub_71782/**/*.json", recursive=True
                )
                for raw_filepath in raw_filepaths:
                    with open(raw_filepath) as f:
                        json_data = json.load(f)

                        document: str = ""
                        for sentence in json_data["sents"]:
                            document += sentence["source_cleaned"] + " "
                        contents.append(document)
            case _:
                raise ValueError(f"Invalid dataset ID: {args.dataset}")
        # Save parsed dataset
        parsing_df = pl.DataFrame({"content": contents, "_is_related": "_"})
        parsing_df.write_csv(parsing_filepath)

    if args.filter:
        print("* Filtering the dataset using GPT...")
        print(f"- Length of contents before filtering: {parsing_df.shape[0]}")

        for i, parsing_sr in enumerate(parsing_df.iter_rows(named=True)):
            if parsing_sr["_is_related"] != "_":
                continue

            prompt: str = (
                "Answer Y only if the following content is related to business, "
                "economics, finance, accounting, investment, or related area.\n"
                f"Content: {parsing_sr['content'][:500]}"
            )
            messages = [
                {
                    "role": "system",
                    "content": "You're an exprt discriminator fluent in Korean.",
                },
                {"role": "user", "content": prompt},
            ]
            response = generate_gpt(
                messages=messages, model="gpt-4o-mini", temperature=0.1
            )
            parsing_df[i, "_is_related"] = "Y" if response == "Y" else "N"

            if i % 200 == 0:
                parsing_df.write_csv(parsing_filepath)

        parsing_df = parsing_df.filter(pl.col("_is_related") == "Y")
        print(f"- Length of contents after filtering: {parsing_df.shape[0]}")

    parsing_df.write_csv(parsed_filepath)
    os.remove(parsing_filepath)


if __name__ == "__main__":
    main()
