import polars as pl
from glob import glob


def main():
    generated_csvs = glob("data/generated/*.csv")

    merged_df = pl.DataFrame()
    for generated_csv in generated_csvs:
        print(f"Processing {generated_csv}...")
        csv_df = pl.read_csv(generated_csv)
        merged_df = pl.concat([merged_df, csv_df], how="diagonal")

    print(merged_df.shape)
    merged_df = merged_df.unique(subset=["question"])
    print(merged_df.shape)
    merged_df.write_csv("data/generated_merged.csv")


if __name__ == "__main__":
    main()
