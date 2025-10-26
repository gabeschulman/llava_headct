import json
import os

import polars as pl

CONDITION_COLS = [
    "cancer",
    "hydrocephalus",
    "edema",
    "dementia",
    "IPH",
    "IVH",
    "SDH",
    "EDH",
    "SAH",
    "ICH",
    "fracture",
    "hematoma",
]


def main():
    config = json.load(open("../config.json"))
    datapath = config["dataset"]["datapath"]
    image_files = config["dataset"]["image_files"]
    processed_files = config["dataset"]["processed_files"]
    radiology_report = config["dataset"]["radiology_report"]

    reports_df = pl.read_csv(radiology_report, infer_schema_length=10000)

    for k in ["train", "test", "val"]:
        image_df = pl.read_csv(os.path.join(datapath, image_files[k])).with_columns(
            pl.col("accession_num").cast(pl.String)
        )
        merged_df = image_df.join(reports_df, on="accession_num")
        merged_df = merged_df.with_columns(
            pl.struct(CONDITION_COLS)
            .alias("conditions")
            .map_elements(
                lambda elt: ", ".join([cond for cond in elt.keys() if elt[cond] == 1]),
                return_dtype=pl.String,
            )
            .map_elements(
                lambda elt: "Conditions: " + elt if elt != "" else "Conditons: none",
                return_dtype=pl.String,
            )
        )

        processed_fp = os.path.join(datapath, processed_files[k])
        merged_df.write_parquet(processed_fp)


if __name__ == "__main__":
    main()
