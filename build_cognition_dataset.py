# cognition dataset builder
import pandas as pd


COGNITION_QUESTIONS = [
    "Disability: Difficulty Concentrating",
    "In the past 7 days, I had trouble concentrating.",
    "In the past 2 weeks, how often have you been bothered by trouble concentrating on things, such as reading the newspaper or watching television."
]


def build_cognition_matrix(df):

    out = (
        df.groupby(
            [
                "person_id",
                "question"
            ]
        )
        .first()
        .reset_index()
    )

    matrix = out.pivot(
        index="person_id",
        columns="question",
        values="answer"
    )

    matrix = matrix.apply(
        lambda x:
        x.astype("category").cat.codes
    )

    matrix = matrix.fillna(0)

    matrix["cognition_latent"] = (
        matrix.mean(axis=1)
    )

    return (
        matrix[
            ["cognition_latent"]
        ]
        .reset_index()
    )


def build_analysis_dataset(
    cognition_df,
    weather_df,
    demographic_df
):

    weather_df = weather_df.copy()

    weather_df["date"] = pd.to_datetime(
        weather_df["date"]
    )

    weather_df["zip3"] = (
        weather_df["zip3"]
        .astype(str)
        .str.zfill(3)
    )

    demographic_df["zip3"] = (
        demographic_df["zip3"]
        .astype(str)
        .str.zfill(3)
    )

    merged = demographic_df.merge(
        weather_df,
        on="zip3",
        how="left"
    )

    merged = merged.merge(
        cognition_df,
        on="person_id",
        how="left"
    )

    return merged


if __name__ == "__main__":

    cognition_survey = pd.read_parquet(
        "cognition_survey.parquet"
    )

    cognition_latent = (
        build_cognition_matrix(
            cognition_survey
        )
    )

    cognition_latent.to_parquet(
        "df_cognition.parquet",
        index=False
    )

    print(
        cognition_latent.shape
    )
