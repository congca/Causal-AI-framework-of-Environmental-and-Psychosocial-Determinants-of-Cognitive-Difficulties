# attention dataset builder
import pandas as pd
import numpy as np


RESPONSE_MAP = {
    "Never": 0,
    "Rarely": 1,
    "Sometimes": 2,
    "Often": 3,
    "Always": 4,
}


def classify_question(question):

    q = str(question).lower()

    if "concentrat" in q or "focus" in q:
        return "attention"

    if (
        "fatigue" in q
        or "tired" in q
        or "energy" in q
    ):
        return "fatigue"

    if (
        "stress" in q
        or "anxious" in q
        or "anger" in q
        or "irritable" in q
        or "depress" in q
    ):
        return "stress"

    if (
        "cope" in q
        or "control" in q
        or "piling" in q
        or "overwhelmed" in q
    ):
        return "coping"

    if (
        "sleep" in q
        or "rest" in q
    ):
        return "sleep"

    if (
        "isolated" in q
        or "alone" in q
        or "companionship" in q
        or "left out" in q
    ):
        return "social"

    return None


def clean_answers(df):

    out = df.copy()

    out["answer"] = out["answer"].replace({
        "PMI: Skip": np.nan,
        "PMI: Prefer Not To Answer": np.nan
    })

    out["answer_num"] = (
        out["answer"]
        .map(RESPONSE_MAP)
    )

    return out


def build_attention_latent(df):

    out = df.copy()

    out["category"] = (
        out["question"]
        .apply(classify_question)
    )

    out = clean_answers(out)

    out = out.dropna(
        subset=[
            "answer_num",
            "category"
        ]
    )

    latent = (
        out
        .groupby(
            [
                "person_id",
                "date",
                "category"
            ]
        )["answer_num"]
        .mean()
        .reset_index()
    )

    latent = latent.pivot(
        index=[
            "person_id",
            "date"
        ],
        columns="category",
        values="answer_num"
    ).reset_index()

    return latent


if __name__ == "__main__":

    survey = pd.read_parquet(
        "survey_responses.parquet"
    )

    survey["date"] = pd.to_datetime(
        survey["date"]
    )

    attention_features = (
        build_attention_latent(
            survey
        )
    )

    attention_features.to_parquet(
        "df_attention_latent.parquet",
        index=False
    )

    print(
        "saved:",
        attention_features.shape
    )
