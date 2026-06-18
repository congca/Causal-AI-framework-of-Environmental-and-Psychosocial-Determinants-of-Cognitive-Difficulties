# psychological feature construction
import pandas as pd


RENAME_MAP = {
    "Income: Annual Income": "income",
    "Education Level: Highest Grade": "education",

    "Smoking: Smoke Frequency": "smoking_freq",
    "Smoking: 100 Cigs Lifetime": "smoking_100",

    "Alcohol: Drink Frequency Past Year": "alcohol",

    "Overall Health: General Mental Health": "mental_health",
    "Overall Health: General Social": "social",
    "Overall Health: Medical Form Confidence": "literacy",

    "How often do you feel lack companionship?": "loneliness",

    "gender_concept_id": "gender",
    "race_concept_id": "race",
    "ethnicity_concept_id": "ethnicity"
}


HEALTH_MAP = {
    "Excellent": 5,
    "Very Good": 4,
    "Good": 3,
    "Fair": 2,
    "Poor": 1
}


LITERACY_MAP = {
    "Extremely": 5,
    "Quite A Bit": 4,
    "Somewhat": 3,
    "A Little Bit": 2,
    "Not At All": 1
}


LONELY_MAP = {
    "Never": 1,
    "Rarely": 2,
    "Sometimes": 3,
    "Often": 4,
    "Always": 5
}


def clean_text(x):

    if pd.isna(x):
        return None

    x = str(x)

    x = x.replace(
        "General Mental Health: ",
        ""
    )

    x = x.replace(
        "General Social: ",
        ""
    )

    x = x.replace(
        "Medical Form Confidence: ",
        ""
    )

    x = x.replace(
        "Excllent",
        "Excellent"
    )

    if "PMI" in x:
        return None

    return x.strip()


def build_psychological_features(df):

    out = df.copy()

    out = out.rename(
        columns=RENAME_MAP
    )

    for v in [
        "mental_health",
        "social",
        "literacy",
        "loneliness"
    ]:

        if v in out.columns:
            out[v] = (
                out[v]
                .apply(clean_text)
            )

    out["mental_health"] = (
        out["mental_health"]
        .map(HEALTH_MAP)
    )

    out["social"] = (
        out["social"]
        .map(HEALTH_MAP)
    )

    out["literacy"] = (
        out["literacy"]
        .map(LITERACY_MAP)
    )

    out["loneliness"] = (
        out["loneliness"]
        .map(LONELY_MAP)
    )

    return out


if __name__ == "__main__":

    survey_df = pd.read_parquet(
        "final_df.parquet"
    )

    psy = build_psychological_features(
        survey_df
    )

    psy.to_parquet(
        "df_psy.parquet",
        index=False
    )

    print(
        "saved:",
        psy.shape
    )
