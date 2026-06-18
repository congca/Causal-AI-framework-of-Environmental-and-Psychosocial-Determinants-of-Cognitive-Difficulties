import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.mediation import Mediation


def load_data():
    attention = pd.read_csv("df_attention.csv")
    weather = pd.read_csv("weather_pm25_zip3_daily_2018_2024_clean.csv")
    baseline = pd.read_parquet("final_df.parquet")

    attention["date"] = pd.to_datetime(attention["date"])
    weather["date"] = pd.to_datetime(weather["date"])

    attention["zip3"] = attention["zip3"].astype(str).str.zfill(3)
    weather["zip3"] = weather["zip3"].astype(str).str.zfill(3)

    df = attention.merge(
        weather,
        on=["zip3", "date"],
        how="left"
    )

    df = df.merge(
        baseline,
        on="person_id",
        how="left"
    )

    df["elderly"] = (df["age"] >= 65).astype(int)

    return df


def total_effect(df):

    model = smf.logit(
        "attention ~ temp_shock + age",
        data=df.dropna(
            subset=["attention", "temp_shock"]
        )
    ).fit()

    return model


def pm25_mediation(df):

    d = df[[
        "attention",
        "temp_shock",
        "pm25",
        "age"
    ]].dropna()

    mediator = smf.ols(
        "pm25 ~ temp_shock + age",
        data=d
    ).fit()

    outcome = smf.logit(
        "attention ~ temp_shock + pm25 + age",
        data=d
    ).fit()

    med = Mediation(
        smf.logit(
            "attention ~ temp_shock + pm25 + age",
            data=d
        ),
        smf.ols(
            "pm25 ~ temp_shock + age",
            data=d
        ),
        exposure="temp_shock",
        mediator="pm25"
    )

    return mediator, outcome, med.fit(n_rep=100)


def fatigue_mediation(df):

    d = df[[
        "attention",
        "temp_shock",
        "fatigue",
        "age"
    ]].dropna()

    mediator = smf.ols(
        "fatigue ~ temp_shock + age",
        data=d
    ).fit()

    outcome = smf.logit(
        "attention ~ temp_shock + fatigue + age",
        data=d
    ).fit(disp=0)

    return mediator, outcome


def coping_mediation(df):

    d = df[[
        "attention",
        "temp_shock",
        "coping",
        "age"
    ]].dropna()

    mediator = smf.ols(
        "coping ~ temp_shock + age",
        data=d
    ).fit()

    outcome = smf.logit(
        "attention ~ temp_shock + coping + age",
        data=d
    ).fit(disp=0)

    return mediator, outcome


def age_heterogeneity(df):

    model = smf.logit(
        "attention ~ temp_shock + elderly + temp_shock:elderly",
        data=df.dropna(
            subset=["attention", "temp_shock"]
        )
    ).fit()

    return model


def lag_analysis(df):

    df = df.sort_values(
        ["person_id", "date"]
    )

    df["temp_shock_lag1"] = (
        df.groupby("person_id")["temp_shock"]
        .shift(1)
    )

    model = smf.logit(
        "attention ~ temp_shock_lag1 + age",
        data=df.dropna(
            subset=[
                "attention",
                "temp_shock_lag1",
                "age"
            ]
        )
    ).fit()

    return model


if __name__ == "__main__":

    df = load_data()

    total = total_effect(df)
    print(total.summary())

    pm_xm, pm_my, pm_med = pm25_mediation(df)
    print(pm_med.summary())

    fatigue_xm, fatigue_my = fatigue_mediation(df)
    coping_xm, coping_my = coping_mediation(df)

    age_model = age_heterogeneity(df)
    lag_model = lag_analysis(df)

    print("analysis complete")
