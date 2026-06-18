# attention outcome analyses
import pandas as pd
import statsmodels.formula.api as smf


def create_temperature_features(df):

    df = df.sort_values(
        ["zip3", "date"]
    )

    if "temp" in df.columns:

        df["temp_7day"] = (
            df.groupby("zip3")["temp"]
            .transform(
                lambda x:
                x.rolling(
                    7,
                    min_periods=3
                ).mean()
            )
        )

        df["extreme_cold"] = (
            df["temp"] < -2
        ).astype(int)

    if "TMAX" in df.columns:

        df["extreme_heat"] = (
            df["TMAX"]
            >
            df["TMAX"].quantile(
                0.95
            )
        ).astype(int)

    return df


def run_extreme_heat(df):

    model = smf.logit(
        "attention ~ extreme_heat",
        data=df
    ).fit()

    return model


def run_extreme_cold(df):

    model = smf.logit(
        "attention ~ extreme_cold",
        data=df
    ).fit()

    return model


def run_age_interaction(df):

    model = smf.logit(
        """
        attention
        ~ temp_shock * age
        """,
        data=df
    ).fit()

    return model


def run_lag_model(df):

    df = df.sort_values(
        ["person_id", "date"]
    )

    df["temp_shock_lag1"] = (
        df.groupby("person_id")
        ["temp_shock"]
        .shift(1)
    )

    model = smf.logit(
        """
        attention
        ~ temp_shock_lag1
        + age
        """,
        data=df.dropna(
            subset=[
                "temp_shock_lag1"
            ]
        )
    ).fit()

    return model
