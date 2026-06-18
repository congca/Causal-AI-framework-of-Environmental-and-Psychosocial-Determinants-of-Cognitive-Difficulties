# cognition outcome analyses
import pandas as pd
import statsmodels.formula.api as smf


def prepare_dataset(
    cognition_df,
    weather_df,
    covariates_df
):

    cognition_df["date"] = pd.to_datetime(
        cognition_df["date"]
    )

    weather_df["date"] = pd.to_datetime(
        weather_df["date"]
    )

    cognition_df["zip3"] = (
        cognition_df["zip3"]
        .astype(str)
        .str.zfill(3)
    )

    weather_df["zip3"] = (
        weather_df["zip3"]
        .astype(str)
        .str.zfill(3)
    )

    df = cognition_df.merge(
        weather_df,
        on=["zip3", "date"],
        how="left"
    )

    df = df.merge(
        covariates_df,
        on="person_id",
        how="left"
    )

    return df


def run_total_effect(df):

    model = smf.ols(
        """
        cognition
        ~ snow_shock_z
        + pm25
        + age
        """,
        data=df.dropna(
            subset=[
                "cognition",
                "snow_shock_z"
            ]
        )
    ).fit()

    return model


def run_mental_health_model(df):

    model = smf.ols(
        """
        cognition
        ~ snow_shock_z
        + mental_health
        + age
        """,
        data=df.dropna(
            subset=[
                "cognition",
                "mental_health"
            ]
        )
    ).fit()

    return model


def run_social_model(df):

    model = smf.ols(
        """
        cognition
        ~ snow_shock_z
        + social
        + age
        """,
        data=df.dropna(
            subset=[
                "cognition",
                "social"
            ]
        )
    ).fit()

    return model


def run_pm25_model(df):

    model = smf.ols(
        """
        cognition
        ~ snow_shock_z
        + pm25
        + age
        """,
        data=df.dropna(
            subset=[
                "pm25"
            ]
        )
    ).fit()

    return model


def run_age_heterogeneity(df):

    df = df.copy()

    df["elderly"] = (
        df["age"] >= 65
    ).astype(int)

    model = smf.ols(
        """
        cognition
        ~ snow_shock_z
        * elderly
        """,
        data=df
    ).fit()

    return model


if __name__ == "__main__":

    cognition_df = pd.read_parquet(
        "df_cognition.parquet"
    )

    weather_df = pd.read_csv(
        "weather_pm25_zip3_daily_2018_2024_clean.csv"
    )

    covariates_df = pd.read_parquet(
        "df_main.parquet"
    )

    analysis_df = prepare_dataset(
        cognition_df,
        weather_df,
        covariates_df
    )

    total_model = run_total_effect(
        analysis_df
    )

    print(
        total_model.summary()
    )
