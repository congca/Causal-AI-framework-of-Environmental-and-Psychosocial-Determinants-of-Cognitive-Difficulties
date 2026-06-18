# CVAE mechanism model
import pandas as pd
import statsmodels.formula.api as smf


def demean_tw(df, col):

    return (
        df[col]
        - df.groupby("zip3")[col].transform("mean")
        - df.groupby("date")[col].transform("mean")
        + df[col].mean()
    )


def build_weather_features(df):

    df = df.sort_values(
        ["zip3", "date"]
    )

    df["snow_lag1"] = (
        df.groupby("zip3")["SNOW"]
        .shift(1)
    )

    df["temp_shock"] = (
        df["SNOW"]
        - df["snow_lag1"]
    )

    df["blizzard"] = (
        (
            df["SNOW"]
            >
            df["SNOW"].quantile(0.90)
        )
        &
        (df["SNOW"] > 0)
    ).astype(int)

    return df


def prepare_core_dataset(df):

    keep = [
        "person_id",
        "date",
        "zip3",
        "cognition",
        "age",
        "pm25",
        "temp_shock",
        "blizzard"
    ]

    out = (
        df[keep]
        .dropna()
        .copy()
    )

    out["cognition_r"] = (
        demean_tw(
            out,
            "cognition"
        )
    )

    out["temp_shock_r"] = (
        demean_tw(
            out,
            "temp_shock"
        )
    )

    out["blizzard_r"] = (
        demean_tw(
            out,
            "blizzard"
        )
    )

    out["pm25_r"] = (
        demean_tw(
            out,
            "pm25"
        )
    )

    return out


def run_total_effect(df):

    model = smf.ols(
        "cognition_r ~ temp_shock_r + blizzard_r",
        data=df
    ).fit()

    return model


def run_pm25_mediation(df):

    mediator = smf.ols(
        "pm25_r ~ temp_shock_r + blizzard_r",
        data=df
    ).fit()

    outcome = smf.ols(
        """
        cognition_r
        ~ temp_shock_r
        + blizzard_r
        + pm25_r
        """,
        data=df
    ).fit()

    return mediator, outcome


def run_psychological_mediation(
    df,
    mediator
):

    m_a = smf.ols(
        f"{mediator} ~ blizzard_r",
        data=df
    ).fit()

    m_b = smf.ols(
        f"""
        cognition_r
        ~ blizzard_r
        + {mediator}
        """,
        data=df
    ).fit()

    indirect = (
        m_a.params["blizzard_r"]
        *
        m_b.params[mediator]
    )

    return indirect


def run_biomarker_mediation(
    df,
    biomarker
):

    m_a = smf.ols(
        f"{biomarker} ~ temp_shock_r",
        data=df
    ).fit()

    m_b = smf.ols(
        f"""
        cognition_r
        ~ temp_shock_r
        + {biomarker}
        """,
        data=df
    ).fit()

    indirect = (
        m_a.params["temp_shock_r"]
        *
        m_b.params[biomarker]
    )

    return indirect


def run_behavior_mediation(
    df,
    behavior
):

    m_a = smf.ols(
        f"{behavior} ~ blizzard_r",
        data=df
    ).fit()

    m_b = smf.ols(
        f"""
        cognition_r
        ~ blizzard_r
        + {behavior}
        """,
        data=df
    ).fit()

    indirect = (
        m_a.params["blizzard_r"]
        *
        m_b.params[behavior]
    )

    return indirect


def run_age_heterogeneity(df):

    df = df.copy()

    df["elderly"] = (
        df["age"] >= 65
    ).astype(int)

    model = smf.ols(
        """
        cognition_r
        ~ blizzard_r
        * elderly
        """,
        data=df
    ).fit()

    return model


if __name__ == "__main__":

    core = pd.read_parquet(
        "df_core.parquet"
    )

    total_model = (
        run_total_effect(core)
    )

    print(
        total_model.summary()
    )
