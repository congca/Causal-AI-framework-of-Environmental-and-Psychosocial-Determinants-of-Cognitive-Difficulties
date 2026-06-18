import pandas as pd
import matplotlib.pyplot as plt


def plot_mediation_results(df):

    plt.figure(
        figsize=(6, 4)
    )

    df = df.sort_values(
        "indirect_effect"
    )

    plt.barh(
        df["mediator"],
        df["indirect_effect"]
    )

    plt.xlabel(
        "Indirect Effect"
    )

    plt.tight_layout()

    plt.savefig(
        "mediation_results.png",
        dpi=300
    )


def plot_temperature_response(df):

    plt.figure(
        figsize=(6, 4)
    )

    plt.scatter(
        df["temp_shock"],
        df["cognition"],
        alpha=0.2
    )

    plt.xlabel(
        "Temperature Shock"
    )

    plt.ylabel(
        "Cognition"
    )

    plt.tight_layout()

    plt.savefig(
        "temperature_response.png",
        dpi=300
    )
