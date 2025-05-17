import pandas as pd
import os

RESULTS_PATH = "data/wandb_export_2025-05-16T02_54_03.211-07_00.csv"
METHODS = ["random_sample_3_examples", "just_name_it", "vibe_based_disguise", "stylistic_clustering", "stylistic_clustering_resample"]
SCORE = "heuristic_diff"

def shorten_model_name(model_name):
    return model_name.split("/")[-1]

def main(method):
    df = pd.read_csv(RESULTS_PATH)
    print("Total rows:", len(df))
    df = df[df["method"] == method]
    print(f"Total rows with method {method}:", len(df))

    # shorten model names
    df["source_model"] = df["source_model"].apply(shorten_model_name)
    df["target_model"] = df["target_model"].apply(shorten_model_name)

    # Create pivot table for heatmap
    pivot_df = df.pivot_table(
        values=SCORE,
        index='source_model',
        columns='target_model',
    )

    # Sort index and columns alphabetically
    pivot_df = pivot_df.sort_index(axis=0)  # Sort rows
    pivot_df = pivot_df.sort_index(axis=1, ascending=False)  # Sort columns in reverse

    # Plot heatmap using Plotly
    import plotly.graph_objects as go

    fig = go.Figure(data=go.Heatmap(
        z=pivot_df.values,
        x=pivot_df.columns,
        y=pivot_df.index,
        text=pivot_df.values,
        texttemplate='%{text:.2f}',  # This only affects the displayed text
        textfont={"size": 25},
        colorscale='PiYG',
        colorbar=dict(title='Similarity Score'),
        zmin=-0.5,
        zmax=0.5
    ))

    fig.update_layout(
        title=f'{SCORE} Heatmap (method:{method})',
        xaxis=dict(title='Target Model'),
        yaxis=dict(title='Source Model'),
        height=1000,
        width=1400,
        font=dict(size=25)
    )

    # make plots/heuristics folder
    save_path = f"plots/{SCORE}/{method}"
    os.makedirs(save_path, exist_ok=True)

    fig.show()
    # save
    fig.write_html(f"{save_path}/heatmap.html")
    #save to png
    fig.write_image(f"{save_path}/heatmap.png")
    #save to pdf
    fig.write_image(f"{save_path}/heatmap.pdf")
    #save to svg
    fig.write_image(f"{save_path}/heatmap.svg")


if __name__ == "__main__":
    for method in METHODS:
        main(method)



