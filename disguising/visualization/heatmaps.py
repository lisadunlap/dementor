'''
Heatmaps all in one plot
'''
import pandas as pd
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

RESULTS_PATH = "data/wandb_export_2025-05-16T02_54_03.211-07_00.csv"
METHODS = ["random_sample_3_examples", "just_name_it", "vibe_based_disguise", "stylistic_clustering", "stylistic_clustering_resample"]
SCORE = "heuristic_diff"

def shorten_model_name(model_name):
    return model_name.split("/")[-1]

def create_heatmap(pivot_df, method, row, col):
    return go.Heatmap(
        z=pivot_df.values,
        x=pivot_df.columns,
        y=pivot_df.index,
        text=pivot_df.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 20},
        colorscale='PiYG',
        zmin=-0.5,
        zmax=0.5,
        showscale=(row == 1 and col == 2)  # Only show colorbar for the last plot
    )

def main():
    df = pd.read_csv(RESULTS_PATH)
    print("Total rows:", len(df))

    fig = make_subplots(rows=2, cols=3, subplot_titles=METHODS,
                        shared_xaxes=True, shared_yaxes=True,
                        vertical_spacing=0.1, horizontal_spacing=0.05)
    fig.update_annotations(font_size=25)

    for i, method in enumerate(METHODS):
        method_df = df[df["method"] == method]
        print(f"Total rows with method {method}:", len(method_df))

        # Shorten model names
        method_df["source_model"] = method_df["source_model"].apply(shorten_model_name)
        method_df["target_model"] = method_df["target_model"].apply(shorten_model_name)

        # Create pivot table for heatmap
        pivot_df = method_df.pivot_table(
            values=SCORE,
            index='source_model',
            columns='target_model',
        )

        # Sort index and columns alphabetically
        pivot_df = pivot_df.sort_index(axis=0)  # Sort rows
        pivot_df = pivot_df.sort_index(axis=1, ascending=False)  # Sort columns in reverse

        row = i // 3 + 1
        col = i % 3 + 1

        heatmap = create_heatmap(pivot_df, method, row, col)
        fig.add_trace(heatmap, row=row, col=col)

        fig.update_xaxes(title_text="Target Model" if row == 2 else "", row=row, col=col)
        fig.update_yaxes(title_text="Source Model" if col == 1 else "", row=row, col=col)

    fig.update_layout(
        height=1200,
        width=1800,
        # title_text=f'{SCORE} Heatmaps',
        font=dict(size=20)
    )

    # Update colorbar
    fig.update_layout(coloraxis=dict(colorbar=dict(title='Similarity Score', y=0.5)))

    # make plots/heuristics folder
    save_path = f"plots/{SCORE}"
    os.makedirs(save_path, exist_ok=True)

    fig.show()
    # save
    fig.write_html(f"{save_path}/heatmaps.html")
    fig.write_image(f"{save_path}/heatmaps.png")
    fig.write_image(f"{save_path}/heatmaps.pdf")
    fig.write_image(f"{save_path}/heatmaps.svg")

if __name__ == "__main__":
    main()