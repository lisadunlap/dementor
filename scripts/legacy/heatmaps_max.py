'''
Heatmaps all in one plot
'''
import pandas as pd
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

RESULTS_PATH = "data/combined_3_methods.csv"
METHODS = ["random_sample_5_examples", "stylistic_clustering", "embedding_clustering"]
# SCORE = "heuristic_diff"
# MIN_SCORE = -0.5
# MAX_SCORE = 0.5
# SCORE = "diff_avg_embedding_distance"
# MIN_SCORE = -1
# MAX_SCORE = 1
SCORE = "heuristic_diff_max"
MIN_SCORE = -1
MAX_SCORE = 1
SUBFOLDER_NAME = "3_methods"
OUT_PATH = f"plots/{SCORE}/{SUBFOLDER_NAME}"

n_methods = len(METHODS)
n_cols = 3
n_rows = math.ceil(n_methods / n_cols)

cell_size = 600  # adjust if needed

def shorten_model_name(model_name):
    return model_name.split("/")[-1]

# def create_heatmap(pivot_df, method, row, col):
#     return go.Heatmap(
#         z=pivot_df.values,
#         x=pivot_df.columns,
#         y=pivot_df.index,
#         text=pivot_df.values,
#         texttemplate='%{text:.2f}',
#         textfont={"size": 20},
#         colorscale='PiYG',
#         zmin=MIN_SCORE,
#         zmax=MAX_SCORE,
#         showscale=(row == 1 and col == 2)  # Only show colorbar for the last plot
#     )
def create_heatmap(pivot_df, label_df, method, row, col):
    text_matrix = [
        [
            f"{label_df.iloc[i, j]}<br>({pivot_df.iloc[i, j] * 100:.1f}%)" if pivot_df.iloc[i, j] > 0 else "/"
            for j in range(pivot_df.shape[1])
        ]
        for i in range(pivot_df.shape[0])
    ]

    return go.Heatmap(
        z=pivot_df.values,
        x=pivot_df.columns,
        y=pivot_df.index,
        text=text_matrix,
        texttemplate='%{text}',
        textfont={"size": 12},
        colorscale='PiYG',
        zmin=MIN_SCORE,
        zmax=MAX_SCORE,
        showscale=(row == 1 and col == 2)
    )


def main():
    df = pd.read_csv(RESULTS_PATH)
    print("Total rows:", len(df))

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=METHODS,
                        shared_xaxes=True, shared_yaxes=True,
                        vertical_spacing=0.1, horizontal_spacing=0.05)
    fig.update_annotations(font_size=25)

    for i, method in enumerate(METHODS):
        method_df = df[df["method"] == method].copy()
        print(f"Total rows with method {method}:", len(method_df))

        # Shorten model names
        method_df["source_model"] = method_df["model"].apply(shorten_model_name)
        method_df["target_model"] = method_df["disguise_as"].apply(shorten_model_name)

        # Create pivot table for heatmap
        # pivot_df = method_df.pivot_table(
        #     values=SCORE,
        #     index='source_model',
        #     columns='target_model',
        # )
        # Create pivot table for heatmap values
        pivot_df = method_df.pivot_table(
            values=SCORE,
            index='source_model',
            columns='target_model',
        ).sort_index(axis=0).sort_index(axis=1, ascending=False)

        # Create pivot table for labels
        label_df = method_df.pivot_table(
            values="max_heuristic_diff_name",
            index="source_model",
            columns="target_model",
            aggfunc="first"
        ).reindex(index=pivot_df.index, columns=pivot_df.columns)


        # Sort index and columns alphabetically
        pivot_df = pivot_df.sort_index(axis=0)  # Sort rows
        pivot_df = pivot_df.sort_index(axis=1, ascending=False)  # Sort columns in reverse

        row = i // n_cols + 1
        col = i % n_cols + 1

        heatmap = create_heatmap(pivot_df, label_df, method, row, col)
        fig.add_trace(heatmap, row=row, col=col)

        fig.update_xaxes(title_text="Target Model" if row == 2 else "", row=row, col=col)
        fig.update_yaxes(title_text="Source Model" if col == 1 else "", row=row, col=col)

    fig.update_layout(
        height=cell_size * n_rows * 1.2,
        width=cell_size * n_cols * 1.2,
        font=dict(size=20),
        coloraxis=dict(colorbar=dict(title='Similarity Score', y=0.5))
    )
    fig.add_annotation(
        text="Target Model",
        x=0.5,
        y=-0.38,
        showarrow=False,
        xref="paper",
        yref="paper",
        font=dict(size=24),
    )

    # Update colorbar
    fig.update_layout(coloraxis=dict(colorbar=dict(title='Similarity Score', y=0.5)))

    # make plots/heuristics folder
    # if os.path.exists(OUT_PATH) and bool(os.listdir(OUT_PATH)):
    #     raise ValueError(f"Folder {OUT_PATH} already exists")
    os.makedirs(OUT_PATH, exist_ok=True)

    # fig.show()
    # save
    # fig.write_html(f"{OUT_PATH}/heatmaps.html")
    fig.write_image(f"{OUT_PATH}/heatmaps.png")
    fig.write_image(f"{OUT_PATH}/heatmaps.pdf")
    # fig.write_image(f"{OUT_PATH}/heatmaps.svg")

if __name__ == "__main__":
    main()