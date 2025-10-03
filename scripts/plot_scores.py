
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import glob
import wandb
import os

def load_metrics_files(results_dir):
    """Load all metrics files and combine them into a single dataframe."""
    data = []
    for file in glob.glob(str(Path(results_dir) / "**" / "*_scores_metrics.csv"), recursive=True):
        df = pd.read_csv(file)
        # Extract model names from filename
        filename = Path(file).stem.replace('_scores_metrics', '')
        source_model, target_model = filename.split('_as_')
        
        # Get semantic and stylistic scores
        semantic_score = df[df['metric'] == 'semantic_score_mean']['value'].iloc[0]
        stylistic_score = df[df['metric'] == 'stylistic_score_mean']['value'].iloc[0]
        
        data.append({
            'source_model': source_model,
            'target_model': target_model,
            'semantic_score': semantic_score,
            'stylistic_score': stylistic_score
        })
    
    return pd.DataFrame(data)

def create_plots(df):
    """Create two bar plots: one for semantic scores and one for stylistic scores."""
    # Create semantic scores plot
    fig_semantic = go.Figure()
    
    # Add bars for each model pair
    for idx, row in df.iterrows():
        fig_semantic.add_trace(go.Bar(
            name=f"{row['source_model']} → {row['target_model']}",
            x=['Semantic Score'],
            y=[row['semantic_score']],
            text=[f"{row['semantic_score']:.2f}"],
            textposition='auto',
        ))
    
    fig_semantic.update_layout(
        title='Semantic Scores by Model Pair (1-4 scale)',
        yaxis_title='Score',
        barmode='group',
        showlegend=True,
        yaxis=dict(range=[0, 4])  # Scores are on a 1-4 scale
    )
    
    # Create stylistic scores plot
    fig_stylistic = go.Figure()
    
    # Add bars for each model pair
    for idx, row in df.iterrows():
        fig_stylistic.add_trace(go.Bar(
            name=f"{row['source_model']} → {row['target_model']}",
            x=['Stylistic Score'],
            y=[row['stylistic_score']],
            text=[f"{row['stylistic_score']:.2f}"],
            textposition='auto',
        ))
    
    fig_stylistic.update_layout(
        title='Stylistic Scores by Model Pair (1-4 scale)',
        yaxis_title='Score',
        barmode='group',
        showlegend=True,
        yaxis=dict(range=[0, 4])  # Scores are on a 1-4 scale
    )
    
    # Log plots to wandb
    wandb.log({
        "semantic_scores_plot": fig_semantic,
        "stylistic_scores_plot": fig_stylistic
    })

def main():
    # Initialize wandb
    try:
        wandb.login()
    except Exception:
        pass
    
    wandb.init(
        project="dementor-disguise",
        name="score_comparison",
        config={},
    )
    
    # Load data from the call_center directory
    results_dir = "/home/nazcol/dementor-sync/data/results/call_center"
    df = load_metrics_files(results_dir)
    
    if df.empty:
        print("No metrics files found!")
        wandb.finish()
        return
        
    print("\nLoaded data:")
    print(df)
    print("\nGenerating plots...")
    create_plots(df)
    
    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    main()
