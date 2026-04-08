import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load data
df = pd.read_csv('outputs/kfold_results.csv')

# Capitalize model types for the legend
df['model_type'] = df['model_type'].str.upper()

# Set up the matplotlib figure
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Plot settings
sns.set_theme(style="whitegrid", context="talk")
palette = {'AE': '#E24A33', 'DAE': '#348ABD', 'VAE': '#988ED5'}

# 1. Mean Loss
sns.lineplot(
    data=df, x='latent_dim', y='mean_loss', hue='model_type',
    marker='o', ax=axes[0], linewidth=3, markersize=10, palette=palette
)
axes[0].set_title('Mean Loss (Lower is Better)', fontweight='bold')
axes[0].set_ylabel('Mean Loss')
axes[0].set_xlabel('Latent Dimension')
axes[0].set_xticks(df['latent_dim'].unique())

# 2. Mean PSNR
sns.lineplot(
    data=df, x='latent_dim', y='mean_psnr', hue='model_type',
    marker='s', ax=axes[1], linewidth=3, markersize=10, palette=palette
)
axes[1].set_title('Mean PSNR (Higher is Better)', fontweight='bold')
axes[1].set_ylabel('Mean PSNR (dB)')
axes[1].set_xlabel('Latent Dimension')
axes[1].set_xticks(df['latent_dim'].unique())
axes[1].set_ylim(13, 23.5) # Expand y-axis to prevent cutoff

# Annotation for Slide 1 points
axes[1].annotate('VAE dominant at\nhigh compression', xy=(16, 19.19), xytext=(24, 15.5),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8),
                 horizontalalignment='center', fontsize=12)
axes[1].annotate('AE catches up at\nlower compression', xy=(64, 21.6), xytext=(58, 20.0),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8),
                 horizontalalignment='right', fontsize=12)

# 3. Mean SSIM
sns.lineplot(
    data=df, x='latent_dim', y='mean_ssim', hue='model_type',
    marker='D', ax=axes[2], linewidth=3, markersize=10, palette=palette
)
axes[2].set_title('Mean SSIM (Higher is Better)', fontweight='bold')
axes[2].set_ylabel('Mean SSIM')
axes[2].set_xlabel('Latent Dimension')
axes[2].set_xticks(df['latent_dim'].unique())

plt.tight_layout()

# Save the figure
output_path = 'outputs/kfold_metrics_slide1_plot.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved successfully to {output_path}")
