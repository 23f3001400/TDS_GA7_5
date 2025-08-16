import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import io

# Set random seed for reproducible results
np.random.seed(42)

# Generate realistic synthetic data for marketing campaign effectiveness
n_campaigns = 150

# Generate data
campaign_data = {
    'campaign_id': range(1, n_campaigns + 1),
    'marketing_spend': np.random.lognormal(mean=8.5, sigma=0.8, size=n_campaigns) * 100,  # Marketing spend in thousands
    'conversion_rate': np.random.beta(2, 8, n_campaigns) * 100,  # Conversion rate percentage
    'campaign_type': np.random.choice(['Social Media', 'Email', 'PPC', 'Display', 'Content'], n_campaigns),
    'duration_days': np.random.randint(7, 91, n_campaigns),  # Campaign duration in days
    'target_audience_size': np.random.randint(1000, 50000, n_campaigns)
}

# Create correlation between spend and conversion rate with some noise
for i in range(n_campaigns):
    # Higher spend generally leads to better conversion rates, but with diminishing returns
    base_conversion = min(15, campaign_data['marketing_spend'][i] / 500)
    noise = np.random.normal(0, 2)
    campaign_data['conversion_rate'][i] = max(0.1, base_conversion + noise)

# Create DataFrame
df = pd.DataFrame(campaign_data)

# Set Seaborn style and context for professional appearance
sns.set_style("whitegrid")
sns.set_context("talk", font_scale=1.1)

# Create the figure with exact dimensions for 512x512 output
fig, ax = plt.subplots(figsize=(8, 8))
fig.set_size_inches(8, 8)

# Create the scatterplot
scatter_plot = sns.scatterplot(
    data=df,
    x='marketing_spend',
    y='conversion_rate',
    hue='campaign_type',
    size='duration_days',
    sizes=(50, 200),
    alpha=0.7,
    palette='viridis',
    ax=ax
)

# Customize the plot
ax.set_title('Marketing Campaign Effectiveness Analysis\nSpend vs Conversion Rate by Campaign Type', 
          fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Marketing Spend (Thousands $)', fontsize=12, fontweight='semibold')
ax.set_ylabel('Conversion Rate (%)', fontsize=12, fontweight='semibold')

# Improve legend
handles, labels = ax.get_legend_handles_labels()

# Split legend into two parts: hue and size
hue_legend = ax.legend(handles[:5], labels[:5], title='Campaign Type', 
                       loc='upper left', bbox_to_anchor=(0, 1))
hue_legend.set_title('Campaign Type', prop={'weight': 'bold'})

# Add size legend
size_legend_elements = [
    plt.scatter([], [], s=50, c='gray', alpha=0.7, label='Short (7-30 days)'),
    plt.scatter([], [], s=125, c='gray', alpha=0.7, label='Medium (31-60 days)'),
    plt.scatter([], [], s=200, c='gray', alpha=0.7, label='Long (61-90 days)')
]
size_legend = ax.legend(handles=size_legend_elements, title='Duration', 
                        loc='lower right', bbox_to_anchor=(1, 0))
size_legend.set_title('Duration', prop={'weight': 'bold'})

# Add the hue legend back
ax.add_artist(hue_legend)

# Add trend line
z = np.polyfit(df['marketing_spend'], df['conversion_rate'], 1)
p = np.poly1d(z)
ax.plot(df['marketing_spend'].sort_values(), p(df['marketing_spend'].sort_values()), 
         "r--", alpha=0.8, linewidth=2, label='Trend Line')

# Customize grid and spines
ax.grid(True, alpha=0.3)
sns.despine(left=False, bottom=False)

# Adjust layout to prevent clipping
fig.tight_layout()

# Save to a buffer first
buf = io.BytesIO()
fig.savefig(buf, format='png', dpi=64, facecolor='white', edgecolor='none', 
            bbox_inches='tight')
buf.seek(0)

# Load image and resize to exactly 512x512
img = Image.open(buf)
img_resized = img.resize((512, 512), Image.Resampling.LANCZOS)
img_resized.save('chart.png', 'PNG')
buf.close()

# Display some summary statistics
print("Marketing Campaign Effectiveness Analysis")
print("=" * 50)
print(f"Total Campaigns Analyzed: {len(df)}")
print(f"Average Marketing Spend: ${df['marketing_spend'].mean():.2f}K")
print(f"Average Conversion Rate: {df['conversion_rate'].mean():.2f}%")
print(f"Best Performing Campaign Type: {df.groupby('campaign_type')['conversion_rate'].mean().idxmax()}")
print(f"Correlation (Spend vs Conversion): {df['marketing_spend'].corr(df['conversion_rate']):.3f}")

plt.show()
