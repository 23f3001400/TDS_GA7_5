import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Generate synthetic dataset
np.random.seed(42)
n_campaigns = 120
campaign_data = {
    'marketing_spend': np.random.uniform(10, 100, n_campaigns),
    'conversion_rate': np.random.uniform(1, 20, n_campaigns),
    'campaign_type': np.random.choice(['Social Media', 'Email', 'PPC', 'Display'], n_campaigns),
    'duration_days': np.random.randint(7, 60, n_campaigns)
}

# Correlate spend and conversion
for i in range(n_campaigns):
    base_conversion = campaign_data['marketing_spend'][i] * 0.15 + np.random.normal(0, 2)
    campaign_data['conversion_rate'][i] = max(0.5, min(25, base_conversion))

df = pd.DataFrame(campaign_data)

# ✅ Create Seaborn barplot
plt.figure(figsize=(8, 6))
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)

sns.barplot(
    data=df,
    x="campaign_type",
    y="conversion_rate",
    estimator=np.mean,
    errorbar=("ci", 95),
    palette="Set2"
)

plt.title("Average Conversion Rate by Campaign Type", fontsize=16, fontweight="bold", pad=15)
plt.xlabel("Campaign Type", fontsize=13, fontweight="semibold")
plt.ylabel("Average Conversion Rate (%)", fontsize=13, fontweight="semibold")

plt.tight_layout()

# ✅ Save as PNG
plt.savefig("barplot_chart.png", dpi=100, bbox_inches="tight")
plt.show()
