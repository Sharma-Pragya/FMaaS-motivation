import matplotlib.pyplot as plt
import numpy as np

# Model names
models = [
    "chronos_tiny (8M)",
    "chronos_mini (20M)",
    "chronos_small (46M)",
    "chronos_base (200M)",
    "chronos_large (716M)",
    "Moirai_Small (14M)",
    "Moirai_Base (91M)",
    "Moirai_Large (311M)",
]

# Inference times in seconds
A2_times = np.array([0.3816, 0.3684, 0.5336, 0.9255, 1.6672, 0.5088, 0.7071, 1.3158])
A16_times = np.array([0.1018, 0.1289, 0.1814, 0.5040, 1.2564, 0.1260, 0.1768, 0.3277])
A100_times = np.array([0.1410, 0.1073, 0.1360, 0.2219, 0.4038, 0.1407, 0.1936, 0.3149])

# Convert to milliseconds
A2_ms = A2_times * 1000
A16_ms = A16_times * 1000
A100_ms = A100_times * 1000

# Prepare unsorted model-GPU triplets: group by model, preserve A2-A16-A100 order per model
combined_models_grouped = []
combined_times_grouped = []
combined_markers_grouped = []

# Respect order in `models` list, grouping A2, A16, A100 together per model
for i, model in enumerate(models):
    combined_models_grouped.extend([model] * 3)
    combined_times_grouped.extend([A2_ms[i], A16_ms[i], A100_ms[i]])
    combined_markers_grouped.extend(['o', '^', 'P'])  # A2, A16, A100

# Indices for plotting
x_grouped = np.arange(len(combined_models_grouped))

# Plot
fig, ax = plt.subplots(figsize=(12, 10))
bar_color = "#9467bd"
bars = ax.barh(x_grouped, combined_times_grouped, color=bar_color, height=0.6)

# Add markers and text
for i, bar in enumerate(bars):
    ax.scatter(bar.get_width(), bar.get_y() + bar.get_height() / 2,
               marker=combined_markers_grouped[i], color='black', s=60)
    ax.text(bar.get_width() * 1.05, bar.get_y() + bar.get_height() / 2,
            f"{combined_times_grouped[i]:.1f} ms", va='center', fontsize=9)
# Dummy markers for legend
ax.scatter([], [], marker='o', color='black', label='A2')
ax.scatter([], [], marker='^', color='black', label='A16')
ax.scatter([], [], marker='P', color='black', label='A100')

# Axis formatting
ax.set_yticks(x_grouped)
ax.set_yticklabels(combined_models_grouped, fontsize=10)
ax.set_xlabel("Avg. Inference Time (ms)")
ax.set_title("Inference Time for Foundation model for Timeseries", fontsize=14)
ax.set_xscale("log")
ax.grid(True, axis='x', linestyle='--', alpha=0.6)
ax.legend(title="GPU Type", loc="upper right")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_linewidth(1)
# ax.spines['bottom'].set_color('black')
plt.tight_layout(pad=2.0)
plt.show()
plt.savefig("inference_time_ts.pdf", dpi=300, bbox_inches='tight')