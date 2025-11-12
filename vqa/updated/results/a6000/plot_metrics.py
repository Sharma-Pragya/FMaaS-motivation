import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV
df = pd.read_csv("combined_metrics.csv")

# Extract short model names for cleaner plots
df["short_model_name"] = df["model_name"].apply(lambda x: x.split("/")[-1])

# List of numeric metrics to plot individually
metrics = [
    "model_load_duration_sec", "gpu_load_memory_mb", "avg_cpu_memory_usage_mb",
    "avg_cpu_usage_percent", "avg_gpu_usage_percent", "avg_gpu_memory_usage_mb",
    "prompt_eval_rate_tokens_per_sec", "ttft_ms", "throughput_tokens_per_sec",
    "accuracy", "average_latency_ms", "total_inference_time_sec"
]

# Plot 1: Boxplot for each metric across models
for metric in metrics:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="short_model_name", y=metric, data=df)
    plt.xticks(rotation=45, ha='right')
    plt.title(f"{metric} Across Models")
    plt.tight_layout()
    plt.grid(True)
    plt.show()
    plt.savefig(f"{metric}_boxplot.png")  # Save each plot as a PNG

# Plot 2: Scatter plot of Accuracy vs Latency
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x="average_latency_ms",
    y="accuracy",
    hue="short_model_name",
    data=df,
    palette="tab10",
    s=100
)
plt.title("Accuracy vs. Latency")
plt.xlabel("Average Latency (ms)")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
plt.savefig("accuracy_vs_latency.png")
