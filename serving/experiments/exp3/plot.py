import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

base_dir = './'

all_data = []

for req_rate_folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, req_rate_folder)

    if os.path.isdir(folder_path):
        file_path = os.path.join(folder_path, 'request_latency_results.csv')

        if os.path.isfile(file_path):
            df = pd.read_csv(file_path)

            df['other_time'] = (
                df['end_to_end_latency(ms)']
                - df['proc_time(ms)']
                - df['swap_time(ms)']
            )

            df['req_rate'] = req_rate_folder
            all_data.append(df)

combined_df = pd.concat(all_data, ignore_index=True)

# Group by req_rate and task
grouped_df = (
    combined_df
    .groupby(['req_rate', 'task'])
    .agg({
        'proc_time(ms)': 'mean',
        'swap_time(ms)': 'mean',
        'other_time': 'mean'
    })
    .reset_index()
)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

# Hatch patterns per task
task_hatches = {
    'ecgclass': '',
    'gestureclass': '//'
}

bar_width = 0.35
req_rates = sorted(grouped_df['req_rate'].unique())
x = range(len(req_rates))

for i, task in enumerate(task_hatches.keys()):
    task_df = grouped_df[grouped_df['task'] == task]

    proc = task_df['proc_time(ms)'].values
    swap = task_df['swap_time(ms)'].values
    other = task_df['other_time'].values

    offset = [pos + (i - 0.5) * bar_width for pos in x]

    ax.bar(offset, proc, bar_width,
           label='Proc Time' if i == 0 else "",
           color='lightblue',
           hatch=task_hatches[task])

    ax.bar(offset, swap, bar_width,
           bottom=proc,
           label='Swap Time' if i == 0 else "",
           color='green',
           hatch=task_hatches[task])

    ax.bar(offset, other, bar_width,
           bottom=proc + swap,
           label='Other Time' if i == 0 else "",
           color='orange',
           hatch=task_hatches[task])

    # Annotations
    for j in range(len(proc)):
        total = proc[j] + swap[j] + other[j]
        ax.text(offset[j], proc[j]/2, f'{proc[j]:.4f}',
                ha='center', va='bottom', fontsize=9)
        ax.text(offset[j], proc[j] + swap[j]/2, f'{swap[j]:.4f}',
                ha='center', va='bottom', fontsize=9)
        ax.text(offset[j], proc[j] + swap[j] + other[j]/2, f'{other[j]:.4f}',
                ha='center', va='bottom', fontsize= 9)
        ax.text(offset[j], total + 0.1, f'{total:.2f}',
                ha='center', va='bottom', fontsize=9)

# X-axis
ax.set_xticks(x)
ax.set_xticklabels(req_rates)
ax.set_xlabel('Request Rate')
ax.set_ylabel('Time (ms)')
ax.set_title('Latency Breakdown vs Request Rate')

# Legends
time_legend = ax.legend(loc='upper center')

task_legend = [
    Patch(facecolor='lightgray', hatch='', label='ECG Classification'),
    Patch(facecolor='lightgray', hatch='//', label='Gesture Classification')
]

ax.legend(handles=task_legend,loc='upper right')
ax.add_artist(time_legend)

plt.tight_layout()
plt.savefig('stacked_bar_plot.png')