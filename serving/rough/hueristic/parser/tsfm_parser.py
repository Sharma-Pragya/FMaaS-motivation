import pandas as pd
import json

# Load CSV
df = pd.read_csv("TSFM.csv")

# Normalize names
df['backbone'] = df['backbone'].str.lower()
df['decoder'] = df['decoder'].str.lower()
df['task_name'] = df['task_name'].str.lower()
df['device'] = df['device'].str.upper()

# Group by backbone-decoder-task
grouped = df.groupby(['backbone', 'decoder', 'task_name'])

components = {}
pipelines = {}
latency = {}
metric = {}

for i, ((backbone, decoder, task), group) in enumerate(grouped, start=1):
    # ---- components ----
    if backbone not in components:
        components[backbone] = {'mem': float(group['backbone memory(MB)'].iloc[0])}

    dec_key = f"{decoder}_{backbone}_{task}"
    components[dec_key] = {'mem': float(group['decoder memory(MB)'].iloc[0])}

    task_key = f"{task}_{backbone}_{decoder}"
    components[task_key] = {'mem': float(group['inference mem peak(MB)'].mean())}

    # ---- pipelines ----
    pid = f"p{i}"
    pipelines[pid] = {'backbone': backbone, 'decoder': decoder, 'task': task}

    # ---- latency (per device) ----
    lat_dict = {}
    for _, row in group.iterrows():
        lat_dict[row['device']] = round(float(row['inference time(ms)']), 5)
    latency[pid] = lat_dict

    # ---- metric ----
    metric[pid] = round(float(group['result'].mean()), 2)

def write_one_line_dict(name, data, f):
    f.write(f"{name}={{\n")
    for k, v in data.items():
        f.write(f"    {repr(k)}:{repr(v)},\n")
    f.write("}\n\n")

with open("profiler.py", "w") as f:
    write_one_line_dict('components', components,f)
    write_one_line_dict('pipelines', pipelines,f)
    write_one_line_dict('latency', latency, f)
    write_one_line_dict('metric', metric,  f)

