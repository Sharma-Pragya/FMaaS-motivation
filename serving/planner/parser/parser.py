import pandas as pd

def write_one_line_dict(name, data, f):
    f.write(f"{name}={{\n")
    for k, v in data.items():
        f.write(f"    {repr(k)}:{repr(v)},\n")
    f.write("}\n\n")

def parse_csv_to_dict(csv_path):
    df = pd.read_csv(csv_path)

    # Normalize names
    df['backbone'] = df['backbone'].str.lower()
    df['decoder'] = df['decoder'].fillna('none')
    df['decoder'] = df['decoder'].str.lower()
    df['task_name'] = df['task_name'].str.lower()
    df['device'] = df['device'].str.upper()
    df['tpc_count'] = df['tpc_count'].astype(int)
    df['inference_batch_size'] = df['inference_batch_size'].astype(int)

    grouped = df.groupby(['backbone', 'decoder', 'task_name'])

    components = {}
    pipelines = {}
    latency = {}
    metric = {}

    BACKBONE_TYPES = {
        'chronos': ['chronostiny', 'chronosmini', 'chronossmall', 'chronosbase', 'chronoslarge'],
        'moment':  ['momentsmall', 'momentbase', 'momentlarge'],
        'papagei': ['papageip', 'papageis', 'papageissvri'],
    }
    _backbone_type_lookup = {b: t for t, bs in BACKBONE_TYPES.items() for b in bs}

    for i, ((backbone, decoder, task), group) in enumerate(grouped, start=1):
        # ---- components ----
        if backbone not in components:
            entry = {'mem': float(group['backbone_load_mem_mb'].iloc[0])}
            btype = _backbone_type_lookup.get(backbone)
            if btype:
                entry['type'] = btype
            components[backbone] = entry

        if decoder != 'none':
            dec_key = f"{decoder}_{backbone}_{task}"
            components[dec_key] = {'mem': float(group['decoder_load_mem_mb'].iloc[0])}

        if decoder == 'none':
            task_key = f"{task}_{backbone}"
        else:
            task_key = f"{task}_{backbone}_{decoder}"
        components[task_key] = {'mem': float(group['peak_gpu_mem_mb'].mean())}

        # ---- pipelines ----
        pid = f"p{i}"
        pipelines[pid] = {'backbone': backbone, 'decoder': decoder, 'task': task}

        # ---- latency: pid -> device -> tpc -> batch_size -> {avg_latency_ms, backbone_mean_ms, backbone_ms_per_sample, throughput_rps} ----
        lat_dict = {}
        for _, row in group.iterrows():
            dev = row['device']
            tpc = int(row['tpc_count'])
            bs = int(row['inference_batch_size'])
            lat_dict.setdefault(dev, {}).setdefault(tpc, {})[bs] = {
                'avg_latency_ms': round(float(row['avg_latency_ms']), 5),
                'backbone_mean_ms': round(float(row['backbone_mean_ms']), 5),
                'backbone_ms_per_sample': round(float(row['backbone_ms_per_sample']), 5),
                'throughput_rps': round(float(row['throughput_rps']), 5),
            }
        latency[pid] = lat_dict

    return components, pipelines, latency, metric

components = {}
pipelines = {}
latency = {}
metric = {}

for name in ["profile.csv"]:
    c, p, l, m = parse_csv_to_dict(name)
    components.update(c)
    pipelines.update(p)
    latency.update(l)
    metric.update(m)


with open("profiler.py", "w") as f:
    write_one_line_dict('components', components, f)
    write_one_line_dict('pipelines', pipelines, f)
    write_one_line_dict('latency', latency, f)
    write_one_line_dict('metric', metric, f)
