components={
    'momentlarge':{'mem': 1461.95456, 'type': 'moment'},
    'mlp_momentlarge_diasbp':{'mem': 0.525824},
    'diasbp_momentlarge_mlp':{'mem': 1405.481},
}

pipelines={
    'p1':{'backbone': 'momentlarge', 'decoder': 'mlp', 'task': 'diasbp'},
}

latency={
    'p1':{'NVIDIA A2': {5: {1: {'avg_latency_ms': 33.4097, 'backbone_mean_ms': 32.6868, 'backbone_ms_per_sample': 32.6868, 'throughput_rps': 29.931}, 2: {'avg_latency_ms': 45.6198, 'backbone_mean_ms': 44.9209, 'backbone_ms_per_sample': 22.4605, 'throughput_rps': 43.841}, 4: {'avg_latency_ms': 84.0004, 'backbone_mean_ms': 83.3146, 'backbone_ms_per_sample': 20.8287, 'throughput_rps': 47.619}, 8: {'avg_latency_ms': 170.002, 'backbone_mean_ms': 168.8225, 'backbone_ms_per_sample': 21.1028, 'throughput_rps': 47.058}, 16: {'avg_latency_ms': 328.6938, 'backbone_mean_ms': 327.5337, 'backbone_ms_per_sample': 20.4709, 'throughput_rps': 48.678}, 32: {'avg_latency_ms': 674.8761, 'backbone_mean_ms': 673.7541, 'backbone_ms_per_sample': 21.0548, 'throughput_rps': 47.416}}, 2: {1: {'avg_latency_ms': 56.5583, 'backbone_mean_ms': 55.7563, 'backbone_ms_per_sample': 55.7563, 'throughput_rps': 17.681}, 2: {'avg_latency_ms': 90.9899, 'backbone_mean_ms': 90.2092, 'backbone_ms_per_sample': 45.1046, 'throughput_rps': 21.98}, 4: {'avg_latency_ms': 173.3413, 'backbone_mean_ms': 172.5594, 'backbone_ms_per_sample': 43.1399, 'throughput_rps': 23.076}, 8: {'avg_latency_ms': 341.1485, 'backbone_mean_ms': 340.2168, 'backbone_ms_per_sample': 42.5271, 'throughput_rps': 23.45}, 16: {'avg_latency_ms': 671.9638, 'backbone_mean_ms': 671.0483, 'backbone_ms_per_sample': 41.9405, 'throughput_rps': 23.811}, 32: {'avg_latency_ms': 1342.0269, 'backbone_mean_ms': 1341.03, 'backbone_ms_per_sample': 41.9072, 'throughput_rps': 23.845}}, 1: {1: {'avg_latency_ms': 104.6139, 'backbone_mean_ms': 103.8112, 'backbone_ms_per_sample': 103.8112, 'throughput_rps': 9.559}, 2: {'avg_latency_ms': 172.9937, 'backbone_mean_ms': 172.2048, 'backbone_ms_per_sample': 86.1024, 'throughput_rps': 11.561}, 4: {'avg_latency_ms': 335.2359, 'backbone_mean_ms': 334.4493, 'backbone_ms_per_sample': 83.6123, 'throughput_rps': 11.932}, 8: {'avg_latency_ms': 663.8259, 'backbone_mean_ms': 663.0025, 'backbone_ms_per_sample': 82.8753, 'throughput_rps': 12.051}, 16: {'avg_latency_ms': 1314.6674, 'backbone_mean_ms': 1313.7519, 'backbone_ms_per_sample': 82.1095, 'throughput_rps': 12.17}, 32: {'avg_latency_ms': 2631.8504, 'backbone_mean_ms': 2630.8233, 'backbone_ms_per_sample': 82.2132, 'throughput_rps': 12.159}}}},
}

metric={
}

