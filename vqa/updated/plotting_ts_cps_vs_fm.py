import matplotlib.pyplot as plt

# Data
pipelines = [1,2,3,4,5,6,7,8,9,10]

# Individual pipeline memory (no sharing, cumulative)
memory_no_sharing = [
    296.39,  # A
    296.39 + 373.71,  # A+B
    296.39 + 373.71 + 810.22,  # A+B+C
    296.39 + 373.71 + 810.22 + 679.37,  # A+B+C+D
    296.39 + 373.71 + 810.22 + 679.37 + 373.71,  # +E
    296.39 + 373.71 + 810.22 + 679.37 + 373.71 + 608.80,  # +F
    296.39 + 373.71 + 810.22 + 679.37 + 373.71 + 608.80 + 777.10,  # +G
    296.39 + 373.71 + 810.22 + 679.37 + 373.71 + 608.80 + 777.10 + 692.95,  # +H
    296.39 + 373.71 + 810.22 + 679.37 + 373.71 + 608.80 + 777.10 + 692.95 + 777.10,  # +I
    296.39 + 373.71 + 810.22 + 679.37 + 373.71 + 608.80 + 777.10 + 692.95 + 777.10 + 810.22 # +J
]

# YOLO shared across pipelines (cumulative)
yolo = 262.40
memory_yolo_shared = [
    memory_no_sharing[0],  # A
    memory_no_sharing[1] - yolo,  # A+B
    memory_no_sharing[2] - (2*yolo),  # A+B+C
    memory_no_sharing[3] - (3*yolo),  # A+B+C+D
    memory_no_sharing[4] - (4*yolo),  # A+B+C+D+E
    memory_no_sharing[5] - (5*yolo),  # +F
    memory_no_sharing[6] - (6*yolo),  # +G
    memory_no_sharing[7] - (7*yolo),  # +H
    memory_no_sharing[8] - (8*yolo),  # +I
    memory_no_sharing[9] - (9*yolo)   # +J
]

# Moondream single model memory (constant)
moondream_memory = [3691.65] * len(pipelines)

# Plot clearly
plt.figure(figsize=(12, 7))

plt.plot(pipelines, memory_no_sharing, marker='o', linestyle='-', label='TS Models (No YOLO shared)')
plt.plot(pipelines, memory_yolo_shared, marker='s', linestyle='--', label='TS Models (YOLO shared)')
plt.plot(pipelines, moondream_memory, marker='^', linestyle='-.', label='Moondream')

# Labels & aesthetics
plt.xlabel('Cumulative Pipelines', fontsize=14)
plt.ylabel('GPU Memory Utilization (MB)', fontsize=14)
# plt.title('GPU Memory Comparison: Task-Specific vs. Foundation Model (Moondream)', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

plt.tight_layout()
plt.show()
plt.savefig("ts_vs_fm.png")