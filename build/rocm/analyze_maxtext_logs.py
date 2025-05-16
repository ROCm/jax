import json, re, glob
import numpy as np

summary = {}
for log in glob.glob("logs_*.log"):
    model = log.replace("logs_", "").replace(".log", "")
    times = []
    with open(log) as f:
        for line in f:
            m = re.search(r"completed step: \d+, seconds: ([\d.]+)", line)
            if m:
                times.append(float(m.group(1)))
    if times:
        summary[model] = {
            "median_step_time": round(float(np.median(times)), 3),
            "steps_counted": len(times)
        }

with open("summary.json", "w") as f:
    json.dump(summary, f, indent=2)
