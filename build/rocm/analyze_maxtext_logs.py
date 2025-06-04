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
        times_np = np.array(times)
        step_info = [{"step": n, "time": t} for n, t in enumerate(times)]
        summary[model] = {
            "steps": step_info,
            "min_step_time": round(float(np.min(times_np)), 3),
            "q25_step_time": round(float(np.percentile(times_np, 25)), 3),
            "median_step_time": round(float(np.median(times_np)), 3),
            "mean_step_time": round(float(np.mean(times_np)), 3),
            "q75_step_time": round(float(np.percentile(times_np, 75)), 3),
            "max_step_time": round(float(np.max(times_np)), 3),
            "steps_counted": len(times)
        }

with open("summary.json", "w") as f:
    json.dump(summary, f, indent=2)