# scripts/plot_training.py
import pandas as pd, matplotlib.pyplot as plt, os
CSV = os.path.join("logs","training_metrics.csv")

df = pd.read_csv(CSV)
plt.figure(figsize=(8,5))
plt.plot(df["episodes"], df["ep_rew_mean"])
plt.title("Recompensa media por episodio")
plt.xlabel("Episodios"); plt.ylabel("Mean reward"); plt.grid(True); plt.tight_layout()
out = os.path.join("logs","mean_reward.png")
plt.savefig(out, dpi=150)
print("Guardado:", out)
