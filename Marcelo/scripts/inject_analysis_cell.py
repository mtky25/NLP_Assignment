"""Inject the analysis code into cell 22 of marcelo_notebook.ipynb."""
import json
from pathlib import Path

NB_PATH = Path(r"C:\Users\marce\Desktop\Code\NLP_Assignment\marcelo_notebook.ipynb")

CODE = r'''# ── Load metrics_only.xlsx and run the trade-off analysis ────────────────
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

XLSX = os.path.join(os.getcwd(), "metrics_only.xlsx")
df = pd.read_excel(XLSX)
df = df.sort_values("Accuracy", ascending=False).reset_index(drop=True)

# Compute Pareto frontier (maximise Accuracy, minimise Time, minimise RAM)
def _pareto(df):
    A, T, R = df["Accuracy"].values, df["Mean Time (s)"].values, df["Total RAM (GB)"].values
    flags = []
    for i in range(len(df)):
        dom = False
        for j in range(len(df)):
            if i == j: continue
            if (A[j] >= A[i] and T[j] <= T[i] and R[j] <= R[i] and
                (A[j] > A[i] or T[j] < T[i] or R[j] < R[i])):
                dom = True; break
        flags.append(not dom)
    return flags
df["Pareto"] = _pareto(df)

# Ranking table
display_cols = ["Experiment", "Accuracy", "Mean Time (s)", "Total RAM (GB)",
                "Acc / GB", "Efficiency (acc/(t·GB))", "Composite Score", "Pareto"]
ranking = df[display_cols].copy()
ranking_display = ranking.style.format({
    "Accuracy": "{:.2%}", "Acc / GB": "{:.2%}",
    "Mean Time (s)": "{:.2f}", "Total RAM (GB)": "{:.2f}",
    "Efficiency (acc/(t·GB))": "{:.4f}", "Composite Score": "{:.3f}",
}).background_gradient(subset=["Accuracy", "Composite Score"], cmap="RdYlGn") \
  .background_gradient(subset=["Mean Time (s)", "Total RAM (GB)"], cmap="RdYlGn_r")
display(ranking_display)

# ── Plots ─────────────────────────────────────────────────────────────────
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3})
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Model comparison — Accuracy × Time × RAM", fontsize=14, fontweight="bold")

# (1) Bubble: Accuracy × Time × RAM
ax = axes[0, 0]
colors = ["#2ca02c" if p else "#1f77b4" for p in df["Pareto"]]
sizes  = (df["Total RAM (GB)"] * 50).values
ax.scatter(df["Mean Time (s)"], df["Accuracy"], s=sizes, c=colors,
           alpha=0.65, edgecolors="black", linewidths=0.7)
for _, r in df.iterrows():
    ax.annotate(r["Experiment"].split("/")[0][:12],
                (r["Mean Time (s)"], r["Accuracy"]),
                fontsize=7, xytext=(4, 4), textcoords="offset points")
ax.set_xlabel("Mean time per question (s)")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy × Time (bubble = RAM, green = Pareto)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

# (2) Composite Score bar
ax = axes[0, 1]
order = df.sort_values("Composite Score", ascending=True)
bar_colors = ["#2ca02c" if p else "#7393b3" for p in order["Pareto"]]
ax.barh(order["Experiment"], order["Composite Score"], color=bar_colors, edgecolor="black", linewidth=0.5)
ax.set_xlabel("Composite Score (0.5·acc − 0.25·time − 0.25·RAM, normalised)")
ax.set_title("Composite score ranking")
for i, v in enumerate(order["Composite Score"]):
    ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=7)

# (3) Heatmap — accuracy per theme
ax = axes[0, 2]
theme_cols = ["Ent Acc.", "Sci Acc.", "Anc Acc.", "Math Acc.", "News Acc.", "Phil Acc."]
theme_labels = ["Ent", "Sci", "Anc", "Math", "News", "Phil"]
mat = df[theme_cols].values
im = ax.imshow(mat, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(len(theme_labels))); ax.set_xticklabels(theme_labels)
ax.set_yticks(range(len(df)));            ax.set_yticklabels(df["Experiment"], fontsize=7)
ax.set_title("Accuracy per theme")
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        ax.text(j, i, f"{mat[i,j]:.0%}", ha="center", va="center",
                fontsize=7, color="black")
plt.colorbar(im, ax=ax, fraction=0.04)

# (4) Stacked bar — time decomposition
ax = axes[1, 0]
order2 = df.sort_values("Mean Time (s)")
ax.barh(order2["Experiment"], order2["Mean Search (s)"], color="#4c78a8", label="Search (RAG)")
ax.barh(order2["Experiment"], order2["Mean Reasoning (s)"],
        left=order2["Mean Search (s)"], color="#f58518", label="Reasoning")
other = order2["Mean Time (s)"] - order2["Mean Search (s)"] - order2["Mean Reasoning (s)"]
ax.barh(order2["Experiment"], other.clip(lower=0),
        left=order2["Mean Search (s)"] + order2["Mean Reasoning (s)"],
        color="#bab0ab", label="Other")
ax.set_xlabel("Mean time per question (s)")
ax.set_title("Time decomposition")
ax.legend(loc="lower right", fontsize=8)

# (5) Pareto scatter — Accuracy × RAM
ax = axes[1, 1]
ax.scatter(df["Total RAM (GB)"], df["Accuracy"], s=80,
           c=["#2ca02c" if p else "#7393b3" for p in df["Pareto"]],
           edgecolors="black", linewidths=0.7, alpha=0.85)
for _, r in df.iterrows():
    ax.annotate(r["Experiment"].split("/")[0][:12],
                (r["Total RAM (GB)"], r["Accuracy"]),
                fontsize=7, xytext=(4, 4), textcoords="offset points")
pareto_pts = df[df["Pareto"]].sort_values("Total RAM (GB)")
ax.plot(pareto_pts["Total RAM (GB)"], pareto_pts["Accuracy"],
        color="#2ca02c", linestyle="--", linewidth=1.5, label="Pareto frontier")
ax.set_xlabel("Total RAM (GB)")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy × RAM with Pareto frontier")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.legend(loc="lower right", fontsize=8)

# (6) Bar — Accuracy per theme, averaged across all experiments
ax = axes[1, 2]
theme_avg = df[theme_cols].mean()
ax.bar(theme_labels, theme_avg.values, color="#9b59b6", edgecolor="black", linewidth=0.5)
ax.set_ylabel("Mean accuracy across all 10 experiments")
ax.set_title("Theme difficulty profile")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
for i, v in enumerate(theme_avg.values):
    ax.text(i, v + 0.01, f"{v:.0%}", ha="center", fontsize=8)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.show()

# ── Textual summary ───────────────────────────────────────────────────────
best_acc   = df.iloc[0]
best_score = df.sort_values("Composite Score", ascending=False).iloc[0]
best_eff   = df.sort_values("Efficiency (acc/(t·GB))", ascending=False).iloc[0]
lightest   = df.sort_values("Total RAM (GB)").iloc[0]
fastest    = df.sort_values("Mean Time (s)").iloc[0]
pareto_set = df[df["Pareto"]]["Experiment"].tolist()
gemma_rows = df[df["Inference Model"] == "gemma3:4b"]
gemma_spread = (gemma_rows["Accuracy"].max() - gemma_rows["Accuracy"].min()) * 100 if len(gemma_rows) else 0

print("=" * 78)
print("KEY FINDINGS")
print("=" * 78)
print(f"• Highest accuracy : {best_acc['Experiment']:<28} {best_acc['Accuracy']:.2%}  "
      f"({best_acc['Mean Time (s)']:.2f}s, {best_acc['Total RAM (GB)']:.2f} GB)")
print(f"• Best composite   : {best_score['Experiment']:<28} score={best_score['Composite Score']:.3f}")
print(f"• Best efficiency  : {best_eff['Experiment']:<28} eff={best_eff['Efficiency (acc/(t·GB))']:.4f}")
print(f"• Lightest in RAM  : {lightest['Experiment']:<28} {lightest['Total RAM (GB)']:.2f} GB")
print(f"• Fastest          : {fastest['Experiment']:<28} {fastest['Mean Time (s)']:.2f} s/question")
print(f"• Pareto frontier  : {', '.join(pareto_set)}")
print(f"• Math bottleneck  : average accuracy across all 10 experiments = "
      f"{df['Math Acc.'].mean():.1%} (worst theme)")
print(f"• Fallback impact  : keeping gemma3:4b as inference and varying the fallback "
      f"swings accuracy by {gemma_spread:.1f} pp")
'''

nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
target = nb["cells"][22]
assert target["cell_type"] == "code", f"cell 22 is {target['cell_type']}"
target["source"] = CODE.splitlines(keepends=True)
target["outputs"] = []
target["execution_count"] = None
NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("Injected analysis code into cell 22.")
