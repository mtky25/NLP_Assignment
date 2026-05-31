import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

# Set style
plt.style.use('seaborn-v0_8-deep')

def extract_size(model_name):
    if pd.isna(model_name):
        return None
    match = re.search(r'[:\-]?(\d+\.?\d*)b', str(model_name).lower())
    if match:
        return float(match.group(1))
    name_lower = str(model_name).lower()
    if "llama3.2" in name_lower: return 3
    if "gemma2" in name_lower: return 9
    if "gemma3" in name_lower: return 4
    return None

sheets_dir = "../plots/sheets/"
# approach_name -> {text_file, speech_file, type}
approaches_config = {
    "Sentence Transformer only\n(+ Wikipedia RAG)": {
        "text": ("bi_cross_encoder_text.xlsx", "excel"),
        "speech": ("bi_cross_encoder_speech.xlsx", "excel")
    },
    "LLM Reasoning (+ Wikipedia RAG)\n": {
        "text": ("../Gianpaolo/gianpaolo_text_results.csv", "csv"),
        "speech": ("../Gianpaolo/gianpaolo_speech_results.csv", "csv")
    },
    "All Local (LLM on CPU + \nLocal Vector-DB RAG)": {
        "combined": ("metrics_only.xlsx", "excel")
    }
}

cat_label_map = {
    'entertainment_mean_question_accuracy': 'Entertainment',
    'ancient_history_mean_question_accuracy': 'History',
    'science_nature_mean_question_accuracy': 'Science',
    'maths_mean_question_accuracy': 'Maths',
    'news_mean_question_accuracy': 'News',
    'philosophy_psychology_mean_question_accuracy': 'Philo',
    'Ent Acc.': 'Entertainment',
    'Sci Acc.': 'Science',
    'Anc Acc.': 'History',
    'Math Acc.': 'Maths',
    'News Acc.': 'News',
    'Phil Acc.': 'Philo'
}

cat_cols_std = [
    'entertainment_mean_question_accuracy',
    'ancient_history_mean_question_accuracy',
    'science_nature_mean_question_accuracy',
    'maths_mean_question_accuracy',
    'news_mean_question_accuracy',
    'philosophy_psychology_mean_question_accuracy'
]
cat_cols_metrics = [
    'Ent Acc.', 'Sci Acc.', 'Anc Acc.', 'Math Acc.', 'News Acc.', 'Phil Acc.'
]

all_comparison_data = []

for approach_name, config in approaches_config.items():
    approach_results = {"name": approach_name, "text": None, "speech": None}
    
    if "combined" in config:
        filename, filetype = config["combined"]
        path = f"{sheets_dir}{filename}"
        df = pd.read_excel(path)
        
        for mode in ["text", "speech"]:
            mode_df = df[df['Mode'] == mode]
            if not mode_df.empty:
                acc_col = 'Accuracy'
                best_row = mode_df.loc[mode_df[acc_col].idxmax()]
                
                size = best_row.get('inference_model_size')
                if pd.isna(size) or size is None:
                    size = extract_size(best_row.get('Inference Model'))
                
                valid_cols = [c for c in cat_cols_metrics if c in df.columns]
                category_results = [(cat_label_map.get(c, c), best_row[c]) for c in valid_cols if not pd.isna(best_row[c])]
                
                approach_results[mode] = {
                    "size": size,
                    "overall_acc": best_row[acc_col],
                    "category_accs": [r[1] for r in category_results],
                    "category_results": category_results,
                    "total_time": best_row.get('Mean Time (s)'),
                    "search_time": best_row.get('Mean Search (s)'),
                    "reasoning_time": best_row.get('Mean Reasoning (s)')
                }
    else:
        for mode in ["text", "speech"]:
            filename, filetype = config[mode]
            try:
                if filetype == "csv":
                    df = pd.read_csv(filename)
                    df.columns = df.columns.str.strip()
                else:
                    df = pd.read_excel(f"{sheets_dir}{filename}" if not filename.startswith("..") else filename)
                
                acc_col = 'mean_question_accuracy'
                if not df.empty:
                    best_row = df.loc[df[acc_col].idxmax()]
                    size = best_row.get('inference_model_size')
                    if pd.isna(size) or size is None:
                        size = extract_size(best_row.get('inference_model'))
                    
                    valid_cols = [c for c in cat_cols_std if c in df.columns]
                    category_results = [(cat_label_map.get(c, c), best_row[c]) for c in valid_cols if not pd.isna(best_row[c])]
                    
                    approach_results[mode] = {
                        "size": size,
                        "overall_acc": best_row[acc_col],
                        "category_accs": [r[1] for r in category_results],
                        "category_results": category_results,
                        "total_time": best_row.get('mean_time'),
                        "search_time": best_row.get('mean_search_time'),
                        "reasoning_time": best_row.get('mean_reasoning_time')
                    }
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    all_comparison_data.append(approach_results)

# Filter out approaches that don't have text mode
sorted_text_best_with_meta = sorted([d for d in all_comparison_data if d["text"] is not None], key=lambda x: x['text']['size'])
sorted_text_best = [d["text"] for d in sorted_text_best_with_meta]

df_random = pd.read_excel(f"{sheets_dir}random_benchmark_results.xlsx")
random_baseline = df_random['mean_question_accuracy'].max()

# Ticks for log scale
ticks = sorted(list(set([d["text"]["size"] for d in sorted_text_best_with_meta])))
tick_labels = [f"{int(t)}B" for t in ticks]

# --- Plot 1: Box Plot (Original) ---
fig1, ax1 = plt.subplots(figsize=(12, 7))
ax1.set_xscale('log')
ax1.axhline(y=random_baseline, color='grey', linestyle='--', alpha=0.6, label=f'Random Baseline\nMean Acc: {random_baseline:.3f}')

for i, approach in enumerate(sorted_text_best_with_meta):
    item = approach["text"]
    pos = item['size']
    same_size_count = sum(1 for x in sorted_text_best_with_meta[:i] if x['text']['size'] == item['size'])
    if same_size_count > 0:
        pos = pos * (1 + 0.15 * same_size_count) 

    width = 0.15 * pos 
    bp = ax1.boxplot([item['category_accs']], positions=[pos], widths=width, patch_artist=True, medianprops={'visible': False})
    color = plt.cm.viridis(i / len(sorted_text_best_with_meta))
    for patch in bp['boxes']:
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_label(f"{approach['name']}\nMean Acc: {item['overall_acc']:.3f}")

ax1.set_xlabel('Model Size (Billion Parameters) - Log Scale')
ax1.set_ylabel('Accuracy')
ax1.set_title('Model Performance vs. Model Size (text mode)')
ax1.grid(True, which="both", ls="-", alpha=0.3)
ax1.set_ylim(0, 1.05)
ax1.set_xticks(ticks)
ax1.set_xticklabels(tick_labels, rotation=45)
ax1.set_xlim(min(ticks) * 0.7, max(ticks) * 1.5)
ax1.legend(title="Approaches", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('performance_plot.png')

# --- Plot 2: Category Bar Chart ---
fig2, ax2 = plt.subplots(figsize=(14, 8))
ax2.set_xscale('log')
ax2.axhline(y=random_baseline, color='grey', linestyle='--', alpha=0.6, label=f'Random Guesser Baseline ({random_baseline:.3f})')
unique_categories = sorted(list(set(cat for item in sorted_text_best for cat, acc in item['category_results'])))
cat_colors = plt.cm.Set2(np.linspace(0, 1, len(unique_categories)))
cat_color_map = {cat: color for cat, color in zip(unique_categories, cat_colors)}
num_cats = len(unique_categories)
log_width = 0.04 

for i, approach in enumerate(sorted_text_best_with_meta):
    item = approach["text"]
    pos = item['size']
    same_size_count = sum(1 for x in sorted_text_best_with_meta[:i] if x['text']['size'] == item['size'])
    center_pos = np.log10(pos) + (same_size_count * 0.4) 
    start_pos = center_pos - (num_cats * log_width) / 2
    item_cats = {cat: acc for cat, acc in item['category_results']}
    for j, cat in enumerate(unique_categories):
        if cat in item_cats:
            acc = item_cats[cat]
            bar_pos = 10**(start_pos + j * log_width + log_width/2)
            bar_width = bar_pos * 0.08 
            ax2.bar(bar_pos, acc, width=bar_width, color=cat_color_map[cat], label=cat if i == 0 else "") 

    # Add approach label above/near the bars
    ax2.text(10**center_pos, max(item['category_accs']) + 0.02,
             approach["name"].replace('\n', ' '), ha='center', va='bottom', fontsize=9)

ax2.set_xlabel('Model Size (Billion Parameters) - Log Scale')
ax2.set_ylabel('Accuracy')
ax2.set_title('Model Performance per Category (text mode)')
ax2.grid(True, which="both", ls="-", alpha=0.2)
ax2.set_ylim(0, 1.05)
ax2.set_xticks(ticks)
ax2.set_xticklabels(tick_labels, rotation=45)
ax2.set_xlim(min(ticks) * 0.6, max(ticks) * 2.5) 
handles, labels = ax2.get_legend_handles_labels()
unique_handles_labels = {}
for h, l in zip(handles, labels):
    if l not in unique_handles_labels:
        unique_handles_labels[l] = h
ax2.legend(unique_handles_labels.values(), unique_handles_labels.keys(), title="Categories", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('category_performance_plot.png')

# --- Plot 3: Text vs Speech Comparison ---
fig3, ax3 = plt.subplots(figsize=(14, 8))
ax3.set_xscale('log')
ax3.axhline(y=random_baseline, color='grey', linestyle='--', alpha=0.6, label='Random Baseline')

for i, approach in enumerate(sorted_text_best_with_meta):
    if approach["text"] is None or approach["speech"] is None:
        continue
    
    pos = approach["text"]["size"]
    same_size_count = sum(1 for x in sorted_text_best_with_meta[:i] if x['text']['size'] == pos)
    center_pos = np.log10(pos) + (same_size_count * 0.4)
    
    # Text Box
    text_pos = 10**(center_pos - 0.05)
    width = text_pos * 0.1
    bp_text = ax3.boxplot([approach["text"]["category_accs"]], positions=[text_pos], widths=width, patch_artist=True, medianprops={'visible': False})
    for patch in bp_text['boxes']:
        patch.set_facecolor('#66b3ff') # Blue for text
        patch.set_alpha(0.7)
        if i == 0: patch.set_label('Text Mode')
    
    # Speech Box
    speech_pos = 10**(center_pos + 0.05)
    bp_speech = ax3.boxplot([approach["speech"]["category_accs"]], positions=[speech_pos], widths=width, patch_artist=True, medianprops={'visible': False})
    for patch in bp_speech['boxes']:
        patch.set_facecolor('#ff9999') # Red for speech
        patch.set_alpha(0.7)
        if i == 0: patch.set_label('Speech Mode')
    
    # Add approach label above/near the boxes
    ax3.text(10**center_pos, max(max(approach["text"]["category_accs"]), max(approach["speech"]["category_accs"])) + 0.02, 
             approach["name"].replace('\n', ' '), ha='center', va='bottom', fontsize=9)

ax3.set_xlabel('Model Size (Billion Parameters) - Log Scale')
ax3.set_ylabel('Accuracy')
ax3.set_title('Performance Comparison: Text Mode vs. Speech Mode')
ax3.grid(True, which="both", ls="-", alpha=0.2)
ax3.set_ylim(0, 1.1)
ax3.set_xticks(ticks)
ax3.set_xticklabels(tick_labels, rotation=45)
ax3.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig('text_vs_speech_comparison.png')

print("All 3 plots saved successfully.")
