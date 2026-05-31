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
    "Sentence Transformer only (+ Wikipedia RAG)": {
        "text": ("bi_cross_encoder_text.xlsx", "excel"),
        "speech": ("bi_cross_encoder_speech.xlsx", "excel")
    },
    "LLM Reasoning (+ Wikipedia RAG)": {
        "text": ("../Gianpaolo/gianpaolo_text_results.csv", "csv"),
        "speech": ("../Gianpaolo/gianpaolo_speech_results.csv", "csv")
    },
    "All Local (LLM on CPU + Local Vector-DB RAG)": {
        "combined": ("metrics_only.xlsx", "excel")
    }
}

cat_label_map = {
    'entertainment_mean_question_accuracy': 'Entertainment',
    'ancient_history_mean_question_accuracy': 'History',
    'science_nature_mean_question_accuracy': 'Science',
    'maths_mean_question_accuracy': 'Maths',
    'news_mean_question_accuracy': 'News',
    'philosophy_psychology_mean_question_accuracy': 'Philosophy',
    'Ent Acc.': 'Entertainment',
    'Sci Acc.': 'Science',
    'Anc Acc.': 'History',
    'Math Acc.': 'Maths',
    'News Acc.': 'News',
    'Phil Acc.': 'Philosophy'
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
        
        # Get canonical size from text mode
        text_df = df[df['Mode'] == 'text']
        base_size = None
        if not text_df.empty:
            best_text = text_df.loc[text_df['Accuracy'].idxmax()]
            base_size = best_text.get('inference_model_size')
            if pd.isna(base_size) or base_size is None:
                base_size = extract_size(best_text.get('Inference Model'))
        
        for mode in ["text", "speech"]:
            mode_df = df[df['Mode'] == mode]
            if not mode_df.empty:
                acc_col = 'Accuracy'
                best_row = mode_df.loc[mode_df[acc_col].idxmax()]
                
                # Use base_size if available, otherwise fallback to row size
                size = base_size if base_size is not None else best_row.get('inference_model_size')
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
                    "reasoning_time": best_row.get('Mean Reasoning (s)'),
                    "transcription_time": best_row.get('Mean Transcription (s)'),
                    "cat_times": {
                        "total": [best_row[c] for c in ['Ent Time (s)', 'Sci Time (s)', 'Anc Time (s)', 'Math Time (s)', 'News Time (s)', 'Phil Time (s)'] if c in best_row and not pd.isna(best_row[c])],
                        "search": [best_row[c] for c in ['Ent Search (s)', 'Sci Search (s)', 'Anc Search (s)', 'Math Search (s)', 'News Search (s)', 'Phil Search (s)'] if c in best_row and not pd.isna(best_row[c])],
                        "reasoning": [best_row[c] for c in ['Ent Reason (s)', 'Sci Reason (s)', 'Anc Reason (s)', 'Math Reason (s)', 'News Reason (s)', 'Phil Reason (s)'] if c in best_row and not pd.isna(best_row[c])],
                        "transcription": []
                    }
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
                        "reasoning_time": best_row.get('mean_reasoning_time'),
                        "transcription_time": best_row.get('mean_transcription_time'),
                        "cat_times": {
                            "total": [best_row[c] for c in ['entertainment_mean_time', 'ancient_history_mean_time', 'science_nature_mean_time', 'maths_mean_time', 'news_mean_time', 'philosophy_psychology_mean_time'] if c in best_row and not pd.isna(best_row[c])],
                            "search": [best_row[c] for c in ['entertainment_mean_search_time', 'ancient_history_mean_search_time', 'science_nature_mean_search_time', 'maths_mean_search_time', 'news_mean_search_time', 'philosophy_psychology_mean_search_time'] if c in best_row and not pd.isna(best_row[c])],
                            "reasoning": [best_row[c] for c in ['entertainment_mean_reasoning_time', 'ancient_history_mean_reasoning_time', 'science_nature_mean_reasoning_time', 'maths_mean_reasoning_time', 'news_mean_reasoning_time', 'philosophy_psychology_mean_reasoning_time'] if c in best_row and not pd.isna(best_row[c])],
                            "transcription": [best_row[c] for c in ['entertainment_mean_transcription_time', 'ancient_history_mean_transcription_time', 'science_nature_mean_transcription_time', 'maths_mean_transcription_time', 'news_mean_transcription_time', 'philosophy_psychology_mean_transcription_time'] if c in best_row and not pd.isna(best_row[c])]
                        }
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

# Shared plotting parameters
GROUP_OFFSET = 0.2 # Increased offset for better separation in log space

# Unified Color Palette (Pleasant Pastels)
C_RED = '#e3bcc1'    # Speech Mode, Total Time, Approach 1
C_BLUE = '#99c9df'   # Text Mode, Reasoning, Approach 2
C_GREEN = '#c8f8e8'  # Search, Approach 3
C_ORANGE = '#ffdfbf' # Transcription
C_PURPLE = '#C1A7E2' # Category 5
C_GOLD = '#91979e'   # Category 6

APPROACH_COLORS = [C_RED, C_BLUE, C_GREEN]
CAT_COLORS = [C_RED, C_BLUE, C_GREEN, C_ORANGE, C_PURPLE, C_GOLD]
TIMING_COLORS = [C_ORANGE, C_GREEN, C_BLUE, C_RED] # Transcription, Search, Reasoning, Total

# --- Plot 1: Box Plot (Original) ---
fig1, ax1 = plt.subplots(figsize=(14, 8))
ax1.set_xscale('log')
ax1.axhline(y=random_baseline, color='grey', linestyle='--', alpha=0.6, label=f'Random Guesser Baseline\nMean Acc: {random_baseline:.3f}')

for i, approach in enumerate(sorted_text_best_with_meta):
    item = approach["text"]
    pos = item['size']
    same_size_count = sum(1 for x in sorted_text_best_with_meta[:i] if x['text']['size'] == item['size'])
    # Center the FIRST model on the tick, and shift subsequent ones to the RIGHT
    center_pos = np.log10(pos) + (same_size_count * GROUP_OFFSET)
    pos_plot = 10**center_pos

    width = 0.2 * pos_plot
    bp = ax1.boxplot([item['category_accs']], positions=[pos_plot], widths=width, patch_artist=True, medianprops={'visible': False})
    color = APPROACH_COLORS[i % len(APPROACH_COLORS)]
    for patch in bp['boxes']:
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_label(f"{approach['name']}\nMean Acc: {item['overall_acc']:.3f}")

ax1.set_xlabel('Model Size (Billion Parameters) - Log Scale')
ax1.set_ylabel('Question Accuracy')
ax1.set_title('Model Performance vs. Model Size (text mode)')
ax1.grid(True, which="both", ls="-", alpha=0.3)
ax1.set_ylim(0, 1.05)
ax1.set_xticks(ticks)
ax1.set_xticklabels(tick_labels, rotation=45)
ax1.set_xlim(min(ticks) * 0.4, max(ticks) * 2.5)
ax1.legend(title="Approaches", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('performance_plot.png')

# --- Plot 2: Category Bar Chart ---
fig2, ax2 = plt.subplots(figsize=(14, 8))
ax2.set_xscale('log')
ax2.axhline(y=random_baseline, color='grey', linestyle='--', alpha=0.6, label=f'Random Guesser Baseline ({random_baseline:.3f})')
unique_categories = sorted(list(set(cat for item in sorted_text_best for cat, acc in item['category_results'])))
cat_color_map = {cat: color for cat, color in zip(unique_categories, CAT_COLORS)}
num_cats = len(unique_categories)
log_width_cat = 0.04 

for i, approach in enumerate(sorted_text_best_with_meta):
    item = approach["text"]
    pos = item['size']
    same_size_count = sum(1 for x in sorted_text_best_with_meta[:i] if x['text']['size'] == item['size'])
    center_pos = np.log10(pos) + (same_size_count * GROUP_OFFSET)
    start_pos = center_pos - (num_cats * log_width_cat) / 2
    item_cats = {cat: acc for cat, acc in item['category_results']}
    for j, cat in enumerate(unique_categories):
        if cat in item_cats:
            acc = item_cats[cat]
            bar_pos = 10**(start_pos + j * log_width_cat + log_width_cat/2)
            bar_width = bar_pos * 0.08 
            ax2.bar(bar_pos, acc, width=bar_width, color=cat_color_map[cat], label=cat if i == 0 else "") 

    # Add approach label dynamically above the bars
    ax2.text(10**center_pos, max(item['category_accs']) + 0.02,
             approach["name"].replace('\n', ' '), ha='center', va='bottom', fontsize=9)

ax2.set_xlabel('Model Size (Billion Parameters) - Log Scale')
ax2.set_ylabel('Question Accuracy')
ax2.set_title('Model Performance per Category (text mode)')
ax2.grid(True, which="both", ls="-", alpha=0.2)
ax2.set_ylim(0, 1.05)
ax2.set_xticks(ticks)
ax2.set_xticklabels(tick_labels, rotation=45)
ax2.set_xlim(min(ticks) * 0.4, max(ticks) * 2.5)
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
ax3.axhline(y=random_baseline, color='grey', linestyle='--', alpha=0.6, label='Random Guesser Baseline')

for i, approach in enumerate(sorted_text_best_with_meta):
    if approach["text"] is None or approach["speech"] is None:
        continue
    
    pos = approach["text"]["size"]
    same_size_count = sum(1 for x in sorted_text_best_with_meta[:i] if x['text']['size'] == pos)
    center_pos = np.log10(pos) + (same_size_count * GROUP_OFFSET)
    
    # Text Box
    text_pos = 10**(center_pos - 0.05)
    width = text_pos * 0.1
    bp_text = ax3.boxplot([approach["text"]["category_accs"]], positions=[text_pos], widths=width, patch_artist=True, medianprops={'visible': False})
    for patch in bp_text['boxes']:
        patch.set_facecolor(C_BLUE) # Blue for text
        patch.set_alpha(0.7)
        if i == 0: patch.set_label('Text Mode')
    
    # Speech Box
    speech_pos = 10**(center_pos + 0.05)
    bp_speech = ax3.boxplot([approach["speech"]["category_accs"]], positions=[speech_pos], widths=width, patch_artist=True, medianprops={'visible': False})
    for patch in bp_speech['boxes']:
        patch.set_facecolor(C_RED) # Red for speech
        patch.set_alpha(0.7)
        if i == 0: patch.set_label('Speech Mode')
    
    # Add approach label above/near the boxes
    max_acc = max(max(approach["text"]["category_accs"]), max(approach["speech"]["category_accs"]))
    ax3.text(10**center_pos, max_acc + 0.02, 
             approach["name"].replace('\n', ' '), ha='center', va='bottom', fontsize=9)

ax3.set_xlabel('Model Size (Billion Parameters) - Log Scale')
ax3.set_ylabel('Question Accuracy')
ax3.set_title('Text Mode vs. Speech Mode - Performance Comparison')
ax3.grid(True, which="both", ls="-", alpha=0.2)
ax3.set_ylim(0, 1.1)
ax3.set_xticks(ticks)
ax3.set_xticklabels(tick_labels, rotation=45)
ax3.set_xlim(min(ticks) * 0.4, max(ticks) * 2.5)
ax3.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig('text_vs_speech_comparison.png')

# --- Plot 4: Timing Comparison ---
fig4, ax4 = plt.subplots(figsize=(14, 8))
ax4.set_xscale('log')
ax4.axhline(y=30, color='red', linestyle='--', alpha=0.6, label='30s Time Limit')

time_categories = ['transcription', 'search', 'reasoning', 'total']
time_labels = ['Transcription', 'Search', 'Reasoning', 'Total Time']
num_time_cats = len(time_categories)
log_width_time = 0.06

for i, approach in enumerate(sorted_text_best_with_meta):
    # Use speech mode for timing to show transcription if available
    item = approach["speech"] if approach["speech"] is not None else approach["text"]
    if item is None: continue
    
    pos = item['size']
    same_size_count = sum(1 for x in sorted_text_best_with_meta[:i] if x['text']['size'] == item['size'])
    # First model at size S is centered on S, subsequent ones shift right
    center_pos = np.log10(pos) + (same_size_count * GROUP_OFFSET)
    start_pos = center_pos - (num_time_cats * log_width_time) / 2
    
    max_h = 0
    for j, cat in enumerate(time_categories):
        mean_val = item.get(f"{cat}_time", 0)
        if mean_val is None: mean_val = 0
        
        cat_vals = item.get("cat_times", {}).get(cat, [])
        if cat_vals and len(cat_vals) > 0:
            max_h = max(max_h, max(cat_vals))
        else:
            max_h = max(max_h, mean_val)
        
        bar_pos = 10**(start_pos + j * log_width_time + log_width_time/2)
        bar_width = bar_pos * 0.1 # Restored original width (10% of position)
        
        ax4.bar(bar_pos, mean_val, width=bar_width, color=TIMING_COLORS[j], label=time_labels[j] if i == 0 else "")
        
        if cat_vals and len(cat_vals) > 0:
            ymin = min(cat_vals)
            ymax = max(cat_vals)
            ax4.errorbar(bar_pos, mean_val, yerr=[[max(0, mean_val - ymin)], [max(0, ymax - mean_val)]], 
                         fmt='none', ecolor='black', capsize=3, alpha=0.3)

    # Add approach label dynamically above the tallest bar/whisker
    ax4.text(10**center_pos, max_h + 1, approach["name"].replace('\n', ' '), ha='center', va='bottom', fontsize=9)

ax4.set_xlabel('Model Size (Billion Parameters) - Log Scale')
ax4.set_ylabel('Time (seconds)')
ax4.set_title('Inference Time Breakdown (speech mode)')
ax4.grid(True, which="both", ls="-", alpha=0.2)
ax4.set_ylim(0, 32) # Set y-lim to 32
ax4.set_xticks(ticks)
ax4.set_xticklabels(tick_labels, rotation=45)
ax4.set_xlim(min(ticks) * 0.4, max(ticks) * 2.5)

handles, labels = ax4.get_legend_handles_labels()
unique_handles_labels = {}
for h, l in zip(handles, labels):
    if l not in unique_handles_labels:
        unique_handles_labels[l] = h
ax4.legend(unique_handles_labels.values(), unique_handles_labels.keys(), title="Time Categories", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('timing_plot.png')

print("All 4 plots saved successfully.")
