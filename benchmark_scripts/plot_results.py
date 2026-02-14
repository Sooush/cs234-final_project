import json
import matplotlib.pyplot as plt
import re
import os


file_path = "./results_aime24_avg1-1/Qwen__Qwen2.5-7B-Instruct/samples_aime24_2026-02-11T18-36-40.550919.jsonl"
output_image = "aime24_avg1.png"

# --- CONFIGURATION ---
# Default to 1, or override via environment variable
ANSWERS_PER_QUEST = int(os.environ.get('ANSWERS_PER_QUEST', 1)) 

def load_and_clean_jsonl(file_path):
    """
    Parses a JSONL file line-by-line, stripping 'Problem' and 'Solution' 
    text to save memory.
    """
    data_list = []
    # Regex to strip large text blocks
    pattern = re.compile(r'("(Problem|Solution)":\s*")(\\.|[^"])*(")')
    
    with open(file_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            # Remove bulky text before parsing JSON
            cleaned_line = pattern.sub(r'\1\4', line)
            try:
                data_list.append(json.loads(cleaned_line))
            except json.JSONDecodeError as e:
                print(f"Skipping malformed line: {e}")
                
    return data_list


try:
    # data is now a list of dictionaries (one per line)
    data = load_and_clean_jsonl(file_path)
    
    # Logic: If each line is a separate doc_id, we treat each line as 1 attempt.
    # We group them if there are multiple attempts per doc_id, 
    # or just plot them directly.
    
    problem_results = []
    for entry in data:
        # Using the key "exact_match" as provided in your snippet
        # Defaulting to 0 if the key is missing
        score = entry.get('exact_match', 0)
        #print(f"score {score}")
        problem_results.append(score)

    # --- Plotting ---
    plt.figure(figsize=(12, 6))
    x_axis = range(1, len(problem_results) + 1)
    
    # Visual Coding: Green for 1 (Correct), Red for 0 (Incorrect)
    colors = ['#2ecc71' if s >= 1 else '#e74c3c' for s in problem_results]
    
    plt.bar(x_axis, problem_results, color=colors, edgecolor='black', alpha=0.8)
    
    plt.xlabel("AIME24 Problem Number (doc_id)")
    plt.ylabel("Result (1 = Correct, 0 = Incorrect)")
    plt.title(f"Model Performance per doc_id (Configured: {ANSWERS_PER_QUEST} per quest)")
    
    plt.xticks(x_axis)
    plt.yticks([0, 1]) # Since it's a binary match
    
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    #plt.show()

    # Save to file
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    print(f"Success! Plot saved as {output_image}")
	
    # Close the plot
    plt.close()

    # Summary Stats
    total_correct = sum(problem_results)
    print(f"--- Summary ---")
    print(f"Total Problems Processed: {len(problem_results)}")
    print(f"Total Correct: {total_correct}")
    print(f"Accuracy: {(total_correct/len(problem_results))*100:.2f}%")

except Exception as e:
    print(f"Error processing file: {e}")
