import os
import matplotlib.pyplot as plt
from collections import Counter
from scipy import stats
import numpy as np
import nltk
from nltk.corpus import brown

nltk.download('brown', quiet=True)

words = [w.lower() for w in brown.words() if w.isalpha()]

script_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(script_dir, 'output')
os.makedirs(out_dir, exist_ok=True)

# Q1
word_counts = Counter(words)
sorted_counts = word_counts.most_common()
total_tokens = len(words)
vocab_size = len(word_counts)

print("Q1: Word Frequencies - Top 10 Words")
print(f"Total tokens (N): {total_tokens}")
print(f"Vocabulary size (|V|): {vocab_size}\n")
print(f"{'Rank':<8}{'Word':<15}{'Frequency':<10}{'Proportion':<10}")
for i, (word, freq) in enumerate(sorted_counts[:10], 1):
    prop = freq / total_tokens
    print(f"{i:<8}{word:<15}{freq:<10}{prop:.5f}")
print("\n" + "="*50 + "\n")

# Q2
ranks = np.arange(1, vocab_size + 1)
freqs = np.array([count for word, count in sorted_counts])
log_ranks = np.log10(ranks)
log_freqs = np.log10(freqs)

slope, intercept, r_value, p_value, std_err = stats.linregress(log_ranks, log_freqs)
print("Q2: Regression Results")
print(f"Slope: {slope:.4f}")
print(f"Intercept: {intercept:.4f}")
print(f"R-squared: {r_value**2:.4f}")
print("\n" + "="*50 + "\n")

plt.figure(figsize=(8, 6))
plt.scatter(log_ranks, log_freqs, s=2, alpha=0.5, label='Words')
plt.plot(log_ranks, slope * log_ranks + intercept, color='red', label=f'Fit (slope={slope:.3f})')
plt.xlabel('log10(Rank)')
plt.ylabel('log10(Frequency)')
plt.title("Zipf's Law (Brown Corpus)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'zipf_loglog.png'))
plt.close()

# Q3
print("Q3: Top 20 Words:")
for i, (w, f) in enumerate(sorted_counts[:20], 1):
    print(f"{i}. {w} ({f})")
print("\nBottom 20 Words:")
for i, (w, f) in enumerate(sorted_counts[-20:], vocab_size - 19):
    print(f"{i}. {w} ({f})")
print("\n" + "="*50 + "\n")

# Q4
sizes = [10000, 50000, 100000, 500000, 1000000]
sizes = [s for s in sizes if s <= len(words)]
if len(words) not in sizes:
    sizes.append(len(words))

ttrs = []
print("Q4: TTR vs. Corpus Size")
for size in sizes:
    sub_words = words[:size]
    unique_words = len(set(sub_words))
    ttr = unique_words / size
    ttrs.append(ttr)
    print(f"Corpus size: {size:<8} | Unique types: {unique_words:<6} | TTR: {ttr:.5f}")
print("\n" + "="*50 + "\n")

plt.figure(figsize=(8, 6))
plt.plot(sizes, ttrs, marker='o', color='blue')
plt.xlabel('Corpus Size (Tokens)')
plt.ylabel('TTR')
plt.title('TTR vs. Corpus Size')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'ttr_vs_size.png'))
plt.close()

# Q5
def get_mattr_fast(tokens, window_size=1000):
    n = len(tokens)
    if n < window_size:
        return len(set(tokens)) / n
    counts = Counter(tokens[:window_size])
    unique = len(counts)
    ttr_sum = unique / window_size
    for i in range(1, n - window_size + 1):
        old_w = tokens[i - 1]
        counts[old_w] -= 1
        if counts[old_w] == 0:
            unique -= 1
        new_w = tokens[i + window_size - 1]
        if counts[new_w] == 0:
            unique += 1
        counts[new_w] += 1
        ttr_sum += unique / window_size
    return ttr_sum / (n - window_size + 1)

mattrs = []
print("Q5: MATTR vs. Corpus Size (window=1000)")
for size in sizes:
    sub_words = words[:size]
    mattr_val = get_mattr_fast(sub_words, 1000)
    mattrs.append(mattr_val)
    print(f"Corpus size: {size:<8} | MATTR: {mattr_val:.5f}")
print("\n" + "="*50 + "\n")

plt.figure(figsize=(8, 6))
plt.plot(sizes, ttrs, marker='o', label='Standard TTR')
plt.plot(sizes, mattrs, marker='s', label='MATTR (w=1000)')
plt.xlabel('Corpus Size')
plt.ylabel('Ratio')
plt.title('Standard TTR vs. MATTR')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'mattr_vs_size.png'))
plt.close()
