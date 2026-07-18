import math
import random
import numpy as np
from collections import Counter, defaultdict
import nltk
from nltk.corpus import brown

nltk.download("brown", quiet=True)

raw_sents = brown.sents()
sents = []
for s in raw_sents:
    words = [w.lower() for w in s if w.isalpha()]
    if words:
        sents.append(["<s>"] + words + ["</s>"])

random.seed(42)
random.shuffle(sents)
split_idx = int(0.8 * len(sents))
train_sents = sents[:split_idx]
test_sents = sents[split_idx:]

vocab = set()
for s in train_sents:
    for w in s:
        if w != "<s>":
            vocab.add(w)
vocab_list = sorted(list(vocab))
V = len(vocab_list)

# Q6
unigram_counts = Counter()
for s in train_sents:
    for w in s:
        if w != "<s>":
            unigram_counts[w] += 1

N_train = sum(unigram_counts.values())

bigram_counts = defaultdict(Counter)
context_counts = Counter()
for s in train_sents:
    for i in range(len(s) - 1):
        w1, w2 = s[i], s[i + 1]
        bigram_counts[w1][w2] += 1
        context_counts[w1] += 1

print("Q6: MLE Estimates")
print("Top 5 Unigrams:")
for w, c in unigram_counts.most_common(5):
    print(f"  {w}: count={c}, prob={c / N_train:.5f}")

print("Top 5 Bigrams:")
all_bigrams = []
for w1 in bigram_counts:
    for w2, c in bigram_counts[w1].items():
        all_bigrams.append((w1, w2, c))
all_bigrams.sort(key=lambda x: x[2], reverse=True)
for w1, w2, c in all_bigrams[:5]:
    prob = c / context_counts[w1]
    print(f"  {w1} -> {w2}: count={c}, prob={prob:.5f}")
print("\n" + "=" * 50 + "\n")

# Q7
print("Q7: Laplace Smoothing explanation")
print(
    "Raw MLE is insufficient for unseen bigrams because any test bigram not seen in training"
)
print(
    "gets probability 0, causing the entire sequence probability to be 0 and perplexity to be infinity."
)
print(
    "Laplace smoothing adds 1 to all bigram counts (and |V| to denominator) to guarantee non-zero probabilities."
)
print("\n" + "=" * 50 + "\n")

# Q8
log_sum_uni = 0.0
total_words = 0
for s in test_sents:
    for w in s[1:]:
        count = unigram_counts.get(w, 0)
        prob = (count + 1) / (N_train + V)
        log_sum_uni += math.log2(prob)
        total_words += 1
pp_uni = 2 ** (-log_sum_uni / total_words)

log_sum_bi = 0.0
total_bi = 0
for s in test_sents:
    for i in range(1, len(s)):
        w1, w2 = s[i - 1], s[i]
        c_bi = bigram_counts[w1].get(w2, 0)
        c_ctx = context_counts.get(w1, 0)
        prob = (c_bi + 1) / (c_ctx + V)
        log_sum_bi += math.log2(prob)
        total_bi += 1
pp_bi = 2 ** (-log_sum_bi / total_bi)

print("Q8: Model Perplexities")
print(f"{'Model':<20}{'Perplexity':<10}")
print("-" * 30)
print(f"{'Unigram (Laplace)':<20}{pp_uni:.2f}")
print(f"{'Bigram (Laplace)':<20}{pp_bi:.2f}")
print("\n" + "=" * 50 + "\n")

# Q9
print("Q9: Generated Sentences (Bigram Ancestral Sampling)")
np.random.seed(42)
random.seed(42)
for sentence_idx in range(5):
    tokens = []
    w1 = "<s>"
    for _ in range(50):
        successors = bigram_counts[w1]
        seen_words = list(successors.keys())
        ctx_count = context_counts.get(w1, 0)

        seen_probs = []
        for w in seen_words:
            c = successors[w]
            seen_probs.append((c + 1) / (ctx_count + V))

        num_unseen = V - len(seen_words)
        unseen_total_prob = num_unseen / (ctx_count + V)

        choices = seen_words + ["__UNSEEN__"]
        probs = seen_probs + [unseen_total_prob]
        probs = np.array(probs)
        probs = probs / probs.sum()

        chosen = np.random.choice(choices, p=probs)

        if chosen == "__UNSEEN__":
            while True:
                candidate = random.choice(vocab_list)
                if candidate not in successors:
                    w2 = candidate
                    break
        else:
            w2 = chosen

        if w2 == "</s>":
            break
        tokens.append(w2)
        w1 = w2
    print(f"  Sentence {sentence_idx + 1}: {' '.join(tokens)}")
print("\n" + "=" * 50 + "\n")

# Q10
trigram_counts = defaultdict(Counter)
trigram_context_counts = Counter()
for s in train_sents:
    padded = ["<s>"] + s
    for i in range(2, len(padded)):
        ctx = (padded[i - 2], padded[i - 1])
        w = padded[i]
        trigram_counts[ctx][w] += 1
        trigram_context_counts[ctx] += 1

log_sum_tri = 0.0
total_tri = 0
for s in test_sents:
    padded = ["<s>"] + s
    for i in range(2, len(padded)):
        ctx = (padded[i - 2], padded[i - 1])
        w = padded[i]
        c_tri = trigram_counts[ctx].get(w, 0)
        c_ctx = trigram_context_counts.get(ctx, 0)
        prob = (c_tri + 1) / (c_ctx + V)
        log_sum_tri += math.log2(prob)
        total_tri += 1
pp_tri = 2 ** (-log_sum_tri / total_tri)

print("Q10: Perplexity Comparison")
print(f"{'Model':<20}{'Perplexity':<10}")
print("-" * 30)
print(f"{'Unigram (Laplace)':<20}{pp_uni:.2f}")
print(f"{'Bigram (Laplace)':<20}{pp_bi:.2f}")
print(f"{'Trigram (Laplace)':<20}{pp_tri:.2f}")
