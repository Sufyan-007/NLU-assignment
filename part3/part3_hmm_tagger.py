import os
import math
import random
import numpy as np
from collections import Counter, defaultdict
import nltk
from nltk.corpus import brown

nltk.download('brown', quiet=True)
nltk.download('universal_tagset', quiet=True)

raw_sents = brown.tagged_sents(tagset='universal')
sents = [[(w.lower(), t) for w, t in s] for s in raw_sents]

random.seed(42)
random.shuffle(sents)
split_idx = int(0.8 * len(sents))
train_sents = sents[:split_idx]
test_sents = sents[split_idx:]

tag_set = set()
vocab = set()
word_tag_counts = defaultdict(set)

for sent in train_sents:
    for word, tag in sent:
        tag_set.add(tag)
        vocab.add(word)
        word_tag_counts[word].add(tag)

tags = sorted(list(tag_set))
num_tags = len(tags)
tag_to_idx = {tag: i for i, tag in enumerate(tags)}
idx_to_tag = {i: tag for i, tag in enumerate(tags)}
V_size = len(vocab) + 1

# Q11
transition_counts = defaultdict(Counter)
tag_counts = Counter()

emission_counts = defaultdict(Counter)
tag_word_totals = Counter()

start_counts = Counter()

for sent in train_sents:
    if len(sent) > 0:
        start_counts[sent[0][1]] += 1
    
    prev_tag = '<START>'
    for word, tag in sent:
        transition_counts[prev_tag][tag] += 1
        tag_counts[prev_tag] += 1
        
        emission_counts[tag][word] += 1
        tag_word_totals[tag] += 1
        
        prev_tag = tag

def get_trans_prob(t1, t2):
    count = transition_counts[t1].get(t2, 0)
    total = tag_counts[t1]
    return (count + 1) / (total + num_tags)

def get_emit_prob(t, w):
    count = emission_counts[t].get(w, 0)
    total = tag_word_totals[t]
    return (count + 1) / (total + V_size)

print("Q11: Transitions and Emissions")
print("Top 5 transitions from NOUN:")
noun_transitions = []
for t in tags:
    p = get_trans_prob('NOUN', t)
    noun_transitions.append((t, p))
noun_transitions.sort(key=lambda x: x[1], reverse=True)
for t, p in noun_transitions[:5]:
    c = transition_counts['NOUN'].get(t, 0)
    print(f"  NOUN -> {t}: count={c}, prob={p:.5f}")

print("\nTop 5 emissions from VERB:")
verb_emissions = emission_counts['VERB'].most_common(5)
for w, c in verb_emissions:
    p = get_emit_prob('VERB', w)
    print(f"  VERB -> {w}: count={c}, prob={p:.5f}")
print("\n" + "="*50 + "\n")

# Q12
def viterbi(words):
    n = len(words)
    if n == 0:
        return []
        
    score = np.full((num_tags, n), -np.inf)
    backpointer = np.zeros((num_tags, n), dtype=int)
    
    start_total = sum(start_counts.values())
    for j in range(num_tags):
        tag = idx_to_tag[j]
        s_count = start_counts.get(tag, 0)
        p_start = (s_count + 1) / (start_total + num_tags)
        p_emit = get_emit_prob(tag, words[0])
        score[j, 0] = math.log(p_start) + math.log(p_emit)
        
    for i in range(1, n):
        word = words[i]
        for j in range(num_tags):
            tag = idx_to_tag[j]
            p_emit = get_emit_prob(tag, word)
            
            best_val = -np.inf
            best_k = 0
            for k in range(num_tags):
                prev_tag = idx_to_tag[k]
                p_trans = get_trans_prob(prev_tag, tag)
                val = score[k, i-1] + math.log(p_trans)
                if val > best_val:
                    best_val = val
                    best_k = k
            
            score[j, i] = best_val + math.log(p_emit)
            backpointer[j, i] = best_k
            
    best_last = np.argmax(score[:, n-1])
    
    path = [0] * n
    path[-1] = best_last
    for i in range(n-2, -1, -1):
        path[i] = backpointer[path[i+1], i+1]
        
    return [idx_to_tag[idx] for idx in path]

print("Q12: Viterbi implementation is written as a function")
print("\n" + "="*50 + "\n")

# Q13
correct = 0
total = 0
correct_seen = 0
total_seen = 0
correct_oov = 0
total_oov = 0

errors = []

for idx, sent in enumerate(test_sents):
    words = [w for w, t in sent]
    gold = [t for w, t in sent]
    pred = viterbi(words)
    
    for i in range(len(words)):
        w = words[i]
        g = gold[i]
        p = pred[i]
        is_oov = (w not in vocab)
        
        total += 1
        if g == p:
            correct += 1
        else:
            errors.append({
                'word': w,
                'gold': g,
                'pred': p,
                'is_oov': is_oov
            })
            
        if is_oov:
            total_oov += 1
            if g == p:
                correct_oov += 1
        else:
            total_seen += 1
            if g == p:
                correct_seen += 1

print("Q13: Evaluation Results")
print(f"Overall Accuracy: {correct/total*100:.2f}% ({correct}/{total})")
print(f"Seen Words Accuracy: {correct_seen/total_seen*100:.2f}% ({correct_seen}/{total_seen})")
print(f"OOV Words Accuracy: {correct_oov/total_oov*100:.2f}% ({correct_oov}/{total_oov})")
print("\n" + "="*50 + "\n")

# Q14
print("Q14: Error Analysis (10 Mis-tagged Tokens)")
print(f"Total error tokens: {len(errors)}")
oov_errors = [e for e in errors if e['is_oov']]
ambig_errors = [e for e in errors if not e['is_oov'] and len(word_tag_counts[e['word']]) > 1]
other_errors = [e for e in errors if not e['is_oov'] and len(word_tag_counts[e['word']]) <= 1]

print(f"OOV errors: {len(oov_errors)} ({len(oov_errors)/len(errors)*100:.1f}%)")
print(f"Ambiguity errors: {len(ambig_errors)} ({len(ambig_errors)/len(errors)*100:.1f}%)")
print(f"Other errors: {len(other_errors)} ({len(other_errors)/len(errors)*100:.1f}%)")
print()

selected = []
for e in oov_errors[:3]:
    selected.append((e['word'], e['gold'], e['pred'], "OOV word - emission is uniform, transition determines tag"))
for e in ambig_errors[:4]:
    possible_tags = sorted(list(word_tag_counts[e['word']]))
    selected.append((e['word'], e['gold'], e['pred'], f"Lexical ambiguity - can be {', '.join(possible_tags)}"))
for e in other_errors[:3]:
    selected.append((e['word'], e['gold'], e['pred'], "Sparse transition context or tag sequence mismatch"))

print(f"{'#':<4}{'Token':<18}{'Gold':<8}{'Pred':<8}{'Reason'}")
print("-" * 80)
for i, (w, g, p, r) in enumerate(selected, 1):
    print(f"{i:<4}{w:<18}{g:<8}{p:<8}{r}")
