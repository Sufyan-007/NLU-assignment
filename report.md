# NLU Assignment 1 Report

**Name:** Sufyan Gazdhar  
**Roll number:** G25AIT1174  
**Corpus:** Brown Corpus from NLTK  

---

## Part 1: Zipf’s Law and Corpus Analysis

### Q1: Word Frequencies
Tokenization of the Brown corpus was performed by lowercasing all words and removing punctuation or numerical digits to ensure that only alphabetical tokens are counted.
- Total tokens in corpus: 981,716
- Unique word types (vocabulary size): 40,234

The table below lists the top 10 most frequent words in the corpus:

| Rank | Word | Frequency | Proportion |
|---|---|---|---|
| 1 | the | 69,971 | 7.13% |
| 2 | of | 36,412 | 3.71% |
| 3 | and | 28,853 | 2.94% |
| 4 | to | 26,158 | 2.66% |
| 5 | a | 23,195 | 2.36% |
| 6 | in | 21,337 | 2.17% |
| 7 | that | 10,594 | 1.08% |
| 8 | is | 10,109 | 1.03% |
| 9 | was | 9,815 | 1.00% |
| 10 | he | 9,548 | 0.97% |

---

### Q2: Log-Log Plot and Fit
A plot was constructed representing the log of word rank on the x-axis and the log of word frequency on the y-axis. A linear regression line was fitted to this data.
- Slope: -1.3588
- Intercept: 6.1917
- R-squared: 0.9714

Zipf's law predicts a slope of approximately -1. The empirical slope of -1.3588 represents a moderate deviation from this prediction. This steeper decline is expected in clean corpora where punctuation and functional characters have been removed.

The plot is saved in `part1/output/zipf_loglog.png`.

---

### Q3: Top-20 vs. Bottom-20 Words
The top 20 words in the corpus are:
the, of, and, to, a, in, that, is, was, he, for, it, with, as, his, on, be, at, by, i.

The bottom 20 words consist of words appearing only once (hapax legomena), such as masterly, beggary, hubris, plumbed, bathos, besmirched, and extremis.

**Analysis of Grammatical Categories:**
- The top-20 words are closed-class words (function words), which include prepositions, determiners, pronouns, and conjunctions. These elements are syntactically necessary but lack lexical meaning.
- The bottom-20 words are open-class words (content words) like nouns, verbs, and adjectives.
- This patterns with the type-token distinction: closed-class categories have very few types but account for a vast majority of running tokens. Conversely, open-class words contribute most of the vocabulary types, but many of these types are rare and appear only as single tokens.

---

### Q4: TTR vs. Corpus Size
The Type-Token Ratio (TTR) was calculated at different slice sizes of the corpus:

| Corpus Size (Tokens) | Unique Words | TTR |
|---|---|---|
| 10,000 | 2,470 | 0.24700 |
| 50,000 | 8,036 | 0.16072 |
| 100,000 | 12,235 | 0.12235 |
| 500,000 | 30,049 | 0.06010 |
| 981,716 | 40,234 | 0.04098 |

TTR decreases significantly as the corpus size grows. This trend follows Heap's Law, which states that vocabulary size grows sub-linearly with respect to running tokens. As more text is processed, incoming words are increasingly repetitions of already-seen types rather than novel vocabulary.

The plot is saved in `part1/output/ttr_vs_size.png`.

---

### Q5: Standardized/Moving-Average TTR (MATTR)
MATTR was computed using a window size of 1000 tokens:

| Corpus Size (Tokens) | Standard TTR | MATTR (Window=1000) |
|---|---|---|
| 10,000 | 0.24700 | 0.44783 |
| 50,000 | 0.16072 | 0.46638 |
| 100,000 | 0.12235 | 0.46924 |
| 500,000 | 0.06010 | 0.45648 |
| 981,716 | 0.04098 | 0.43879 |

Unlike standard TTR, MATTR remains highly stable (averaging between 0.44 and 0.47) across all corpus sizes. By computing and averaging TTR over constant window sizes, MATTR controls for text length.

The plot is saved in `part1/output/mattr_vs_size.png`.

---

## Part 2: Language Model

### Q6: MLE Unigram & Bigram Estimates
- Training set: 80% (45,412 sentences)
- Test set: 20% (11,354 sentences)
- Vocabulary size: 36,888

**Top 5 Unigrams:**
1. the (count=55780, prob=0.06717)
2. `</s>` (count=45412, prob=0.05469)
3. of (count=29164, prob=0.03512)
4. and (count=23041, prob=0.02775)
5. to (count=20879, prob=0.02514)

**Top 5 Bigrams:**
1. of -> the (count=7723, prob=0.26481)
2. `<s>` -> the (count=5685, prob=0.12519)
3. in -> the (count=4880, prob=0.28463)
4. to -> the (count=2809, prob=0.13454)
5. `<s>` -> he (count=2402, prob=0.05289)

---

### Q7: Laplace Smoothing
MLE assigns a probability of 0 to unseen n-grams. When evaluating on a test set, a single zero-probability bigram causes the joint probability of the entire test sequence to become 0, making perplexity undefined. Laplace smoothing adds 1 to all transition counts to ensure all transitions receive a non-zero probability.

---

### Q8: Perplexity Table
The perplexity values calculated on the test set are as follows:

| Model | Perplexity |
|---|---|
| Unigram (with Laplace) | 1143.58 |
| Bigram (with Laplace) | 4972.39 |

**Discussion:**
In theory, bigram models should have lower perplexity because they incorporate contextual sequence information. However, with add-1 (Laplace) smoothing, bigram perplexity is higher. This occurs because add-1 smoothing adds 1 to every vocabulary word (36,888 words) for every context. This over-allocates probability mass to unseen transitions, leaving less probability for the actual seen transitions in the test set.

---

### Q9: Generated Sentences
Five sentences generated using ancestral sampling from the bigram model:

1. *nice coordinator arbeitskommando keith identical gynecological digiorgio concrete walking chromatography reviews atmospheres assented coconut greenhouse here threat arlen yokuts foully waite restorative grotesque sheered knot adjectives effie reynolds nubile kneeling dunes gortonists nobility concierge clutched portia combining parsimonious offal investors beautifully slimed validated danube polyphosphate cautions whit locales paulus fleetest*
2. *bygone bellow harvests lifting cennino heresy comprising portraying knight sickness percent eluded phraseology paid geological isolating calluses errors upswing idealization embroiled sniper populated jersey woodshed grimly morocco bolster hayes attachment microsomal rail izvestia bugs giggled mexicans glides telescoping punk slavs discounted involutions dignitaries imaging yori verloop interviewing rotary quite pawtucket*
3. *what grievances dice toilet symptomatic cliches berger contriving dreams egregiously reviving broil prepares poultry sparta units incomprehension widthwise aikin counters varied island nra copley locomotive saner eclectic sidney accompanying intrinsic tents exported till consitutional luxurious tilghman founded dreams plaintiff elemental versus unofficial aberration moriarty supernatural anglican corduroy peculiarly matlowsky horrifying*
4. *borrow however cauterize chisholm successive buttoned untidiness dawns dedifferentiated starve welcomed encomiums inwardly unformed riboflavin giving versed frequented membrane quiescent pittsboro schemes tripled shotgun culminate impending halt brooked nonsingular anthea williams heart groomed admit calibrated bounded hawksley burch ate naps caldwell transfered hone kob subtended godless versailles dennis squawk hushed*
5. *squirted recognized fishermen cohen comenico rural ozagen riddled registrants southerners blundered commonplaces brassnose rape note contradicts imply flashy firm validly sharpened dimensional reviewer fascination kochaneks snowy imputed carletonian seekonk wes commercially birdwhistell vied alloys coattails hobo enforcement recently subtypes strenuous glycosides ragging boughs empty pontiac abused promotion iodination simca leases*

---

### Q10: Trigram Model Perplexity and Discussion
The perplexity values for all three models:

| Model | Perplexity |
|---|---|
| Unigram (with Laplace) | 1143.58 |
| Bigram (with Laplace) | 4972.39 |
| Trigram (with Laplace) | 19180.68 |

The trigram model yields the highest perplexity. This is due to three factors:
1. **Sparsity:** The context count space is extremely large (36,888 squared). Almost all trigram contexts are unseen in training.
2. **Smoothing:** The high sparsity causes add-1 smoothing to dominate the distribution, distributing almost all probability mass uniformly across unseen words.
3. **History:** Conditioning on a longer history increases parameter variance. Without back-off or interpolation smoothing, this leads to overfitting on the training set.

---

## Part 3: HMM POS Tagger with Viterbi

### Q11: Transitions and Emissions
- Training set: 45,872 sentences
- Test set: 11,468 sentences
- POS tags: 12 (Universal Tagset)

**Top 5 transitions from NOUN:**
- NOUN -> . (count=62550, prob=0.28457)
- NOUN -> ADP (count=53785, prob=0.24469)
- NOUN -> VERB (count=34998, prob=0.15922)
- NOUN -> NOUN (count=33069, prob=0.15045)
- NOUN -> CONJ (count=13156, prob=0.05986)

**Top 5 emissions from VERB:**
- VERB -> is (count=8113, prob=0.04244)
- VERB -> was (count=7908, prob=0.04136)
- VERB -> be (count=5133, prob=0.02685)
- VERB -> had (count=4091, prob=0.02140)
- VERB -> are (count=3466, prob=0.01813)

---

### Q12: Viterbi Algorithm
The Viterbi algorithm was implemented from scratch using log-probabilities to avoid floating-point underflow. The tagger maintains:
1. A **score matrix** of size (number of tags) by (sentence length) representing the best path probabilities.
2. A **backpointer matrix** of the same size tracking the path transitions.

Optimal tags are resolved by backtracking from the highest scoring tag at the final word position.

---

### Q13: Accuracy Evaluation
Evaluation of the tagger on the 20% test split:

| Category | Correct | Total | Accuracy |
|---|---|---|---|
| Overall | 217,778 | 232,177 | 93.80% |
| Seen Words | 215,692 | 227,127 | 94.97% |
| OOV (Unseen) Words | 2,086 | 5,050 | 41.31% |

**Comments:**
Accuracy on seen words is significantly higher than on unseen words. Seen words have strong emission weights learned in training. For unseen words, the tag emission probabilities are uniform, meaning the tagger must rely entirely on transition probabilities, which limits prediction performance.

---

### Q14: Error Analysis
Total error tokens in the test set: 14,399
- OOV errors: 2,964 (20.6%)
- Ambiguity errors: 6,760 (46.9%)
- Transition/other errors: 4,675 (32.5%)

A selection of 10 representative errors:

| # | Token | Gold Tag | Pred Tag | Reason |
|---|---|---|---|---|
| 1 | non-partisan | ADJ | DET | Unseen word (OOV), tagged based on transition probabilities only |
| 2 | chi-chi | NOUN | . | Unseen word (OOV), tagged based on transition probabilities only |
| 3 | durrell's | NOUN | DET | Unseen word (OOV), tagged based on transition probabilities only |
| 4 | average | VERB | NOUN | Word can be ADJ, NOUN, or VERB; model selected the dominant tag |
| 5 | depending | ADP | VERB | Word can be ADP or VERB; transition scores favored VERB |
| 6 | very | ADV | ADJ | Word can be ADV or ADJ; transition scores favored ADJ |
| 7 | offer | NOUN | VERB | Lexical ambiguity between noun and verb usage |
| 8 | $200 | NOUN | ADP | Sparse transition pattern, rare tag sequence |
| 9 | tempting | VERB | NOUN | Sparse transition pattern, rare tag sequence |
| 10 | spartan | NOUN | DET | Sparse transition pattern, rare tag sequence |

**Summary:**
The most frequent errors are **ambiguity errors** (46.9%). Taggers struggle with words that function as multiple parts of speech depending on sentence context, and the limited bigram history is often insufficient to resolve these ambiguities correctly.
