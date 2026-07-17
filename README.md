# NLU Assignment 1

## Corpus

**Brown Corpus** (via NLTK) — approximately 1 million tokens, 500 text samples across 15 genres.
The corpus is downloaded automatically on first run.

## Prerequisites

- Python 3.8+
- pip

## Setup

```bash
pip install -r requirements.txt
```

## Running

### Part 1: Zipf's Law and Corpus Analysis (Q1–Q5)

```bash
python part1/part1_zipf.py
```

Outputs solutions for Q1–Q5 to the console.
Plots are saved to `part1/output/`.

### Part 2: Language Model (Q6–Q10)

```bash
python part2/part2_language_model.py
```

Outputs solutions for Q6–Q10 to the console.

### Part 3: HMM POS Tagger with Viterbi (Q11–Q14)

```bash
python part3/part3_hmm_tagger.py
```

Outputs solutions for Q11–Q14 to the console.

## Project Structure

```
NLU assignment/
├── requirements.txt
├── README.md
├── part1/
│   ├── part1_zipf.py
│   └── output/               # Generated plots (Q2, Q4, Q5)
├── part2/
│   ├── part2_language_model.py
│   └── output/
└── part3/
    ├── part3_hmm_tagger.py
    └── output/
```

## Notes

- All scripts use the Brown Corpus from NLTK with automatic download.
- Part 2 uses an 80/20 sentence-level train/test split (random seed 42).
- Part 3 uses the Universal POS tagset (12 tags) and implements Viterbi from scratch.
