# Evaluating Multiple Samples with a Single Cartridge

## Overview

Yes! The evaluation script now fully supports evaluating multiple samples at the same time with a single cartridge trained on the whole dataset. The script will:

1. Process all entries in your JSON file sequentially
2. Track results per entry (so you can see how each fact performs)
3. Aggregate results across all entries
4. Export per-entry summaries

## How It Works

When you load a JSON file with multiple AKEW entries, the script:

1. **Processes each entry**: Generates test cases for each fact
2. **Tracks per-entry results**: Stores metrics separately for each entry
3. **Aggregates overall**: Calculates summary statistics across all entries
4. **Exports detailed breakdowns**: Provides both per-entry and overall summaries

## Example Usage

### Basic: Evaluate Multiple Samples

```bash
# Load a JSON file with multiple entries (e.g., test3.json with 3 facts)
python comprehensive_eval.py \
    --data-file samples/test3.json \
    --cartridge-ids "your-cartridge-id-trained-on-all-data" \
    --base-url http://localhost:10210 \
    --output results/multi_sample_eval.json
```

### What You Get

The script will generate:

1. **Console Output**:
   - Progress for each entry (Entry 1/3, Entry 2/3, etc.)
   - Per-entry summary table
   - Overall summary across all entries

2. **JSON Export** (`results/multi_sample_eval.json`):
   ```json
   {
     "summary": {
       "Efficacy": {...},
       "Generalization": {...},
       "Locality": {...},
       "Portability": {...},
       "OVERALL": {...}
     },
     "detailed_results": [...],  // All test cases
     "results_by_type": {...},   // Grouped by test type
     "results_by_entry": {       // NEW: Per-entry breakdown
       "8": {
         "entry_id": 8,
         "subject": "Wellington",
         "results": [...],
         "num_tests": 15
       },
       "14780": {
         "entry_id": 14780,
         "subject": "Another Subject",
         "results": [...],
         "num_tests": 15
       }
     },
     "entry_summaries": {        // NEW: Per-entry metrics
       "8": {
         "entry_id": 8,
         "subject": "Wellington",
         "rouge_l": 0.85,
         "bert": 0.92,
         "exact_match_rate": 80.0,
         "success_rate": 85.0,
         ...
       }
     }
   }
   ```

3. **CSV Exports**:
   - `*_summary.csv`: Overall metrics by test type
   - `*_per_entry_summary.csv`: **NEW** - Metrics for each entry separately
   - `*_detailed.csv`: All test cases with entry_id and entry_subject columns

## Example: Evaluating 3 Facts

Let's say you have `test3.json` with 3 facts:

```json
[
  {
    "case_id": 8,
    "requested_rewrite": {
      "subject": "Wellington",
      "target_new": {"str": "Sheffield"},
      "target_true": {"str": "Sydney"},
      ...
    },
    ...
  },
  {
    "case_id": 9,
    "requested_rewrite": {
      "subject": "Go Hyeon-jeong",
      "target_new": {"str": "French"},
      "target_true": {"str": "Korean"},
      ...
    },
    ...
  },
  {
    "case_id": 10,
    "requested_rewrite": {
      "subject": "Another Subject",
      ...
    },
    ...
  }
]
```

Run:
```bash
python comprehensive_eval.py \
    --data-file samples/test3.json \
    --cartridge-ids "your-cartridge-id" \
    --output results/three_facts_eval.json
```

**Output includes**:
- Per-entry metrics: How well each of the 3 facts was edited
- Overall metrics: Average across all 3 facts
- Detailed breakdown: Which test cases passed/failed for each fact

## Per-Entry Summary Table

When evaluating multiple entries, you'll see:

```
 PER-ENTRY SUMMARY 
================================================================================

Entry 8 (Wellington):
  Overall        | ROUGE-L: 0.8523 | BERTScore: 0.9123 | EM: 80.0% | Judge: 4.2/5.0 | Success: 85.0% | Old Leak: 5.0%

Entry 9 (Go Hyeon-jeong):
  Overall        | ROUGE-L: 0.7845 | BERTScore: 0.8934 | EM: 75.0% | Judge: 3.9/5.0 | Success: 80.0% | Old Leak: 10.0%

Entry 10 (Another Subject):
  Overall        | ROUGE-L: 0.8123 | BERTScore: 0.9012 | EM: 78.0% | Judge: 4.1/5.0 | Success: 82.0% | Old Leak: 8.0%

 COMPREHENSIVE EVALUATION SUMMARY 
================================================================================
...
```

## Use Cases

### 1. Compare Individual Facts
See which facts are edited better/worse:
```bash
# Check per_entry_summary.csv
cat results/multi_sample_eval_per_entry_summary.csv
```

### 2. Identify Problem Cases
Find entries with low success rates:
```python
import json
results = json.load(open('results/multi_sample_eval.json'))
for entry_id, summary in results['entry_summaries'].items():
    if summary['success_rate'] < 80:
        print(f"Low success: {summary['subject']} ({summary['success_rate']:.1f}%)")
```

### 3. Aggregate Analysis
Get overall performance across all facts:
```python
overall = results['summary']['OVERALL']
print(f"Overall Success Rate: {overall['success_rate']:.1f}%")
```

## Important Notes

1. **Single Cartridge**: All entries are evaluated with the same cartridge ID(s)
2. **Sequential Processing**: Entries are processed one at a time (not parallel)
3. **Independent Evaluation**: Each entry's test cases are evaluated independently
4. **Aggregated Results**: Overall metrics are averages across all entries

## Performance

- **Time**: Scales linearly with number of entries
- **API Calls**: One per test case (can be many for multiple entries)
- **Memory**: All results stored in memory (fine for reasonable dataset sizes)

## Tips

1. **Use `--quiet` for batch processing**:
   ```bash
   python comprehensive_eval.py --data-file samples/all_facts.json --quiet --no-judge
   ```

2. **Export to analyze later**:
   ```bash
   python comprehensive_eval.py --data-file samples/test3.json --output results/batch_eval.json
   ```

3. **Compare entries**:
   ```python
   import pandas as pd
   df = pd.read_csv('results/multi_sample_eval_per_entry_summary.csv')
   print(df.sort_values('Success Rate %', ascending=False))
   ```
