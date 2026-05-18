# CEDIS Chief Complaint Classification Integration

## Overview

The CEDIS (Canadian Emergency Department Information System) classifier has been integrated into the medevac_db pipeline. It classifies free-text chief complaints (`ReasonforVisitDSC`) into standardized CEDIS codes and categories.

## Location

- **Classifier code:** `src/medevac_db/cedis_classifier/`
- **Documentation:** `docs/CEDIS_CLASSIFIER_GUIDE.md` and `docs/CEDIS_CLASSIFIER_SUMMARY.md`
- **Classification scripts:** `scripts/classify_chiefcomplaints*.py`

## What It Does

Takes chief complaints like:
- "Chest pain and shortness of breath"
- "Motor vehicle accident - head injury"
- "Pregnancy - contractions"

And classifies them to:
- **CEDIS Code:** e.g., `[003]`, `[407]`, `[458]`
- **Description:** e.g., "Chest pain—cardiac features"
- **Category:** e.g., "Cardiac", "Trauma", "Obstetric/Gynecologic"
- **Confidence:** 0.0-1.0 score
- **Method:** How it was classified (rule/fuzzy/tfidf)

## Classification Methods

The classifier uses a multi-method ensemble:

1. **Rule-based matching** - Expert patterns for high-confidence cases
2. **Fuzzy string matching** - Handles typos using `rapidfuzz`
3. **TF-IDF semantic similarity** - Captures meaning through term frequency
4. **Medical terminology expansion** - Expands abbreviations (MVA, SOB, CTX, etc.)

## Usage

### For PHI Data (with medevac)

```bash
# After running process_all_data.py
python scripts/classify_chiefcomplaints.py
```

This classifies chief complaints in:
- `data/final/encounters_for_deid.csv`
- `data/final/ed_encounters_for_deid.csv`
- `data/final/encounters_ed_merged_for_deid.csv`

Output: Same files with `_cedis` suffix and 5 new columns:
- `cedis_code`
- `cedis_description`
- `cedis_category`
- `cedis_confidence`
- `cedis_method`

### For Commercial Data

```bash
# After running process_commercial_data.py
python scripts/classify_chiefcomplaints_commercial.py
```

This classifies chief complaints in:
- `data/final/commercial/encounters_for_deid.csv`
- `data/final/commercial/ed_encounters_for_deid.csv`
- `data/final/commercial/encounters_ed_merged_for_deid.csv`

## Output Format

Original encounters file + 5 new columns:

| Column | Description |
|--------|-------------|
| `cedis_code` | CEDIS V2.0 code (e.g., "003", "407") |
| `cedis_description` | Full description (e.g., "Chest pain—cardiac features") |
| `cedis_category` | High-level category (e.g., "Cardiac", "Trauma") |
| `cedis_confidence` | Confidence score 0.0-1.0 |
| `cedis_method` | Classification method (rule/fuzzy/tfidf/none) |

## Integration with Pipeline

### Option 1: Manual Run (Current)

Run CEDIS classification as a separate step after the main pipeline:

```bash
# PHI pipeline
python scripts/process_all_data.py
python scripts/classify_chiefcomplaints.py

# Commercial pipeline
python scripts/process_commercial_data.py
python scripts/classify_chiefcomplaints_commercial.py
```

### Option 2: Automatic Integration (Future)

Add CEDIS classification as a pipeline step in `process_all_data.py` and `process_commercial_data.py`.

## Dependencies

The classifier requires:
- `rapidfuzz>=3.0` - Fast fuzzy string matching
- `scikit-learn>=1.0` - TF-IDF semantic similarity
- `tqdm>=4.65` - Progress bars

These are now in `pyproject.toml` and `requirements.txt`.

## CEDIS Code Structure

180+ codes across 15 categories:
- **Cardiac** - Chest pain, palpitations, cardiac arrest
- **Trauma** - Burns, bites, assault, falls, head injury, MVA
- **Respiratory** - SOB, cough, hemoptysis
- **Neurologic** - Headache, seizure, stroke symptoms
- **Obstetric/Gynecologic** - Pregnancy, vaginal bleeding
- **Gastrointestinal** - Abdominal pain, N/V, GI bleeding
- **Genitourinary** - Dysuria, hematuria, flank pain
- **And more...**

## Validation & Tuning

The classifier was developed and tested in the medevac_pipeline_project. For this integration:

1. **Test on sample data first:**
   ```python
   from medevac_db.cedis_classifier import CEDISClassifier
   
   classifier = CEDISClassifier()
   result = classifier.classify("chest pain and SOB", return_top_k=3)
   
   for pred in result:
       print(f"[{pred['cedis_code']}] {pred['description']}")
       print(f"  Confidence: {pred['confidence']:.3f} | Method: {pred['method']}")
   ```

2. **Review confidence distributions** - Check the log output after classification

3. **Add Alaska-specific patterns** if needed - Edit `src/medevac_db/cedis_classifier/rule_matcher.py`

## Benefits for This Project

✅ **Unified processing** - Works on both PHI and commercial data  
✅ **Standardized codes** - Enables cross-dataset comparison  
✅ **No LLM required** - Fast, deterministic, works offline  
✅ **High quality** - Multi-method ensemble with medical terminology  
✅ **Auditable** - Shows confidence and classification method  
✅ **Maintainable** - Easy to add custom rules for Alaska context

## Next Steps

1. **Test on your data** - Run on a few encounter files and review
2. **Check classifications** - Are they reasonable? What's the confidence distribution?
3. **Add custom patterns** if needed - Alaska-specific terms or local patterns
4. **Decide integration** - Keep as separate step or add to main pipeline?
5. **Update analysis scripts** - Use CEDIS categories in downstream analysis

## Questions?

See the detailed guides:
- `docs/CEDIS_CLASSIFIER_GUIDE.md` - Quick start and usage
- `docs/CEDIS_CLASSIFIER_SUMMARY.md` - Technical implementation details
- `src/medevac_db/cedis_classifier/README.md` - API documentation
