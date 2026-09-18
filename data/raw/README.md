# data/raw/ — PHI input data (gitignored)

This directory holds the raw, non-de-identified chief complaint CSV pulled from the source system on the PHI machine. **No files in this directory are committed to git.**

## Expected file

Place your raw chief complaint export here and point `local_paths.cfg` at it:

```
[paths]
raw_cc     = data/raw/chief_complaints_phi.csv
cc_column  = ReasonforVisitDSC
```

The file must have at minimum a free-text chief complaint column (name configured via `cc_column`).
