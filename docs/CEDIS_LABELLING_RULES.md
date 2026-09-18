# CEDIS Gold-Standard Labelling Rules

These rules were developed during manual gold-standard labelling of the medevac chief complaint dataset. They resolve common ambiguities where the raw free-text does not uniquely identify a CEDIS code.

**First matching rule wins.** If no rule applies, use the most specific CEDIS code that fits and avoid 866 (Minor complaints NOS) when a better option exists.

---

## General

| Presentation | CEDIS Code |
|---|---|
| Establish care | 866 Minor complaints NOS |
| Clearance to fly, return-to-work note | 866 Minor complaints NOS |
| Joint aches or pains (non-specific) | 866 Minor complaints NOS |
| Well visit / physical / WWV / annual check | 889 Well visit |

---

## Trauma

| Presentation | CEDIS Code |
|---|---|
| Falls with no other clear descriptor | 802 Major trauma—blunt |
| ATV or MVC without injury descriptor | 802 Major trauma—blunt |
| ATV or MVC with injury descriptor AND EtOH/AMS present | 802 Major trauma—blunt |
| Trauma involving >1 anatomical sites (upper, lower, head, torso/abdomen) | 802 Major trauma—blunt |
| "Trauma 1" or "Trauma 2" (triage designation) | 802 Major trauma—blunt |
| Assault without injury descriptor | 802 Major trauma—blunt |

---

## Cardiovascular / Vital Signs

| Presentation | CEDIS Code |
|---|---|
| Chest pain without any qualifier | 3 Chest pain—cardiac features |
| Hypotension / low blood pressure | 7 General weakness |
| Dehydrated / dehydration | 7 General weakness |
| Very sick child / ill-appearing child | 7 General weakness |
| AICD or defibrillator fired | 8 Syncope/pre-syncope |

---

## Neurological / Psychiatric

| Presentation | CEDIS Code |
|---|---|
| Dizziness | 403 Vertigo |
| Altered mental status / AMS | 401 Altered level of consciousness |
| Hanging / rope around neck | 351 Depression/suicidal/deliberate self-harm |
| Overdose in context of suicidal attempt | 752 Overdose ingestion |

---

## Respiratory / Infectious

| Presentation | CEDIS Code |
|---|---|
| Pneumonia | 651 Shortness of breath |
| Epiglottitis | 651 Shortness of breath |
| Sinus infection / sinusitis | 154 URTI complaints |
| Sepsis without localising symptoms | 852 Fever |
| Diarrhoea and fever | 852 Fever |
| COVID testing or exposure | 851 Exposure to communicable disease |

---

## Gastrointestinal

| Presentation | CEDIS Code |
|---|---|
| Gastroenteritis | 257 Nausea and/or vomiting |
| Vomiting and diarrhoea | 257 Nausea and/or vomiting |
| GI bleed without clear haematemesis or BRBPR | 260 Blood in stool/melena |
| Small bowel obstruction (SBO) | 251 Abdominal pain |
| CHAM sheet complaint (Digestive, Musculoskeletal) without specific descriptor | 999 Unknown |

---

## Genitourinary / Reproductive

| Presentation | CEDIS Code |
|---|---|
| STI testing | 303 Genital discharge/lesion |
| Penis pain | 304 Penile swelling |
| Prenatal follow-up (no gestational age given) | 458 Pregnancy issues, >20 weeks |
| Postpartum blood pressure check or postpartum visit | 458 Pregnancy issues, >20 weeks |

---

## Administrative / Follow-up

| Presentation | CEDIS Code |
|---|---|
| Follow-up or return visit for a known/chronic condition, no new complaint | 888 Follow-up/Return Visit |
| Video/telehealth consult, referral, named provider, or NP | 891 Planned telehealth |
| Pre-operative visit | 855 Direct referral for consultation |
| Status post (s/p) surgical procedure | 865 Post-operative complications |

---

## Local Code Extensions

Standard CEDIS V2.0 ends at code 869. Codes **888–899** are reserved as local extensions for this project, defined in `data/cedis_codes.csv` under the domain **General and Minor**. Codes 870–887 are left as a buffer for any future standard CEDIS additions.

### Code reservation table

| Range | Status |
|---|---|
| 801–806 | Standard CEDIS — Trauma |
| 851–869 | Standard CEDIS — General and Minor |
| 870–887 | Buffer — reserved for future standard CEDIS expansion |
| **888–899** | **Local extensions — this project** |
| 999 | Unknown / unclassifiable |

### Assigned local codes

| Code | Label | Rationale |
|---|---|---|
| 888 | Follow-up/Return Visit | Unified code for any follow-up or return visit for a known/chronic condition, with no new presenting complaint. Originally split out a "specific planned service" case as 890 Specified care; that split was retired — all follow-up/return visits, planned-service or not, now unify under 888. |
| 889 | Well visit | Well-child, physical, WWV, annual check — no acute complaint |
| 891 | Planned telehealth | Video/telehealth consult, referral, named provider, or NP visit — consolidates what was previously split across 855 (Direct referral for consultation) for these presentations |
| 892–899 | *Unassigned* | Available for future local extensions |

**Retired:** `890 Specified care` — folded into 888 (see rationale above). Not present in `data/cedis_codes.csv`; do not reuse this number for a different concept, since historical runs before this change may still reference it in old output CSVs.

These rules are also embedded in the LLM system prompts as the `CODING_RULES` constant in `scripts/llm_cedis_second_labeller.py` and imported by `scripts/llm_cedis_panel.py`.
