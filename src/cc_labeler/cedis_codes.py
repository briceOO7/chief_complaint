"""
CEDIS Presenting Complaint Codes (V2.0)

Canadian Emergency Department Information System (CEDIS)
Source: Canadian Association of Emergency Physicians (CAEP)
        Canadian Institute for Health Information (CIHI)
Effective Date: April 2012
"""

CEDIS_CODES = {
    # Cardiovascular (001-012)
    "001": {"code": "001", "category": "Cardiovascular", "description": "Cardiac arrest (non-traumatic)"},
    "002": {"code": "002", "category": "Cardiovascular", "description": "Cardiac arrest (traumatic)"},
    "003": {"code": "003", "category": "Cardiovascular", "description": "Chest pain—cardiac features"},
    "004": {"code": "004", "category": "Cardiovascular", "description": "Chest pain—non-cardiac features"},
    "005": {"code": "005", "category": "Cardiovascular", "description": "Palpitations/irregular heart beat"},
    "006": {"code": "006", "category": "Cardiovascular", "description": "Hypertension"},
    "007": {"code": "007", "category": "Cardiovascular", "description": "General weakness"},
    "008": {"code": "008", "category": "Cardiovascular", "description": "Syncope/pre-syncope"},
    "009": {"code": "009", "category": "Cardiovascular", "description": "Edema, generalized"},
    "010": {"code": "010", "category": "Cardiovascular", "description": "Bilateral leg swelling/edema"},
    "011": {"code": "011", "category": "Cardiovascular", "description": "Cool pulseless limb"},
    "012": {"code": "012", "category": "Cardiovascular", "description": "Unilateral reddened hot limb"},

    # ENT - Ears (051-056)
    "051": {"code": "051", "category": "ENT - Ears", "description": "Earache"},
    "052": {"code": "052", "category": "ENT - Ears", "description": "Foreign body, ear"},
    "053": {"code": "053", "category": "ENT - Ears", "description": "Loss of hearing"},
    "054": {"code": "054", "category": "ENT - Ears", "description": "Tinnitus"},
    "055": {"code": "055", "category": "ENT - Ears", "description": "Discharge, ear"},
    "056": {"code": "056", "category": "ENT - Ears", "description": "Ear injury"},

    # ENT - Mouth/Throat/Neck (101-107)
    "101": {"code": "101", "category": "ENT - Mouth/Throat/Neck", "description": "Dental/gum problem"},
    "102": {"code": "102", "category": "ENT - Mouth/Throat/Neck", "description": "Facial trauma"},
    "103": {"code": "103", "category": "ENT - Mouth/Throat/Neck", "description": "Sore throat"},
    "104": {"code": "104", "category": "ENT - Mouth/Throat/Neck", "description": "Neck swelling/pain"},
    "105": {"code": "105", "category": "ENT - Mouth/Throat/Neck", "description": "Neck trauma"},
    "106": {"code": "106", "category": "ENT - Mouth/Throat/Neck", "description": "Difficulty swallowing/dysphagia"},
    "107": {"code": "107", "category": "ENT - Mouth/Throat/Neck", "description": "Facial pain (non-traumatic/non-dental)"},

    # ENT - Nose (151-155)
    "151": {"code": "151", "category": "ENT - Nose", "description": "Epistaxis"},
    "152": {"code": "152", "category": "ENT - Nose", "description": "Nasal congestion/hay fever"},
    "153": {"code": "153", "category": "ENT - Nose", "description": "Foreign body, nose"},
    "154": {"code": "154", "category": "ENT - Nose", "description": "URTI complaints"},
    "155": {"code": "155", "category": "ENT - Nose", "description": "Nasal trauma"},

    # Environmental (201-206)
    "201": {"code": "201", "category": "Environmental", "description": "Frostbite/cold injury"},
    "202": {"code": "202", "category": "Environmental", "description": "Noxious inhalation"},
    "203": {"code": "203", "category": "Environmental", "description": "Electrical injury"},
    "204": {"code": "204", "category": "Environmental", "description": "Chemical exposure"},
    "205": {"code": "205", "category": "Environmental", "description": "Hypothermia"},
    "206": {"code": "206", "category": "Environmental", "description": "Near drowning"},

    # Gastrointestinal (251-267)
    "251": {"code": "251", "category": "Gastrointestinal", "description": "Abdominal pain"},
    "252": {"code": "252", "category": "Gastrointestinal", "description": "Anorexia"},
    "253": {"code": "253", "category": "Gastrointestinal", "description": "Constipation"},
    "254": {"code": "254", "category": "Gastrointestinal", "description": "Diarrhea"},
    "255": {"code": "255", "category": "Gastrointestinal", "description": "Foreign body in rectum"},
    "256": {"code": "256", "category": "Gastrointestinal", "description": "Groin pain/mass"},
    "257": {"code": "257", "category": "Gastrointestinal", "description": "Nausea and/or vomiting"},
    "258": {"code": "258", "category": "Gastrointestinal", "description": "Rectal/perineal pain"},
    "259": {"code": "259", "category": "Gastrointestinal", "description": "Vomiting blood"},
    "260": {"code": "260", "category": "Gastrointestinal", "description": "Blood in stool/melena"},
    "261": {"code": "261", "category": "Gastrointestinal", "description": "Jaundice"},
    "262": {"code": "262", "category": "Gastrointestinal", "description": "Hiccoughs"},
    "263": {"code": "263", "category": "Gastrointestinal", "description": "Abdominal mass/distention"},
    "264": {"code": "264", "category": "Gastrointestinal", "description": "Anal/rectal trauma"},
    "265": {"code": "265", "category": "Gastrointestinal", "description": "Oral/esophageal foreign body"},
    "266": {"code": "266", "category": "Gastrointestinal", "description": "Feeding difficulties in newborn"},
    "267": {"code": "267", "category": "Gastrointestinal", "description": "Neonatal jaundice"},

    # Genitourinary (301-310)
    "301": {"code": "301", "category": "Genitourinary", "description": "Flank pain"},
    "302": {"code": "302", "category": "Genitourinary", "description": "Hematuria"},
    "303": {"code": "303", "category": "Genitourinary", "description": "Genital discharge/lesion"},
    "304": {"code": "304", "category": "Genitourinary", "description": "Penile swelling"},
    "305": {"code": "305", "category": "Genitourinary", "description": "Scrotal pain and/or swelling"},
    "306": {"code": "306", "category": "Genitourinary", "description": "Urinary retention"},
    "307": {"code": "307", "category": "Genitourinary", "description": "UTI complaints"},
    "308": {"code": "308", "category": "Genitourinary", "description": "Oliguria"},
    "309": {"code": "309", "category": "Genitourinary", "description": "Polyuria"},
    "310": {"code": "310", "category": "Genitourinary", "description": "Genital trauma"},

    # Mental Health (351-360)
    "351": {"code": "351", "category": "Mental Health", "description": "Depression/suicidal/deliberate self-harm"},
    "352": {"code": "352", "category": "Mental Health", "description": "Anxiety/situational crisis"},
    "353": {"code": "353", "category": "Mental Health", "description": "Hallucinations/delusions"},
    "354": {"code": "354", "category": "Mental Health", "description": "Insomnia"},
    "355": {"code": "355", "category": "Mental Health", "description": "Violent/homicidal behaviour"},
    "356": {"code": "356", "category": "Mental Health", "description": "Social problem"},
    "358": {"code": "358", "category": "Mental Health", "description": "Bizarre behaviour"},
    "359": {"code": "359", "category": "Mental Health", "description": "Concern for patient's welfare"},
    "360": {"code": "360", "category": "Mental Health", "description": "Pediatric disruptive behaviour"},

    # Neurologic (401-411)
    "401": {"code": "401", "category": "Neurologic", "description": "Altered level of consciousness"},
    "402": {"code": "402", "category": "Neurologic", "description": "Confusion"},
    "403": {"code": "403", "category": "Neurologic", "description": "Vertigo"},
    "404": {"code": "404", "category": "Neurologic", "description": "Headache"},
    "405": {"code": "405", "category": "Neurologic", "description": "Seizure"},
    "406": {"code": "406", "category": "Neurologic", "description": "Gait disturbance/ataxia"},
    "407": {"code": "407", "category": "Neurologic", "description": "Head injury"},
    "408": {"code": "408", "category": "Neurologic", "description": "Tremor"},
    "409": {"code": "409", "category": "Neurologic", "description": "Extremity weakness/symptoms of CVA"},
    "410": {"code": "410", "category": "Neurologic", "description": "Sensory loss/paresthesia"},
    "411": {"code": "411", "category": "Neurologic", "description": "Floppy child"},

    # OB/GYN (451-460)
    "451": {"code": "451", "category": "OB/GYN", "description": "Menstrual problems"},
    "452": {"code": "452", "category": "OB/GYN", "description": "Foreign body, vagina"},
    "453": {"code": "453", "category": "OB/GYN", "description": "Vaginal discharge"},
    "454": {"code": "454", "category": "OB/GYN", "description": "Sexual assault"},
    "455": {"code": "455", "category": "OB/GYN", "description": "Vaginal bleed"},
    "456": {"code": "456", "category": "OB/GYN", "description": "Labial swelling"},
    "457": {"code": "457", "category": "OB/GYN", "description": "Pregnancy issues, <20 weeks"},
    "458": {"code": "458", "category": "OB/GYN", "description": "Pregnancy issues, >20 weeks"},
    "460": {"code": "460", "category": "OB/GYN", "description": "Vaginal pain/dyspareunia"},

    # Ophthalmology (502-511)
    "502": {"code": "502", "category": "Ophthalmology", "description": "Chemical exposure, eye"},
    "503": {"code": "503", "category": "Ophthalmology", "description": "Foreign body, eye"},
    "504": {"code": "504", "category": "Ophthalmology", "description": "Visual disturbance"},
    "505": {"code": "505", "category": "Ophthalmology", "description": "Eye pain"},
    "506": {"code": "506", "category": "Ophthalmology", "description": "Red eye, discharge"},
    "507": {"code": "507", "category": "Ophthalmology", "description": "Photophobia"},
    "508": {"code": "508", "category": "Ophthalmology", "description": "Diplopia"},
    "509": {"code": "509", "category": "Ophthalmology", "description": "Periorbital swelling"},
    "510": {"code": "510", "category": "Ophthalmology", "description": "Eye trauma"},
    "511": {"code": "511", "category": "Ophthalmology", "description": "Re-check eye"},

    # Orthopedic (551-559)
    "551": {"code": "551", "category": "Orthopedic", "description": "Back pain"},
    "552": {"code": "552", "category": "Orthopedic", "description": "Traumatic back/spine injury"},
    "553": {"code": "553", "category": "Orthopedic", "description": "Amputation"},
    "554": {"code": "554", "category": "Orthopedic", "description": "Upper extremity pain"},
    "555": {"code": "555", "category": "Orthopedic", "description": "Lower extremity pain"},
    "556": {"code": "556", "category": "Orthopedic", "description": "Upper extremity injury"},
    "557": {"code": "557", "category": "Orthopedic", "description": "Lower extremity injury"},
    "558": {"code": "558", "category": "Orthopedic", "description": "Joint(s) swelling"},
    "559": {"code": "559", "category": "Orthopedic", "description": "Pediatric gait disorder/painful walk"},

    # Respiratory (651-660)
    "651": {"code": "651", "category": "Respiratory", "description": "Shortness of breath"},
    "652": {"code": "652", "category": "Respiratory", "description": "Respiratory arrest"},
    "653": {"code": "653", "category": "Respiratory", "description": "Cough/congestion"},
    "654": {"code": "654", "category": "Respiratory", "description": "Hyperventilation"},
    "655": {"code": "655", "category": "Respiratory", "description": "Hemoptysis"},
    "656": {"code": "656", "category": "Respiratory", "description": "Respiratory foreign body"},
    "657": {"code": "657", "category": "Respiratory", "description": "Allergic reaction"},
    "658": {"code": "658", "category": "Respiratory", "description": "Stridor"},
    "659": {"code": "659", "category": "Respiratory", "description": "Wheezing—no other complaints"},
    "660": {"code": "660", "category": "Respiratory", "description": "Apneic spells in infants"},

    # Skin (701-717)
    "701": {"code": "701", "category": "Skin", "description": "Bite"},
    "702": {"code": "702", "category": "Skin", "description": "Sting"},
    "703": {"code": "703", "category": "Skin", "description": "Abrasion"},
    "704": {"code": "704", "category": "Skin", "description": "Laceration/puncture"},
    "705": {"code": "705", "category": "Skin", "description": "Burn"},
    "706": {"code": "706", "category": "Skin", "description": "Blood and body fluid exposure"},
    "707": {"code": "707", "category": "Skin", "description": "Pruritus"},
    "708": {"code": "708", "category": "Skin", "description": "Rash"},
    "709": {"code": "709", "category": "Skin", "description": "Localized swelling/redness"},
    "710": {"code": "710", "category": "Skin", "description": "Wound check"},
    "711": {"code": "711", "category": "Skin", "description": "Other skin conditions"},
    "712": {"code": "712", "category": "Skin", "description": "Lumps, bumps, calluses"},
    "713": {"code": "713", "category": "Skin", "description": "Redness/tenderness, breast"},
    "714": {"code": "714", "category": "Skin", "description": "Rule out infestation"},
    "715": {"code": "715", "category": "Skin", "description": "Cyanosis"},
    "716": {"code": "716", "category": "Skin", "description": "Spontaneous bruising"},
    "717": {"code": "717", "category": "Skin", "description": "Foreign body, skin"},

    # Substance Misuse (751-753)
    "751": {"code": "751", "category": "Substance Misuse", "description": "Substance misuse/intoxication"},
    "752": {"code": "752", "category": "Substance Misuse", "description": "Overdose ingestion"},
    "753": {"code": "753", "category": "Substance Misuse", "description": "Substance withdrawal"},

    # Trauma (801-806)
    "801": {"code": "801", "category": "Trauma", "description": "Major trauma—penetrating"},
    "802": {"code": "802", "category": "Trauma", "description": "Major trauma—blunt"},
    "803": {"code": "803", "category": "Trauma", "description": "Isolated chest trauma—penetrating"},
    "804": {"code": "804", "category": "Trauma", "description": "Isolated chest trauma—blunt"},
    "805": {"code": "805", "category": "Trauma", "description": "Isolated abdominal trauma—penetrating"},
    "806": {"code": "806", "category": "Trauma", "description": "Isolated abdominal trauma—blunt"},

    # General and Minor (851-869, 999)
    "851": {"code": "851", "category": "General and Minor", "description": "Exposure to communicable disease"},
    "852": {"code": "852", "category": "General and Minor", "description": "Fever"},
    "853": {"code": "853", "category": "General and Minor", "description": "Hyperglycemia"},
    "854": {"code": "854", "category": "General and Minor", "description": "Hypoglycemia"},
    "855": {"code": "855", "category": "General and Minor", "description": "Direct referral for consultation"},
    "856": {"code": "856", "category": "General and Minor", "description": "Dressing change"},
    "857": {"code": "857", "category": "General and Minor", "description": "Removal staples/sutures"},
    "858": {"code": "858", "category": "General and Minor", "description": "Cast check"},
    "859": {"code": "859", "category": "General and Minor", "description": "Imaging tests"},
    "860": {"code": "860", "category": "General and Minor", "description": "Medical device problem"},
    "861": {"code": "861", "category": "General and Minor", "description": "Prescription/medication request"},
    "862": {"code": "862", "category": "General and Minor", "description": "Ring removal"},
    "863": {"code": "863", "category": "General and Minor", "description": "Abnormal lab values"},
    "864": {"code": "864", "category": "General and Minor", "description": "Pallor/anemia"},
    "865": {"code": "865", "category": "General and Minor", "description": "Post-operative complications"},
    "866": {"code": "866", "category": "General and Minor", "description": "Minor complaints NOS"},
    "867": {"code": "867", "category": "General and Minor", "description": "Inconsolable crying"},
    "868": {"code": "868", "category": "General and Minor", "description": "Congenital problem in children"},
    "869": {"code": "869", "category": "General and Minor", "description": "Newly Born"},
    "999": {"code": "999", "category": "General and Minor", "description": "Unknown"},
}


def get_cedis_by_code(code: str) -> dict | None:
    """Get CEDIS entry by code number."""
    return CEDIS_CODES.get(str(code).zfill(3))


def get_cedis_by_category(category: str) -> list[dict]:
    """Get all CEDIS codes for a specific category."""
    return [e for e in CEDIS_CODES.values() if e["category"] == category]


def get_all_categories() -> list[str]:
    """Return all unique CEDIS category names in code order."""
    seen = {}
    for e in CEDIS_CODES.values():
        cat = e["category"]
        if cat not in seen:
            seen[cat] = True
    return list(seen.keys())
