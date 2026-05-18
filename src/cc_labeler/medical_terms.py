"""
Medical terminology dictionary for chief complaint preprocessing.

Contains:
- Common medical abbreviations and their expansions
- Synonym mappings for medical terms
- Body part anatomical terms
"""

# Medical abbreviations commonly found in chief complaints
MEDICAL_ABBREVIATIONS = {
    # Vital signs & measurements
    'sob': 'shortness of breath',
    'sobs': 'shortness of breath',
    'ams': 'altered mental status',
    'loc': 'loss of consciousness',
    'aloc': 'altered level of consciousness',
    
    # Cardiovascular
    'cp': 'chest pain',
    'mi': 'myocardial infarction',
    'cva': 'stroke',
    'dvt': 'deep vein thrombosis',
    'htn': 'hypertension',
    
    # Trauma
    'mva': 'motor vehicle accident',
    'mvc': 'motor vehicle collision',
    'mvt': 'motor vehicle trauma',
    'gsh': 'gunshot',
    'gsw': 'gunshot wound',
    'fx': 'fracture',
    'atv': 'all terrain vehicle',
    
    # OB/GYN
    'ctx': 'contractions',
    'ctxs': 'contractions',
    'ptl': 'preterm labor',
    'srom': 'spontaneous rupture of membranes',
    'nst': 'non stress test',
    'ob': 'obstetric',
    'iufd': 'intrauterine fetal death',
    'sab': 'spontaneous abortion',
    'edd': 'estimated due date',
    'rpnv': 'routine prenatal visit',
    
    # Respiratory
    'uri': 'upper respiratory infection',
    'urti': 'upper respiratory tract infection',
    'lrti': 'lower respiratory tract infection',
    'copd': 'chronic obstructive pulmonary disease',
    'asthma exac': 'asthma exacerbation',
    
    # GI/GU
    'abd': 'abdominal',
    'uti': 'urinary tract infection',
    'n/v': 'nausea and vomiting',
    'nv': 'nausea and vomiting',
    'brbpr': 'bright red blood per rectum',
    
    # Neurologic
    'sz': 'seizure',
    'szs': 'seizures',
    'ha': 'headache',
    'tha': 'thunderclap headache',
    'tia': 'transient ischemic attack',
    
    # General
    'etoh': 'alcohol',
    'od': 'overdose',
    'lac': 'laceration',
    'lacs': 'lacerations',
    'fb': 'foreign body',
    'wnl': 'within normal limits',
    'n/a': '',
    'unk': 'unknown',
    'c/o': 'complaining of',
    
    # Pediatric
    'ped': 'pediatric',
    'peds': 'pediatric',
    'yo': 'year old',
    'yom': 'year old male',
    'yof': 'year old female',
    'nb': 'newborn',
    
    # Alaska-specific
    'snowgo': 'snowmobile',
    'snowmachine': 'snowmobile',
}

# Symptom synonyms - map variations to canonical form
SYMPTOM_SYNONYMS = {
    # Pain descriptors
    'painful': 'pain',
    'ache': 'pain',
    'aching': 'pain',
    'discomfort': 'pain',
    'soreness': 'pain',
    'hurting': 'pain',
    
    # Breathing
    'difficulty breathing': 'shortness of breath',
    'trouble breathing': 'shortness of breath',
    'cant breathe': 'shortness of breath',
    "can't breathe": 'shortness of breath',
    'breathing problems': 'shortness of breath',
    'dyspnea': 'shortness of breath',
    
    # Consciousness
    'passed out': 'loss of consciousness',
    'blacked out': 'loss of consciousness',
    'unresponsive': 'altered level of consciousness',
    'unconscious': 'loss of consciousness',
    'drowsy': 'altered level of consciousness',
    'lethargic': 'altered level of consciousness',
    
    # Bleeding
    'bleeding': 'hemorrhage',
    'blood': 'hemorrhage',
    'bloody': 'hemorrhage',
    
    # Injury
    'injured': 'injury',
    'hurt': 'injury',
    'trauma': 'injury',
    'wounded': 'injury',
    
    # Pregnancy
    'pregnant': 'pregnancy',
    'expecting': 'pregnancy',
    'gravid': 'pregnancy',
    'labor': 'contractions',
    'labour': 'contractions',
    
    # Mental status
    'confused': 'confusion',
    'disoriented': 'confusion',
    'not making sense': 'confusion',
    'dizzy': 'dizziness',
    'lightheaded': 'dizziness',
    'light headed': 'dizziness',
    
    # GI symptoms
    'vomit': 'vomiting',
    'vomits': 'vomiting',
    'throwing up': 'vomiting',
    'puking': 'vomiting',
    'nausea': 'nausea',
    'nauseous': 'nausea',
    'diarrhea': 'diarrhea',
    'loose stools': 'diarrhea',
}

# Body part anatomical terms
BODY_PARTS = {
    'anatomical': [
        'head', 'face', 'eye', 'ear', 'nose', 'mouth', 'throat', 'neck',
        'chest', 'thorax', 'breast', 'abdomen', 'belly', 'stomach', 'back',
        'spine', 'pelvis', 'groin', 'hip', 'buttock',
        'arm', 'shoulder', 'elbow', 'wrist', 'hand', 'finger',
        'leg', 'thigh', 'knee', 'ankle', 'foot', 'toe',
        'skin', 'scalp', 'jaw', 'tooth', 'teeth', 'tongue',
    ],
    'organs': [
        'heart', 'lung', 'liver', 'kidney', 'bladder', 'bowel',
        'brain', 'spleen', 'pancreas',
    ],
}

# Trauma mechanism keywords
TRAUMA_MECHANISMS = [
    'fall', 'fell', 'fallen',
    'crash', 'collision', 'accident',
    'hit', 'struck', 'impact',
    'assault', 'attacked', 'fight',
    'cut', 'laceration', 'stab',
    'burn', 'burned', 'scalded',
    'fracture', 'broken', 'break',
    'gunshot', 'shot', 'bullet',
    'rollover', 'rolled',
    'atv', 'snowmobile', 'snowmachine', 'honda',
    'injury', 'injured', 'trauma',
    'amputation', 'amputated',
]

# OB/GYN specific terms
OBGYN_TERMS = [
    'pregnancy', 'pregnant', 'prenatal',
    'contractions', 'labor', 'labour',
    'delivery', 'birth', 'delivering',
    'miscarriage', 'abortion',
    'bleeding', 'spotting',
    'vaginal', 'cervical',
    'weeks gestation', 'weeks pregnant',
    'trimester',
    'preeclampsia', 'eclampsia',
    'postpartum', 'preterm',
]

# Age-related modifiers (for demographic extraction, not matching)
AGE_MODIFIERS = [
    'pediatric', 'child', 'infant', 'newborn', 'neonate', 'baby',
    'elderly', 'geriatric', 'senior',
    'adolescent', 'teen', 'teenager',
    'adult', 'year old', 'yo', 'month old', 'mo',
]

# Gender modifiers (for demographic extraction, not matching)
GENDER_MODIFIERS = [
    'male', 'female', 'man', 'woman', 'boy', 'girl',
]


def expand_abbreviations(text):
    """
    Expand common medical abbreviations in text.
    
    Args:
        text: Input text with potential abbreviations
        
    Returns:
        Text with abbreviations expanded
    """
    if not text:
        return text
    
    text_lower = text.lower()
    result = text_lower
    
    # Sort by length (longest first) to handle multi-word abbreviations
    sorted_abbrevs = sorted(MEDICAL_ABBREVIATIONS.items(), 
                           key=lambda x: len(x[0]), 
                           reverse=True)
    
    for abbrev, expansion in sorted_abbrevs:
        # Use word boundaries to avoid partial matches
        import re
        pattern = r'\b' + re.escape(abbrev) + r'\b'
        result = re.sub(pattern, expansion, result)
    
    return result


def normalize_symptoms(text):
    """
    Normalize symptom descriptions to canonical forms.
    
    Args:
        text: Input text with symptom descriptions
        
    Returns:
        Text with normalized symptom terms
    """
    if not text:
        return text
    
    result = text.lower()
    
    # Sort by length (longest first) for better matching
    sorted_synonyms = sorted(SYMPTOM_SYNONYMS.items(), 
                            key=lambda x: len(x[0]), 
                            reverse=True)
    
    for synonym, canonical in sorted_synonyms:
        import re
        # Use word boundaries but allow for partial matches in phrases
        result = result.replace(synonym, canonical)
    
    return result
