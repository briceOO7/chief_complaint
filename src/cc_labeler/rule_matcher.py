"""
Rule-based matching for CEDIS classification.

Implements expert rules for high-confidence classification of specific
complaint types (trauma, OB/GYN, mental health, etc.)
"""

import re
from typing import List, Dict, Optional, Tuple


class RuleMatcher:
    """Rule-based classifier for CEDIS codes."""
    
    def __init__(self):
        """Initialize rule-based matcher with pattern definitions."""
        self.rules = self._define_rules()
    
    def _define_rules(self) -> List[Dict]:
        """
        Define classification rules with priority ordering.
        
        Rules are checked in order, so more specific rules should come first.
        
        Returns:
            List of rule dictionaries with patterns and target codes
        """
        rules = []
        
        # ===== CARDIAC ARREST (HIGHEST PRIORITY) =====
        rules.append({
            'name': 'cardiac_arrest_non_traumatic',
            'patterns': [
                r'\bcardiac arrest\b',
                r'\bheart.*stopped\b',
                r'\bcode blue\b',
                r'\bunresponsive.*no pulse\b',
            ],
            'cedis_code': '001',
            'confidence': 0.95,
            'exclude_patterns': [r'\btrauma\b', r'\binjur', r'\baccident\b'],
        })
        
        rules.append({
            'name': 'cardiac_arrest_traumatic',
            'patterns': [
                r'\btraumatic.*arrest\b',
                r'\bcardiac arrest.*trauma\b',
            ],
            'cedis_code': '002',
            'confidence': 0.95,
        })
        
        # ===== RESPIRATORY ARREST =====
        rules.append({
            'name': 'respiratory_arrest',
            'patterns': [
                r'\brespiratory arrest\b',
                r'\bnot breathing\b',
                r'\bstopped breathing\b',
                r'\bapnea\b',
            ],
            'cedis_code': '652',
            'confidence': 0.95,
        })
        
        # ===== MAJOR TRAUMA =====
        rules.append({
            'name': 'major_trauma_penetrating',
            'patterns': [
                r'\bgunshot\b',
                r'\bgsw\b',
                r'\bstab\b.*\bwound\b',
                r'\bpenetrating.*trauma\b',
                r'\bshot\b',
            ],
            'cedis_code': '801',
            'confidence': 0.90,
        })
        
        rules.append({
            'name': 'major_trauma_blunt_multi',
            'patterns': [
                r'\bmultiple.*trauma\b',
                r'\bmulti.*trauma\b',
                r'\bmajor.*trauma\b',
                r'\bmva\b.*\bmultiple\b',
                r'\bmotor vehicle accident\b.*\bmultiple\b',
                r'\brollover\b',
                r'\bhigh.*speed\b',
            ],
            'cedis_code': '802',
            'confidence': 0.85,
        })
        
        # ===== HEAD/NEURO TRAUMA =====
        rules.append({
            'name': 'head_injury',
            'patterns': [
                r'\bhead.*injur',
                r'\bhead.*trauma',
                r'\bskull.*fracture\b',
                r'\bconcussion\b',
                r'\btbi\b',
                r'\btraumatic brain injur',
                r'\bfall.*head\b',
                r'\bhit.*head\b',
                r'\bbump.*head\b',
            ],
            'cedis_code': '407',
            'confidence': 0.85,
        })
        
        # ===== OBSTETRIC (HIGH PRIORITY) =====
        rules.append({
            'name': 'pregnancy_early',
            'patterns': [
                r'\bpregnancy.*\b(\d+)\s*weeks?\b',  # Will check if < 20 weeks
                r'\bmiscarriage\b',
                r'\bspontaneous abortion\b',
                r'\bearly.*pregnancy\b',
                r'\bectopic\b',
            ],
            'cedis_code': '457',  # <20 weeks
            'confidence': 0.85,
            'custom_check': lambda text: self._check_pregnancy_weeks(text, max_weeks=20),
        })
        
        rules.append({
            'name': 'pregnancy_late',
            'patterns': [
                r'\bcontractions?\b',
                r'\blabou?r\b',
                r'\bdelivery\b',
                r'\bpreeclampsia\b',
                r'\bpreterm.*labou?r\b',
                r'\brupture.*membranes?\b',
            ],
            'cedis_code': '458',  # >20 weeks
            'confidence': 0.85,
        })
        
        rules.append({
            'name': 'vaginal_bleeding',
            'patterns': [
                r'\bvaginal.*bleed',
                r'\bvaginal.*hemorrhage',
                r'\bspotting\b',
            ],
            'cedis_code': '455',
            'confidence': 0.80,
        })
        
        # ===== CHEST PAIN =====
        rules.append({
            'name': 'chest_pain_cardiac',
            'patterns': [
                r'\bchest.*pain\b.*\b(sob|shortness|breath|radiating|crushing|pressure)\b',
                r'\b(crushing|pressure).*chest\b',
                r'\bangina\b',
                r'\bmi\b',
                r'\bheart attack\b',
            ],
            'cedis_code': '003',
            'confidence': 0.80,
        })
        
        rules.append({
            'name': 'chest_pain_non_cardiac',
            'patterns': [
                r'\bchest.*pain\b',
                r'\bchest.*discomfort\b',
            ],
            'cedis_code': '004',
            'confidence': 0.65,
        })
        
        # ===== SHORTNESS OF BREATH =====
        rules.append({
            'name': 'shortness_of_breath',
            'patterns': [
                r'\bshortness.*breath\b',
                r'\bdifficulty.*breathing\b',
                r'\btrouble.*breathing\b',
                r'\bsob\b',
                r'\bdyspnea\b',
                r'\bcant.*breathe\b',
                r'\brespiratory distress\b',
            ],
            'cedis_code': '651',
            'confidence': 0.80,
        })
        
        # ===== SEIZURE =====
        rules.append({
            'name': 'seizure',
            'patterns': [
                r'\bseizure\b',
                r'\bconvulsion\b',
                r'\bfitting\b',
                r'\bseizing\b',
            ],
            'cedis_code': '405',
            'confidence': 0.90,
        })
        
        # ===== STROKE/CVA =====
        rules.append({
            'name': 'stroke_symptoms',
            'patterns': [
                r'\bcva\b',
                r'\bstroke\b',
                r'\b(left|right).*weak',
                r'\bfacial.*droop',
                r'\bslurred.*speech\b',
                r'\bhemiparesis\b',
            ],
            'cedis_code': '409',
            'confidence': 0.85,
        })
        
        # ===== ALTERED MENTAL STATUS =====
        rules.append({
            'name': 'altered_consciousness',
            'patterns': [
                r'\baltered.*mental.*status\b',
                r'\baltered.*level.*consciousness\b',
                r'\bams\b',
                r'\baloc\b',
                r'\bunresponsive\b',
            ],
            'cedis_code': '401',
            'confidence': 0.85,
        })
        
        rules.append({
            'name': 'confusion',
            'patterns': [
                r'\bconfused\b',
                r'\bconfusion\b',
                r'\bdisoriented\b',
                r'\bnot making sense\b',
            ],
            'cedis_code': '402',
            'confidence': 0.80,
        })
        
        # ===== ABDOMINAL PAIN =====
        rules.append({
            'name': 'abdominal_pain',
            'patterns': [
                r'\babdominal.*pain\b',
                r'\bstomach.*pain\b',
                r'\bbelly.*pain\b',
                r'\babd.*pain\b',
            ],
            'cedis_code': '251',
            'confidence': 0.75,
        })
        
        # ===== NAUSEA/VOMITING =====
        rules.append({
            'name': 'nausea_vomiting',
            'patterns': [
                r'\bnausea\b',
                r'\bvomiting\b',
                r'\bvomit\b',
                r'\bthrowing up\b',
                r'\bn/v\b',
            ],
            'cedis_code': '257',
            'confidence': 0.75,
        })
        
        # ===== FEVER =====
        rules.append({
            'name': 'fever',
            'patterns': [
                r'\bfever\b',
                r'\bfebrile\b',
                r'\bhigh temperature\b',
                r'\bhyperthermia\b',
            ],
            'cedis_code': '852',
            'confidence': 0.75,
        })
        
        # ===== FRACTURES/ORTHOPEDIC =====
        rules.append({
            'name': 'back_injury',
            'patterns': [
                r'\bback.*injur',
                r'\bspine.*injur',
                r'\bback.*trauma',
                r'\bc-?spine\b',
            ],
            'cedis_code': '552',
            'confidence': 0.80,
        })
        
        rules.append({
            'name': 'back_pain',
            'patterns': [
                r'\bback.*pain\b',
                r'\blower back\b',
            ],
            'cedis_code': '551',
            'confidence': 0.70,
        })
        
        rules.append({
            'name': 'upper_extremity_injury',
            'patterns': [
                r'\b(arm|hand|wrist|shoulder|elbow).*injur',
                r'\b(arm|hand|wrist|shoulder|elbow).*fracture',
                r'\bfall.*arm\b',
            ],
            'cedis_code': '556',
            'confidence': 0.75,
        })
        
        rules.append({
            'name': 'lower_extremity_injury',
            'patterns': [
                r'\b(leg|foot|ankle|knee|hip).*injur',
                r'\b(leg|foot|ankle|knee|hip).*fracture',
                r'\bfall.*leg\b',
            ],
            'cedis_code': '557',
            'confidence': 0.75,
        })
        
        # ===== MENTAL HEALTH =====
        rules.append({
            'name': 'suicidal',
            'patterns': [
                r'\bsuicidal\b',
                r'\bself.*harm\b',
                r'\boverdose\b.*\bintentional\b',
                r'\bwants.*die\b',
            ],
            'cedis_code': '351',
            'confidence': 0.90,
        })
        
        rules.append({
            'name': 'anxiety',
            'patterns': [
                r'\banxiety\b',
                r'\bpanic.*attack\b',
                r'\banxious\b',
            ],
            'cedis_code': '352',
            'confidence': 0.75,
        })
        
        # ===== DIABETES =====
        rules.append({
            'name': 'hyperglycemia',
            'patterns': [
                r'\bhyperglycemi',
                r'\bhigh.*blood.*sugar\b',
                r'\bhigh.*glucose\b',
            ],
            'cedis_code': '853',
            'confidence': 0.85,
        })
        
        rules.append({
            'name': 'hypoglycemia',
            'patterns': [
                r'\bhypoglycemi',
                r'\blow.*blood.*sugar\b',
                r'\blow.*glucose\b',
            ],
            'cedis_code': '854',
            'confidence': 0.85,
        })
        
        # ===== SUBSTANCE MISUSE =====
        rules.append({
            'name': 'intoxication',
            'patterns': [
                r'\bintoxicat',
                r'\bdrunk\b',
                r'\betoh\b',
                r'\balcohol\b.*\babuse\b',
                r'\bsubstance.*abuse\b',
            ],
            'cedis_code': '751',
            'confidence': 0.80,
        })
        
        rules.append({
            'name': 'overdose',
            'patterns': [
                r'\boverdose\b',
                r'\bod\b',
                r'\bopioid\b.*\boverdose\b',
            ],
            'cedis_code': '752',
            'confidence': 0.85,
        })
        
        # ===== ALLERGIC REACTION =====
        rules.append({
            'name': 'allergic_reaction',
            'patterns': [
                r'\ballergic.*reaction\b',
                r'\banaphylaxis\b',
                r'\bhives\b',
                r'\bswelling\b.*\ballerg',
            ],
            'cedis_code': '657',
            'confidence': 0.85,
        })
        
        # ===== BURNS =====
        rules.append({
            'name': 'burn',
            'patterns': [
                r'\bburn\b',
                r'\bburned\b',
                r'\bscald',
            ],
            'cedis_code': '705',
            'confidence': 0.85,
        })
        
        # ===== LACERATIONS =====
        rules.append({
            'name': 'laceration',
            'patterns': [
                r'\blaceration\b',
                r'\bcut\b',
                r'\bgash\b',
            ],
            'cedis_code': '704',
            'confidence': 0.80,
        })
        
        return rules
    
    def _check_pregnancy_weeks(self, text: str, max_weeks: int = 20) -> bool:
        """
        Check if pregnancy weeks mentioned is below max_weeks.
        
        Args:
            text: Text to check
            max_weeks: Maximum weeks for positive match
            
        Returns:
            True if weeks mentioned and < max_weeks, False otherwise
        """
        # Look for patterns like "28 weeks", "28 wks", "28 weeks gestation"
        pattern = r'(\d+)\s*(?:weeks?|wks?)\s*(?:gestation|pregnant|preg)?'
        match = re.search(pattern, text, re.IGNORECASE)
        
        if match:
            weeks = int(match.group(1))
            return weeks < max_weeks
        
        return False
    
    def match(self, text: str) -> List[Tuple[str, float, str]]:
        """
        Apply rules to text and return matching CEDIS codes with confidence.
        
        Args:
            text: Preprocessed chief complaint text
            
        Returns:
            List of tuples (cedis_code, confidence, rule_name) sorted by confidence
        """
        if not text:
            return []
        
        text_lower = text.lower()
        matches = []
        
        for rule in self.rules:
            # Check exclude patterns first
            if 'exclude_patterns' in rule:
                if any(re.search(pattern, text_lower) for pattern in rule['exclude_patterns']):
                    continue
            
            # Check if any pattern matches
            matched = False
            for pattern in rule['patterns']:
                if re.search(pattern, text_lower):
                    matched = True
                    break
            
            if matched:
                # Apply custom check if present
                if 'custom_check' in rule:
                    if not rule['custom_check'](text_lower):
                        continue
                
                matches.append((
                    rule['cedis_code'],
                    rule['confidence'],
                    rule['name']
                ))
        
        # Sort by confidence (highest first)
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches
    
    def get_best_match(self, text: str, min_confidence: float = 0.7) -> Optional[Tuple[str, float, str]]:
        """
        Get single best rule-based match above confidence threshold.
        
        Args:
            text: Preprocessed chief complaint text
            min_confidence: Minimum confidence threshold
            
        Returns:
            Tuple of (cedis_code, confidence, rule_name) or None
        """
        matches = self.match(text)
        
        if matches and matches[0][1] >= min_confidence:
            return matches[0]
        
        return None
