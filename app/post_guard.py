"""
post_guard.py – Classifier-based post-generation safety backstop for PACE.

Architecture (three layers, cheapest first):
─────────────────────────────────────────────────────────────────────────────
Layer 0 │ Fast regex — three tiers
        │  0a _HARD_HIGH   → immediate HIGH (skip classifiers)
        │  0b _HARD_MEDIUM → MEDIUM floor
        │  0c _HARD_LOW    → LOW floor (prevents classifier over-escalation)
─────────────────────────────────────────────────────────────────────────────
Layer 1 │ Suicidality classifier  (sentinetyd/suicidality)
        │ ELECTRA fine-tuned on Twitter/Reddit suicide ideation data.
        │ Binary: LABEL_0 = safe, LABEL_1 = suicidal signal.
        │ Only runs when regex already found something (floor > NONE).
─────────────────────────────────────────────────────────────────────────────
Layer 2 │ Mental-health severity classifier  (KevSun/mentalhealth_LM)
        │ BERT fine-tuned on psychiatrist-labelled data.
        │ Labels 0–5: 0 = minimal severity … 5 = serious/immediate intervention.
        │ Only runs when regex already found something (floor > NONE).
─────────────────────────────────────────────────────────────────────────────
Layer 3 │ Output safety scan (regex)
        │ Scans the LLM's response for accidentally harmful content.
─────────────────────────────────────────────────────────────────────────────



Environment variables (all optional):
  DISTRESS_CLASSIFIER_SUICIDE   model id for Layer 1  (default: sentinetyd/suicidality)
  DISTRESS_CLASSIFIER_SEVERITY  model id for Layer 2  (default: KevSun/mentalhealth_LM)
  DISTRESS_SUICIDE_THRESHOLD    float 0-1  (default: 0.65)   ← raised from 0.55
  DISTRESS_SEVERITY_HIGH        int        (default: 4)
  DISTRESS_SEVERITY_MEDIUM      int        (default: 3)       ← raised from 2
  DISTRESS_ENABLED              "false" to disable entirely (default: true)
"""

from __future__ import annotations

import os
import re
import logging
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

logger = logging.getLogger(__name__)


# Config


SUICIDE_MODEL          = os.getenv("DISTRESS_CLASSIFIER_SUICIDE",  "sentinetyd/suicidality")
SEVERITY_MODEL         = os.getenv("DISTRESS_CLASSIFIER_SEVERITY", "KevSun/mentalhealth_LM")
SUICIDE_THRESHOLD      = float(os.getenv("DISTRESS_SUICIDE_THRESHOLD",       "0.65"))  # raised
SEVERITY_HIGH          = int(os.getenv("DISTRESS_SEVERITY_HIGH",             "4"))
SEVERITY_MEDIUM        = int(os.getenv("DISTRESS_SEVERITY_MEDIUM",           "3"))     # raised from 2
MEDIUM_URGENT_SEVERITY = int(os.getenv("DISTRESS_MEDIUM_URGENT_SEVERITY",    "4"))
MEDIUM_URGENT_SUICIDE  = float(os.getenv("DISTRESS_MEDIUM_URGENT_SUICIDE",   "0.45"))
ENABLED                = os.getenv("DISTRESS_ENABLED", "true").lower() != "false"



# Severity enum


class DistressLevel(str, Enum):
    NONE   = "none"
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"



# Layer 0a – Hard HIGH patterns
# These are ALWAYS high risk — skip classifiers, go straight to HIGH.
# Only genuinely alarming, unambiguous language belongs here.


_HARD_HIGH: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        # Explicit suicidal language
        r"\b(suicid|kill\s+\w*self|end\s+(my|their|his|her)\s+life)\b",
        r"\b(overdos|self.?harm|cutting\s+\w*self|hurt\s+(him|her|them)self)\b",
        r"\bthreaten(ed|ing)?\s+to\s+(hurt|harm|kill|cut)\s+(him|her|them)self\b",
        r"\b(wants?\s+to\s+die|don.?t\s+want\s+to\s+(live|be\s+here|exist|be\s+alive)|doesn.?t\s+want\s+to\s+(live|be\s+alive|exist|be\s+here))\b",

        # Weapons / dangerous items found on the teen
        r"\bfound\s+(hidden\s+)?(a\s+)?(knife|knives|gun|weapon|pills?\s+stash|meds?\s+stash|blades?)\b",
        r"\b(pills?|meds?|medications?)\s+(hidden|stashed|in\s+(her|his|their)\s+room)\b",

        # Physical abuse or assault
        r"\b(abuse[ds]?|molest|assault)\b",
        r"\b(hit|beat|chok|strangle)\s+(my|the)\s+(kid|child|teen|son|daughter)\b",

        # Actually missing / ran away (specific, not just "went out")
        r"\b(ran\s+away\s+from\s+home|actually\s+ran\s+away|gone\s+missing|can.?t\s+find\s+(him|her|them))\b",

        # Suicidal farewell signals — specific phrases, not common words
        r"\b(better\s+off\s+without\s+(me|him|her|them)|nobody\s+will\s+miss|world\s+would\s+be\s+better)\b",
        r"\b(made?\s+me\s+promise\s+to\s+take\s+care|promised?\s+to\s+look\s+after)\b",
        r"\bwon.?t\s+need\s+(my|her|his|their)\s+(stuff|things|belongings)\s+anymore\b",
        r"\b(said\s+final\s+goodbye|like\s+(she|he|they)\s+wasn.?t\s+coming\s+back)\b",
        r"\b(finally\s+found\s+a\s+way|make\s+(the\s+)?(pain|it)\s+stop)\b",
        r"\bwon.?t\s+(be\s+)?(a\s+)?problem\s+(anymore|much\s+longer)\b",
        r"\b(won.?t\s+have\s+to\s+worry|you.?ll\s+be\s+fine\s+without)\b",
    ]
]



# Layer 0b – Hard MEDIUM patterns
# Sets a MEDIUM floor. Only specific, concrete concerning behaviours.
# Removed: "lately", "recently", "for a while", "for weeks" — too broad.
# Removed: "withdrawn", "scared", "overwhelmed" — too common in normal messages.


_HARD_MEDIUM: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        # Concrete eating/sleeping disruption (not just "tired")
        r"\bstopped?\s+(eating|sleeping|going\s+to\s+school)\b",
        r"\b(sleeping\s+all\s+day|up\s+all\s+night|reversed?\s+sleep\s+schedule)\b",
        r"\b(won.?t\s+(get\s+up|leave\s+(her|his|their)\s+(bed|room)))\b",

        # Physical threat or violence — specific, not just angry words
        r"\bthreaten(ed|ing)?\s+to\s+(run\s+away|hurt\s+(me|themselves|someone))\b",
        r"\b(physically\s+(attacked|hit|hurt)|threw\s+(something|things)\s+at)\b",
        r"\b(police|called\s+911|emergency\s+services)\b",

        # Substance use — specific substances, not just "drinking"
        r"\b(drugs?|vaping|smoking\s+weed|marijuana)\b",
        r"\b(drinking\s+alcohol|found\s+alcohol|found\s+drugs?)\b",

        # School refusal — concrete, not just "missed a day"
        r"\bwon.?t\s+go\s+to\s+school\b",
        r"\brefus(es?|ing)\s+to\s+go\s+to\s+school\b",
        r"\b(expelled?|suspended\s+from\s+school|dropped?\s+out)\b",
        r"\b(missing|skipping)\s+school\s+(every|all|most)\b",  # needs qualifier now

        # Parent minimising paired with a disclosure — specific phrases
        r"\bmaybe\s+it.?s\s+nothing\s+but\b",   # "maybe it's nothing but he..."
        r"\bprobably\s+overreacting\s+but\b",
        r"\bmight\s+be\s+nothing\s+but\b",

        # Suicidal farewell signals (behavioural, not verbal — duplicated from HIGH
        # because they may appear in milder phrasing that doesn't hit HIGH patterns)
        r"\bgiving\s+away\s+(her|his|their)\s+(\w+\s+)?(things|stuff|belongings|possessions|items)\b",
        r"\bgave\s+up\s+on\s+(everything|all)\b",
    ]
]



# Layer 0c – Hard LOW patterns
# Parental burnout/exhaustion — LOW so classifiers can't over-escalate.


_HARD_LOW: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\b(exhausted|burnt?\s+out|burnout|worn\s+out)\b",
        r"\bnothing\s+(i\s+)?(say|do|try)\s+(gets?|works?|helps?|through)\b",
        r"\b(tried\s+everything|nothing\s+(works?|is\s+working)|at\s+a\s+loss)\b",
        r"\b(i\s+don.?t\s+know\s+what\s+to\s+do|don.?t\s+know\s+how\s+to\s+handle)\b",
        r"\b(so\s+frustrated|so\s+tired\s+of|fed\s+up)\b",
        r"\b(giving\s+up|can.?t\s+do\s+this\s+anymore|at\s+my\s+wits\s+end)\b",
    ]
]


def _hard_high(text: str) -> bool:
    return any(p.search(text) for p in _HARD_HIGH)

def _hard_medium(text: str) -> bool:
    return any(p.search(text) for p in _HARD_MEDIUM)

def _hard_low(text: str) -> bool:
    return any(p.search(text) for p in _HARD_LOW)



# Prefilter — broad concern signals
# Too vague for a hard floor but enough to warrant running the ML classifiers.
# If prefilter matches AND regex found nothing, classifiers run but capped at MEDIUM.


_PREFILTER: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\b(not eating|hasn.t (been )?(eating|eaten)|barely (eating|eaten)|skipping meals|refuses? to eat|won.t eat|not eaten properly)\b",
        r"\b(not sleeping|hasn.t (been )?sleeping|barely sleeping)\b",
        r"\b(withdrawn|isolated|isolating|not talking|stopped talking|barely speaks?)\b",
        r"\b(really worried|very concerned|something is wrong|not herself|not himself|not themselves)\b",
        r"\b(days? now|weeks? now|for days?|for weeks?)\b",
        r"\b(hopeless|worthless|nobody cares|doesn.t care about anything)\b",
    ]
]

def _prefilter(text: str) -> bool:
    return any(p.search(text) for p in _PREFILTER)





# Lazy classifier loading  (downloaded on first use, not at startup)


@lru_cache(maxsize=1)
def _load_suicide_classifier():
    from transformers import pipeline as hf_pipeline
    logger.info(f"[BACKSTOP] Loading suicide classifier: {SUICIDE_MODEL}")
    return hf_pipeline(
        "text-classification",
        model=SUICIDE_MODEL,
        device=-1,
        truncation=True,
        max_length=512,
    )

@lru_cache(maxsize=1)
def _load_severity_classifier():
    from transformers import (
        BertTokenizer,
        BertForSequenceClassification,
        AutoConfig,
        pipeline as hf_pipeline,
    )
    logger.info(f"[BACKSTOP] Loading severity classifier: {SEVERITY_MODEL}")
    config = AutoConfig.from_pretrained(
        SEVERITY_MODEL,
        num_labels=6,
        problem_type="single_label_classification",
    )
    tokenizer = BertTokenizer.from_pretrained(SEVERITY_MODEL, use_fast=True)
    model = BertForSequenceClassification.from_pretrained(
        SEVERITY_MODEL,
        config=config,
        ignore_mismatched_sizes=True,
    )
    return hf_pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=-1,
        truncation=True,
        max_length=512,
    )



# Layer 3 – output safety regex


_OUTPUT_UNSAFE: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\b(step[\s\-]by[\s\-]step|instructions?\s+to|how\s+to)\s+(harm|hurt|kill|abuse)\b",
        r"\b(suicide\s+method|ways?\s+to\s+(die|end\s+(your|their)\s+life))\b",
        r"\b(acquire|obtain|buy)\s+(weapon|gun|knife|drugs?)\b",
    ]
]



# Result types


@dataclass
class DistressResult:
    level: DistressLevel = DistressLevel.NONE
    suicide_score: float = 0.0
    severity_label: int = 0
    triggered_by: str = "none"


@dataclass
class BackstopResult:
    triggered: bool
    level: DistressLevel
    regenerate: bool = False
    override_system: str = ""
    static_response: str = ""



# Distress-aware system prompts


DISTRESS_SYSTEM_HIGH = """
You are PACE, a compassionate support assistant.
The parent is describing a crisis involving self-harm, suicidal ideation, or physical danger.

Your response must follow this exact structure:
1. One sentence acknowledging how frightening this is. Be direct and warm.
2. One sentence saying this needs professional support right now, not communication coaching.
3. These two resources, exactly as written:
   Crisis Lifeline: call or text 988
   Crisis Text Line: text HOME to 741741
4. One short sentence letting them know you are here if they want to talk.

Hard rules:
- Do NOT say "I can only imagine" or "I want you to know"
- Do NOT ask any questions
- Do NOT give parenting advice or phone scripts
- Do NOT introduce worries they did not express
- Under 80 words total
"""

DISTRESS_SYSTEM_MEDIUM = """
You are PACE, a compassionate parenting communication coach.
The parent is describing a serious situation that goes beyond communication coaching —
such as school refusal, substance use, physical threats, or a major behavioural change.

Your response must:
1. Acknowledge their concern in one warm sentence — only reflect what they said.
2. Say clearly that this sounds like a situation where professional support would help.
3. Name one specific resource: a family therapist or school counsellor.
   Do NOT suggest a pediatrician or general doctor.
4. End with one sentence keeping the door open for when they are ready to talk about communication.

Hard rules:
- Do NOT ask open-ended questions
- Do NOT give parenting scripts or communication techniques
- Do NOT say "I can only imagine" or "I want you to know"
- Do NOT introduce worries they did not express
- Under 80 words total
"""

DISTRESS_SYSTEM_MEDIUM_URGENT = """
You are PACE, a compassionate support assistant.
The parent is describing behaviour that may be a warning sign of serious emotional distress
in their child — even if they have not used explicit crisis language.

Your response must:
1. Acknowledge their concern in one direct, warm sentence.
2. Say clearly that this specific behaviour is worth taking seriously and acting on soon.
3. Recommend a therapist or child psychologist — not a general doctor.
4. Keep the door open for communication help once they have support in place.

Hard rules:
- Do NOT minimise this as a normal parenting concern
- Do NOT ask open-ended questions
- Do NOT give communication scripts
- Do NOT introduce fears they did not express
- Under 80 words total
"""

DISTRESS_SYSTEM_LOW = """
You are PACE, a compassionate parenting communication coach.
The parent sounds exhausted or at their limit.

Your response must:
1. Acknowledge their exhaustion directly in one or two warm sentences.
2. Validate that feeling stuck is normal — only reflect what they said.
3. Ask ONE gentle question about what specifically feels hardest right now.

Hard rules:
- Do NOT give techniques, scripts, or advice yet
- Do NOT say "I can only imagine" or "I can see that"
- Do NOT introduce worries they did not express
- Under 60 words total
"""

OUTPUT_UNSAFE_FALLBACK = (
    "I want to make sure I'm being helpful and safe here. "
    "If you or your child is in any danger, please contact emergency services "
    "or the 988 Suicide & Crisis Lifeline (call or text 988). "
    "I'm here to support healthy communication — would you like to tell me "
    "more about what's happening?"
)

_LEVEL_TO_SYSTEM = {
    DistressLevel.HIGH:   DISTRESS_SYSTEM_HIGH,
    DistressLevel.MEDIUM: DISTRESS_SYSTEM_MEDIUM,
    DistressLevel.LOW:    DISTRESS_SYSTEM_LOW,
}



# Core classifier logic


def _run_classifiers(text: str) -> DistressResult:
    result = DistressResult()

    # Layer 1 – suicidality (ELECTRA binary classifier)
    try:
        clf = _load_suicide_classifier()
        out = clf(text)
        label = out[0]["label"]
        score = float(out[0]["score"])
        logger.debug(f"[BACKSTOP] Suicide → {label} ({score:.3f})")

        if label == "LABEL_1" and score >= SUICIDE_THRESHOLD:
            result.level         = DistressLevel.HIGH
            result.suicide_score = score
            result.triggered_by  = "suicide_classifier"
            return result

    except Exception as e:
        logger.warning(f"[BACKSTOP] Suicide classifier failed: {e}")

    # Layer 2 – severity (BERT 0-5 labels)
    try:
        clf = _load_severity_classifier()
        out = clf(text)
        raw = out[0]["label"]
        sev = int(raw.split("_")[-1])
        logger.debug(f"[BACKSTOP] Severity → {raw} (sev={sev})")

        result.severity_label = sev
        result.triggered_by   = "severity_classifier"

        if sev >= SEVERITY_HIGH:
            result.level = DistressLevel.HIGH
        elif sev >= SEVERITY_MEDIUM:
            result.level = DistressLevel.MEDIUM
        elif sev >= 1:
            result.level = DistressLevel.LOW

    except Exception as e:
        logger.warning(f"[BACKSTOP] Severity classifier failed: {e}")

    return result



# Public API


_LEVEL_ORDER = {
    DistressLevel.NONE:   0,
    DistressLevel.LOW:    1,
    DistressLevel.MEDIUM: 2,
    DistressLevel.HIGH:   3,
}

def _max_level(a: DistressLevel, b: DistressLevel) -> DistressLevel:
    return a if _LEVEL_ORDER[a] >= _LEVEL_ORDER[b] else b


def check_input(user_text: str) -> BackstopResult:
    """
    Run before the LLM (after BART guard passes).

    Pipeline:
      1. Regex sets a floor level (NONE / LOW / MEDIUM / HIGH)
      2. ML classifiers ONLY run if regex floor > NONE
         (prevents normal messages from getting crisis prompts via classifier alone)
      3. Final level = max(regex_level, classifier_level)

    This means:
      - "My daughter has been on TikTok lately" - regex=NONE - classifiers skip - NONE 
      - "My son stopped eating and won't go to school" - regex=MEDIUM - classifiers run 
      - "She said she wants to die" - regex=HIGH - skip classifiers - HIGH 
      - "I'm exhausted, tried everything" - regex=LOW - classifiers run, max=LOW/MEDIUM 
    """
    if not ENABLED:
        return BackstopResult(triggered=False, level=DistressLevel.NONE)

    #  Step 1: Regex floor 
    if _hard_high(user_text):
        regex_level = DistressLevel.HIGH
    elif _hard_medium(user_text):
        regex_level = DistressLevel.MEDIUM
    elif _hard_low(user_text):
        regex_level = DistressLevel.LOW
    else:
        regex_level = DistressLevel.NONE

    logger.debug(f"[BACKSTOP] Regex floor: {regex_level.value} for: '{user_text[:60]}'")

    #  Step 2: ML classifiers gate
    # Classifiers run on ALL messages — regex only sets a floor.
    # The threshold (SEVERITY_MEDIUM=3, SEVERITY_HIGH=4) ensures normal messages
    # like "TikTok lately" (score 0-1) never trigger, while genuinely concerning
    # messages (score 3+) are caught regardless of phrasing.
    # Only _HARD_HIGH skips classifiers — we already know it's a crisis.

    # For _HARD_HIGH, skip classifiers — we already know it's HIGH.
    if regex_level == DistressLevel.HIGH:
        system_prompt = _LEVEL_TO_SYSTEM[DistressLevel.HIGH]
        return BackstopResult(
            triggered=True,
            level=DistressLevel.HIGH,
            regenerate=True,
            override_system=system_prompt,
        )

    # Run classifiers on all messages (regex floor is just a minimum, not a gate).
    clf_result = _run_classifiers(user_text)
    classifier_level = clf_result.level
    logger.debug(
        f"[BACKSTOP] Classifier: {classifier_level.value} "
        f"via={clf_result.triggered_by} "
        f"suicide_score={clf_result.suicide_score:.3f} severity={clf_result.severity_label}"
    )

    # Step 3: Final level = max(regex, classifier) 
    final_level = _max_level(regex_level, classifier_level)

    if final_level == DistressLevel.NONE:
        return BackstopResult(triggered=False, level=DistressLevel.NONE)

    logger.info(
        f"[BACKSTOP] Final level={final_level.value} "
        f"(regex={regex_level.value}, classifier={classifier_level.value}) "
        f"text='{user_text[:60]}'"
    )

    #  Step 4: For MEDIUM, pick between general and urgent prompt
    if final_level == DistressLevel.MEDIUM:
        is_urgent = (
            clf_result.severity_label >= MEDIUM_URGENT_SEVERITY or
            clf_result.suicide_score  >= MEDIUM_URGENT_SUICIDE
        )
        system_prompt = DISTRESS_SYSTEM_MEDIUM_URGENT if is_urgent else DISTRESS_SYSTEM_MEDIUM
        logger.info(
            f"[BACKSTOP] MEDIUM prompt={'URGENT' if is_urgent else 'GENERAL'} "
            f"(severity={clf_result.severity_label} suicide_score={clf_result.suicide_score:.3f})"
        )
    else:
        system_prompt = _LEVEL_TO_SYSTEM[final_level]

    return BackstopResult(
        triggered=True,
        level=final_level,
        regenerate=True,
        override_system=system_prompt,
    )


def check_output(llm_response: str) -> BackstopResult:
    """
    Run after LLM generates a response to catch accidentally unsafe output.
    Uses regex only — fast and deterministic.
    """
    if any(p.search(llm_response) for p in _OUTPUT_UNSAFE):
        logger.warning("[BACKSTOP] Unsafe content in LLM output — using safe fallback.")
        return BackstopResult(
            triggered=True,
            level=DistressLevel.HIGH,
            regenerate=False,
            static_response=OUTPUT_UNSAFE_FALLBACK,
        )
    return BackstopResult(triggered=False, level=DistressLevel.NONE)