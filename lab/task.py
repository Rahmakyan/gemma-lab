"""The task: triage of patient messages at a primary-care clinic.

Clinics get many portal messages and voicemails each day. A nurse sorts
them by urgency and sends each to a team. The messages are health data
(HIPAA, GDPR), so a clinic cannot send them to a public API. We teach a
local Gemma with fake messages that Gemini labels.

QUESTIONS is in the shape of the Gemini Decisions API (POST /v1beta/decisions),
one of each type, so the benchmark covers all three: `noul` is yes/no,
`choice` picks one key of `criteria`, `score` picks a level of an ordered
list (the answer is the probability-weighted level).

DIMENSIONS are where triage is likely to go wrong (the book's section 1.4):
each scenario is one tuple, one value of each dimension. `urgency` is what
the writer of the fake message aims for, not the label: the label is what
Gemini decides.
"""

DOMAIN = "patient messages to a primary-care clinic (patient-portal messages and voicemail transcripts)"

QUESTIONS = {
    "triage": {
        "type": "choice",
        "instructions": "How urgently must the clinic act on this patient message?",
        "criteria": {
            "emergency": "A possible life-threatening problem now: the patient must call emergency services or go to the emergency department.",
            "same_day": "A clinician must respond today: new or worse symptoms, a medication reaction, or an urgent question about a result.",
            "routine": "A clinical question or request that can wait one to three working days.",
            "admin": "No clinical need: appointments, bills, forms or records.",
        },
    },
    "route": {
        "type": "choice",
        "instructions": "Which team at the clinic must handle this patient message first?",
        "criteria": {
            "nurse": "Symptoms, triage and care advice.",
            "doctor": "Diagnosis, test results and treatment decisions.",
            "pharmacy": "Prescription refills, dose questions and side effects of a medication.",
            "front_desk": "Appointments, forms and requests for records.",
            "billing": "Bills, insurance and payments.",
        },
    },
    "callback_today": {
        "type": "noul",
        "instructions": "Must a clinician phone this patient back today?",
    },
    "severity": {
        "type": "score",
        "instructions": "How severe are the patient's symptoms, as the message describes them?",
        "criteria": [
            "none: no symptoms are mentioned",
            "mild: small discomfort, daily life goes on",
            "moderate: clear symptoms that affect daily life",
            "severe: strong symptoms that need care soon",
            "critical: symptoms that may threaten life",
        ],
    },
}

DIMENSIONS = {
    "topic": [
        "chest pain or palpitations",
        "breathing problems",
        "stomach or bowel problems",
        "headache, dizziness or numbness",
        "fever or infection",
        "mental health or mood",
        "injury or fall",
        "skin problem or rash",
        "pregnancy",
        "a sick child",
        "prescription refill",
        "side effect of a medication",
        "question about a test result",
        "appointment or scheduling",
        "bill or insurance",
        "forms, letters or medical records",
    ],
    "urgency": ["emergency", "same_day", "routine", "admin"],
    "sender": ["the patient", "a parent of a young child", "an adult child of an elderly patient", "a partner or caregiver"],
    "clarity": [
        "clear and complete",
        "vague",
        "key details missing",
        "a serious symptom hidden inside a routine request",
        "the writer plays down a serious symptom",
        "the writer is very worried about a minor problem",
    ],
    "style": ["formal", "casual", "text-message style with typos", "non-native English", "long and rambling", "very short"],
    "channel": ["patient-portal message", "voicemail transcript"],
}

# Boundary cases: reported apart from the rest (the book's challenge set),
# so the routine majority does not hide failures on them.
CHALLENGE = {
    "a serious symptom hidden inside a routine request",
    "the writer plays down a serious symptom",
    "the writer is very worried about a minor problem",
}

ADMIN_TOPICS = {"appointment or scheduling", "bill or insurance", "forms, letters or medical records"}


def plausible(t: dict) -> bool:
    """Drops combinations that make no sense (the book's human review of
    tuples, written down as rules)."""
    admin_topic = t["topic"] in ADMIN_TOPICS
    hidden = t["clarity"] == "a serious symptom hidden inside a routine request"
    if admin_topic and t["urgency"] in ("emergency", "same_day") and not hidden:
        return False
    if not admin_topic and t["urgency"] == "admin" and t["topic"] not in ("prescription refill", "question about a test result"):
        return False
    if t["topic"] == "a sick child" and t["sender"] != "a parent of a young child":
        return False
    if t["clarity"] == "the writer plays down a serious symptom" and t["urgency"] in ("routine", "admin"):
        return False
    if t["clarity"] == "the writer is very worried about a minor problem" and t["urgency"] == "emergency":
        return False
    if hidden and t["urgency"] in ("routine", "admin"):
        return False
    return True
