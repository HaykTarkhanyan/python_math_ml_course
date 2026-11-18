"""Shared corpus for the L41 (RAG retrieval) figures.

The running example is the cheese factory from the classification chapter, now with a
documentation problem. Equipment manuals are in English, which is realistic for industrial
kit; the Armenian material lives in ARM_* and is used only by the ATE-2 / Armenian section.

Every figure script imports from here so the same chunks appear throughout the deck.
"""

# --- the indexed corpus (each string is one chunk) -------------------------------------
CHUNKS = [
    "The Lori press operates at 2.5 bar during the first pressing stage.",
    "Press maintenance requires checking the hydraulic gauge weekly. Readings above 3 bar "
    "indicate a fault in the press. Every press reading must be logged by the operator and "
    "reviewed monthly.",
    "Cheese brining follows the pressing stage. The brine tank is kept at 12 degrees for "
    "Lori cheese.",
    "Pasteurisation holds the milk at 72 degrees for fifteen seconds before the culture is added.",
    "The starter culture is added once the milk has cooled to 32 degrees.",
    "Rennet coagulates the milk into curd within forty minutes.",
    "Curd is cut into cubes of roughly one centimetre before draining.",
    "Whey is drained through the perforated base of the vat.",
    "Model PRS-400 replaces the older PRS-220 press on line two.",
    "Ripening cellars are held at 10 degrees and 85 percent humidity.",
    "Operators must wear cut-resistant gloves when handling the curd knives.",
    "The vacuum packer seals wheels at 0.8 bar residual pressure.",
    "Milk arriving from suppliers is tested for antibiotics before unloading.",
    "Annual calibration of all pressure gauges is contracted to an external lab.",
    # Near-miss part numbers. Added after the first measurement run: without competing
    # identifiers the "exact identifier" query had nothing to be confused by, so it was
    # not a real test of anything.
    "Model PRS-220 was withdrawn from service after the 2019 recall.",
    "The PRS-380 press is rated for harder cheeses and higher pressures.",
    "Spare parts for the PRS-410 are no longer stocked by the supplier.",
    "Line three uses a VAC-400 vacuum unit rather than a press.",
]

# Short labels for plotting (chunks are too long for an axis tick).
CHUNK_LABELS = [
    "press 2.5 bar", "press maintenance", "brining", "pasteurisation", "starter culture",
    "rennet", "curd cutting", "whey draining", "PRS-400 model", "ripening cellar",
    "gloves / safety", "vacuum packing", "milk testing", "gauge calibration",
    "PRS-220 recall", "PRS-380 press", "PRS-410 parts", "VAC-400 unit",
]

# --- the three queries the deck keeps returning to -------------------------------------
# Target is an index into CHUNKS. Every `note` below states a MEASURED outcome, not an
# intended one - see probe_retrieval_claims.py and _learnings/. The first draft of this
# dict asserted results that the measurement then contradicted.
QUERIES = {
    "direct": {
        "text": "What pressure should the press run at for Lori cheese?",
        "target": 0,
        "note": "words overlap: BM25 rank 2, dense rank 2 - the easy case, both work",
    },
    "paraphrase": {
        "text": "How warm is the room where the cheese matures?",
        "target": 9,
        "note": "no content word shared: BM25 scores EXACTLY 0.000 (rank 17/18), dense rank 4",
    },
    "identifier": {
        "text": "PRS-400",
        "target": 8,
        "note": "exact id: BM25 rank 1, dense rank 1 - dense is NOT worse here at this scale",
    },
}

# --- BM25 worked example ---------------------------------------------------------------
# Three candidate chunks, ranked by hand on the slide. Indices into CHUNKS.
WORKED_DOCS = [0, 1, 2]
WORKED_QUERY_TERMS = ["lori", "press", "bar"]

# Corpus statistics for the worked example. These describe a HYPOTHETICAL 4,000-document
# factory archive - they are stated parameters, not measurements. The BM25 arithmetic
# performed on them is exact.
CORPUS_N = 4000
DOC_FREQ = {"lori": 120, "press": 380, "bar": 210}
AVGDL = 20.0

# Standard BM25 parameters. Robertson & Zaragoza (2009) report 1.2 < k1 < 2 and
# 0.5 < b < 0.8 as broadly good; these are the usual defaults inside that range.
K1 = 1.5
B = 0.75

# --- Armenian material (section 5 only) ------------------------------------------------
ARM_QUERY = "Ի՞նչ ճնշման տակ է աշխատում մամլիչը"
ARM_PASSAGES = [
    "Լոռի պանրի մամլիչն աշխատում է 2.5 բար ճնշման տակ:",
    "Աղաջրի տակառը պահվում է 12 աստիճանում:",
    "Կաթը պաստերացվում է 72 աստիճանում տասնհինգ վայրկյան:",
    "Հասունացման նկուղում պահվում է 10 աստիճան ջերմություն:",
]
ARM_LABELS = ["answer (press, 2.5 bar)", "brine tank", "pasteurisation", "ripening cellar"]


def tokenize(text):
    """Lowercase whitespace tokenisation, punctuation stripped.

    Deliberately naive and stemless: the deck uses this to show that plain BM25 treats
    'press', 'pressing' and 'pressure' as three unrelated terms.
    """
    out = []
    for raw in text.lower().split():
        word = raw.strip(".,;:!?()[]\"'")
        if word:
            out.append(word)
    return out
