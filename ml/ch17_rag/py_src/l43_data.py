"""Shared material for the L43 (generation and evaluation) figures.

L43 reuses the L41 cheese-factory corpus so the running example stays continuous, and adds
the material that only the generation/evaluation lecture needs:

  * two contradictory revisions of the same procedure (the "sources disagree" failure mode);
  * a small labelled evaluation set, in two flavours - questions copied from a chunk's own
    wording (what a naive question generator produces) and questions phrased the way a user
    actually asks (the "synthetic eval sets are too easy" experiment);
  * the answer-relevance material: one question, one good answer, one vague answer, and the
    questions you would get back if you asked a model to reconstruct the question;
  * the HyDE hypothetical document.

IMPORTANT - written before measuring. Both question sets below were fixed and committed
BEFORE `l43_probe_claims.py` was run for the first time, precisely so that the outcome could
not be tuned. Whatever the probe reported is what the deck says. See
_learnings/2026-08-10-2016_rag-figure-contradicted-its-own-claim.md for why this matters.
"""

from l41_data import CHUNKS, tokenize  # noqa: F401  (tokenize re-exported for figure scripts)

# --- the corpus L43 retrieves over -----------------------------------------------------
# L41's 18 chunks plus two revisions of the same press procedure. The revisions are the
# point of the "two sources contradict each other" frame: both are relevant, both retrieve,
# and only one of them is current.
REVISION_CHUNKS = [
    "Press manual revision 2, issued 2019: the Lori press is held at 2.5 bar for the whole "
    "of the pressing stage.",
    "Press manual revision 4, issued 2024: the Lori press starts at 2.5 bar and is raised "
    "to 3.2 bar after twenty minutes.",
]

CORPUS = CHUNKS + REVISION_CHUNKS

# Indices into CHUNKS that the cold open depends on.
IDX_PRESS_ANSWER = 0     # "The Lori press operates at 2.5 bar during the first pressing stage."
IDX_PRESS_FAULT = 1      # "...Readings above 3 bar indicate a fault in the press..."
IDX_PRS400 = 8           # "Model PRS-400 replaces the older PRS-220 press on line two."
IDX_CELLAR = 9           # "Ripening cellars are held at 10 degrees and 85 percent humidity."

COLD_OPEN_QUERY = "What pressure should the Lori press run at?"

# --- the evaluation set ----------------------------------------------------------------
# Eight chunks, two questions each. `copied` imitates what you get when you hand a chunk to
# a model and say "write a question this passage answers": it reuses the passage's own
# nouns. `asked` is how the same information gets requested by someone who has never read
# the passage. Both were written by hand, in one pass, before any measurement.
EVAL_SET = [
    # (gold chunk index, question copied from the chunk, question as a user would ask it)
    (0,  "At what pressure does the Lori press operate during the first pressing stage?",
         "How hard does the machine squeeze the cheese?"),
    (3,  "At what temperature is the milk held during pasteurisation, and for how long?",
         "How is the raw milk made safe before anything else happens?"),
    (4,  "At what temperature is the starter culture added to the milk?",
         "When do we put the bacteria in?"),
    (5,  "How long does rennet take to coagulate the milk into curd?",
         "How long before the liquid turns solid?"),
    (6,  "Into what size cubes is the curd cut before draining?",
         "How small do we chop it up?"),
    (9,  "At what temperature and humidity are ripening cellars held?",
         "How warm is the room where the cheese matures?"),
    (10, "What must operators wear when handling the curd knives?",
         "What protective equipment is needed around the sharp tools?"),
    (11, "At what residual pressure does the vacuum packer seal wheels?",
         "What setting is used for wrapping the finished product?"),
]

# --- answer relevance (RAGAS) ----------------------------------------------------------
# The metric asks a model to reconstruct the question from the answer, then measures how
# close the reconstruction lands to the real question. No model call happens here: the
# reconstructions below are written by hand to stand in for that step. The COSINE NUMBERS
# ARE REAL - they come from intfloat/multilingual-e5-small.
AR_QUESTION = "What pressure should the Lori press run at?"

AR_GOOD_ANSWER = ("The Lori press runs at 2.5 bar during the first pressing stage.")
AR_GOOD_BACKQUESTIONS = [
    "What pressure does the Lori press run at?",
    "At what pressure does the first pressing stage happen?",
    "How much pressure is used on the Lori press?",
]

AR_VAGUE_ANSWER = ("Pressing is an important step in cheese production and the settings "
                   "depend on the equipment and the recipe.")
AR_VAGUE_BACKQUESTIONS = [
    "Why is pressing important in cheese production?",
    "What does the choice of settings depend on?",
    "What affects cheese production settings?",
]

# --- faithfulness (RAGAS) --------------------------------------------------------------
# The cold-open answer, split into the statements a decomposition step would produce, each
# labelled with whether the retrieved context supports it. The context is the measured
# dense top-3: rev.4 (2024), rev.2 (2019), and the original press chunk.
#
# The sting is in statement 2. It IS supported - by the 2019 revision, which is in the
# context and is out of date. Statement-level verification checks each claim against the
# context; it never checks the context against itself. Arithmetic on the slide is exact.
COLD_OPEN_ANSWER = (
    "The Lori press runs at 2.5 bar. The pressure stays at 2.5 bar throughout the "
    "pressing stage. This is the setting used for Lori cheese, and the same setting "
    "applies to every press in the factory."
)
FAITH_STATEMENTS = [
    ("The Lori press runs at 2.5 bar.",                             True),
    ("The pressure stays at 2.5 bar throughout the pressing stage.", True),
    ("This is the setting used for Lori cheese.",                    True),
    ("The same setting applies to every press in the factory.",      False),
]

# --- HyDE ------------------------------------------------------------------------------
# Hypothetical Document Embeddings: instead of embedding the question, write the answer you
# expect to find and embed that. This passage is fabricated - the numbers in it are guesses,
# which is the whole idea. Written before measuring, one attempt, no retries.
HYDE_QUERY = "How warm is the room where the cheese matures?"
HYDE_DOCUMENT = ("The maturing room is kept cool and damp. Temperature is held at around "
                 "11 degrees with high relative humidity so the rind does not dry out.")

# --- multi-hop -------------------------------------------------------------------------
# Answering this needs TWO chunks, and the second is only findable once you have read the
# first: nothing in the question contains the string "PRS-400".
MULTIHOP_QUERY = "What pressure does the press that replaced the PRS-220 run at?"
MULTIHOP_NEEDED = [IDX_PRS400, IDX_PRESS_ANSWER]

# --- aggregation (the "do not use RAG" case) -------------------------------------------
# Stated parameters of the hypothetical archive, not measurements. The arithmetic is exact.
N_INCIDENT_REPORTS = 400
N_MENTIONING_PRESS = 37
TOP_K = 5
