"""The word/phrase list for the Armenian semantic-tree project (project 4 of ch09).

Source of truth for the dataset. `embed_armenian_words.py` reads this and writes
`data/armenian_words.npz`, so students never download a model.

Design: ten clean semantic families give the ground truth for ARI/AMI. On top of that sit
four kinds of planted probe, each answering a question the flat families cannot:

  ORTHO_PROBES   words that START alike but mean unrelated things. If the tree groups
                 these, the model is reading spelling, not meaning. Measured on the first
                 draft: dzu/dzi (egg/horse) merged at cosine 0.13 - tighter than any real
                 family pair in the whole tree.
  PHRASE_FORMS   the same concept as a bare word and as a short natural phrase. Sentence
                 encoders are trained on sentences, so this is the controlled test of
                 whether context repairs the orthographic collapse.
  SYNONYM_PAIRS  pairs that SHOULD merge first. The control: if these do not merge before
                 the orthographic collisions, the space is not usable at all.
  POLYSEMY       one surface form, several unrelated senses, each in a full sentence with
                 a matching anchor sentence. The hardest test in the set.

INSTRUCTOR REVIEW WANTED on spelling, on family assignment for the handful of words that
could sit in two families, and above all on the ORTHO_PROBES - they only work if the
near-collisions are ones a native reader agrees are near-collisions.
"""

# ---------------------------------------------------------------- families (ground truth)
# EVERY probe word below also lives in a family. That is deliberate and it is the whole
# experiment: a probe word with no semantic home has nowhere to go but its orthographic
# neighbours, so its position would prove nothing. The first draft had exactly that bug -
# ձեռք was probe-only, drifted to the ձ- cluster, and the result was uninterpretable.
FAMILIES = {
    "ուտելիք": ["հաց", "պանիր", "մածուն", "լավաշ", "խորոված", "տոլմա", "գաթա", "ապուր",
                "ձու", "գարեջուր"],
    "կենդանիներ": ["կատու", "շուն", "ձի", "կով", "ոչխար", "արջ", "գայլ", "աղվես", "ձուկ"],
    "ընտանիք": ["մայր", "հայր", "քույր", "եղբայր", "տատիկ", "պապիկ", "որդի", "դուստր"],
    "գույներ": ["կարմիր", "կապույտ", "դեղին", "կանաչ", "սև", "սպիտակ", "մոխրագույն",
                "նարնջագույն"],
    "քաղաքներ": ["Երևան", "Գյումրի", "Վանաձոր", "Դիլիջան", "Աշտարակ", "Կապան", "Իջևան",
                 "Գորիս"],
    "մասնագիտություններ": ["ուսուցիչ", "բժիշկ", "վարորդ", "ինժեներ", "երգիչ", "նկարիչ",
                           "իրավաբան", "հաշվապահ"],
    "եղանակ": ["անձրև", "ձյուն", "արև", "քամի", "ամպ", "մառախուղ", "կարկուտ", "գարուն"],
    "տրանսպորտ": ["մեքենա", "ավտոբուս", "գնացք", "ինքնաթիռ", "հեծանիվ", "մետրո",
                  "մարշրուտկա", "տաքսի"],
    "երաժշտություն": ["դուդուկ", "զուռնա", "քանոն", "թառ", "դհոլ", "երգ", "պար",
                      "նվագախումբ"],
    "դպրոց": ["գիրք", "տետր", "գրիչ", "դասարան", "քննություն", "դասագիրք", "համալսարան",
              "մատիտ"],
    "մարմին": ["ձեռք", "ոտք", "գլուխ", "աչք", "ականջ", "քիթ", "մազ", "սիրտ"],
    "բնություն": ["քար", "արտ", "սերմ", "լեռ", "գետ", "ծառ", "ծաղիկ", "անտառ"],
}

# ------------------------------------------------------- probe 1: spelling vs meaning
# Each group shares an opening, and the members belong to DIFFERENT families above.
# Words marked (+) are not in any family and exist only as probe material.
# Measured on the draft: the collisions do NOT track shared letters, they track a shared
# leading SUBWORD TOKEN. Check with model.tokenizer.tokenize(word). That is why the groups
# below are split into "collides" and "control" - the control groups share letters but
# tokenize differently, and they do not collide. Students are meant to discover this rule.
ORTHO_PROBES = [
    {
        "name": "ձ-",
        "words": ["ձու", "ձի", "ձուկ", "ձյուն"],
        "senses": ["egg", "horse", "fish", "snow"],
        "families": ["ուտելիք", "կենդանիներ", "ուտելիք", "եղանակ"],
        "expect": "collides",
        "why": "all four tokenize as ['▁ձ', ...] - one shared leading token, four "
               "different families. ձու+ձի merged at 0.505, tighter than most synonym pairs",
    },
    {
        "name": "ձեռք (the control for ձ-)",
        "words": ["ձեռք", "ձի"],
        "senses": ["hand", "horse"],
        "families": ["մարմին", "կենդանիներ"],
        "expect": "does NOT collide",
        "why": "ձեռք starts with the same letter but tokenizes as a SINGLE token ['▁ձեռք'], "
               "so it shares nothing with ձի. This pair is what refutes the naive "
               "'it groups by spelling' explanation",
    },
    {
        "name": "գար-",
        "words": ["գարուն", "գարեջուր"],
        "senses": ["spring (season)", "beer"],
        "families": ["եղանակ", "ուտելիք"],
        "expect": "collides",
        "why": "both contain the token 'գար'",
    },
    {
        "name": "ար- (control)",
        "words": ["արջ", "արև", "արտ"],
        "senses": ["bear", "sun", "field"],
        "families": ["կենդանիներ", "եղանակ", "բնություն"],
        "expect": "does NOT collide",
        "why": "share two letters but tokenize as ['▁ար','ջ'] / ['▁արեւ'] / ['▁','արտ'] - "
               "no shared leading token",
    },
    {
        "name": "ք- (control)",
        "words": ["քար", "քամի", "քիթ"],
        "senses": ["stone", "wind", "nose"],
        "families": ["բնություն", "եղանակ", "մարմին"],
        "expect": "does NOT collide",
        "why": "['▁քար'] / ['▁','քա','մի'] / ... - different tokenizations despite the "
               "shared first letter",
    },
    {
        "name": "մա- (control)",
        "words": ["մայր", "մածուն", "մատիտ"],
        "senses": ["mother", "matsoun", "pencil"],
        "families": ["ընտանիք", "ուտելիք", "դպրոց"],
        "expect": "does NOT collide",
        "why": "['▁մայր'] / ['▁մ','ած','ուն'] / ['▁մատ','իտ']",
    },
]

# --------------------------------------- probe 2: does context repair the collisions?
# Same concept, bare word vs a short natural phrase. Compare the two trees.
PHRASE_FORMS = {
    "ձու": "հավի ձու",
    "ձի": "արագ ձի",
    "ձուկ": "թարմ ձուկ",
    "ձյուն": "ձյուն է գալիս",
    "ձեռք": "աջ ձեռք",
    "արջ": "անտառի արջ",
    "արև": "պայծառ արև",
    "արտ": "ցորենի արտ",
    "գարուն": "գարնան առաջին օրը",
    "գարեջուր": "սառը գարեջուր",
    "սերմ": "ծաղկի սերմ",
    "քար": "մեծ քար",
    "քիթ": "երկար քիթ",
    "քամի": "ուժեղ քամի",
    "մատիտ": "սուր մատիտ",
    "մածուն": "տնական մածուն",
    "մայր": "հոգատար մայր",
    "գաթա": "քաղցր գաթա",
    "գնացք": "գիշերային գնացք",
    "գայլ": "քաղցած գայլ",
    "գիրք": "հետաքրքիր գիրք",
    "կատու": "սև կատու",
    "կով": "կաթ տվող կով",
}

# --------------------------------------------- probe 3: the control (should merge first)
SYNONYM_PAIRS = [
    ("մեքենա", "ավտոմեքենա", "same thing, one is the fuller form"),
    ("համակարգիչ", "կոմպյուտեր", "native coinage vs loanword"),
    ("ինքնաթիռ", "օդանավ", "both aircraft"),
    ("ուրախ", "երջանիկ", "happy / joyful"),
    ("տուն", "բնակարան", "house / flat - close but not identical, a useful middle case"),
    ("ուսուցիչ", "դասատու", "teacher, two words in daily use"),
]

# ------------------------------------------------- probe 4: one form, several meanings
# Full sentences: the first draft showed bare tokens and short phrases do NOT separate
# these senses, while sentences do. That failure is the point of the words-vs-phrases task.
POLYSEMY = {
    "Արա": [
        {"sense": "թագավոր",
         "text": "Արա Գեղեցիկը հայոց լեգենդար թագավորն էր։",
         "anchor": "Հին Հայաստանի թագավորը ղեկավարում էր իր երկիրը։"},
        {"sense": "դիմելաձև",
         "text": "Արա, այ տղա, ինչո՞ւ ես ուշացել։",
         "anchor": "Ընկերս ինձ կանչեց և բարևեց փողոցում։"},
        {"sense": "հրամայական",
         "text": "Արա քո գործը և մի ծուլացիր։",
         "anchor": "Նա ամբողջ օրը աշխատում է գրասենյակում։"},
    ],
    "սեր": [
        {"sense": "զգացմունք",
         "text": "Նրանց սերը տևեց ամբողջ կյանքը։",
         "anchor": "Նա սիրահարվեց և ամուսնացավ։"},
        {"sense": "կաթնամթերք",
         "text": "Սուրճին մի քիչ սեր ավելացրու։",
         "anchor": "Կաթից պատրաստում են կարագ և պանիր։"},
    ],
    "մարտ": [
        {"sense": "ամիս",
         "text": "Մարտը գարնան առաջին ամիսն է։",
         "anchor": "Ապրիլից հետո գալիս է մայիսը։"},
        {"sense": "ճակատամարտ",
         "text": "Մարտը տևեց մինչև լույս։",
         "anchor": "Զինվորները կռվեցին թշնամու դեմ։"},
    ],
    "բաց": [
        {"sense": "բացված",
         "text": "Դուռը բաց էր, և մենք ներս մտանք։",
         "anchor": "Խանութը փակ էր, պատուհանները՝ նույնպես։"},
        {"sense": "գույնի երանգ",
         "text": "Նա հագել էր բաց կապույտ վերնաշապիկ։",
         "anchor": "Պատերը ներկված էին մուգ կանաչ գույնով։"},
    ],
}


def all_leaves():
    """Every word that goes into the main dendrogram, with its family label.

    Probe words that belong to no family are labelled '(probe)' - they still appear as
    leaves, because seeing WHERE they land is the whole point.
    """
    leaves, labels = [], []
    for fam, words in FAMILIES.items():
        leaves += words
        labels += [fam] * len(words)

    in_family = set(leaves)
    for group in ORTHO_PROBES:
        for w in group["words"]:
            if w not in in_family:
                leaves.append(w)
                labels.append("(probe)")
                in_family.add(w)

    for a, b, _ in SYNONYM_PAIRS:
        for w in (a, b):
            if w not in in_family:
                leaves.append(w)
                labels.append("(probe)")
                in_family.add(w)

    return leaves, labels


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    leaves, labels = all_leaves()
    print(f"{len(FAMILIES)} families, {len(leaves)} leaves total")
    print(f"  from families : {sum(len(v) for v in FAMILIES.values())}")
    print(f"  probe-only    : {labels.count('(probe)')}")
    print(f"phrase forms    : {len(PHRASE_FORMS)}")
    print(f"polysemy sets   : {len(POLYSEMY)} "
          f"({sum(len(v) for v in POLYSEMY.values())} sentences + as many anchors)")
    dupes = {w for w in leaves if leaves.count(w) > 1}
    if dupes:
        raise SystemExit(f"duplicate leaves would self-match at similarity 1.0: {dupes}")
    print("no duplicate leaves")
