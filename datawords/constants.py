from gensim.models.phrases import ENGLISH_CONNECTOR_WORDS

CONNECTOR_WORDS = {
    "en": ENGLISH_CONNECTOR_WORDS,
    "es": frozenset(
        {
            "un",
            "una",
            "y",
            "para",
            "por",
            "en",
            "el",
            "asi",
            "con",
            "si",
            "asi",
            "aunque",
            "de",
            "sino",
            "pero",
            "sin",
            "como",
            "segun",
            "ahora",
            "que",
            "cuando",
            "durante",
            "entonces",
            "hasta",
            "luego",
        }
    ),
}
