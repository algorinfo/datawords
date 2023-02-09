ALPHANUMERIC_REGEX = r"[a-zA-Z0-9:/\#]+"
# ALPHANUMERIC_ACCENT_REGEX = r"[A-zÀ-ú0-9:/\-\_\#]+"
ALPHANUMERIC_ACCENT_REGEX = r"[A-zÀ-ú0-9]+"
WORDS_ACCENT_REGEX = r"[A-zÀ-ú:/\#-]+"
WORDS_REGEX = r"[a-zA-Z:/\#]+"
URL_REGEX = r"https?://\S+"

CONNECTOR_WORDS = {
    "en": frozenset(
        {
            "a",
            "an",
            "and",
            "at",
            "by",
            "for",
            "from",
            "in",
            "of",
            "on",
            "or",
            "the",
            "to",
            "with",
            "without",
        }
    ),
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

