def clean(text):
    text = " ".join(text.lower().split())
    text = text.replace("\n", " ")
    return text
