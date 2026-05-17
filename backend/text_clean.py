def clean(text):
    text = " ".join(text.split())
    text = text.replace("\n", " ")
    return text