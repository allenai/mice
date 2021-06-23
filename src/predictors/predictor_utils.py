def clean_text(text, special_chars=["\n", "\t"]):
    for char in special_chars:
        text = text.replace(char, " ")
    return text

