import markovify

with open("poems.txt", 'r', encoding = 'utf-8', errors='ignore') as f:
    text = f.read()
    f.close()

text_model = markovify.Text(text)

def generate(start):
    try:
        return text_model.make_sentence_with_start(start)
    except:
        return None


