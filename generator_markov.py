import markovify

markov_params = open('model_params/markov_params.json').read()
rev_markov_params = open('model_params/rev_markov_params.json').read()

model = markovify.Text.from_json(markov_params)
rev_model = markovify.Text.from_json(rev_markov_params)

#Generator
def generate(start):
    try:
        prefix = rev_model.make_sentence_with_start(start, tries = 10)
        postfix = model.make_sentence_with_start(start, tries = 10)
        prefix = ' '.join(prefix.split()[::-1][1:-1]) + ' '
        out = prefix + postfix
        for i in range(1, len(out)):
            if out[i].isupper():
                out = out[:i-1] + '\n' + out[i:]
    except:
        out = None
    return out

