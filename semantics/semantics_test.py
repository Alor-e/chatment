import gc
import json
from sentence_transformers import SentenceTransformer, util

# formers = ['sentence-transformers/sentence-t5-xxl', 'sentence-transformers/all-mpnet-base-v2']
formers = ['sentence-transformers/all-mpnet-base-v2']
intents = None

with open('../data/data.json', 'r') as f:
    intents = json.load(f)

assert intents is not None
texts = []
items = {}
correct_intents = {}
for key, value in intents.items():
    for example in value:
        items[example] = key
        texts.append(example)
        correct_intents[key] = 0

pairs = {}
for transformer in formers:
    model = SentenceTransformer(transformer)

    # Compute embeddings
    embeddings = model.encode(texts, convert_to_tensor=True)

    # Compute cosine-similarities for each sentence with each other sentence
    cosine_scores = util.cos_sim(embeddings, embeddings)

    # Find the pairs with the highest cosine similarity scores

    pairs[transformer] = []
    for i in range(len(cosine_scores) - 1):
        for j in range(i + 1, len(cosine_scores)):
            pairs[transformer].append({'index': [i, j], 'score': cosine_scores[i][j]})

    # Sort scores in decreasing order
    pairs[transformer] = sorted(pairs[transformer], key=lambda x: x['score'], reverse=True)

    del model
    gc.collect()

# TODO: Find for each intent how many are correct, to see if each model is better or worse for specific intents
for index in range(0, (len(items.keys()) ** 2) // 20, 1000):
    if index == 0:
        continue
    for key, values in pairs.items():
        incorrect = 0
        correct = 0
        for value in values[:index]:
            i, j = value['index']
            if items[texts[i]] == items[texts[j]]:
                correct += 1
            else:
                incorrect += 1
            # print("{} \t\t {} \t\t Score: {:.4f}".format(sentences[i], sentences[j], pair['score']))
        print(f"{key}: {correct}/{correct + incorrect}, {correct * 100 / (correct + incorrect)}%")
    print("\n")
