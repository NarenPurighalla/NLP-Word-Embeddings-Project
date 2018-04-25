from gensim.models import KeyedVectors

N = 5 # top n words

wordlist = ["director","great","shot","news","man"]
corpora = ["imdb","reuters"]
windows = ["3","5","7","9"]
archs   = ["cbow","skip"]

def add_nested(d, keys, value):
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value

print("Loading models...")
values = [
    [['imdb','3','cbow'], KeyedVectors.load_word2vec_format("models/model-imdb-3-cbow", binary=False, limit=100000)],
    [['imdb','5','cbow'], KeyedVectors.load_word2vec_format("models/model-imdb-5-cbow", binary=False, limit=100000)],
    [['imdb','7','cbow'], KeyedVectors.load_word2vec_format("models/model-imdb-7-cbow", binary=False, limit=100000)],
    [['imdb','9','cbow'], KeyedVectors.load_word2vec_format("models/model-imdb-9-cbow", binary=False, limit=100000)],
    [['imdb','3','skip'], KeyedVectors.load_word2vec_format("models/model-imdb-3-skip", binary=False, limit=100000)],
    [['imdb','5','skip'], KeyedVectors.load_word2vec_format("models/model-imdb-5-skip", binary=False, limit=100000)],
    [['imdb','7','skip'], KeyedVectors.load_word2vec_format("models/model-imdb-7-skip", binary=False, limit=100000)],
    [['imdb','9','skip'], KeyedVectors.load_word2vec_format("models/model-imdb-9-skip", binary=False, limit=100000)],
    [['reuters','3','cbow'], KeyedVectors.load_word2vec_format("models/model-reuters-3-cbow", binary=False, limit=100000)],
    [['reuters','5','cbow'], KeyedVectors.load_word2vec_format("models/model-reuters-5-cbow", binary=False, limit=100000)],
    [['reuters','7','cbow'], KeyedVectors.load_word2vec_format("models/model-reuters-7-cbow", binary=False, limit=100000)],
    [['reuters','9','cbow'], KeyedVectors.load_word2vec_format("models/model-reuters-9-cbow", binary=False, limit=100000)],
    [['reuters','3','skip'], KeyedVectors.load_word2vec_format("models/model-reuters-3-skip", binary=False, limit=100000)],
    [['reuters','5','skip'], KeyedVectors.load_word2vec_format("models/model-reuters-5-skip", binary=False, limit=100000)],
    [['reuters','7','skip'], KeyedVectors.load_word2vec_format("models/model-reuters-7-skip", binary=False, limit=100000)],
    [['reuters','9','skip'], KeyedVectors.load_word2vec_format("models/model-reuters-9-skip", binary=False, limit=100000)]]

models = {}
for keys, value in values:
    add_nested(models, keys, value)

similarity_dict = {}
for word in wordlist:
    for c in corpora:
        for w in windows:
            for a in archs:
                add_nested(similarity_dict, [word,c,w,a], None)

for corpus in models:
    for win in models[corpus]:
        for arch in models[corpus][win]:
            for word in wordlist:
                similarity = models[corpus][win][arch].most_similar(word,topn=N)
                print(corpus,win,arch,word,similarity)
                similarity_dict[word][corpus][win][arch] = similarity

def compare_neighbors(n1,n2):
    neighbors1 = [n[0] for n in n1]
    neighbors2 = [n[0] for n in n2]

    result = 0
    for neighbor in neighbors1:
        if neighbor in neighbors2:
            result = result + 1

    return result/N

def compare_by(variable,words):
    if variable == "c":
        for word in words:
            print("###",word,"###")
            for i in range(len(corpora)-1):
                c1 = corpora[i]
                for c2 in corpora[i+1:]:
                    print("\t",c1,"/",c2)
                    for win in windows:
                        for arch in archs:
                            n1 = similarity_dict[word][c1][win][arch]
                            n2 = similarity_dict[word][c2][win][arch]
                            print("\t\t",win,arch,compare_neighbors(n1,n2))
    elif variable == "w":
        for word in words:
            print("###",word,"###")
            for i in range(len(windows)-1):
                w1 = windows[i]
                for w2 in windows[i+1:]:
                    print("\t",w1,"/",w2)
                    for corpus in corpora:
                        for arch in archs:
                            n1 = similarity_dict[word][corpus][w1][arch]
                            n2 = similarity_dict[word][corpus][w2][arch]
                            print("\t\t",corpus,arch,compare_neighbors(n1,n2))
    elif variable == "a":
        for word in words:
            print("###",word,"###")
            for i in range(len(archs)-1):
                a1 = archs[i]
                for a2 in archs[i+1:]:
                    print("\t",a1,"/",a2)
                    for corpus in corpora:
                        for win in windows:
                            n1 = similarity_dict[word][corpus][win][a1]
                            n2 = similarity_dict[word][corpus][win][a2]
                            print("\t\t",corpus,win,compare_neighbors(n1,n2))
    else:
        print("Could not understand:",variable,", please use 'c', 'w', or 'a'")
