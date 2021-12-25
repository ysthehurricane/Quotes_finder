from gensim.models.doc2vec import Doc2Vec
from nltk.tokenize import word_tokenize
import pandas as pd
import math
import os

root_path = os.path.abspath(os.curdir)

class simQuotes:
    def __init__(self, inputQuote):
        self.model = Doc2Vec.load(os.path.join(root_path,"quotes_d2v.model"))
        self.quotes = pd.read_csv(os.path.join(root_path,"quotes.csv"))['quote'] 
        self.inputQuote = inputQuote
        
    def similarityFind(self):
        sim_quotes = dict()
        test_data = word_tokenize(self.inputQuote.lower())
        v1 = self.model.infer_vector(test_data)
        similar_doc = self.model.dv.most_similar([v1])

        for d_ in similar_doc[0:5]:
            val = self.quotes[int(d_[0])]
            sim_quotes[val] = math.ceil(d_[1]*100)
        return sim_quotes

# obj = simQuotes("Dreams are for the real")
# sim_ = obj.simQuotes()
# sim_




