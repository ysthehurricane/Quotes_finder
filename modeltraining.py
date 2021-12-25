from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import pandas as pd
import os

df = pd.read_csv(os.path.join(root_path,"quotes.csv"))
data = list(df["quote"][:1500])
print(data[0].lower())

tagged_data = [TaggedDocument(words=word_tokenize(text.lower()), tags=[str(i)]) for i, text in enumerate(data)]

max_epochs = 50
vec_size = 20
alpha = 0.025

model = Doc2Vec(vector_size=vec_size,
                alpha=alpha, 
                min_alpha=0.00025,
                min_count=1,
                workers = 16,
                dm =1)
  
model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    
    print('iteration {0}'.format(epoch))
    
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=20)
    
    # decrease the learning rate
    model.alpha -= 0.0002
    
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

model.save(os.path.join(root_path,"quotes_d2v.model"))
print("Model Saved")





