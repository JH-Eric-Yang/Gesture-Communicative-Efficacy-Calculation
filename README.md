# Commmunicative efficacy calculatation

TThis repository includes scripts to calculate entropy and semantic similarity of gestures based on interpretations of naive interpreters.

The semantic similarity calculation (calculate_semantic_similarity.py) is based on <a href="https://nlp.stanford.edu/projects/glove/">GLoVe: Global Vectors for Word Representation</a>. To use the code, please download pre-trained word vectors from the GLoVe project page into the project folder. The script will automatically convert the GLoVe model into Word2Vec format.

If you see the error:

`EOFError: unexpected end of input; is count incorrect or file otherwise damaged?`

This is due to the conversion error of Gensim. Please open the glove_word2vec.txt and make sure the row number is 2196017 for the glove_840B_300d model.