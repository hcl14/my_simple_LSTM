# my_simple_LSTM


My attempt of realization of a simple QA LSTM with attention and cosine measure between question and answer, following the project https://github.com/sujitpal/dl-models-for-qa <br>
but trying to use cosine distance instead of incorrect Dense layer. Python3 is needed because of pickle highest protocol. Memory requirements are >6 GB RAM.<br>

DOES NOT WORK PROPERLY AT THE MOMENT - trains only to 75%(i.e. outputs all false. The same problem as in https://github.com/sujitpal/dl-models-for-qa). I keep working to make it train properly.<br>


![picture](pic.png?raw=true)



I use SciQ dataset: http://data.allenai.org/sciq/  (already included in the data directory, ad processed data is already pickled).

I use my custom word2vec embeddings with keyphrases (keyphrase segmentation code included).

keypharses file: https://drive.google.com/file/d/1qHr02DYcctdASRZ-YbglQ51g8urbCwf8/view?usp=sharing  <br>
embeddings file (you can import Gensim and use ordinary Google word2vec as well, but you need to throw out keyphrase preprocessing): https://drive.google.com/file/d/1TrDb_peOjP3KOyOapHH3OXx03MHrYnux/view?usp=sharing  <br>


You can yse keyphrases file to segment your new data, function call is commented out at the moment. Program already loads preprocessed data from `processed_input.pickle`.



Literature: LSTM-BASED DEEP LEARNING MODELS FOR NON-FACTOID ANSWER SELECTION, Ming Tan, Cicero dos Santos, Bing Xiang & Bowen Zhou  https://arxiv.org/abs/1511.04108
