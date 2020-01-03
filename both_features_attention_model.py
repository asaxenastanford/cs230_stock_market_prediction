# import pandas as pd
# import numpy as np
# import tensorflow as tf
# import re
# from nltk.corpus import stopwords
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import median_absolute_error as mae
# from sklearn.metrics import mean_squared_error as mse
# from sklearn.metrics import r2_score as r2
# from sklearn.metrics import accuracy_score as acc
# from sklearn.metrics import f1_score
# from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt

# from keras_self_attention import SeqSelfAttention
# from keras.models import Sequential
# from keras import initializers
# from keras.layers import Dropout, Activation, Embedding, Convolution1D, MaxPooling1D, Input, Dense, merge, BatchNormalization, Flatten, Reshape, Concatenate
# from keras.layers.recurrent import LSTM, GRU
# from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
# from keras.models import Model
# from keras.optimizers import Adam, SGD, RMSprop
# from keras import regularizers
# from keras.layers import *
# from keras.layers import add

# import time
# import nltk
# # nltk.download('stopwords')

# dj = pd.read_csv("/home/ubuntu/.../DowJones.csv") #read in stock prices
# news = pd.read_csv("/home/ubuntu/.../News.csv") #read in news data
# news = news[news.Date.isin(dj.Date)]

# # ## Inspect the data
# dj.head()
# dj.isnull().sum()
# news.isnull().sum()
# news.head()
# print(dj.shape)
# print(news.shape)

# # Compare the number of unique dates. We want matching values.
# print(len(set(dj.Date)))
# print(len(set(news.Date)))
# print(len(set(dj.Date)))
# print(len(set(news.Date)))

# # Calculate the difference in opening prices between the following and current day.
# # The model will try to predict how much the Open value will change beased on the news.
# dj = dj.set_index('Date').diff(periods=1)
# dj['Date'] = dj.index
# dj = dj.reset_index(drop=True)
# # Remove unneeded features
# dj = dj.drop(['High','Low','Close','Volume','Adj Close'], 1)
# dj.head()

# # Remove top row since it has a null value.
# dj = dj[dj.Open.notnull()]

# # Check if there are any more null values.
# dj.isnull().sum()

# # Create a list of the opening prices and their corresponding daily headlines from the news
# price = []
# headlines = []

# for row in dj.iterrows():
#     daily_headlines = []
#     date = row[1]['Date']
#     price.append(row[1]['Open'])
#     for row_ in news[news.Date==date].iterrows():
#         daily_headlines.append(row_[1]['News'])
    
#     # Track progress
#     headlines.append(daily_headlines)
#     if len(price) % 500 == 0:
#         print(len(price))

# # Compare lengths to ensure they are the same
# print(len(price))
# print(len(headlines))

# # Compare the number of headlines for each day
# print(max(len(i) for i in headlines))
# print(min(len(i) for i in headlines))

# # A list of contractions from http://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
# contractions = { 
# "ain't": "am not",
# "aren't": "are not",
# "can't": "cannot",
# "can't've": "cannot have",
# "'cause": "because",
# "could've": "could have",
# "couldn't": "could not",
# "couldn't've": "could not have",
# "didn't": "did not",
# "doesn't": "does not",
# "don't": "do not",
# "hadn't": "had not",
# "hadn't've": "had not have",
# "hasn't": "has not",
# "haven't": "have not",
# "he'd": "he would",
# "he'd've": "he would have",
# "he'll": "he will",
# "he's": "he is",
# "how'd": "how did",
# "how'll": "how will",
# "how's": "how is",
# "i'd": "i would",
# "i'll": "i will",
# "i'm": "i am",
# "i've": "i have",
# "isn't": "is not",
# "it'd": "it would",
# "it'll": "it will",
# "it's": "it is",
# "let's": "let us",
# "ma'am": "madam",
# "mayn't": "may not",
# "might've": "might have",
# "mightn't": "might not",
# "must've": "must have",
# "mustn't": "must not",
# "needn't": "need not",
# "oughtn't": "ought not",
# "shan't": "shall not",
# "sha'n't": "shall not",
# "she'd": "she would",
# "she'll": "she will",
# "she's": "she is",
# "should've": "should have",
# "shouldn't": "should not",
# "that'd": "that would",
# "that's": "that is",
# "there'd": "there had",
# "there's": "there is",
# "they'd": "they would",
# "they'll": "they will",
# "they're": "they are",
# "they've": "they have",
# "wasn't": "was not",
# "we'd": "we would",
# "we'll": "we will",
# "we're": "we are",
# "we've": "we have",
# "weren't": "were not",
# "what'll": "what will",
# "what're": "what are",
# "what's": "what is",
# "what've": "what have",
# "where'd": "where did",
# "where's": "where is",
# "who'll": "who will",
# "who's": "who is",
# "won't": "will not",
# "wouldn't": "would not",
# "you'd": "you would",
# "you'll": "you will",
# "you're": "you are"
# }

# def clean_text(text, remove_stopwords = True):
#     '''Remove unwanted characters and format the text to create fewer nulls word embeddings'''
    
#     # Convert words to lower case
#     text = text.lower()
    
#     # Replace contractions with their longer forms 
#     if True:
#         text = text.split()
#         new_text = []
#         for word in text:
#             if word in contractions:
#                 new_text.append(contractions[word])
#             else:
#                 new_text.append(word)
#         text = " ".join(new_text)
    
#     # Format words and remove unwanted characters
#     text = re.sub(r'&amp;', '', text) 
#     text = re.sub(r'0,0', '00', text) 
#     text = re.sub(r'[_"\-;%()|.,+&=*%.,!?:#@\[\]]', ' ', text)
#     text = re.sub(r'\'', ' ', text)
#     text = re.sub(r'\$', ' $ ', text)
#     text = re.sub(r'u s ', ' united states ', text)
#     text = re.sub(r'u n ', ' united nations ', text)
#     text = re.sub(r'u k ', ' united kingdom ', text)
#     text = re.sub(r'j k ', ' jk ', text)
#     text = re.sub(r' s ', ' ', text)
#     text = re.sub(r' yr ', ' year ', text)
#     text = re.sub(r' l g b t ', ' lgbt ', text)
#     text = re.sub(r'0km ', '0 km ', text)
    
#     # Optionally, remove stop words
#     if remove_stopwords:
#         text = text.split()
#         stops = set(stopwords.words("english"))
#         text = [w for w in text if not w in stops]
#         text = " ".join(text)

#     return text

# # Clean the headlines
# clean_headlines = []

# for daily_headlines in headlines:
#     clean_daily_headlines = []
#     for headline in daily_headlines:
#         clean_daily_headlines.append(clean_text(headline))
#     clean_headlines.append(clean_daily_headlines)

# # Take a look at some headlines to ensure everything was cleaned well
# clean_headlines[0]

# # Find the number of times each word was used and the size of the vocabulary
# word_counts = {}

# for date in clean_headlines:
#     for headline in date:
#         for word in headline.split():
#             if word not in word_counts:
#                 word_counts[word] = 1
#             else:
#                 word_counts[word] += 1
            
# print("Size of Vocabulary:", len(word_counts))

# # Load GloVe's embeddings
# embeddings_index = {}
# with open('/home/ubuntu/newsheadline/glove.840B.300d.txt', encoding='utf-8') as f:
#     for line in f:
#         values = line.split(' ')
#         word = values[0]
#         embedding = np.asarray(values[1:], dtype='float32')
#         embeddings_index[word] = embedding

# print('Word embeddings:', len(embeddings_index))

# # Find the number of words that are missing from GloVe, and are used more than our threshold.
# missing_words = 0
# threshold = 10

# for word, count in word_counts.items():
#     if count > threshold:
#         if word not in embeddings_index:
#             missing_words += 1
            
# missing_ratio = round(missing_words/len(word_counts),4)*100
            
# print("Number of words missing from GloVe:", missing_words)
# print("Percent of words that are missing from vocabulary: {}%".format(missing_ratio))

# # Limit the vocab that we will use to words that appear ≥ threshold or are in GloVe

# #dictionary to convert words to integers
# vocab_to_int = {} 

# value = 0
# for word, count in word_counts.items():
#     if count >= threshold or word in embeddings_index:
#         vocab_to_int[word] = value
#         value += 1

# # Special tokens that will be added to our vocab
# codes = ["<UNK>","<PAD>"]   

# # Add codes to vocab
# for code in codes:
#     vocab_to_int[code] = len(vocab_to_int)

# # Dictionary to convert integers to words
# int_to_vocab = {}
# for word, value in vocab_to_int.items():
#     int_to_vocab[value] = word

# usage_ratio = round(len(vocab_to_int) / len(word_counts),4)*100

# print("Total Number of Unique Words:", len(word_counts))
# print("Number of Words we will use:", len(vocab_to_int))
# print("Percent of Words we will use: {}%".format(usage_ratio))

# # Need to use 300 for embedding dimensions to match GloVe's vectors.
# embedding_dim = 300

# nb_words = len(vocab_to_int)
# # Create matrix with default values of zero
# word_embedding_matrix = np.zeros((nb_words, embedding_dim))
# for word, i in vocab_to_int.items():
#     if word in embeddings_index:
#         word_embedding_matrix[i] = embeddings_index[word]
#     else:
#         # If word not in GloVe, create a random embedding for it
#         new_embedding = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))
#         embeddings_index[word] = new_embedding
#         word_embedding_matrix[i] = new_embedding

# # Check if value matches len(vocab_to_int)
# print(len(word_embedding_matrix))


## Major modifications to baseline start here 

gold = pd.read_csv("/home/ubuntu/.../models/gld.csv") #read in gold data
gold = gold.reindex(index=gold.index[::-1])
gold = gold[gold.Date.isin(dj.Date)]
gold.reset_index(drop=True, inplace=True)
gold.drop_duplicates('Date')

gold = gold.set_index('Date').diff(periods=1)
gold['Date'] = gold.index
gold = gold.reset_index(drop=True)
gold = gold[gold.Open.notnull()]

gold_prices = []
for row in gold.iterrows():
    gold_prices.append(row[1]['Open'])

# Change the text from words to integers
# If word is not in vocab, replace it with <UNK> (unknown)
word_count = 0
unk_count = 0

int_headlines = []

for date in clean_headlines:
    int_daily_headlines = []
    for headline in date:
        int_headline = []
        for word in headline.split():
            word_count += 1
            if word in vocab_to_int:
                int_headline.append(vocab_to_int[word])
            else:
                int_headline.append(vocab_to_int["<UNK>"])
                unk_count += 1
        int_daily_headlines.append(int_headline)
    int_headlines.append(int_daily_headlines)

unk_percent = round(unk_count/word_count,4)*100

print("Total number of words in headlines:", word_count)
print("Total number of UNKs in headlines:", unk_count)
print("Percent of words that are UNK: {}%".format(unk_percent))

# Find the length of headlines
lengths = []
for date in int_headlines:
    for headline in date:
        lengths.append(len(headline))

# Create a dataframe so that the values can be inspected
lengths = pd.DataFrame(lengths, columns=['counts'])


lengths.describe()

# Limit the length of a day's news to 200 words, and the length of any headline to 16 words.
# These values are chosen to not have an excessively long training time and 
# balance the number of headlines used and the number of words from each headline.
max_headline_length = 16
max_daily_length = 200
pad_headlines = []

# add 'blank headlines'
past_headlines = 3
past_padded_headlines = []
for j in range(past_headlines - 1):
    daily_headline = []
    for i in range(max_daily_length):
        pad = vocab_to_int["<PAD>"]
        daily_headline.append(pad)
    past_padded_headlines.append(daily_headline)

for date in int_headlines:
    pad_daily_headlines = []
    for headline in date:
        # Add headline if it is less than max length
        if len(headline) <= max_headline_length:
            for word in headline:
                pad_daily_headlines.append(word)
        # Limit headline if it is more than max length  
        else:
            headline = headline[:max_headline_length]
            for word in headline:
                pad_daily_headlines.append(word)
    
    # Pad daily_headlines if they are less than max length
    if len(pad_daily_headlines) < max_daily_length:
        for i in range(max_daily_length-len(pad_daily_headlines)):
            pad = vocab_to_int["<PAD>"]
            pad_daily_headlines.append(pad)
    # Limit daily_headlines if they are more than max length
    else:
        pad_daily_headlines = pad_daily_headlines[:max_daily_length]
    pad_headlines.append(pad_daily_headlines)
    past_padded_headlines.append(pad_daily_headlines)

concat_past_padded_headlines = [] 
counter = 0
for i in range(len(pad_headlines)):
    all_relevant_headlines = []
    for j in range(past_headlines):
        all_relevant_headlines.append(past_padded_headlines[counter + j])
    counter += 1
    all_relevant_headlines_concatenated = [item for sublist in all_relevant_headlines for item in sublist]
    concat_past_padded_headlines.append(all_relevant_headlines_concatenated)
    
# Normalize opening prices (target values)
max_price = max(price)
min_price = min(price)
mean_price = np.mean(price)
def normalize(price):
    return ((price-min_price)/(max_price-min_price))

# Normalize gold prices (target values)
gold_max_price = max(gold_prices)
gold_min_price = min(gold_prices)
gold_mean_price = np.mean(gold_prices)
def normalize_gold(price):
    return ((price-gold_min_price)/(gold_max_price-gold_min_price))

norm_price = []
for p in price:
    norm_price.append(normalize(p))

gold_norm_price = []
for p in gold_prices:
    gold_norm_price.append(normalize_gold(p))

# Check that normalization worked well
print(min(norm_price))
print(max(norm_price))
print(np.mean(norm_price))

# Split data into training and testing sets.
# Validating data will be created during training.
# X: numerical headlines, Y: normalized pries 
# x_train, x_test, y_train, y_test = train_test_split(pad_headlines, norm_price, test_size = 0.15, random_state = 2)
x_train, x_test, x_gold_train, x_gold_test, y_train, y_test = train_test_split(concat_past_padded_headlines, gold_norm_price, norm_price, test_size = 0.15, random_state = 2)

x_train = np.array(x_train)
x_test = np.array(x_test)
x_gold_train = np.array(x_gold_train)
x_gold_test = np.array(x_gold_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


print("X Train:")
print(x_train)
print(type(x_train))
print("X Gold Train:")
print(x_gold_train)
print(type(x_gold_train))
print("Y Train: ")
print(y_train)

# Check the lengths
print(len(x_train))
print(len(x_test))
print(len(x_gold_train))
print(len(x_gold_test))
print('finished')

filter_length1 = 3
filter_length2 = 5
dropout = 0.5
learning_rate = 0.001
weights = initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=2)
nb_filter = 16
rnn_output_size = 128
hidden_dims = 128
wider = True
deeper = True
in_length = max_daily_length*past_headlines

if wider == True:
    nb_filter *= 2
    rnn_output_size *= 2
    hidden_dims *= 2

def build_model():
    
    headlines_one = Input(shape=(in_length,), dtype='int64', name='headlines_one')

    embedding_layer_one = Embedding(nb_words,embedding_dim,weights=[word_embedding_matrix],input_length=in_length)(headlines_one)
    
    dropout1_one = Dropout(dropout)(embedding_layer_one)
    
    conv1_one = Convolution1D(filters = nb_filter,kernel_size = filter_length1,padding = 'same',activation = 'relu')(dropout1_one)
    
    dropout2_one = Dropout(dropout)(conv1_one)
    
    bidirectional1_one = Bidirectional(LSTM(rnn_output_size,return_sequences=True,activation=None,kernel_initializer=weights,dropout = dropout))(dropout2_one)
    
#     bidirectional1_two = Bidirectional(LSTM(rnn_output_size,return_sequences=True,activation=None,kernel_initializer=weights,dropout = dropout))(bidirectional1_one)
    
    attention1_one = SeqSelfAttention(attention_activation='sigmoid')(bidirectional1_one)
    
    flatten_one = Flatten()(attention1_one)
    
    gold_input = Input(shape=(1,))
        
    conc = Concatenate()([flatten_one,gold_input])
    
    dropout3_one = Dropout(dropout)(conc)
    
    output_model_one = Dense(1, kernel_initializer = weights, name='output')(dropout3_one)
    
    model_one = Model(inputs=[headlines_one,gold_input], outputs=output_model_one, name='functional_model_one')
    
    model_one.compile(loss='mean_squared_error', optimizer=Adam(lr=learning_rate, clipvalue=1.0))
    
    return model_one

model = build_model()
print(model.summary())

# Use grid search to help find a better model
for deeper in [False, True]:
    for wider in [False, True]:
        for learning_rate in [0.001]:
            for dropout in [0.3, 0.5]: 
                model = build_model()
                print()
                print("Current model: Deeper={}, Wider={}, LR={}, Dropout={}".format(
                    deeper,wider,learning_rate,dropout))
                print()
                save_best_weights = '..._deeper={}_wider={}_lr={}_dropout={}.h5'.format(
                    deeper,wider,learning_rate,dropout)

                callbacks = [ModelCheckpoint(save_best_weights, monitor='val_loss', save_best_only=True),
                             EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto'),
                             ReduceLROnPlateau(monitor='val_loss', factor=0.2, verbose=1, patience=3)]

                history = model.fit([x_train,x_gold_train],
                                    y_train,
                                    batch_size=32,
                                    epochs=100,
                                    validation_split=0.15,
                                    verbose=True,
                                    shuffle=True,
                                    callbacks = callbacks)

                
                # summarize history for loss
                plt.plot(history.history['loss'])
                plt.plot(history.history['val_loss'])
                plt.title('model loss')
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.legend(['train', 'validation'], loc='upper left')
                plt.show()


# Need to rebuild model in case it is different from the model that was trained most recently.
model = build_model()
model.load_weights('..._deeper=False_wider=False_lr=0.001_dropout=0.5.h5')
predictions = model.predict([x_test,x_gold_test], verbose = True)


def unnormalize(price):
    '''Revert values to their unnormalized amounts'''
    price = price*(max_price-min_price)+min_price
    return(price)

unnorm_predictions = []
for pred in predictions:
    unnorm_predictions.append(unnormalize(pred))
    
unnorm_y_test = []
for y in y_test:
    unnorm_y_test.append(unnormalize(y))


print("Summary of actual opening price changes")
print(pd.DataFrame(unnorm_y_test, columns=[""]).describe())
print()
print("Summary of predicted opening price changes")
print(pd.DataFrame(unnorm_predictions, columns=[""]).describe())

# Plot the predicted (blue) and actual (green) values
plt.figure(figsize=(12,4))
plt.plot(unnorm_predictions)
plt.plot(unnorm_y_test)
plt.title("Predicted (blue) vs Actual (green) Opening Price Changes")
plt.xlabel("Testing instances")
plt.ylabel("Change in Opening Price")
plt.savefig("PredictedvActual.png")
plt.savefig("PredictedvActual.png")

# Create lists to measure if opening price increased or decreased
direction_pred = []
for pred in unnorm_predictions:
    if pred >= 0:
        direction_pred.append(1)
    else:
        direction_pred.append(0)
direction_test = []
for value in unnorm_y_test:
    if value >= 0:
        direction_test.append(1)
    else:
        direction_test.append(0)

# Calculate errors
mae_score = mae(unnorm_y_test, unnorm_predictions) #median absolute error
rmse = np.sqrt(mse(y_test, predictions)) # root mean squared error
r2 = r2(unnorm_y_test, unnorm_predictions) #R squared error

print("Median absolute error: {}".format(mae_score))
print("RMSE: {}".format(rmse))
print("R squared error: {}".format(r2))

# Calculate F1
f1 = f1_score(direction_test,direction_pred)
print("F1 Score: {}".format(f1))

# Calculate Confusion Matrix
cm = confusion_matrix(direction_test,direction_pred)
print("CM Score: {}".format(cm))

# Calculate if the predicted direction matched the actual direction
direction = acc(direction_test, direction_pred)
direction = round(direction,4)*100
print("Predicted values matched the actual direction {}% of the time.".format(direction))
