def main():
    text = """Data plays a vital role in our everyday life.
    Directly or indirectly, for daily life decisions, we depend on some data, be it choosing a novel to read from a list of books, buying a thing after considering the budget, and so on.
    Have you ever imagined searching for something on Google or Yahoo generates a lot of data?
    This data is essential to analyze user experiences.
    Getting recommendations on various e-commerce websites after buying a product and tracking parcels during delivery are part of Data Analytics which involves analyzing the raw data to make informed decisions.
    But this raw data does not help make decisions if it has some redundancy, inconsistency, or inaccuracy.
    Therefore, this data needs to be cleaned before considering for analysis."""

    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import Tokenizer

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])

    total_words = len(tokenizer.word_index) + 1

    input_sequences = []
    for sentence in text.split('\n'):
        tokenized_sentence = tokenizer.texts_to_sequences([sentence])[0]

        for i in range(1, len(tokenized_sentence)):
            input_sequences.append(tokenized_sentence[:i+1])

    max_len = max([len(x) for x in input_sequences])

    # add 0's
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    padded_input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='pre')

    x = padded_input_sequences[:,:-1]
    y = padded_input_sequences[:,-1]

    from tensorflow.keras.utils import to_categorical

    input_seq_len = x.shape[1]
    y = to_categorical(y, num_classes=total_words)

    # Model Building
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, LSTM, Dense

    model = Sequential()
    model.add(Embedding(total_words, 100, input_length=input_seq_len))
    model.add(LSTM(150))
    model.add(Dense(total_words, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.build(input_shape=(None, input_seq_len))

    print(model.summary())

    # Fitting the model
    model.fit(x, y, epochs=100)

    # Test the model
    text2 = "Data plays a vital"

    tokenized_text = tokenizer.texts_to_sequences([text2])[0]
    padded_text = pad_sequences([tokenized_text], maxlen=33, padding='pre')
    model.predict(padded_text)

    import numpy as np
    pos = np.argmax(model.predict(padded_text))

    for word, index in tokenizer.word_index.items():
        if index == pos:
            print(word)

if __name__ == "__main__":
    main()