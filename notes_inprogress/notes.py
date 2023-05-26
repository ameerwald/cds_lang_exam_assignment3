# this function is from the notebook form Ross using the pretrained word embedding model 
def create_model(max_sequence_len, total_words):
    input_len = max_sequence_len - 1
    model = Sequential()
    # Add Input Embedding Layer - notice that this is different
    model.add(Embedding(
            total_words,
            embedding_dim,
            embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
            trainable=False,
            input_length=input_len)
    )
    # Add Hidden Layer 1 - LSTM Layer
    model.add(LSTM(500))
    model.add(Dropout(0.2))
    # Add Output Layer
    model.add(Dense(total_words, 
                    activation='softmax'))
    model.compile(loss='categorical_crossentropy', 
                    optimizer='adam')
    
    return model


# alternative way to save 
    #f = open("models/max_sequence_len.txt", "w")
    #f.write(str(max_sequence_len))
    #f.close()

   model.save('my_model.h5')
  model = tf.keras.model.load_model('my_model.h5')
    #loaded_model = tf.keras.saving.load_model("model")
  #model = tf.saved_model.load("model")

      # Save the model
    #save_path = os.path.join("models", "model")
    #model.save(save_path)
