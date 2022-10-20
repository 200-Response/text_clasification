#--------------------------------------------------------------------------
#------------------------- IMPORTS ----------------------------------------
#--------------------------------------------------------------------------
import tensorflow as tf
import os
#--------------------------------------------------------------------------
#---------------------- variables -----------------------------------------
#--------------------------------------------------------------------------
#get current path
currentPath = os.getcwd()


def model_generator(vocab_size, embedding_dim, max_length, training_padded,
                    training_labels, testing_padded, testing_labels):
    #-----------------------------------------------------------------------------
    #-------- AQUI HACEMOS EMBEDING -    NEURAL NETWORK CODE ---------------------
    #-----------------------------------------------------------------------------
    #
    #Un concepto que busca explicar como la informacion segun lo establecido puede ir de un punto a otro como un vector
    # es decir supongamos que testeamos algo que es bueno o malo
    #hay 3 caminos :
    #   * izquierda (malo)
    #   * centro (neutral)
    #   * derecha (bueno)
    #pero si lo representamos como un vector https://www.google.com/search?q=vectores&rlz=1C1CHZN_esMX957MX957&sxsrf=ALiCzsZjUc9_82_CvaW_Gm9ecpIQ5-lZ2g:1664841867297&source=lnms&tbm=isch&sa=X&ved=2ahUKEwiF0-2go8X6AhVcMEQIHRw6BIsQ_AUoAXoECAEQAw&biw=1360&bih=635&dpr=1
    # su posicion cambiaria segun lo establecido y podriamos obtener un texto
    # bueno [1,0]
    #  malo [-1,0],
    # ligeramente bueno inclinqeo q neutras [0.7,0.7]
    #                  Y
    #                  |
    #                  |
    #                  | *[0.7,0.7]
    #                  |
    #                  |
    #                  |
    #                  |
    # [-1,0]-------------------------- X [1,0]

    #NEURAL NETWORK CODE
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size,
                                  embedding_dim,
                                  input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='relu'),
    ])

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()

    #------------------------------------------------------------------------------------------
    #-----AQUI ENTRENAMOS A LA IA - mediante ephochs (ciclos de entrenamiento para una IA)-----
    #------------------------------------------------------------------------------------------
    #
    #  https://ciberseguridad.com/guias/nuevas-tecnologias/machine-learning/epoch/

    num_epochs = 30
    history = model.fit(training_padded,
                        training_labels,
                        epochs=num_epochs,
                        validation_data=(testing_padded, testing_labels),
                        verbose=2)

    model.save(currentPath + "/text_clasification/model_trained_spanish")
    print("Modelo creado exitosamente")