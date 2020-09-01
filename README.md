# Spell-Checker

The objective of this project is to build a model that can take a sentence with spelling mistakes as input, and output the same sentence, but with the mistakes corrected. 

See [article](https://medium.com/@Currie32/creating-a-spell-checker-with-tensorflow-d35b23939f60) that explains how to create the input data (the sentences with spelling mistakes) for this model.

Do not be afraid of changing or adding/removing in the code as you run the notebook. The notebook is an excellent tool for experimenting.

<b>New in TF1.4</b>

If books do not contain all the letters in "letters = ['a','b','c',...]", there will be errors. TF1.4 have Swedish letters, remove these if running English texts.

Running big datasets, e.g. wikipedia, and long sentences will take months on cpu only. Start very small!

<b>TF 1.9 and books2</b>
  
This notebook has been run in Anaconda3 with Tensorflow 1.9. My environment did not support running on the GPU, which would have run about 100 times faster (even on a 2060).
 
Books 2 are Swedish government texts which are free to distribute. Use any texts you like.
 
Note, that the notebook saves checkpoints. It should be relatively easy to add code for reading any such checkpoint, to be inserted before the training code. Use the code from the inferencing part for example. In TF 2.0 it is possible to save weights, and reload and run specific sets of weights any number of times. An other improvement could be to save the model, and reload it. Although, since the version 2.x of TF is very different, it may be more useful to work on a newer codebase.

<b>LSTM TF 2.0</b>

Same use of books folder, but different neural network. One LSTM, unidirectional (and other stuff). This notebook has been run on books2 from scratch. It quickly overfits on such a small sample of text. Allowing caps and adding more texts should improve results, but adding computing time, and the need for memory.

The model seems to work also on TF 2.1. It is not tested on 2.2 or 2.3.

<b>Errors</b>

Lots of errors are caused by large work units on small gpus. You may need to test smaller batch sizes if gpu memory is limited. Try 32 - 16 - 8 - 4 - 2 - 1, in descending order,  to see when the error goes away.
  
There seems to be a rounding error in TF regarding batches or validation. Exchange the following two lines, with their equivalents in the code.

    encoder_inputs = Input(batch_input_shape=(None, max_encoder_seq_length, num_encoder_tokens))
  
    model.fit(
        [encoder_input_data, decoder_input_data],
        decoder_target_data,
        batch_size=batch_size,
        epochs=epochs)
 

<b>Improving LSTM TF 2.0</b>

Changing to bidirectional LSTM is an improvement. Notice the 3 differences for Concatenate/Average in the comments. Replace existing with the following code.

    # Define an input sequence and process it
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder = Bidirectional(LSTM(latent_dim, return_state=True))
    encoder_outputs, forward_h, forward_c, backward_h, backward_c  = encoder(encoder_inputs)
    state_h = Concatenate()([forward_h, backward_h])#Average()([forward_h, backward_h])
    state_c = Concatenate()([forward_c, backward_c])#Average()([forward_c, backward_c])

    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, num_decoder_tokens))

    # decoder
    decoder_lstm = LSTM(latent_dim*2, return_sequences=True, return_state=True)#LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

You might get strange errors about dnn not found or similar. This code can help (insert after import tensorflow as tf).

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if not gpus:
        print("No GPU available")
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

<b>Create environment to run the Jupyter notebook in linux</b>

I may have forgotten something. Something may have changed since. Please study the conda documentation if you have problems.

The original file has been changed to run on TF 1.4.

In a Terminal paste:<br>
conda create -n tensorflow1.4 python=3.5<br>
conda activate tensorflow1.4<br>
conda install tensorflow=1.4    or    pip install tensorflow==1.4<br>
conda install -c anaconda pandas<br>
conda install -c anaconda scikit-learn<br>
conda install anaconda-navigator<br>

Run Anaconda:<br>
anaconda-navigator

Choose Jupyter notebook and navigate to the .ipynb file. You need to have a ./books subfolder, below where the file is, with the books in.


<b>In Windows</b>

Install Anaconda and clone the base installation. Install above apps by opening a terminal in the cloned instance. Possibly you will only be able to install only a few TF versions. Try 1.9 for any code 1.x, or else 2.0 if you are willing to change the code.

When running and you encounter crashes, remove all graphics applications in the Tray, e.g. [manufacturer] settings and [manufacturer] experience.
