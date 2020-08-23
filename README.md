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

<b> LSTM TF 2.0</b>

Same use of books folder, but different neural network. One LSTM, unidirectional (and other stuff). This notebook has been run on books2 from scratch. It quickly overfits on such a small sample of text. Allowing caps and adding more texts should improve results, but adding computing time, and the need for memory.

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
