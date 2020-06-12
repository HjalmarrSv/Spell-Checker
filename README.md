# Spell-Checker

The objective of this project is to build a model that can take a sentence with spelling mistakes as input, and output the same sentence, but with the mistakes corrected. The data that we will use for this project will be twenty popular books from [Project Gutenberg](http://www.gutenberg.org/ebooks/search/?sort_order=downloads). Our model is designed using grid search to find the optimal architecture, and hyperparameter values. The best results, as measured by sequence loss with 15% of our data, were created using a two-layered network with a bi-direction RNN in the encoding layer and Bahdanau Attention in the decoding layer. [FloydHub's](https://www.floydhub.com/) GPU service was used to train the model.

All of the books that I used for training can be found in books.zip.

To view my work most easily, see the .ipynb file.

I wrote an [article](https://medium.com/@Currie32/creating-a-spell-checker-with-tensorflow-d35b23939f60) that explains how to create the input data (the sentences with spelling mistakes) for this model.

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
