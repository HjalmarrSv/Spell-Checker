# Spell-Checker

The objective of this project is to build a model that can take a sentence with spelling mistakes as input, and output the same sentence, but with the mistakes corrected. 


See [article](https://medium.com/@Currie32/creating-a-spell-checker-with-tensorflow-d35b23939f60) that explains how to create the input data (the sentences with spelling mistakes) for this model.

<b>New in TF1.4</b>

If books do not contain all the letters in "letters = ['a','b','c',...]", there will be errors. TF1.4 have Swedish letters, remove these if running English texts.

Running big datasets, e.g. wikipedia, and long sentences will take months on cpu only. Start very small!

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
