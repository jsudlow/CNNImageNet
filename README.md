Written in Python.

The code requires a complex soup of libraries, python packages and bindings and invidual vendor files from nvidia.
 
Here is what I did to make this work on my system.

If you don't have python installed use this
Python 3.7 64 bit

If you have never installed tensorflow get the tf-nightly builds.
pip install tf-nightly

If you do have tensorflow installed - then you need to uninstall all the packages with pip. After that, you need to actually go into your python installation. Find the site-packages directory and delete all the folders that have to do with tensorflow. After that, you can then proceed with the tf-nightly install.

Install tensorboard
pip install tensorboard

Install pydot, pydotplus and graphviz bindings
pip install pydot
pip install pydotplus
pip install graphviz

Go to graphviz'z website and download their binary for your system. The graphviz pip package is only the bindings...you need the actual binaries on your system.

Go to nvdia's cuDNN website. Make an account. Get the GPU computing toolkit. The version that worked for me was 8.04 for cuda 10.1

I'm sure I've forgotten a thing or two along the way. Best of luck.

