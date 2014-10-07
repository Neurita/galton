.. -*- mode: rst -*-

galton
======

Univariate Voxel Analysis and Correlations Tools

Named after [Francis Galton](http://en.wikipedia.org/wiki/Francis_Galton) who created the statistical concept of correlation and widely promoted regression toward the mean.

.. image:: https://secure.travis-ci.org/neurita/galton.png?branch=master
    :target: https://travis-ci.org/neurita/galton
.. image:: https://coveralls.io/repos/neurita/galton/badge.png
    :target: https://coveralls.io/r/neurita/galton


Dependencies
============

Please see the requirements.txt and pip_requirements.txt files.

Install
=======

Before installing it, you need all the requirements installed.
These are listed in the requirements.txt files.
The best way to install them is running the following command:

    for r in \`cat galton/requirements.txt\`; do pip install $r; done

This package uses distutils, which is the default way of installing
python modules. To install in your home directory, use::

    python setup.py install --user

To install for all users on Unix/Linux::

    python setup.py build
    sudo python setup.py install


Development
===========

Code
----

Github
~~~~~~

You can check the latest sources with the command::

    git clone https://www.github.com/Neurita/galton.git

or if you have write privileges::

    git clone git@github.com:Neurita/galton.git

If you are going to create patches for this project, create a branch for it 
from the master branch.

The stable releases are tagged in the repository.


Testing
-------

TODO
