# Multiple librairies are required
To create a conda environement, the requirements_conda.txt file can be used.
<br>
To create a virtual environement with pip, the requirements_pip.txt file can be used.
<br>
<br>

Note that the [torchmeta](https://github.com/tristandeleu/pytorch-meta) package is required but there is some compatibility issue with the latest pytorch releases, at least at time of upload. To solve this, the dataset files import shall be removed from the torchmeta.__init __.py file. 