Creating virtual env in python:

## Ubuntu way of creating a venv
# setup
```
$ sudo apt-get install python3-pip
$ sudo pip3 install virtualenv 
```

# creating env (any one of below options)
```
$ virtualenv myenv
$ virtualenv -p /usr/bin/python2.7 myenv
$ virtualenv -p python3.6 myenv
```

# activate
```
$ source venv/bin/activate
```

# deactivate
```
$ deactivate
```

## Anaconda python

# reference
[Install anaconda python on ubuntu](https://www.digitalocean.com/community/tutorials/how-to-install-the-anaconda-python-distribution-on-ubuntu-16-04)

# setup anaconda
```
$ cd /tmp
$ curl -O https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
$ sha256sum Anaconda3-5.0.1-Linux-x86_64.sh	// cryptographic hash verification
$ bash Anaconda3-5.0.1-Linux-x86_64.sh
$ source ~/.bashrc
$ conda list
```

# create environment
```
$ conda search "^python"	// check versions available
$ ls -ls /usr/bin/python* 	// check versions available
$ conda create --name my_env python=3.6
```

# activate
```
$ source activate my_env
```

# deactivate
```
$ source deactivate
```

# see all environments in conda
```
$ conda info --envs
```

# install new package
```
$ conda install --name my_env35 numpy
```

# remove an venv
```
$ conda remove --name my_env35 --all
```

# update conda(commanline utility for Anaconda)
```
$ conda update conda
```

# update anaconda
```
$ conda update anaconda
```
