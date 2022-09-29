# Demonstration Notebook for public github and data

This directory contains a Dockerfile that simplifies the process of building
and running an example usage of MagNav.jl. 

## Building the Docker image
It can be run by installing Docker, changing to this directory, and building
the image via:

```docker build --tag magnav .```

This will take a while, as it installs julia and MagNav.jl into the image
and downloads support files. The image is large, about 9.5 GB. 

## Starting the Docker container
Once that is completed, either start the container via Docker Desktop
or type the following on the command line:

```docker run -p 8888:8888 -v `pwd`:/home/joyvan/work magnav```

Note that this shares the current directory (volume) with the container
with the -v switch. Alternatively, this can be dropped and one can upload
the Jupyter notebook (`Demo.ipynb`) manually:

```docker run -p 8888:8888 magnav```

Note that if you do this, subsequent changes to the notebook occur inside
the container only; you will need to manually download them from the browser
if you want to keep them on your native filesystem.

## Running the Demonstration

The container will spin up and provide a URL for you to copy into a 
local browser. It will look something like this:
http://127.0.0.1:8888/lab?token=f07a5b18c83b37b899689308e4daae6b8f2eeb8fae6f7840

