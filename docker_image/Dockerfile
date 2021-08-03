FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

ARG UID
ARG GID
ARG USER
ARG GROUP

SHELL [ "/bin/bash", "--login", "-c" ]

# install utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         apt-utils \
         cmake \
         nano \
         bc \
         jq \
         git \
         curl \
         ca-certificates \
         sudo \
         bzip2 \
         libx11-6 \
         git \
         wget \
         ssh-client \
         libjpeg-dev \
         bash-completion \
         libgl1-mesa-dev \
         ffmpeg \
         tmux \
         htop \
         nfs-common \
         cifs-utils \
         zip \
         unzip \
         pydf \
         nnn \
         ncdu \
         aria2 \
         mdadm \
         net-tools \
         uidmap \
         openslide-tools \
         libjemalloc-dev \
         libpng-dev && \
     rm -rf /var/lib/apt/lists/*

# Create a non-root user
ENV HOME=/home/$USER

RUN addgroup --gid $GID $GROUP \
    && adduser --disabled-password \
    --gecos "" \
    --uid $UID \
    --gid $GID \
    --shell /bin/bash \
    --home $HOME \
    $USER 
WORKDIR $HOME
# switch to that user
# USER $USER

# install miniconda
ENV MINICONDA_VERSION=py38_4.8.3
# if you want a specific version (you shouldn't) replace "latest" with that, e.g. ENV MINICONDA_VERSION py38_4.8.3

ENV CONDA_DIR=$HOME/miniconda3
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-$MINICONDA_VERSION-Linux-x86_64.sh -O ~/miniconda.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh

# add conda to path (so that we can just use conda install <package> in the rest of the dockerfile)
ENV PATH=$CONDA_DIR/bin:$PATH

# make conda activate command available from /bin/bash --login shells
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.profile

# make conda activate command available from /bin/bash --interative shells
RUN conda init bash


# build the conda environment
ENV ENV_PREFIX=$HOME/env

RUN conda update --name base --channel defaults conda && \
conda install python==3.8 \
        pytorch==1.8.1=py3.8_cuda10.2_cudnn7.6.5_0 \
        torchvision=0.9.1=py38_cu102 \
        cudatoolkit=10.2.89=hfd86e86_1 \
        scikit-learn==0.24.2=py38hdc147b9_0 \
        matplotlib==3.4.2=py38h578d9bd_0 \ 
        ipykernel==5.5.5=py38hd0cf306_0 \
        pandas==1.2.4=py38h1abd341_0 \
        ipywidgets==7.6.3=pyhd3deb0d_0 \
        umap-learn==0.5.1=py38h578d9bd_1 \
        scikit-image==0.18.1=py38h51da96c_0 \
        tabulate==0.8.9=pyhd8ed1ab_0 \
        colorcet==2.0.6=pyhd8ed1ab_0 \
        datashader==0.13.0=pyh6c4a22f_0 \
        bokeh==2.3.2=py38h578d9bd_0 \
        holoviews==1.14.4=pyhd8ed1ab_0 \
        h5py==3.2.1=mpi_openmpi_py38h45a5288_0 \
        easydict==1.9=py_0 \
        wandb==0.10.31=pyhd8ed1ab_0 \
        tqdm==4.61.0=pyhd8ed1ab_0 \
        openpyxl==3.0.7=pyhd8ed1ab_0 \
        shapely==1.7.1=py38haeee4fe_5 \
        -c pytorch -c nvidia -c conda-forge -y \
    && conda clean --all --yes

ENV SHELL=/bin/bash
RUN pip install --no-cache-dir \ 
         jupyter jupyterlab termcolor tensorboard

# Use C.UTF-8 locale to avoid issues with ASCII encoding
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

CMD ["/bin/bash"]
