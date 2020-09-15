FROM jupyter/base-notebook
RUN conda config --add channels conda-forge && \
    conda config --add channels numba && \
    conda config --add channels anaconda && \
    conda config --env --set always_yes true && \
    conda config --set channel_priority false
RUN conda install -n base bokeh \
osmnx \
networkx \
numpy \
numba \
pyproj \
scipy \
geojson


WORKDIR /home/jovyan/work
COPY stapy /home/jovyan/work/stapy

# set default command to launch when container is run
CMD ["jupyter", "lab", "--ip='0.0.0.0'", "--port=8888", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''"]
