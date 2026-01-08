FROM continuumio/miniconda3:24.1.2-0

SHELL ["bash", "-lc"]

WORKDIR /opt/env
COPY environment.yml /opt/env/environment.yml

RUN conda env create -f /opt/env/environment.yml && \
    conda clean -afy

RUN conda config --set auto_activate_base false

RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate kaggle_ml" >> ~/.bashrc

WORKDIR /workspace
COPY . /workspace

CMD ["bash"]