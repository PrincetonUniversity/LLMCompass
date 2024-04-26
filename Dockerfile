# Start with a base image that includes Miniconda to manage our environment
FROM continuumio/miniconda3

# Set the working directory in the container to /app
WORKDIR /app

# Create the conda environment
COPY environment.yml /app/environment.yml
RUN conda env create -f /app/environment.yml

# Initialize conda in bash shell
RUN echo "source activate llmcompass_ae" > ~/.bashrc
ENV PATH /opt/conda/envs/llmcompass_ae/bin:$PATH

# Clone your GitHub repository
RUN git clone https://github.com/HenryChang213/LLMCompass_ISCA_AE.git /app/LLMCompass_ISCA_AE
RUN cd /app/LLMCompass_ISCA_AE && git submodule init && git submodule update --recursive

# Expose the port your app runs on
EXPOSE 8000


