FROM nvcr.io/nvidia/pytorch:22.07-py3

# Install git
RUN apt-get update && apt-get install -y git

WORKDIR /wdr


# Clone the specific repo
RUN git clone --recurse-submodules https://github.com/HazyResearch/hyena-dna.git ../hyena-dna && cd ../hyena-dna

RUN rm -f huggingface.py
RUN rm -f requirements.txt
RUN rm -f Dockerfile



# Move the contents of the cloned repo to the current working directory
RUN mv ../zb-hyena-dna-scripts/* . && \
    rm -rf ../zb-hyena-dna-scripts


# Install the Python requirements
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Copy everything from the current directory into the image
COPY . .

# Run the additional installations
RUN cd flash-attention && pip install . --no-build-isolation && \
    cd csrc/layer_norm && pip install . --no-build-isolation
