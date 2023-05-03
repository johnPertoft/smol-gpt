FROM python:3.11
RUN pip install --upgrade pip
RUN pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install \
    datasets \
    einops \
    flax \
    matplotlib \
    orbax \
    optax \
    tokenizers
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENTRYPOINT [ "bash" ]
