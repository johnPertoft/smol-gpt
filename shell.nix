{ pkgs ? import <nixpkgs> { } }:
pkgs.mkShell {
  packages = [
    (pkgs.python3.withPackages (p: with p; [
      datasets
      einops
      flax
      ipython
      jax
      jaxlibWithCuda
      matplotlib
      numpy
      optax
      pytest
      pyyaml
      tokenizers
      tqdm
    ]))
  ];
}