{ pkgs ? import <nixpkgs> { } }:
pkgs.mkShell {
  packages = [
    (pkgs.python3.withPackages (p: with p; [
      datasets
      einops
      flax
      ipython
      jax
      jaxlib
      numpy
      optax
      pyyaml
      tokenizers
      tqdm
    ]))
  ];
}