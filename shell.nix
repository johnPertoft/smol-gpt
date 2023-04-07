{ pkgs ? import <nixpkgs> { } }:
pkgs.mkShell {
  packages = [
    (pkgs.python3.withPackages (p: with p; [
      einops
      flax
      jax
      jaxlib
      numpy
      pyyaml
    ]))
  ];
}