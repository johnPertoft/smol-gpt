{
  description = "A Nix flake for a Jax implementation of a GPT model";
  
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-22.11";
    nixpkgs-unstable.url = "github:nixos/nixpkgs/nixos-unstable";
  };
  
  outputs = { self, nixpkgs, nixpkgs-unstable }: 
  let
    system = "x86_64-linux";
    pkgs = nixpkgs.legacyPackages.${system};
    #pkgs = nixpkgs-unstable.legacyPackages.${system};
  in
  {
    devShells.${system}.default = pkgs.mkShell {
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
    };
    
    packages.${system}.default = pkgs.hello;
  };
}
