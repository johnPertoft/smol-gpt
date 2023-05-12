{
  description = "A small GPT implementation in Jax.";
  
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }: flake-utils.lib.eachDefaultSystem (system:
    let
      pkgs = import nixpkgs { 
        inherit system;
        config = { allowUnfree = true; }; 
      };
      
      tensorstore = pkgs.callPackage ./packages/tensorstore.nix { };
      orbax-checkpoint = pkgs.callPackage ./packages/orbax-checkpoint.nix { inherit tensorstore; };
      flax = pkgs.callPackage ./packages/flax.nix { inherit tensorstore; };
    in
    {
      packages.pythonEnvironment = (pkgs.python3.withPackages (ps: [
        ps.datasets
        ps.einops
        ps.jax
        ps.jaxlibWithCuda
        ps.matplotlib
        ps.numpy
        ps.optax
        ps.protobuf
        ps.pytest
        ps.tensorboardx
        ps.tokenizers
        flax
        orbax-checkpoint
      ]));

      devShells.default = pkgs.mkShell {
        buildInputs = [
          self.packages.${system}.pythonEnvironment  
        ];

        #LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
      };
    });
}
