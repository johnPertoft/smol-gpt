{
  description = "A small GPT implementation in Jax.";
  
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }: flake-utils.lib.eachDefaultSystem (system:
    let
      #pkgs = nixpkgs.legacyPackages.${system};
      pkgs = import nixpkgs { 
        inherit system;
        config = { allowUnfree = true; }; 
      };
      
      tensorstore = pkgs.callPackage ./packages/tensorstore.nix { };
      orbax_checkpoint = pkgs.callPackage ./packages/orbax-checkpoint.nix { inherit tensorstore; };
      
      #flax = pkgs.callPackage ./packages/flax.nix { inherit orbax };
    in
    {
      packages.pythonEnvironment = (pkgs.python3.withPackages (ps: [
        ps.datasets
        ps.einops
        #flax
        ps.jax
        ps.jaxlibWithCuda
        ps.matplotlib
        ps.numpy
        ps.optax
        #orbax
        orbax_checkpoint
        tensorstore
        ps.pytest
        ps.tokenizers
      ]));

      devShells.default = pkgs.mkShell {
        buildInputs = [
          self.packages.${system}.pythonEnvironment  
        ];

        #LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
      };
    });
}
