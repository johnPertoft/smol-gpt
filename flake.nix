{
  description = "A small GPT implementation in Jax.";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }: flake-utils.lib.eachDefaultSystem (system:
    let
      pkgs = nixpkgs.legacyPackages.${system};
    in
    {
      packages.pythonEnvironment = (pkgs.python3.withPackages (ps: with ps; [
        datasets
        einops
        #flax
        jax
        jaxlib-bin
        #jaxlibWithCuda
        matplotlib
        numpy
        optax
        pytest
        tokenizers
      ]));
      #datasets
      #einops
      #flax
      #ipython
      #jax
      #jaxlibWithCuda
      #matplotlib
      #numpy
      #optax
      #pytest
      #pyyaml
      #tokenizers

      devShells.default = pkgs.mkShell {
        buildInputs = [
          self.packages.${system}.pythonEnvironment  
        ];
      };
    });
}
