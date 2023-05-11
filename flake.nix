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

      #pkgs = nixpkgs.legacyPackages.${system}.extend (nixpkgs.lib.foldl nixpkgs.lib.composeExtensions (_: _: { }) [
      #  (self: super: {
      #    config = {
      #      allowUnfree = true;
      #    };
      #  })
      #]);
    in
    {
      packages.pythonEnvironment = (pkgs.python3.withPackages (ps: with ps; [
        datasets
        einops
        #flax
        jax
        jaxlibWithCuda
        matplotlib
        numpy
        optax
        pytest
        tokenizers
      ]));

      devShells.default = pkgs.mkShell {
        buildInputs = [
          self.packages.${system}.pythonEnvironment  
        ];
      };
    });
}
