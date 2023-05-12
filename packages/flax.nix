{ pkgs, fetchFromGitHub, tensorstore }:
pkgs.python3Packages.buildPythonPackage rec {
  pname = "flax";
  version = "0.6.5";

  src = fetchFromGitHub {
    owner = "google";
    repo = pname;
    rev = "refs/tags/v${version}";
    hash = "sha256-Vv68BK83gTIKj0r9x+twdhqmRYziD0vxQCdHkYSeTak=";
  };

  # TODO: This is probably dumb.
  # TODO: We've already built orbax-checkpoint, but it didn't seem
  # to work when passing it in. Because it's not matching "orbax"
  # exactly maybe? 
  patches = [
    ./flax.patch
  ];

  doCheck = false;

  # TODO: It doesn't seem to pick up jax and optax which are already
  # included via nixpkgs. How come?
  # Just removing them from the setup.py for now
  propagatedBuildInputs = with pkgs.python3Packages; [
    matplotlib
    msgpack
    numpy
    pyyaml
    rich
    tensorstore
    typing-extensions
  ];
}
