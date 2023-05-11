{ pkgs, fetchFromGitHub, tensorstore }:
pkgs.python3Packages.buildPythonPackage rec {
  pname = "orbax_checkpoint";
  version = "0.2.2";
  format = "pyproject";

  src = fetchFromGitHub {
    owner = "google";
    repo = "orbax";
    rev = "f2e63dfdf724ee59c48ade06549a367717f48c6c";
    hash = "sha256-1J071Bd8dq16NS2c+/j0CtGODyAvOIuQr1Td60Qb0Ro=";
  };
  sourceRoot = "source/checkpoint";

  # TODO: Try patching minimum jax version requirement?

  propagatedBuildInputs = with pkgs.python3Packages; [
    absl-py
    etils
    nest-asyncio
    pyyaml
    tensorstore
    typing-extensions
  ];
}
