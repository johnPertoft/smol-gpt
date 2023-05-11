{ pkgs, fetchPypi, tensorstore }:
pkgs.python3Packages.buildPythonPackage rec {
  pname = "orbax_checkpoint";
  version = "0.2.2";
  format = "wheel";

  src = fetchPypi {
    inherit pname version format;
    dist = "py3";
    python = "py3";
    sha256 = "8e1a385e28d2817a477dcdab601081bebb127b2c0fa3747a5e1a53f29f103bfa"; 
  };

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
