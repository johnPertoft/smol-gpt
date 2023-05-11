{ pkgs, fetchPypi, tensorstore }:
pkgs.python3Packages.buildPythonPackage rec {
  pname = "orbax-checkpoint";
  version = "0.2.2";
  format = "pyproject";

  src = fetchPypi {
    inherit pname version;
    hash = "sha256-n2omDj4u/oXB6XVZnPyNoMaRFh9D+2fFRVfTYmXJUSc=";
  };

  nativeBuildInputs = with pkgs.python3Packages; [ flit-core ];
  
  # TODO: This is probably a dumb idea.
  prePatch = ''
    substituteInPlace pyproject.toml \
        --replace 'jax >= 0.4.8' 'jax >= 0.4.5'
  '';

  propagatedBuildInputs = with pkgs.python3Packages; [
    absl-py
    cached-property
    etils
    jax
    jaxlibWithCuda
    msgpack
    nest-asyncio
    pyyaml
    tensorstore
    typing-extensions
  ];
}
