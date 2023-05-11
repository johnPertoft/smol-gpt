{ pkgs, fetchPypi }:
pkgs.python3Packages.buildPythonPackage rec {
  pname = "orbax";
  version = "0.1.7";
  format = "wheel";

  src = fetchPypi {
    inherit pname version format;
    dist = "py3";
    python = "py3";
    sha256 = "67c7ce52b5476202af84977e8db03dede6c009b5d1f1095acfc175578038449b"; 
  };

  propagatedBuildInputs = with pkgs.python3Packages; [
    etils
    typing-extensions
  ];
}
