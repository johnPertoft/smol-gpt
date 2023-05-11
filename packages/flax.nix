{ pkgs, fetchPypi }:
pkgs.python3Packages.buildPythonPackage rec {
  pname = "flax";
  version = "xxxx";
  format = "wheel";

  src = fetchPypi {
    inherit pname version format;
    dist = "py3";
    python = "py3";
    sha256 = "xxxxxx"; 
  };

  propagatedBuildInputs = with pkgs.python3Packages; [
    #typing-extensions
  ];
}
