{ pkgs, fetchPypi }:
pkgs.python3Packages.buildPythonPackage rec {
  pname = "tensorstore";
  version = "0.1.36";
  format = "wheel";

  src = fetchPypi {
    inherit pname version format;
    dist = "cp310";
    python = "cp310";
    abi = "cp310";
    platform = "manylinux_2_17_x86_64.manylinux2014_x86_64";
    sha256 = "33ad5669e5f3ee705718978f5519d96b25ff43f607730ac473947b0bac4c66d9"; 
  };

  # TODO: This is not actually importable yet.
  # ImportError: libstdc++.so.6: cannot open shared object file: No such file or directory
  #pythonImportsCheck = [
  #  "tensorstore"
  #]; 

  propagatedBuildInputs = with pkgs.python3Packages; [
    numpy
  ];
}
