{ pkgs, fetchPypi }:
pkgs.python3Packages.buildPythonPackage rec {
  pname = "tensorstore";
  version = "0.1.36";
  format = "wheel";

  #"https://files.pythonhosted.org/packages/${dist}/${builtins.substring 0 1 pname}/${pname}/${pname}-${version}-${python}-${abi}-${platform}.whl";
  
  # have: https://files.pythonhosted.org/packages/py2.py3/t/tensorstore/tensorstore-0.1.36-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
  # want: https://files.pythonhosted.org/packages/c7/3f/b41.../tensorstore-0.1.36-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

  src = fetchPypi {
    inherit pname version format;
    dist = "cp310";
    python = "cp310";
    abi = "cp310";
    platform = "manylinux_2_17_x86_64.manylinux2014_x86_64";
    sha256 = "33ad5669e5f3ee705718978f5519d96b25ff43f607730ac473947b0bac4c66d9"; 
  };

  #nativeBuildInputs = with pkgs.python3Packages; [
  #  setuptools_scm
  #];

  #checkInputs = [
  #  pytestCheckHook
  #];

  propagatedBuildInputs = with pkgs.python3Packages; [
    numpy
  ];
}
