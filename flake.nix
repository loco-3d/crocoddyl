{
  description = "optimal control library for robot control under contact sequence.";

  inputs.gepetto.url = "github:gepetto/nix";

  outputs =
    inputs:
    inputs.gepetto.lib.mkFlakoboros inputs (
      { lib, ... }:
      {
        extraDevPyPackages = [ "crocoddyl" ];
        overrideAttrs.crocoddyl = {
          src = lib.fileset.toSource {
            root = ./.;
            fileset = lib.fileset.unions [
              ./benchmark
              ./bindings
              ./CMakeLists.txt
              ./crocoddyl.cmake
              ./doc
              ./examples
              ./include
              ./notebooks
              ./package.xml
              ./src
              ./unittest
            ];
          };
        };
      }
    );
}
