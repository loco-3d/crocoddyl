{
  description = "optimal control library for robot control under contact sequence.";

  inputs = {
    flake-parts.url = "github:hercules-ci/flake-parts";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      systems = inputs.nixpkgs.lib.systems.flakeExposed;
      perSystem =
        {
          pkgs,
          self',
          ...
        }:
        {
          apps.default = {
            type = "app";
            program = pkgs.python3.withPackages (_: [ self'.packages.default ]);
          };
          devShells.default = pkgs.mkShell {
            inputsFrom = [ self'.packages.default ];
            packages = [ (pkgs.python3.withPackages (p: [ p.tomlkit ])) ]; # for "make release"
          };
          packages = {
            default = self'.packages.crocoddyl;
            crocoddyl = pkgs.python3Packages.crocoddyl.overrideAttrs (super: {
              src = pkgs.lib.fileset.toSource {
                root = ./.;
                fileset = pkgs.lib.fileset.unions [
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
              checkInputs = (super.checkInputs or [ ]) ++ [ pkgs.python3Packages.nbformat ];
            });
          };
        };
    };
}
