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
          apps = {
            default = {
              type = "app";
              program = pkgs.python3.withPackages (_: [ self'.packages.default ]);
            };
            jupyter = {
              type = "app";
              program = pkgs.writeShellApplication {
                name = "jupyter-crocoddyl";
                text = "jupyter lab";
                runtimeInputs = [
                  (pkgs.python3.withPackages (p: [
                    p.jupyterlab
                    p.meshcat
                    self'.packages.default
                  ]))
                ];
              };
            };
          };
          devShells.default = pkgs.mkShell {
            inputsFrom = [ self'.packages.default ];
            packages = with pkgs; [
              ffmpeg
              (python3.withPackages (p: [
                p.tomlkit
                p.matplotlib
                p.nbconvert
                p.nbformat
                p.ipykernel
              ]))
            ];
            shellHook = ''
              export PATH=${pkgs.ffmpeg}/bin:$PATH
            '';
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
              checkInputs = (super.checkInputs or [ ]) ++ [
                pkgs.python3Packages.nbconvert
                pkgs.python3Packages.nbformat
                pkgs.python3Packages.ipykernel
                pkgs.python3Packages.matplotlib
                pkgs.ffmpeg
              ];
              preCheck = ''
                export PATH=${pkgs.ffmpeg}/bin:$PATH
              '';
            });
          };
        };
    };
}
