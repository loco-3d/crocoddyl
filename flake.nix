{
  description = "optimal control library for robot control under contact sequence.";

  inputs = {
    flake-parts.url = "github:hercules-ci/flake-parts";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } (
      { lib, self, ... }:
      {
        systems = inputs.nixpkgs.lib.systems.flakeExposed;
        flake.overlays.default = final: prev: {
          crocoddyl = prev.crocoddyl.overrideAttrs (super: {
            patches = [ ];
            src = lib.fileset.toSource {
              root = ./.;
              fileset = lib.fileset.unions [
                ./benchmark
                ./bindings
                ./CMakeLists.txt
                ./colmpc
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
              final.python3Packages.nbconvert
              final.python3Packages.nbformat
              final.python3Packages.ipykernel
              final.python3Packages.matplotlib
              final.ffmpeg
            ];
            preCheck = ''
              export PATH=${final.ffmpeg}/bin:$PATH
            '';
          });
        };
        perSystem =
          {
            pkgs,
            self',
            system,
            ...
          }:
          {
            _module.args.pkgs = import inputs.nixpkgs {
              inherit system;
              overlays = [ self.overlays.default ];
            };
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
              crocoddyl = pkgs.python3Packages.crocoddyl;
            };
          };
      }
    );
}
