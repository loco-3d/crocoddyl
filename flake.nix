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
          system,
          ...
        }:
        {
          # fix crocoddyl python imports on OSX.
          # Remove this for pinocchio > 3.3.1
          _module.args.pkgs = import inputs.nixpkgs {
            inherit system;
            overlays = [
              (final: prev: {
                pinocchio = prev.pinocchio.overrideAttrs {
                  patches = final.lib.optionals final.stdenv.hostPlatform.isDarwin [
                    (final.fetchpatch {
                      url = "https://github.com/stack-of-tasks/pinocchio/pull/2566/commits/4758b80c0f8937b5ddc270d29267feba7f637f0f.patch";
                      hash = "sha256-RaRF1Jo9HfxXiJ3GZAgcvV7UBvCpHCS4TJMOQyC1PvY=";
                    })
                  ];
                };
              })
            ];
          };
          apps.default = {
            type = "app";
            program = pkgs.python3.withPackages (_: [ self'.packages.default ]);
          };
          devShells.default = pkgs.mkShell { inputsFrom = [ self'.packages.default ]; };
          packages = {
            # expose patched pinocchio to dependent packages
            inherit (pkgs) pinocchio;
            default = self'.packages.crocoddyl;
            crocoddyl = pkgs.python3Packages.crocoddyl.overrideAttrs (_: {
              src = pkgs.lib.fileset.toSource {
                root = ./.;
                fileset = pkgs.lib.fileset.unions [
                  ./benchmark
                  ./bindings
                  ./CMakeLists.txt
                  ./doc
                  ./examples
                  ./include
                  ./package.xml
                  ./src
                  ./unittest
                ];
              };
            });
          };
        };
    };
}
