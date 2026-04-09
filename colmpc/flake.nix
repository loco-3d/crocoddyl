{
  description = "Collision avoidance for MPC";

  inputs = {
    gepetto.url = "github:gepetto/nix";
    flake-parts.follows = "gepetto/flake-parts";
    systems.follows = "gepetto/systems";
  };

  outputs =
    inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } (
      { lib, ... }:
      {
        systems = import inputs.systems;
        imports = [
          inputs.gepetto.flakeModule
          {
            flakoboros = {
              overrideAttrs.colmpc = _: {
                src = lib.fileset.toSource {
                  root = ./.;
                  fileset = lib.fileset.unions [
                    ./examples
                    ./include
                    ./python
                    ./tests
                    ./CMakeLists.txt
                    ./package.xml
                    ./pyproject.toml
                  ];
                };
              };
            };
          }
        ];
      }
    );
}
