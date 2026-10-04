{ pkgs, src }:
let
  source = /. + builtins.unsafeDiscardStringContext (toString src);
  manifest = (pkgs.lib.importTOML (source + "/server/Cargo.toml")).package;
in
pkgs.rustPlatform.buildRustPackage {
  pname = manifest.name;
  version = manifest.version;
  src = pkgs.lib.fileset.toSource {
    root = source;
    fileset = pkgs.lib.fileset.unions [
      (source + "/Cargo.toml")
      (source + "/Cargo.lock")
      (source + "/server")
      (source + "/client/Cargo.toml")
      (source + "/client/build.rs")
      (source + "/client/src")
      (source + "/error-context")
      (source + "/protocol")
      (source + "/queries/examples")
    ];
  };
  cargoLock.lockFile = source + "/Cargo.lock";
  cargoBuildFlags = [
    "--package"
    "tensorlane"
    "--bin"
    "tensorlane"
  ];
  cargoTestFlags = [
    "--package"
    "tensorlane"
    "--bin"
    "tensorlane"
  ];
}
