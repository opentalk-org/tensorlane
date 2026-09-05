const PROTO: &str = "../proto/tensorlane.proto";
const INCLUDE: &str = "../proto";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed={PROTO}");

    // compile with protox so no system `protoc` is required
    let descriptors = protox::compile([PROTO], [INCLUDE])?;

    tonic_build::configure()
        .bytes(["."])
        .compile_fds(descriptors)?;

    Ok(())
}
