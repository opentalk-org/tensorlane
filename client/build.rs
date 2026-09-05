const PROTO: &str = "../proto/tensorlane.proto";
const INCLUDE: &str = "../proto";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    pyo3_build_config::add_extension_module_link_args();
    println!("cargo:rerun-if-changed={PROTO}");
    let descriptors = protox::compile([PROTO], [INCLUDE])?;
    tonic_build::configure()
        .bytes(["."])
        .compile_fds(descriptors)?;
    Ok(())
}
