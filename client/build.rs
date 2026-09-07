fn main() -> Result<(), Box<dyn std::error::Error>> {
    pyo3_build_config::add_extension_module_link_args();
    println!("cargo:rerun-if-changed=../proto/tensorlane.proto");
    let descriptors = protox::compile(["../proto/tensorlane.proto"], ["../proto"])?;
    tonic_build::configure().compile_fds(descriptors)?;
    Ok(())
}
