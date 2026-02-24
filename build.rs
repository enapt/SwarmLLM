fn main() {
    // Trigger rebuild when frontend files change
    println!("cargo:rerun-if-changed=frontend/");

    // Compile Cap'n Proto schema for zero-copy tensor serialization.
    // Requires the `capnp` compiler tool to be installed:
    //   Linux:   apt-get install capnproto
    //   macOS:   brew install capnp
    //   Windows: choco install capnproto
    println!("cargo:rerun-if-changed=proto/messages.capnp");

    // Check if capnp tool is available before attempting compilation
    let capnp_available = std::process::Command::new("capnp")
        .arg("--version")
        .output()
        .is_ok();

    if capnp_available {
        capnpc::CompilerCommand::new()
            .src_prefix("proto")
            .file("proto/messages.capnp")
            .run()
            .expect("Cap'n Proto schema compilation failed");
    } else {
        println!(
            "cargo:warning=capnp compiler not found; \
             using manual tensor serialization instead of generated schema"
        );
    }
}
