fn main() {
    // Only track frontend changes when embedding (not in dev mode).
    #[cfg(feature = "embedded")]
    println!("cargo:rerun-if-changed=../../frontend/");
}
