fn main() {
    // Trigger rebuild when frontend files change
    println!("cargo:rerun-if-changed=frontend/");
}
