fn main() {
    // Frontend rebuild tracking moved to crates/swarmllm-frontend/build.rs
    //
    // CUDA library search paths are NOT set here. `candle-flash-attn` is the
    // crate that declares the link (`rustc-link-lib=static=cudart_static`), and
    // rustc resolves a `static=` library while building THAT crate's rlib — a
    // search path emitted from this package arrives too late and fails with
    // `could not find native static library cudart_static`. It lives in
    // vendor/candle-flash-attn/build.rs, whose directives propagate to the
    // final binary link as well.
}
