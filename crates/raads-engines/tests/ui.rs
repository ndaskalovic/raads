//! Compile errors of `#[kernels]` against the engine specs, and generated code that must
//! compile cleanly in strict user crates.

#[test]
fn ui() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/*.rs");
    t.pass("tests/ui/pass/*.rs");
}
