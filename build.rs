fn main() {
    // Re-run if C++ source changes
    println!("cargo:rerun-if-changed=cpp/metrics.cpp");

    let mut build = cc::Build::new();
    build.cpp(true)
        .file("cpp/metrics.cpp")
        .flag_if_supported("-std=c++17")
        .warnings(true);

    build.compile("metrics");

    // Link standard C++ library depending on platform
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "macos" {
        println!("cargo:rustc-link-lib=c++");
    } else if target_os == "windows" {
        // MSVC links the C++ runtime automatically
    } else {
        // Assume libstdc++ on other Unix-like OSes (e.g., Linux)
        println!("cargo:rustc-link-lib=stdc++");
    }
}
