use std::path::{Path, PathBuf};

fn main() {
    let swift = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../swift/.build/out/Products/Release");
    let exe_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/release");
    let _ = std::fs::create_dir_all(&exe_dir);
    let link = exe_dir.join("mlx.metallib");
    let _ = std::fs::remove_file(&link);
    let _ = std::os::unix::fs::symlink(
        swift.join("mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib"), &link);
    println!("cargo:rustc-link-arg={}", swift.join("Cmlx.o").display());
    println!("cargo:rustc-link-lib=c++");
    for f in ["Metal", "Foundation", "CoreFoundation", "Accelerate",
              "CoreML", "MetalKit", "IOSurface", "CoreImage"] {
        println!("cargo:rustc-link-lib=framework={f}");
    }

    let header = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../swift/.build/checkouts/mlx-swift/Source/Cmlx/include/mlx.h");
    println!("cargo:rerun-if-changed={}", header.display());
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("bindings.rs");
    let mut builder = bindgen::Builder::default()
        .header(header.display().to_string())
        .clang_arg(format!("-I{}",
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../swift/.build/checkouts/mlx-swift/Source/Cmlx/include").display()))
        .allowlist_function("mlx_.*")
        .allowlist_type("mlx_.*")
        .allowlist_var("mlx_.*")
        .blocklist_item(".*std.*")
        .default_enum_style(bindgen::EnumVariation::Consts);
    for inc in cxx_includes() {
        builder = builder.clang_arg(format!("-I{inc}"));
    }
    builder
        .generate()
        .expect("bindgen failed")
        .write_to_file(&out)
        .expect("write bindings failed");
}

fn cxx_includes() -> Vec<String> {
    let out = std::process::Command::new("c++")
        .arg("-E")
        .arg("-Wp,-v")
        .arg("-x")
        .arg("c++")
        .arg("-")
        .stderr(std::process::Stdio::piped())
        .output()
        .expect("c++ not found");
    String::from_utf8_lossy(&out.stderr)
        .lines()
        .skip_while(|l| !l.starts_with(" search starts here"))
        .take_while(|l| !l.starts_with("End of search"))
        .map(|l| l.trim().to_string())
        .filter(|l| Path::new(l).exists())
        .collect()
}
