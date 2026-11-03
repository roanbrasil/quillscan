# Minimal image for the quillscan CLI. The build stage needs only the Rust
# toolchain: no OpenCV, no CMake, no C++ compiler beyond what the base image
# already has for libc.
FROM rust:1-slim AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release

FROM gcr.io/distroless/cc-debian12
COPY --from=build /src/target/release/quillscan /quillscan
COPY models /models
ENTRYPOINT ["/quillscan", "--models", "/models"]
