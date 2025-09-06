use std::path::PathBuf;

use anyhow::Result;
use clap::{ArgAction, Parser};

#[derive(Parser, Debug)]
#[command(name = "sketch-vision", about = "Detect edges in images using the Sobel operator.")]
struct Cli {
    /// Input image path (JPEG/PNG/etc.)
    input: PathBuf,

    /// Output image path (PNG recommended)
    #[arg(short, long, default_value = "edges.png")]
    output: PathBuf,

    /// Optional threshold 0..255; if provided, binarizes the edges
    #[arg(short, long)]
    threshold: Option<u8>,

    /// Invert the output image
    #[arg(long, action = ArgAction::SetTrue)]
    invert: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    sketch_vision::save_edges(&cli.input, &cli.output, cli.threshold, cli.invert)?;
    eprintln!(
        "Saved edges to {} (threshold={:?}, invert={})",
        cli.output.display(),
        cli.threshold,
        cli.invert
    );
    Ok(())
}
