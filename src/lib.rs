use anyhow::Result;
use image::{DynamicImage, GrayImage, ImageBuffer, Luma};
use std::path::Path;

extern "C" {
    fn count_nonzero(data: *const u8, len: u32) -> u32;
}

pub fn count_nonzero_bytes(data: &[u8]) -> u32 {
    unsafe { count_nonzero(data.as_ptr(), data.len() as u32) }
}

pub fn sobel_edges(image: &DynamicImage) -> GrayImage {
    let gray: GrayImage = image.to_luma8();
    let (width, height) = gray.dimensions();

    let mut gx: ImageBuffer<Luma<f32>, Vec<f32>> = ImageBuffer::from_pixel(width, height, Luma([0.0]));
    let mut gy: ImageBuffer<Luma<f32>, Vec<f32>> = ImageBuffer::from_pixel(width, height, Luma([0.0]));

    let kx: [[f32; 3]; 3] = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]];
    let ky: [[f32; 3]; 3] = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]];

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let mut sum_x = 0.0f32;
            let mut sum_y = 0.0f32;
            for j in 0..3 {
                for i in 0..3 {
                    let px = gray.get_pixel(x + i - 1, y + j - 1).0[0] as f32;
                    sum_x += kx[j as usize][i as usize] * px;
                    sum_y += ky[j as usize][i as usize] * px;
                }
            }
            gx.put_pixel(x, y, Luma([sum_x]));
            gy.put_pixel(x, y, Luma([sum_y]));
        }
    }

    let mut mag: ImageBuffer<Luma<f32>, Vec<f32>> = ImageBuffer::from_pixel(width, height, Luma([0.0]));
    let mut max_val = 0.0f32;

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let sx = gx.get_pixel(x, y).0[0];
            let sy = gy.get_pixel(x, y).0[0];
            let m = (sx * sx + sy * sy).sqrt();
            if m > max_val { max_val = m; }
            mag.put_pixel(x, y, Luma([m]));
        }
    }

    // Normalize to 0..255
    let scale = if max_val > 0.0 { 255.0 / max_val } else { 0.0 };
    let mut out: GrayImage = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let v = mag.get_pixel(x, y).0[0] * scale;
            let v = v.clamp(0.0, 255.0) as u8;
            out.put_pixel(x, y, Luma([v]));
        }
    }

    out
}

pub fn save_edges(input_path: &Path, output_path: &Path, threshold: Option<u8>, invert: bool) -> Result<()> {
    let img = image::open(input_path)?;
    let mut edges = sobel_edges(&img);

    if let Some(t) = threshold {
        for p in edges.pixels_mut() {
            let v = p[0];
            p[0] = if v >= t { 255 } else { 0 };
        }
    }

    if invert {
        for p in edges.pixels_mut() {
            p[0] = 255u8.saturating_sub(p[0]);
        }
    }

    edges.save(output_path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::count_nonzero_bytes;

    #[test]
    fn counts_nonzero_correctly() {
        let data = [0u8, 1, 2, 0, 3, 0, 0, 4];
        assert_eq!(count_nonzero_bytes(&data), 4);
    }
}
