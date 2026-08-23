use std::fs;

use ndarray::prelude::*;

use ninterp::prelude::*;

/// A row parsed from one of the `mount_blue_sky_*.csv` fixtures: longitude,
/// latitude, elevation (meters). These are crops of a real USGS 1-arcsecond
/// elevation tile around Mount Blue Sky, Colorado (39.5883 N, -105.6438 W).
struct Point {
    lon: f64,
    lat: f64,
    elev: f64,
}

/// Minimal `X,Y,Z` CSV parser; the fixtures are small enough that pulling in
/// a `csv` crate isn't worth it.
fn read_points(path: &str) -> Vec<Point> {
    fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("failed to read {path}: {e}"))
        .lines()
        .skip(1) // header
        .map(|line| {
            let mut fields = line.split(',');
            let lon: f64 = fields.next().unwrap().parse().unwrap();
            let lat: f64 = fields.next().unwrap().parse().unwrap();
            let elev: f64 = fields.next().unwrap().parse().unwrap();
            Point { lon, lat, elev }
        })
        .collect()
}

/// Reshape a flat list of grid points into ninterp's `(grid_x, grid_y, values)`
/// shape: sorted ascending 1-D axes, plus a 2-D array of `values[[x_idx, y_idx]]`.
///
/// The source USGS tile scans latitude from north to south (descending), but
/// ninterp requires strictly-increasing grids, so the latitude axis (and the
/// matching array axis) get reversed here.
fn to_grid(points: &[Point]) -> (Array1<f64>, Array1<f64>, Array2<f64>) {
    let sorted_unique = |mut v: Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v.dedup();
        v
    };
    let grid_x = sorted_unique(points.iter().map(|p| p.lon).collect());
    let grid_y = sorted_unique(points.iter().map(|p| p.lat).collect());

    let mut values = Array2::<f64>::zeros((grid_x.len(), grid_y.len()));
    for p in points {
        let xi = grid_x.partition_point(|&x| x < p.lon);
        let yi = grid_y.partition_point(|&y| y < p.lat);
        values[[xi, yi]] = p.elev;
    }

    (Array1::from(grid_x), Array1::from(grid_y), values)
}

fn main() {
    let coarse = read_points("examples/data/mount_blue_sky_coarse.csv");
    let fine = read_points("examples/data/mount_blue_sky_fine.csv");

    let (grid_x, grid_y, values) = to_grid(&coarse);
    println!(
        "coarse control grid: {} x {} points",
        grid_x.len(),
        grid_y.len()
    );

    // Bilinear would only ever interpolate flat between the 42x42 control
    // points, chopping down real peaks and ridges that fall between samples.
    // A not-a-knot cubic spline instead fits local curvature, so it tracks
    // the terrain's actual bumps instead of flattening them.
    let interp = Interp2D::new(
        grid_x,
        grid_y,
        values,
        strategy::CubicC2::not_a_knot(),
        Extrapolate::Error,
    )
    .expect("coarse grid should be strictly increasing");

    // Interpolate at every fine-grid location and compare against the real,
    // held-out elevation at that point.
    let mut abs_errors = Vec::with_capacity(fine.len());
    let mut export = String::from("lon,lat,true_elev,interp_elev\n");
    for p in &fine {
        let interp_elev = interp.interpolate(&[p.lon, p.lat]).unwrap();
        abs_errors.push((p.elev - interp_elev).abs());
        export.push_str(&format!("{},{},{},{}\n", p.lon, p.lat, p.elev, interp_elev));
    }

    let mean_error = abs_errors.iter().sum::<f64>() / abs_errors.len() as f64;
    let max_error = abs_errors.iter().cloned().fold(0.0, f64::max);
    println!("fine ground-truth points: {}", fine.len());
    println!("mean absolute error: {mean_error:.2} m");
    println!("max absolute error:  {max_error:.2} m");

    fs::write("examples/data/mount_blue_sky_result.csv", export)
        .expect("failed to write results CSV");
    println!("wrote examples/data/mount_blue_sky_result.csv");
}
