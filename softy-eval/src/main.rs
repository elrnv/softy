use std::io::Write;
use std::path::PathBuf;

use anyhow::Result;
use structopt::StructOpt;

const ABOUT: &str = "
Softy is a 3D FEM soft body and cloth simulation engine with two-way frictional contact coupling.";

#[derive(StructOpt)]
#[structopt(author, about = ABOUT, name = "softy")]
struct Opt {
    /// Path to the scene configuration file.
    ///
    /// The file is expected to be in `bincode` format.
    #[structopt(name = "CONFIG", parse(from_os_str))]
    config: PathBuf,

    /// Output simulation file(s).
    ///
    /// If a `.vtk` or `.vtu` file is specified, then the resulting animation is output from
    /// by incrementing the last found numeric value in the path for each new frame.
    /// For instance, an output value `./out_0001.vtk` will be followed by `./out_0002.vtk`,
    /// `./out_0003.vtk` and so on.
    ///
    /// If no number occurs in the file name, one will be appended to the file stem. For instance,
    /// if `./out.vtk` is specified, then the first frame will be written to `./out0.vtk`, then
    /// `./out1.vtk` and so on.
    #[structopt(name = "OUTPUT", parse(from_os_str))]
    output: PathBuf,

    /// Log file path.
    #[structopt(short, long, parse(from_os_str))]
    logfile: Option<PathBuf>,

    /// Number of steps of simulation to run.
    #[structopt(short, long, default_value = "1")]
    steps: u64,
}

pub fn main() -> Result<()> {
    let _ = env_logger::Builder::from_env("SOFTY_LOG").try_init();

    use terminal_size::{terminal_size, Width};
    let app = Opt::clap().set_term_width(if let Some((Width(w), _)) = terminal_size() {
        w as usize
    } else {
        80
    });

    let opt = Opt::from_clap(&app.get_matches());

    let file_stem = if let Some(file_stem) = opt.output.file_stem() {
        file_stem.to_string_lossy().to_string()
    } else {
        anyhow::bail!(
            "Missing output file name in output path: {}",
            opt.output.display()
        )
    };
    let ext = if let Some(ext) = opt.output.extension().and_then(|x| x.to_str()) {
        ext
    } else {
        anyhow::bail!(
            "Missing file extension in output path: {}",
            opt.output.display()
        )
    };

    let config_ext = if let Some(ext) = opt.config.extension().and_then(|x| x.to_str()) {
        ext
    } else {
        anyhow::bail!(
            "Missing file extension in config path: {}",
            opt.config.display()
        )
    };

    // Pre-emptively create the log file. This way we can fail early.
    if let Some(logfile) = opt.logfile.as_ref() {
        let _ = std::fs::File::create(logfile)?;
    }

    let scene_config = match config_ext {
        "bin" => softy::scene::SceneConfig::load_from_bin(opt.config)?,
        "ron" => softy::scene::SceneConfig::load_from_ron(opt.config)?,
        "json" => softy::scene::SceneConfig::load_from_json(opt.config)?,
        _ => anyhow::bail!("Unsupported config extension: '.{}'", config_ext),
    };

    match ext {
        "gltf" | "glb" => anyhow::bail!("glTF output is not currently supported"),
        "vtu" | "vtk" => {
            let mut indexed_digits = file_stem.rmatch_indices(|x| char::is_ascii_digit(&x));
            let mut num_digits = 0;
            let mut first_frame: u64 = 0;
            if let Some((mut prev_i, first_digit)) = indexed_digits.next() {
                // Determine the digits of the last numeric value.
                let mut num_str = format!("{}", first_digit);
                for (i, d) in indexed_digits {
                    if i == prev_i - 1 {
                        num_str.push_str(d);
                    } else {
                        break;
                    }
                    prev_i = i;
                }
                let num_str = num_str.chars().rev().collect::<String>();
                num_digits = num_str.len();
                first_frame = num_str.parse()?;
            }
            let out_path = opt.output.parent().unwrap().to_path_buf();
            let logfile = opt.logfile.as_ref();
            scene_config.run_with(opt.steps, |frame, result, mesh| {
                if let Some(ref logfile) = logfile {
                    let mut f = std::fs::File::options().append(true).open(logfile).unwrap();
                    writeln!(f, "\nFrame {}:\n{}", frame, result).unwrap();
                }
                let mut out_file_name = file_stem.clone();
                out_file_name.push_str(&format!("{:01$}", first_frame + frame, num_digits));
                geo::io::save_mesh(&mesh, out_path.join(out_file_name).with_extension(ext)).is_ok()
            })?;
        }
        _ => anyhow::bail!("Unsupported output file extension: '.{}'", ext),
    }
    Ok(())
}
