use std::io::Write;
use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

const ABOUT: &str = "
Softy is a 3D FEM soft body and cloth simulation engine with two-way frictional contact coupling.";

#[derive(Parser)]
#[clap(author, about = ABOUT, name = "softy")]
struct Opt {
    /// Path to the scene configuration file.
    ///
    /// The file is expected to be in `bincode` format.
    #[clap(name = "CONFIG", parse(from_os_str))]
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
    #[clap(name = "OUTPUT", parse(from_os_str))]
    output: PathBuf,

    /// Log file path.
    #[clap(short, long, parse(from_os_str))]
    logfile: Option<PathBuf>,

    /// Number of steps of simulation to run.
    #[clap(short, long, default_value = "1")]
    steps: u64,
}

pub fn main() {
    if let Err(err) = try_main() {
        eprintln!("{}", err);
        std::process::exit(1);
    }
}

pub fn try_main() -> Result<()> {
    let _ = env_logger::Builder::from_env("SOFTY_LOG").try_init();

    let opt = Opt::parse();

    let mut file_stem = if let Some(file_stem) = opt.output.file_stem() {
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
        "sfrb" | "bin" => softy::scene::Scene::load_from_sfrb(opt.config)?,
        "ron" => softy::scene::Scene::load_from_ron(opt.config)?,
        "json" => softy::scene::Scene::load_from_json(opt.config)?,
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
                file_stem.truncate(file_stem.len() - num_digits);
            }
            let out_path = opt.output.parent().unwrap().to_path_buf();
            let logfile = opt.logfile.as_ref();

            // Write scene config so we know how the following log was created.
            if let Some(ref logfile) = logfile {
                let f = std::fs::File::options().append(true).open(logfile).unwrap();
                let mut buf = std::io::BufWriter::new(f);
                writeln!(buf, "\nConfig:\n").unwrap();
                scene_config.config.write_as_ron(&mut buf).unwrap();
                writeln!(buf).unwrap();
            }

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
