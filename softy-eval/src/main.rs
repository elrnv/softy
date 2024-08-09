use std::io::Write;
use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use clap_verbosity_flag::Verbosity;
use indicatif::{ProgressBar, ProgressStyle};

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

    /// Controls verbosity of printed output
    #[clap(flatten)]
    verbose: Verbosity,
}

fn trimesh_from_mesh(mesh: geo::Mesh<f64>) -> geo::TriMesh<f32> {
    use geo::algo::*;
    use geo::attrib::*;
    use geo::topology::*;
    use geo::vertex_positions::*;

    let mut tet_vertex_map = vec![-1_isize; mesh.num_vertices()];
    let mut tet_vertex_positions = Vec::new();
    let mut tet_indices = Vec::with_capacity(mesh.num_cells());
    let mut tet_cell_attributes: AttribDict<CellIndex> = AttribDict::new();
    let mut tet_vertex_attributes: AttribDict<VertexIndex> = AttribDict::new();

    let mut tri_vertex_map = vec![-1_isize; mesh.num_vertices()];
    let mut tri_vertex_positions = Vec::new();
    let mut tri_indices = Vec::with_capacity(mesh.num_cells());
    let mut tri_face_attributes: AttribDict<FaceIndex> = AttribDict::new();
    let mut tri_vertex_attributes: AttribDict<VertexIndex> = AttribDict::new();

    // Transfer face attributes
    for (name, attrib) in mesh.attrib_dict::<CellIndex>().iter() {
        tri_face_attributes.insert(
            name.to_string(),
            attrib.promote_with(|new, old| {
                // Copy the attribute for every triangle originating from this polygon.
                for (cell, elem) in mesh.cell_iter().zip(old.iter()) {
                    if cell.len() == 3 {
                        new.push_cloned(elem.reborrow()).unwrap();
                    }
                }
            }),
        );
        tet_cell_attributes.insert(
            name.to_string(),
            attrib.duplicate_with(|new, old| {
                // Copy the attribute for every triangle originating from this polygon.
                for (cell, elem) in mesh.cell_iter().zip(old.iter()) {
                    if cell.len() == 4 {
                        new.push_cloned(elem.reborrow()).unwrap();
                    }
                }
            }),
        );
    }

    for cell in mesh.cell_iter() {
        if cell.len() == 3 {
            for &i in cell {
                tri_vertex_map[i] = 0;
            }
        } else if cell.len() == 4 {
            for &i in cell {
                tet_vertex_map[i] = 0;
            }
        }
    }

    for ((tri_v_idx, tet_v_idx), &[x, y, z]) in tri_vertex_map
        .iter_mut()
        .zip(tet_vertex_map.iter_mut())
        .zip(mesh.vertex_position_iter())
    {
        if *tri_v_idx > -1 {
            *tri_v_idx =
                isize::try_from(tri_vertex_positions.len()).expect("less than half of usize::max");
            tri_vertex_positions.push([x as f32, y as f32, z as f32]);
        }
        if *tet_v_idx > -1 {
            *tet_v_idx =
                isize::try_from(tet_vertex_positions.len()).expect("less than half of usize::max");
            tet_vertex_positions.push([x as f32, y as f32, z as f32]);
        }
    }

    for cell in mesh.cell_iter() {
        match *cell {
            [a, b, c] => {
                let tvm = &tri_vertex_map;
                tri_indices.push([
                    usize::try_from(tvm[a]).expect("positive integer"),
                    usize::try_from(tvm[b]).expect("positive integer"),
                    usize::try_from(tvm[c]).expect("positive integer"),
                ]);
            }
            [a, b, c, d] => {
                let tvm = &tet_vertex_map;
                tet_indices.push([
                    usize::try_from(tvm[a]).expect("positive integer"),
                    usize::try_from(tvm[b]).expect("positive integer"),
                    usize::try_from(tvm[c]).expect("positive integer"),
                    usize::try_from(tvm[d]).expect("positive integer"),
                ]);
            }
            _ => {}
        }
    }

    for (name, attrib) in mesh.attrib_dict::<VertexIndex>().iter() {
        tri_vertex_attributes.insert(
            name.to_string(),
            attrib.duplicate_with(|new, old| {
                // Copy the attribute for every triangle originating from this polygon.
                for (new_idx, elem) in tri_vertex_map.iter().zip(old.iter()) {
                    if *new_idx != -1 {
                        new.push_cloned(elem.reborrow()).unwrap();
                    }
                }
            }),
        );
        tet_vertex_attributes.insert(
            name.to_string(),
            attrib.duplicate_with(|new, old| {
                // Copy the attribute for every triangle originating from this polygon.
                for (new_idx, elem) in tet_vertex_map.iter().zip(old.iter()) {
                    if *new_idx != -1 {
                        new.push_cloned(elem.reborrow()).unwrap();
                    }
                }
            }),
        );
    }

    let geo::Mesh {
        attribute_value_cache,
        ..
    } = mesh;

    let mut trimesh = geo::TriMesh {
        vertex_positions: tri_vertex_positions.into(),
        indices: tri_indices.into(),
        vertex_attributes: tri_vertex_attributes,
        face_attributes: tri_face_attributes,
        attribute_value_cache: attribute_value_cache.clone(),
        ..Default::default()
    };
    let tetmesh = geo::TetMesh {
        vertex_positions: tet_vertex_positions.into(),
        indices: tet_indices.into(),
        vertex_attributes: tet_vertex_attributes,
        cell_attributes: tet_cell_attributes,
        attribute_value_cache: attribute_value_cache.clone(),
        ..Default::default()
    };
    trimesh.merge(tetmesh.surface_trimesh());
    trimesh
}

pub fn main() {
    if let Err(err) = try_main() {
        eprintln!("ERROR: {}", err);
        std::process::exit(1);
    }
}

pub fn try_main() -> Result<()> {
    let opt = Opt::parse();

    let _ = env_logger::Builder::new()
        .filter_level(opt.verbose.log_level_filter())
        .init();

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

    use std::sync::atomic::*;
    use std::sync::Arc;

    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
    })
    .expect("Error setting Ctrl-C handler");

    // Progress bar
    let bar = if opt.verbose.is_silent() {
        ProgressBar::hidden()
    } else {
        ProgressBar::new(opt.steps).with_style(
            ProgressStyle::default_bar()
                .progress_chars("=> ")
                .template("{elapsed:4} [{bar:20.cyan}] {pos:>7}/{len:7} {msg}")
                .expect("Failed to render the progress bar."),
        )
    };

    match ext {
        "gltf" | "glb" => {
            let logfile = opt.logfile.as_ref();

            // Write scene config so we know how the following log was created.
            if let Some(ref logfile) = logfile {
                let f = std::fs::File::options().append(true).open(logfile).unwrap();
                let mut buf = std::io::BufWriter::new(f);
                writeln!(buf, "\nConfig:\n").unwrap();
                scene_config.config.write_as_ron(&mut buf).unwrap();
                writeln!(buf).unwrap();
            }

            let mut meshes = Vec::new();

            let result = scene_config.run_with(opt.steps, |frame, mut mesh, result| {
                bar.inc(1);
                if !running.load(Ordering::SeqCst) {
                    return false;
                }
                if let Some(ref logfile) = logfile {
                    let mut f = std::fs::File::options().append(true).open(logfile).unwrap();
                    writeln!(f, "\nFrame {}:", frame).unwrap();
                    if let Some(result) = result {
                        writeln!(f, "{}", result).unwrap();
                    }
                }
                use geo::attrib::Attrib;
                use geo::topology::VertexIndex;
                // Convert vec3(f64) -> vec3(f32) attribs
                for name in ["vel", "friction", "residual", "contact"] {
                    let vec32 = mesh
                        .remove_attrib::<VertexIndex>(name)
                        .expect(&format!("existing {name} attribute"))
                        .into_data()
                        .into_vec::<[f64; 3]>()
                        .expect(&format!("{name} attribute having type [f64; 3]"))
                        .into_iter()
                        .map(|[x, y, z]| [x as f32, y as f32, z as f32])
                        .collect();
                    mesh.insert_attrib_data::<_, VertexIndex>(name, vec32)
                        .expect(&format!("vacant attribute name \"{name}\""));
                }
                // convert mass attrib
                let scalar32 = mesh
                    .remove_attrib::<VertexIndex>("mass")
                    .expect("existing mass attribute")
                    .into_data()
                    .into_vec::<f64>()
                    .expect("mass attribute having type f64")
                    .into_iter()
                    .map(|x| {
                        if x.is_nan() || x.is_infinite() {
                            -1.0 // cast nans and infinite masses to -1.0
                        } else {
                            x as f32
                        }
                    })
                    .collect();
                mesh.insert_attrib_data::<_, VertexIndex>("mass", scalar32)
                    .expect("vacant attribute name \"mass\"");
                for name in ["fixed", "animated"] {
                    let int = mesh
                        .remove_attrib::<VertexIndex>(name)
                        .expect(&format!("existing {name} attribute"))
                        .into_data()
                        .into_vec::<i32>()
                        .expect(&format!("{name} attribute having type i32"))
                        .into_iter()
                        .map(|x| u32::try_from(x).unwrap_or(0))
                        .collect();
                    mesh.insert_attrib_data::<_, VertexIndex>(name, int)
                        .expect(&format!("vacant attribute name \"{name}\""));
                }
                meshes.push((file_stem.to_string(), trimesh_from_mesh(mesh).into()));
                true
            });

            if result.is_ok()
                || result
                    .as_ref()
                    .is_err_and(|x| matches!(x, &softy::scene::SceneError::Interrupted))
            {
                let attrib_config = gltfgen::AttribConfig {
                    // attributes: &r#"{"vel": Vec3(f32), "contact": Vec3(f32), "friction": Vec3(f32), "net_force": Vec3(f32), "residual": Vec3(f32), "mass": f32, "animated": u32, "fixed": u32}"#.parse().unwrap(),
                    attributes: &r#"{"vel": Vec3(f32), "contact": Vec3(f32), "friction": Vec3(f32), "net_force": Vec3(f32), "residual": Vec3(f32)}"#.parse().unwrap(),
                    colors: &gltfgen::AttributeInfo::default(),
                    texcoords: &gltfgen::TextureAttributeInfo::default(),
                    material_attribute: "mtl_id",
                };

                gltfgen::export::export_named_meshes(
                    meshes,
                    attrib_config,
                    gltfgen::export::ExportConfig {
                        textures: Vec::new(),
                        materials: Vec::new(),
                        output: opt.output.into(),
                        time_step: scene_config
                            .config
                            .sim_params
                            .time_step
                            .unwrap_or(1.0 / 24.0),
                        insert_vanishing_frames: false,
                        animate_normals: false,
                        animate_tangents: false,
                        quiet: opt.verbose.is_silent(),
                    },
                );
            }
            result?;
        }
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

            let out_file_name = |frame| {
                let mut out_file_name = file_stem.clone();
                out_file_name.push_str(&format!("{:01$}", first_frame + frame, num_digits));
                out_file_name
            };

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

            scene_config.run_with(opt.steps, |frame, mesh, result| {
                bar.inc(1);
                if !running.load(Ordering::SeqCst) {
                    return false;
                }
                if let Some(ref logfile) = logfile {
                    let mut f = std::fs::File::options().append(true).open(logfile).unwrap();
                    writeln!(f, "\nFrame {}:", frame).unwrap();
                    if let Some(result) = result {
                        writeln!(f, "{}", result).unwrap();
                    }
                }
                geo::io::save_mesh(
                    &mesh,
                    out_path.join(out_file_name(frame)).with_extension(ext),
                )
                .is_ok()
            })?;
        }
        _ => anyhow::bail!("Unsupported output file extension: '.{}'", ext),
    }
    bar.finish();
    Ok(())
}
