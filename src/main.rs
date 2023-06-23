use bytemuck::{Pod, Zeroable};
use nalgebra::{vector, Vector2};
use rand::Rng;
use std::borrow::Cow;
use std::sync::Arc;
use wgpu::SubmissionIndex;

const WORKGROUP_SIZE: u32 = 128;
const INVALID_ID: u32 = 0xffffffff;

#[derive(Pod, Zeroable, Copy, Clone, Debug)]
#[repr(C)]
struct Config {
    num_emitters: u32,
    num_reflectors: u32,
}

#[derive(Pod, Zeroable, Copy, Clone, Debug)]
#[repr(C)]
struct Emitter {
    position: Vector2<f32>,
    angle: f32,
    pad: f32,
}

#[derive(Pod, Zeroable, Copy, Clone, Debug)]
#[repr(C)]
struct Reflector {
    position: Vector2<f32>,
    rcs: f32,
    pad: f32,
}

#[derive(Pod, Zeroable, Copy, Clone, Debug)]
#[repr(C)]
struct Output {
    reflector: u32,
    rssi: f32,
}

async fn run() {
    let mut rng = rand::thread_rng();

    let mut emitters = vec![];
    for _ in 0..1000 {
        emitters.push(Emitter {
            position: vector![rng.gen_range(-1e3..1e3), rng.gen_range(-1e3..1e3)],
            angle: 1.0,
            pad: 0.0,
        });
    }

    let mut reflectors = vec![];
    for _ in 0..1000 {
        reflectors.push(Reflector {
            position: vector![rng.gen_range(-1e3..1e3), rng.gen_range(-1e3..1e3)],
            rcs: 1.0,
            pad: 0.0,
        });
    }

    let instance = wgpu::Instance::default();
    let mut opts = wgpu::RequestAdapterOptions::default();
    opts.power_preference = wgpu::PowerPreference::HighPerformance;
    let adapter = instance.request_adapter(&opts).await.unwrap();

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        )
        .await
        .unwrap();

    let pass = create_compute_pass(&device).await;
    let device = Arc::new(device);

    let gpu_result = calculate_gpu(&device, &queue, &pass, &emitters, &reflectors).await;
    //println!("GPU result: {gpu_result:?}",);

    let cpu_result = calculate_cpu(&emitters, &reflectors);
    //println!("CPU result: {cpu_result:?}",);

    for i in 0..emitters.len() {
        assert_eq!(gpu_result[i].reflector, cpu_result[i].reflector);
        assert!((gpu_result[i].rssi - cpu_result[i].rssi).abs() < 1.0);
    }

    for _ in 0..10 {
        for gpu in [false, true] {
            let start_time = std::time::Instant::now();
            let mut i = 0;
            while start_time.elapsed() < std::time::Duration::from_secs(1) {
                if gpu {
                    std::hint::black_box(
                        calculate_gpu(&device, &queue, &pass, &emitters, &reflectors).await,
                    );
                } else {
                    std::hint::black_box(calculate_cpu(&emitters, &reflectors));
                }
                i += 1;
            }
            let elapsed = start_time.elapsed();
            println!(
                "{} Iteration time: {:.1?} mega-op/s: {:.1?}",
                if gpu { "GPU" } else { "CPU" },
                elapsed / i,
                1e-6 * i as f64 * emitters.len() as f64 * reflectors.len() as f64
                    / elapsed.as_secs_f64()
            );
        }
    }
}

fn calculate_cpu(emitters: &[Emitter], reflectors: &[Reflector]) -> Vec<Output> {
    let mut output = vec![];
    for i in 0..emitters.len() {
        let mut best = Output {
            reflector: INVALID_ID,
            rssi: 0.0,
        };
        for j in 0..reflectors.len() {
            let distance = (emitters[i].position - reflectors[j].position).norm();
            let rssi = reflectors[j].rcs / distance.powi(4);
            if rssi > best.rssi {
                best = Output {
                    reflector: j as u32,
                    rssi,
                };
            }
        }
        output.push(best);
    }

    output
}

struct ComputePass {
    compute_pipeline: wgpu::ComputePipeline,
    config_staging_buffer: wgpu::Buffer,
    config_storage_buffer: wgpu::Buffer,
    emitter_staging_buffer: wgpu::Buffer,
    emitter_storage_buffer: wgpu::Buffer,
    reflector_staging_buffer: wgpu::Buffer,
    reflector_storage_buffer: wgpu::Buffer,
    output_staging_buffer: wgpu::Buffer,
    output_storage_buffer: wgpu::Buffer,
}

async fn create_compute_pass(device: &wgpu::Device) -> ComputePass {
    let buf_size = 1024 * 1024;

    // Loads the shader from WGSL
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &cs_module,
        entry_point: "main",
    });

    let config_size = std::mem::size_of::<Config>() as wgpu::BufferAddress;

    let config_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("config staging buffer"),
        size: config_size,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::MAP_WRITE,
        mapped_at_creation: false,
    });

    let config_storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("config storage buffer"),
        size: config_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let emitter_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("emitter staging buffer"),
        size: buf_size,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::MAP_WRITE,
        mapped_at_creation: false,
    });

    let emitter_storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("emitter storage buffer"),
        size: buf_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let reflector_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reflector staging buffer"),
        size: buf_size,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::MAP_WRITE,
        mapped_at_creation: false,
    });

    let reflector_storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reflector storage buffer"),
        size: buf_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let output_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output staging buffer"),
        size: buf_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let output_storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output storage buffer"),
        size: buf_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    ComputePass {
        compute_pipeline,
        config_staging_buffer,
        config_storage_buffer,
        emitter_staging_buffer,
        emitter_storage_buffer,
        reflector_staging_buffer,
        reflector_storage_buffer,
        output_staging_buffer,
        output_storage_buffer,
    }
}

async fn calculate_gpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pass: &ComputePass,
    emitters: &[Emitter],
    reflectors: &[Reflector],
) -> Vec<Output> {
    let bind_group_layout = pass.compute_pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: pass.config_storage_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: pass.emitter_storage_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: pass.reflector_storage_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: pass.output_storage_buffer.as_entire_binding(),
            },
        ],
    });

    let config = Config {
        num_emitters: emitters.len() as u32,
        num_reflectors: reflectors.len() as u32,
    };

    upload_buffer(
        device,
        &pass.config_staging_buffer,
        bytemuck::bytes_of(&config),
    )
    .unwrap();

    upload_buffer(
        device,
        &pass.emitter_staging_buffer,
        bytemuck::cast_slice(&emitters),
    )
    .unwrap();

    upload_buffer(
        device,
        &pass.reflector_staging_buffer,
        bytemuck::cast_slice(&reflectors),
    )
    .unwrap();

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    encoder.copy_buffer_to_buffer(
        &pass.config_staging_buffer,
        0,
        &pass.config_storage_buffer,
        0,
        pass.config_storage_buffer.size(),
    );

    encoder.copy_buffer_to_buffer(
        &pass.emitter_staging_buffer,
        0,
        &pass.emitter_storage_buffer,
        0,
        pass.emitter_storage_buffer.size(),
    );

    encoder.copy_buffer_to_buffer(
        &pass.reflector_staging_buffer,
        0,
        &pass.reflector_storage_buffer,
        0,
        pass.reflector_storage_buffer.size(),
    );

    let output_cols = (reflectors.len() as u32 + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        cpass.set_pipeline(&pass.compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(output_cols, emitters.len() as u32, 1);
    }

    encoder.copy_buffer_to_buffer(
        &pass.output_storage_buffer,
        0,
        &pass.output_staging_buffer,
        0,
        pass.output_storage_buffer.size(),
    );

    let id = queue.submit(Some(encoder.finish()));

    let partial_outputs: Vec<Output> =
        download_buffer(&device, &pass.output_staging_buffer, id).unwrap();
    let mut outputs = Vec::new();
    outputs.reserve(emitters.len());

    for i in 0..emitters.len() {
        let mut best = Output {
            reflector: INVALID_ID,
            rssi: 0.0,
        };
        for j in 0..output_cols {
            let partial_output = partial_outputs[i * output_cols as usize + j as usize];
            //println!("Emitter {i} col {j} = {partial_output:?}");
            if partial_output.rssi > best.rssi {
                best = partial_output;
            }
        }
        outputs.push(best);
    }

    outputs
}

fn main() {
    env_logger::init();
    pollster::block_on(run());
}

fn download_buffer<T>(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    id: SubmissionIndex,
) -> Result<Vec<T>, wgpu::BufferAsyncError>
where
    T: bytemuck::Pod,
{
    let (sender, receiver) = std::sync::mpsc::channel();
    let buffer_slice = buffer.slice(..);
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
        drop(sender.send(v));
    });
    device.poll(wgpu::Maintain::WaitForSubmissionIndex(id));
    receiver.recv().unwrap()?;
    let view = buffer_slice.get_mapped_range();
    let data = bytemuck::cast_slice(&view).to_vec();
    drop(view);
    buffer.unmap();
    Ok(data)
}

fn upload_buffer(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    data: &[u8],
) -> Result<(), wgpu::BufferAsyncError> {
    let (sender, receiver) = std::sync::mpsc::channel();
    let buffer_slice = buffer.slice(..(data.len() as u64));
    buffer_slice.map_async(wgpu::MapMode::Write, move |v| {
        drop(sender.send(v));
    });
    device.poll(wgpu::Maintain::Wait);
    receiver.recv().unwrap()?;
    let mut view = buffer_slice.get_mapped_range_mut();
    view.copy_from_slice(data);
    drop(view);
    buffer.unmap();
    Ok(())
}

#[cfg(test)]
fn calculate_for_test(emitters: &[Emitter], reflectors: &[Reflector], gpu: bool) -> Vec<Output> {
    if gpu {
        let instance = wgpu::Instance::default();
        let mut opts = wgpu::RequestAdapterOptions::default();
        opts.power_preference = wgpu::PowerPreference::HighPerformance;
        let adapter = pollster::block_on(instance.request_adapter(&opts)).unwrap();

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        ))
        .unwrap();

        let pass = pollster::block_on(create_compute_pass(&device));
        let device = Arc::new(device);

        pollster::block_on(calculate_gpu(
            &device,
            &queue,
            &pass,
            &emitters,
            &reflectors,
        ))
    } else {
        calculate_cpu(&emitters, &reflectors)
    }
}

#[test]
fn test_basic() {
    use assert_float_eq::*;
    let emitters = vec![Emitter {
        position: vector![0.0, 0.0],
        angle: 0.0,
        pad: 0.0,
    }];

    let reflectors = vec![Reflector {
        position: vector![1000.0, 0.0],
        rcs: 1.0,
        pad: 0.0,
    }];

    for gpu in [false, true] {
        let outputs = calculate_for_test(&emitters, &reflectors, gpu);
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].reflector, 0);
        assert_float_relative_eq!(outputs[0].rssi, 1e-12);
    }
}

#[test]
fn test_position() {
    use assert_float_eq::*;
    let emitters = vec![Emitter {
        position: vector![1000.0, 0.0],
        angle: 0.0,
        pad: 0.0,
    }];

    let reflectors = vec![Reflector {
        position: vector![1000.0, 1000.0],
        rcs: 1.0,
        pad: 0.0,
    }];

    for gpu in [false, true] {
        let outputs = calculate_for_test(&emitters, &reflectors, gpu);
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].reflector, 0);
        assert_float_relative_eq!(outputs[0].rssi, 1e-12);
    }
}

#[test]
fn test_two_reflectors() {
    use assert_float_eq::*;
    let emitters = vec![Emitter {
        position: vector![1000.0, 0.0],
        angle: 0.0,
        pad: 0.0,
    }];

    let reflectors = vec![
        Reflector {
            position: vector![1000.0, 1000.0],
            rcs: 1.0,
            pad: 0.0,
        },
        Reflector {
            position: vector![1000.0, 500.0],
            rcs: 1.0,
            pad: 0.0,
        },
    ];

    for gpu in [false, true] {
        let outputs = calculate_for_test(&emitters, &reflectors, gpu);
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].reflector, 1);
        assert_float_relative_eq!(outputs[0].rssi, 1.6e-11);
    }
}

#[test]
fn test_rcs() {
    use assert_float_eq::*;
    let emitters = vec![Emitter {
        position: vector![0.0, 0.0],
        angle: 0.0,
        pad: 0.0,
    }];

    let reflectors = vec![
        Reflector {
            position: vector![1000.0, 0.0],
            rcs: 1.0,
            pad: 0.0,
        },
        Reflector {
            position: vector![2000.0, 0.0],
            rcs: 100.0,
            pad: 0.0,
        },
    ];

    for gpu in [false, true] {
        let outputs = calculate_for_test(&emitters, &reflectors, gpu);
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].reflector, 1);
        assert_float_relative_eq!(outputs[0].rssi, 6.25e-12);
    }
}

#[test]
fn test_two_emitters() {
    use assert_float_eq::*;
    let emitters = vec![
        Emitter {
            position: vector![1000.0, 0.0],
            angle: 0.0,
            pad: 0.0,
        },
        Emitter {
            position: vector![2000.0, 0.0],
            angle: 0.0,
            pad: 0.0,
        },
    ];

    let reflectors = vec![
        Reflector {
            position: vector![0.0, 0.0],
            rcs: 1.0,
            pad: 0.0,
        },
        Reflector {
            position: vector![3000.0, 0.0],
            rcs: 1.0,
            pad: 0.0,
        },
    ];

    for gpu in [false, true] {
        let outputs = calculate_for_test(&emitters, &reflectors, gpu);
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].reflector, 0);
        assert_float_relative_eq!(outputs[0].rssi, 1e-12);
        assert_eq!(outputs[1].reflector, 1);
        assert_float_relative_eq!(outputs[1].rssi, 1e-12);
    }
}

#[test]
fn test_random() {
    use assert_float_eq::*;

    let mut rng = rand::thread_rng();

    let mut emitters = vec![];
    for _ in 0..1000 {
        emitters.push(Emitter {
            position: vector![rng.gen_range(-1e3..1e3), rng.gen_range(-1e3..1e3)],
            angle: rng.gen_range(0.0..std::f32::consts::TAU),
            pad: 0.0,
        });
    }

    let mut reflectors = vec![];
    for _ in 0..1000 {
        reflectors.push(Reflector {
            position: vector![rng.gen_range(-1e3..1e3), rng.gen_range(-1e3..1e3)],
            rcs: rng.gen_range(0.1..10.0),
            pad: 0.0,
        });
    }

    let cpu_outputs = calculate_for_test(&emitters, &reflectors, false);
    let gpu_outputs = calculate_for_test(&emitters, &reflectors, true);

    assert_eq!(cpu_outputs.len(), gpu_outputs.len());
    for i in 0..cpu_outputs.len() {
        assert_eq!(cpu_outputs[i].reflector, gpu_outputs[i].reflector);
        assert_float_relative_eq!(cpu_outputs[i].rssi, gpu_outputs[i].rssi, 1e-3);
    }
}
