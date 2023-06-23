use std::borrow::Cow;
use std::sync::Arc;
use wgpu::SubmissionIndex;

const WORKGROUP_SIZE: u32 = 128;

async fn run() {
    let numbers = (0..513).map(|x| x as f32).collect::<Vec<_>>();

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

    let pass = create_compute_pass(&device, numbers.len()).await;
    let device = Arc::new(device);

    let result = calculate_gpu(&device, &queue, &pass, &numbers).await;
    println!("Result: {result}",);

    assert_eq!(result, calculate_cpu(&numbers));

    for _ in 0..10 {
        for gpu in [false, true] {
            let start_time = std::time::Instant::now();
            let mut i = 0;
            while start_time.elapsed() < std::time::Duration::from_secs(1) {
                if gpu {
                    std::hint::black_box(calculate_gpu(&device, &queue, &pass, &numbers).await);
                } else {
                    std::hint::black_box(calculate_cpu(&numbers));
                }
                i += 1;
            }
            let elapsed = start_time.elapsed();
            println!(
                "{} Iteration time: {:.1?} mega-op/s: {:.1?}",
                if gpu { "GPU" } else { "CPU" },
                elapsed / i,
                1e-6 * i as f64 * numbers.len() as f64 * numbers.len() as f64
                    / elapsed.as_secs_f64()
            );
        }
    }
}

fn calculate_cpu(numbers: &[f32]) -> f64 {
    let mut expected: f64 = 0.0;
    for i in 0..numbers.len() {
        for j in 0..numbers.len() {
            expected += std::hint::black_box(numbers[i] * numbers[j]) as f64;
        }
    }
    expected
}

struct ComputePass {
    compute_pipeline: wgpu::ComputePipeline,
    config_staging_buffer: wgpu::Buffer,
    config_storage_buffer: wgpu::Buffer,
    items_staging_buffer: wgpu::Buffer,
    items_storage_buffer: wgpu::Buffer,
    output_staging_buffer: wgpu::Buffer,
    output_storage_buffer: wgpu::Buffer,
}

async fn create_compute_pass(device: &wgpu::Device, len: usize) -> ComputePass {
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

    let config_size = std::mem::size_of::<u32>() as wgpu::BufferAddress;
    let items_size = len as u64 * std::mem::size_of::<f32>() as wgpu::BufferAddress;
    let output_size = (len * ((len + WORKGROUP_SIZE as usize - 1) / WORKGROUP_SIZE as usize))
        as u64
        * std::mem::size_of::<f32>() as wgpu::BufferAddress;

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

    let items_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("items staging buffer"),
        size: items_size,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::MAP_WRITE,
        mapped_at_creation: false,
    });

    let items_storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("items storage buffer"),
        size: items_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let output_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output staging buffer"),
        size: output_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let output_storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output storage buffer"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    ComputePass {
        compute_pipeline,
        config_staging_buffer,
        config_storage_buffer,
        items_staging_buffer,
        items_storage_buffer,
        output_staging_buffer,
        output_storage_buffer,
    }
}

async fn calculate_gpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pass: &ComputePass,
    numbers: &[f32],
) -> f64 {
    let bind_group_layout = pass.compute_pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: pass.items_storage_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: pass.output_storage_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: pass.config_storage_buffer.as_entire_binding(),
            },
        ],
    });

    upload_buffer(
        device,
        &pass.config_staging_buffer,
        bytemuck::cast_slice(&[numbers.len() as u32]),
    )
    .unwrap();

    upload_buffer(
        device,
        &pass.items_staging_buffer,
        bytemuck::cast_slice(numbers),
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
        &pass.items_staging_buffer,
        0,
        &pass.items_storage_buffer,
        0,
        pass.items_storage_buffer.size(),
    );

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        cpass.set_pipeline(&pass.compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(
            (numbers.len() as u32 + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE,
            numbers.len() as u32,
            1,
        );
    }

    encoder.copy_buffer_to_buffer(
        &pass.output_storage_buffer,
        0,
        &pass.output_staging_buffer,
        0,
        pass.output_storage_buffer.size(),
    );

    let id = queue.submit(Some(encoder.finish()));

    let data: Vec<f32> = download_buffer(&device, &pass.output_staging_buffer, id).unwrap();
    data.iter().copied().map(|x| x as f64).sum::<f64>()
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
    let buffer_slice = buffer.slice(..);
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
