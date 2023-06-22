use std::borrow::Cow;
use std::sync::Arc;
use wgpu::SubmissionIndex;

async fn run() {
    let numbers = (0..1024).map(|x| x as f32).collect::<Vec<_>>();

    execute_gpu(&numbers).await;
}

async fn execute_gpu(numbers: &[f32]) {
    // Instantiates instance of WebGPU
    let instance = wgpu::Instance::default();

    // `request_adapter` instantiates the general connection to the GPU
    let mut opts = wgpu::RequestAdapterOptions::default();
    opts.power_preference = wgpu::PowerPreference::HighPerformance;
    let adapter = instance.request_adapter(&opts).await.unwrap();

    // `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
    //  `features` being the available features.
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

    let pass = create_radar_compute_pass(&device, numbers.len()).await;
    let device = Arc::new(device);

    let result = execute_gpu_inner(&device, &queue, &pass, numbers)
        .await
        .unwrap();
    let sum = result.iter().copied().map(|x| x as f64).sum::<f64>();
    println!("Result: {sum}",);
    assert_eq!(sum, 274341298176.0);

    for _ in 0..100 {
        let n = 100;
        let start_time = std::time::Instant::now();
        for _ in 0..n {
            let _ = execute_gpu_inner(&device, &queue, &pass, numbers).await;
        }
        let elapsed = start_time.elapsed();
        println!("Average time per iteration: {:?}", elapsed / n);
    }
}

struct RadarComputePass {
    compute_pipeline: wgpu::ComputePipeline,
    config_staging_buffer: wgpu::Buffer,
    config_storage_buffer: wgpu::Buffer,
    items_staging_buffer: wgpu::Buffer,
    items_storage_buffer: wgpu::Buffer,
    output_staging_buffer: wgpu::Buffer,
    output_storage_buffer: wgpu::Buffer,
}

async fn create_radar_compute_pass(device: &wgpu::Device, len: usize) -> RadarComputePass {
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
    let output_size = (len * len) as u64 * std::mem::size_of::<f32>() as wgpu::BufferAddress;

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

    RadarComputePass {
        compute_pipeline,
        config_staging_buffer,
        config_storage_buffer,
        items_staging_buffer,
        items_storage_buffer,
        output_staging_buffer,
        output_storage_buffer,
    }
}

async fn execute_gpu_inner(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pass: &RadarComputePass,
    numbers: &[f32],
) -> Option<Vec<f32>> {
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
        cpass.dispatch_workgroups(numbers.len() as u32 / 16, numbers.len() as u32 / 16, 1);
    }

    encoder.copy_buffer_to_buffer(
        &pass.output_storage_buffer,
        0,
        &pass.output_staging_buffer,
        0,
        pass.output_storage_buffer.size(),
    );

    let start_time = std::time::Instant::now();

    // Submits command encoder for processing
    let id = queue.submit(Some(encoder.finish()));
    let buffer_slice = pass.output_staging_buffer.slice(..);
    let data = map_buffer(&device, &buffer_slice, id).unwrap();

    if false {
        let elapsed = start_time.elapsed();
        println!("Elapsed: {}ms", elapsed.as_millis());
    }

    // Since contents are got in bytes, this converts these bytes back to f32
    let result = bytemuck::cast_slice(&data).to_vec();

    // With the current interface, we have to make sure all mapped views are
    // dropped before we unmap the buffer.
    drop(data);
    pass.output_staging_buffer.unmap(); // Unmaps buffer from memory
                                        // If you are familiar with C++ these 2 lines can be thought of similarly to:
                                        //   delete myPointer;
                                        //   myPointer = NULL;
                                        // It effectively frees the memory

    // Returns data from buffer
    return Some(result);
}

fn main() {
    env_logger::init();
    pollster::block_on(run());
}

fn map_buffer<'a>(
    device: &wgpu::Device,
    buffer_slice: &'a wgpu::BufferSlice,
    id: SubmissionIndex,
) -> Result<wgpu::BufferView<'a>, wgpu::BufferAsyncError> {
    let (sender, receiver) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
        drop(sender.send(v));
    });
    device.poll(wgpu::Maintain::WaitForSubmissionIndex(id));
    receiver.recv().unwrap()?;
    Ok(buffer_slice.get_mapped_range())
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
