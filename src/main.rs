use std::borrow::Cow;
use std::sync::Arc;
use wgpu::util::DeviceExt;
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

    let pass = create_radar_compute_pass(&device).await;
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
}

async fn create_radar_compute_pass(device: &wgpu::Device) -> RadarComputePass {
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

    RadarComputePass { compute_pipeline }
}

async fn execute_gpu_inner(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pass: &RadarComputePass,
    numbers: &[f32],
) -> Option<Vec<f32>> {
    let items_storage_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("items Storage Buffer"),
        contents: bytemuck::cast_slice(numbers),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let config_storage_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("config Storage Buffer"),
        contents: bytemuck::cast_slice(&[numbers.len() as u32]),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // Gets the size in bytes of the output buffer.
    let output_size =
        (numbers.len() * numbers.len()) as u64 * std::mem::size_of::<f32>() as wgpu::BufferAddress;

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: output_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let output_storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output Storage Buffer"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let bind_group_layout = pass.compute_pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: items_storage_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_storage_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: config_storage_buffer.as_entire_binding(),
            },
        ],
    });

    // A command encoder executes one or many pipelines.
    // It is to WebGPU what a command buffer is to Vulkan.
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        cpass.set_pipeline(&pass.compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(numbers.len() as u32 / 16, numbers.len() as u32 / 16, 1);
        // Number of cells to run, the (x,y,z) size of item being processed
    }
    // Sets adds copy operation to command encoder.
    // Will copy data from storage buffer on GPU to staging buffer on CPU.
    encoder.copy_buffer_to_buffer(&output_storage_buffer, 0, &staging_buffer, 0, output_size);

    let start_time = std::time::Instant::now();

    // Submits command encoder for processing
    let id = queue.submit(Some(encoder.finish()));
    let buffer_slice = staging_buffer.slice(..);
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
    staging_buffer.unmap(); // Unmaps buffer from memory
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
