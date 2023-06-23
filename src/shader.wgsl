const WORKGROUP_SIZE: u32 = 128u;

@group(0)
@binding(0)
var<storage, read> items: array<f32>;

@group(0)
@binding(1)
var<storage, write> output: array<f32>;

struct Config {
    num_items: u32
};

@group(0)
@binding(2)
var<storage, read> config: Config;

var<workgroup> local_sums: array<f32, WORKGROUP_SIZE>;

@compute
@workgroup_size(128, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>, @builtin(workgroup_id) group_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    if global_id.x < config.num_items && global_id.y < config.num_items {
        local_sums[local_id.x] = calculate(global_id);
    } else {
        local_sums[local_id.x] = 0.0;
    }

    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride /= 2u) {
        workgroupBarrier();

        if local_id.x < stride {
            local_sums[local_id.x] += local_sums[local_id.x + stride] ;
        }
    }

    if local_id.x == 0u {
        output[group_id.x + group_id.y * num_workgroups.x] = local_sums[0];
    }
}

fn calculate(global_id: vec3<u32>) -> f32 {
    let a = items[global_id.x];
    let b = items[global_id.y];
    return log2(max(a, 1.0)) * log2(max(b, 1.0));
}
