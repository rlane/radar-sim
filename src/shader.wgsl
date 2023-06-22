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

@compute
@workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    output[global_id.x + global_id.y * config.num_items] = items[global_id.x] * items[global_id.y];
}
