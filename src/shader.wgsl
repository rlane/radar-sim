const WORKGROUP_SIZE: u32 = 128u;
const INVALID_ID: u32 = 0xffffffffu;

struct Config {
  num_emitters: u32,
  num_reflectors: u32
}

struct Emitter {
  position: vec2<f32>,
  angle: f32,
  beamwidth: f32,
}

struct Reflector {
  position: vec2<f32>,
  rcs: f32,
}

struct Output {
  reflector: u32,
  rssi: f32,
}

@group(0)
@binding(0)
var<storage, read> config: Config;

@group(0)
@binding(1)
var<storage, read> emitters: array<Emitter>;

@group(0)
@binding(2)
var<storage, read> reflectors: array<Reflector>;

@group(0)
@binding(3)
var<storage, write> output: array<Output>;

var<workgroup> local_output: array<Output, WORKGROUP_SIZE>;

@compute
@workgroup_size(128, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>, @builtin(workgroup_id) group_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let emitter_index = global_id.y;
    let reflector_index = global_id.x;

    if emitter_index < config.num_emitters && reflector_index < config.num_reflectors {
        local_output[local_id.x] = calculate(emitter_index, reflector_index);
    } else {
        local_output[local_id.x] = Output(INVALID_ID, 0.0);
    }

    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride /= 2u) {
        workgroupBarrier();

        if local_id.x < stride {
            if local_output[local_id.x].rssi < local_output[local_id.x + stride].rssi {
                local_output[local_id.x] = local_output[local_id.x + stride];
            }
        }
    }

    if local_id.x == 0u {
        output[group_id.x + group_id.y * num_workgroups.x] = local_output[0];
    }
}

fn calculate(emitter_index: u32, reflector_index: u32) -> Output {
    let emitter = emitters[emitter_index];
    let reflector = reflectors[reflector_index];
    let dp = reflector.position - emitter.position;
    let start_bearing = emitter.angle - 0.5 * emitter.beamwidth;
    let end_bearing = emitter.angle + 0.5 * emitter.beamwidth;
    let ray0 = vec2<f32>(cos(start_bearing), sin(start_bearing));
    let ray1 = vec2<f32>(cos(end_bearing), sin(end_bearing));

    if !is_clockwise(ray0, dp) && is_clockwise(ray1, dp) {
        let distance = length(dp);
        let rssi = reflector.rcs / pow(distance, 4.0);
        return Output(reflector_index, rssi);
    } else {
        return Output(INVALID_ID, 0.0);
    }
}

fn is_clockwise(v0: vec2<f32>, v1: vec2<f32>) -> bool {
    return -v0.x * v1.y + v0.y * v1.x > 0.0;
}
