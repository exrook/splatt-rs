@group(0) @binding(0)
var<storage, read> gs: array<Gaussian>;

@group(2) @binding(0)
var<storage, read> controlbuf: ControlData;

@group(2) @binding(1)
var<storage, read_write> tiles: array<TileIndirect>;

@group(3) @binding(0)
var color_out: texture_storage_2d<rgba8unorm, read_write>;

struct Gaussian {
    rot: vec4<f32>,
    pos: vec3<f32>,
    //_pad0: u32,
    n: vec3<f32>,
    //_pad1: u32,
    sh_dc: vec3<f32>,
    //_pad2: u32,
    scale: vec3<f32>,
    opacity: f32,
}

struct ControlData {
    camera: Camera,
}

struct Camera {
    camera_rot: mat3x3<f32>,
    camera_rot_invert: mat3x3<f32>,
    camera_origin: vec3<f32>,
    focal_length: f32,
    camera_4: mat4x4<f32>,
    camera_proj: mat4x4<f32>,
    camera_4_invert: mat4x4<f32>,
}

// should match definition in lib.rs
const TILE_SIZE: u32 = 2048u;

struct TileIndirect {
    next_elem: atomic<u32>,
    area: vec4<f32>,
    splats: array<SplatPtr, TILE_SIZE>
}

struct SplatPtr {
    idx: u32,
    depth: f32,
}

var<push_constant> dd: DispatchData;
struct DispatchData {
    num_tiles: vec2<u32>,
    output_size: vec2<u32>,
    num_splats: u32,
}

fn sigmoid(x: f32) -> f32 {
    if (x >= 0.) {
        return 1. / (1. + exp(-x));
    } else {
        let z = exp(x);
        return z / (1. + z);
    }
}

fn compute_cov3d(log_scale: vec3<f32>, rot: vec4<f32>) -> array<f32, 6> {
    // let modifier = uniforms.scale_modifier;
    let modifier = 1.0;
    let S = mat3x3<f32>(
        exp(log_scale.x) * modifier, 0., 0.,
        0., exp(log_scale.y) * modifier, 0.,
        0., 0., exp(log_scale.z) * modifier,
    );

    let r = rot.x;
    let x = rot.y;
    let y = rot.z;
    let z = rot.w;

    let R = mat3x3<f32>(
        1. - 2. * (y * y + z * z), 2. * (x * y - r * z), 2. * (x * z + r * y),
        2. * (x * y + r * z), 1. - 2. * (x * x + z * z), 2. * (y * z - r * x),
        2. * (x * z - r * y), 2. * (y * z + r * x), 1. - 2. * (x * x + y * y),
    );

    let M = S * R;
    let Sigma = transpose(M) * M;

    return array<f32, 6>(
        Sigma[0][0],
        Sigma[0][1],
        Sigma[0][2],
        Sigma[1][1],
        Sigma[1][2],
        Sigma[2][2],
    );
} 

fn ndc2pix(v: f32, size: u32) -> f32 {
    return ((v + 1.0) * f32(size) - 1.0) * 0.5;
}

fn compute_cov2d(position: vec3<f32>, cov3d: array<f32, 6>) -> vec3<f32> {
    var t = controlbuf.camera.camera_rot_invert * (position - controlbuf.camera.camera_origin);// + vec3(0.5, 0.5,0.0);

    let focal_x = controlbuf.camera.focal_length * 1.0;

    let limx = 01. / focal_x;
    let limy = 01. / focal_x;
    let txtz = t.x / t.z;
    let tytz = t.y / t.z;

    t.x = (clamp(txtz, -limx, limx)) * t.z;
    t.y = (clamp(tytz, -limy, limy)) * t.z;

    let J = mat3x3(
        1.0/t.z, 0.            , -t.x / (t.z * t.z), // 0.,
        0.           , 1.0/t.z , -t.y / (t.z * t.z), // 0.,
        0.           , 0.            , 0.                            , // 0.,
        // 0.           , 0.            , 0.                            , 0.,
    ) * controlbuf.camera.focal_length * 1.0;

    let W = (controlbuf.camera.camera_rot);

    let T = W * J;

    let Vrk = mat3x3(
        cov3d[0], cov3d[1], cov3d[2],
        cov3d[1], cov3d[3], cov3d[4],
        cov3d[2], cov3d[4], cov3d[5],
    );

    var cov = transpose(T) * (Vrk) * T;

    // Apply low-pass filter: every Gaussian should be at least
    // one pixel wide/high. Discard 3rd row and column.
    cov[0][0] += 0.00000010;
    cov[1][1] += 0.00000010;

    return vec3<f32>(cov[0][0], cov[0][1], cov[1][1]);
}

@compute
@workgroup_size(1024)
fn preprocess(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(workgroup_id) wg_id: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>, @builtin(local_invocation_index) local_idx: u32) {
    if global_id.x < dd.num_splats {
        preprocess_splat(global_id.x);
    }
}

@compute
@workgroup_size(1024)
fn precompute(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(workgroup_id) wg_id: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>, @builtin(local_invocation_index) local_idx: u32) {
    if global_id.x < dd.num_splats {
        precompute_splat(global_id.x);
    }
}
@group(1) @binding(0)
var<storage, read_write> splat_depths: array<f32>;
@group(1) @binding(1)
var<storage, read_write> splat_2d_pos: array<vec2<f32>>;
@group(1) @binding(2)
var<storage, read_write> splat_2d_cov_alphas: array<vec4<f32>>;
// pack colors?
@group(1) @binding(3)
var<storage, read_write> splat_colors: array<vec3<f32>>;
@group(1) @binding(4)
var<storage, read_write> splat_3d_cov: array<array<f32, 6>>;
@group(1) @binding(5)
var<storage, read_write> splat_3d_pos_alpha: array<vec4<f32>>;

fn precompute_splat(splat_id: u32) {
    let splat = gs[splat_id];
    splat_3d_pos_alpha[splat_id] = vec4(splat.pos, sigmoid(splat.opacity));
    let color = harmonic_to_rgb(splat.sh_dc);
    splat_colors[splat_id] = color;
    let cov3d = compute_cov3d(splat.scale, splat.rot);
    splat_3d_cov[splat_id] = cov3d;
}

fn isnan(val: f32) -> bool {
    let floatToUint: u32 = bitcast<u32>(val);
    return (floatToUint & 0x7fffffffu) > 0x7f800000u;
}
fn preprocess_splat(splat_id: u32) {
    let pos_alpha = splat_3d_pos_alpha[splat_id];
    let splat_pos = pos_alpha.xyz;

    let cam_pos = transform_world_to_cam(splat_pos);
    let pos_2d = cam_pos.xy;
    let depth = cam_pos.z;

    if depth <= 0.0 {
        // check depth
        return;
    }

    splat_depths[splat_id] = depth;
    splat_2d_pos[splat_id] = pos_2d;

    let cov3d = splat_3d_cov[splat_id];

    // TODO: higher degree sh
    // let color =  harmonic_to_rgb(splat.sh_dc);
    // splat_colors[splat_id] = color
    
    let cov2d = compute_cov2d(splat_pos, cov3d);
    let det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    let det_inv = 1.0 / det;
    let conic = vec3<f32>(cov2d.z * det_inv, -cov2d.y * det_inv, cov2d.x * det_inv);

    splat_2d_cov_alphas[splat_id] = vec4(conic, pos_alpha.a);

    let scale = 2.0;
    var aabb = sqrt(abs(vec2(cov2d.x, cov2d.z))) * scale;

    let aabb_tiles =vec2<i32>(ceil(aabb * vec2<f32>(dd.num_tiles)));
    let splat_tile = vec2<i32>(ceil((cam_pos.xy) * vec2<f32>(dd.num_tiles) - vec2(0.2)));

    let corner1 = clamp(splat_tile - aabb_tiles, vec2(0), vec2<i32>(dd.num_tiles));
    let corner2 = clamp(splat_tile + aabb_tiles, vec2(0), vec2<i32>(dd.num_tiles));

    let size = corner2 - corner1;

    let count = max(0, min(size.x * size.y, 80000)); // I hoped this would prevent GPU hangs. It does not.

    for (var i = 0; i < count; i++) {
        let x = i % (size.x);
        let y = i / (size.x);
        let coord = corner1 + vec2(x,y);
        if all(coord >= vec2(0) && coord < vec2<i32>(dd.num_tiles)) {
            let tile_idx = coord.x + coord.y * i32(dd.num_tiles.x);
        
            let t_idx = atomicAdd(&tiles[tile_idx].next_elem, 1u);
            if t_idx < TILE_SIZE {
                tiles[tile_idx].splats[t_idx] = SplatPtr(splat_id, depth);
            }
        }
    }
}

@compute
@workgroup_size(1024)
fn setup_tiles(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(workgroup_id) wg_id: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>, @builtin(local_invocation_index) local_idx: u32, @builtin(local_invocation_id) local_id: vec3<u32>) {
    let tile_id = global_id.y;
    let tile_y = tile_id / dd.num_tiles.x;
    let tile_x = tile_id % dd.num_tiles.x;
    if tile_id < (dd.num_tiles.x * dd.num_tiles.y) {
        if global_id.x < TILE_SIZE {
            tiles[tile_id].splats[global_id.x] = SplatPtr(0u, 1000001.0 * f32(global_id.x));
        }
        if global_id.x == 0u {
            tiles[tile_id].next_elem = 0u;
        }
    }
}

fn pos_camera_space(pos: vec3<f32>) -> vec3<f32> {
    return controlbuf.camera.camera_rot_invert * (pos - controlbuf.camera.camera_origin);
}

fn transform_world_to_cam(pos: vec3<f32>) -> vec3<f32> {
    let camera_space = pos_camera_space(pos);

    // TODO: handle this
    let aspect = vec2(1.0);

    let z = camera_space.z;
    let reconstruct_pixel = -(camera_space).xy / (camera_space.z);
    let reconstruct_relative_pos = reconstruct_pixel * controlbuf.camera.focal_length + (vec2(0.5) * aspect);
    // let reconstruct_relative_pos = reconstruct_pixel * 2.5 + (vec2(0.5) * aspect);
    let reconstruct_rel_pos = reconstruct_relative_pos / aspect;
    return vec3(reconstruct_rel_pos, z);
}

const INNER_SORT_WORKGROUP_SIZE: u32 = 512u;
var<workgroup> wg_splats: array<SplatPtr, TILE_SIZE>;

@compute
@workgroup_size(512)
fn sort_within_tiles(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(workgroup_id) wg_id: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>, @builtin(local_invocation_index) local_idx: u32) {
    let tile_id = wg_id.x;

    load_wg_splats(tile_id, local_idx);
    workgroupBarrier();
    sort(local_idx);
    store_wg_splats(tile_id, local_idx);

}

fn sort(real_local_idx: u32) {
    // double the size of orange block until we hit the array size
    for (var orange_size = 2u; orange_size <= TILE_SIZE; orange_size *= 2u) { // k is doubled every iteration

        for (var i = 0u; i <= (TILE_SIZE / (INNER_SORT_WORKGROUP_SIZE * 2u)); i += 1u) {
            let local_idx = real_local_idx * (TILE_SIZE / (INNER_SORT_WORKGROUP_SIZE * 2u)) + i;
            // execute orange block
            let jump = orange_size * (local_idx / (orange_size / 2u));
            let idx1 = (local_idx % (orange_size / 2u)) + jump;
            let opponent = (orange_size - (1u + 2u * ((local_idx) % (orange_size / 2u))));
            swap(idx1, idx1 + opponent);
        }
        workgroupBarrier();

        // red blocks, first red block is half the size of orange block, go until size 2
        for (var red_size = orange_size/2u; red_size > 1u; red_size /= 2u)  {
            for (var i = 0u; i <= (TILE_SIZE / (INNER_SORT_WORKGROUP_SIZE * 2u)); i += 1u) {
                let local_idx = real_local_idx * (TILE_SIZE / (INNER_SORT_WORKGROUP_SIZE * 2u)) + i;
                let jump = red_size * (local_idx / (red_size / 2u));
                let idx1 = (local_idx % (red_size / 2u)) + jump;

                let opponent = idx1 + (red_size / 2u);
                swap(idx1, opponent);
            }
            workgroupBarrier();
        }
    }
}

fn swap(lower_idx: u32, upper_idx: u32) {
    let lower = wg_splats[lower_idx];
    let upper = wg_splats[upper_idx];

    if lower.depth > upper.depth {
        wg_splats[lower_idx] = upper;
        wg_splats[upper_idx] = lower;
    }
}

fn load_wg_splats(tile_idx: u32, local_idx: u32) {
    for (var i = 0u; i < (TILE_SIZE / (INNER_SORT_WORKGROUP_SIZE)); i += 1u) {
        let idx = local_idx + i*INNER_SORT_WORKGROUP_SIZE;
        wg_splats[idx] = tiles[tile_idx].splats[idx];
    }
}

fn store_wg_splats(tile_idx: u32, local_idx: u32) {
    for (var i = 0u; i < (TILE_SIZE / (INNER_SORT_WORKGROUP_SIZE)); i += 1u) {
        let idx = local_idx + i*INNER_SORT_WORKGROUP_SIZE;
        tiles[tile_idx].splats[idx] = wg_splats[idx];
    }
}

@compute
@workgroup_size(16, 16)
fn draw_tiles(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(workgroup_id) wg_id: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>, @builtin(local_invocation_index) local_idx: u32) {
    let tile_id = wg_id.xy;
    let tile_idx = tile_id.x + tile_id.y * dd.num_tiles.x;

    let screen_pos = vec2<f32>(global_id.xy) / vec2<f32>(dd.output_size);

    var out_color = vec3(0.0);
    let len = min(tiles[tile_idx].next_elem, TILE_SIZE);
    let tile_ptr = &tiles[tile_idx];

    var T = 1.0;

    for (var i = 0u; i < len; i++) {
        let splat_idx = tiles[tile_idx].splats[i].idx;

        let color = splat_colors[splat_idx];
        let cov2da = splat_2d_cov_alphas[splat_idx];
        let pos = splat_2d_pos[splat_idx];
        let rel_pos = (screen_pos - pos);
        let power = -0.5f * (cov2da.x * (rel_pos.x * rel_pos.x) + cov2da.z * (rel_pos.y * rel_pos.y)) - cov2da.y * (rel_pos.x * rel_pos.y);
        if (power > 0.0f) {
            continue;
        }
        let alpha = min(0.99, cov2da.a * exp(power));
        if alpha < 1.0/255.0 {
            continue;
        }

        out_color += color * alpha * T;
        if T * (1.0 - alpha) < 0.001 {
            break;
        }
        T = T * (1.0 - alpha);
    }
    // let background_color = vec4(vec3(0.0), 0.001);
    // out_color = alpha_blend_straight(vec4(out_color, 1.0 - T), background_color).xyz;

    // DEBUG: show tiles at or near capcity
    // if len >= (7u * TILE_SIZE / 8u) {
    //     out_color.r += 0.2;
    // }
    textureStore(color_out, global_id.xy, vec4(out_color, 1.0));
}

fn alpha_blend_straight(over: vec4<f32>, under: vec4<f32>) -> vec4<f32> {
    let blend_factor = 1.0 - over.a;
    let new_alpha = over.a + under.a * blend_factor;
    let new_color = (over.a * over.rgb + under.rgb * under.a * blend_factor) / new_alpha;
    return vec4(new_color, new_alpha);
}

const SH_C0: f32 = 0.28209479177387814;
fn harmonic_to_rgb(harmonic: vec3<f32>) -> vec3<f32> {
    return saturate(harmonic * SH_C0 + vec3(0.5));
}
