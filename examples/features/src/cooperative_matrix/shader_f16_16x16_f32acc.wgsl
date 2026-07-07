// Cooperative Matrix Multiplication Example (16x16, f16 inputs, f32 accumulator)
//
// Mixed-precision variant: A and B are f16, C accumulates in f32
// (Vulkan config "AB=F16 CR=F32", the preferred shape for ML workloads).
//
// The matrices are stored in row-major order, hence the T-suffixed
// load/store builtins (the unsuffixed ones are column-major).

enable f16;
enable wgpu_cooperative_matrix;

const TILE_SIZE: u32 = 16u;

@group(0) @binding(0)
var<storage, read> matrix_a: array<f16>;

@group(0) @binding(1)
var<storage, read> matrix_b: array<f16>;

@group(0) @binding(2)
var<storage, read_write> matrix_c: array<f32>;

// Dimensions passed as uniforms: M, N, K for C[M,N] = A[M,K] * B[K,N]
@group(0) @binding(3)
var<uniform> dimensions: vec4<u32>; // x=M, y=N, z=K, w=stride

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let K = dimensions.z;
    let stride = dimensions.w;

    let tile_row = workgroup_id.x * TILE_SIZE;
    let tile_col = workgroup_id.y * TILE_SIZE;

    let c_offset = tile_row * stride + tile_col;
    var c_tile = coopLoadT<coop_mat16x16<f32, C>>(&matrix_c[c_offset], stride);

    for (var k: u32 = 0u; k < K; k += TILE_SIZE) {
        let a_tile = coopLoadT<coop_mat16x16<f16, A>>(&matrix_a[tile_row * K + k], K);
        let b_tile = coopLoadT<coop_mat16x16<f16, B>>(&matrix_b[k * stride + tile_col], stride);
        c_tile = coopMultiplyAdd(a_tile, b_tile, c_tile);
    }

    coopStoreT(c_tile, &matrix_c[c_offset], stride);
}
