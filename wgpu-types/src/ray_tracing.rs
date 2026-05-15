use alloc::vec::Vec;

#[cfg(any(feature = "serde", test))]
use serde::{Deserialize, Serialize};

#[cfg(doc)]
use crate::{Features, VertexFormat};

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Descriptor for all size defining attributes of a single triangle geometry inside a bottom level acceleration structure.
pub struct BlasTriangleGeometrySizeDescriptor {
    /// Format of a vertex position, must be [`VertexFormat::Float32x3`]
    /// with just [`Features::EXPERIMENTAL_RAY_QUERY`]
    /// but [`Features::EXTENDED_ACCELERATION_STRUCTURE_VERTEX_FORMATS`] adds more.
    pub vertex_format: crate::VertexFormat,
    /// Number of vertices.
    pub vertex_count: u32,
    /// Format of an index. Only needed if an index buffer is used.
    /// If `index_format` is provided `index_count` is required.
    pub index_format: Option<crate::IndexFormat>,
    /// Number of indices. Only needed if an index buffer is used.
    /// If `index_count` is provided `index_format` is required.
    pub index_count: Option<u32>,
    /// Flags for the geometry.
    pub flags: AccelerationStructureGeometryFlags,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Descriptor for size-defining attributes of one AABB geometry group in a bottom level acceleration structure.
///
/// Each primitive is one axis-aligned bounding box, typically stored as two `vec3<f32>` min/max corners.
pub struct BlasAABBGeometrySizeDescriptor {
    /// Number of AABB primitives in this geometry.
    pub primitive_count: u32,
    /// Flags for the geometry.
    pub flags: AccelerationStructureGeometryFlags,
}

/// Minimum stride for AABB geometry (24 bytes: two `vec3<f32>`).
pub const AABB_GEOMETRY_MIN_STRIDE: crate::BufferAddress = 24;

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Descriptor for all size defining attributes of all geometries inside a bottom level acceleration structure.
pub enum BlasGeometrySizeDescriptors {
    /// Triangle geometry version.
    Triangles {
        /// Descriptor for each triangle geometry.
        descriptors: Vec<BlasTriangleGeometrySizeDescriptor>,
    },
    /// AABB geometry version.
    AABBs {
        /// Descriptor for each AABB geometry.
        descriptors: Vec<BlasAABBGeometrySizeDescriptor>,
    },
}

#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Update mode for acceleration structure builds.
pub enum AccelerationStructureUpdateMode {
    /// Always perform a full build.
    Build,
    /// If possible, perform an incremental update.
    ///
    /// Not advised for major topology changes.
    /// (Useful for e.g. skinning)
    PreferUpdate,
}

#[repr(C)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Descriptor for creating a bottom level acceleration structure.
pub struct CreateBlasDescriptor<L> {
    /// Label for the bottom level acceleration structure.
    pub label: L,
    /// Flags for the bottom level acceleration structure.
    pub flags: AccelerationStructureFlags,
    /// Update mode for the bottom level acceleration structure.
    pub update_mode: AccelerationStructureUpdateMode,
}

impl<L> CreateBlasDescriptor<L> {
    /// Takes a closure and maps the label of the blas descriptor into another.
    pub fn map_label<K>(&self, fun: impl FnOnce(&L) -> K) -> CreateBlasDescriptor<K> {
        CreateBlasDescriptor {
            label: fun(&self.label),
            flags: self.flags,
            update_mode: self.update_mode,
        }
    }
}

#[repr(C)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Descriptor for creating a top level acceleration structure.
pub struct CreateTlasDescriptor<L> {
    /// Label for the top level acceleration structure.
    pub label: L,
    /// Number of instances that can be stored in the acceleration structure.
    pub max_instances: u32,
    /// Flags for the bottom level acceleration structure.
    pub flags: AccelerationStructureFlags,
    /// Update mode for the bottom level acceleration structure.
    pub update_mode: AccelerationStructureUpdateMode,
}

impl<L> CreateTlasDescriptor<L> {
    /// Takes a closure and maps the label of the blas descriptor into another.
    pub fn map_label<K>(&self, fun: impl FnOnce(&L) -> K) -> CreateTlasDescriptor<K> {
        CreateTlasDescriptor {
            label: fun(&self.label),
            flags: self.flags,
            update_mode: self.update_mode,
            max_instances: self.max_instances,
        }
    }
}

/// Descriptor for `Device::create_partitioned_tlas`.
///
/// Sized via `build_sizes_input` and allocated through wgpu-hal so the
/// resulting `wgpu::Tlas` owns its acceleration-structure storage. The
/// initial AS storage is zero-initialised; the partitioned-AS build path
/// treats that as an empty AS on the first
/// `build_partitioned_acceleration_structures` call.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CreatePartitionedTlasDescriptor<L> {
    /// Label for the partitioned TLAS.
    pub label: L,
    /// Maximum number of instances the TLAS will hold.
    pub max_instances: u32,
    /// Build performance / memory hints. Surface-matched with the
    /// later partitioned-AS build descriptor.
    pub flags: AccelerationStructureFlags,
    /// Update mode for the TLAS.
    pub update_mode: AccelerationStructureUpdateMode,
    /// Upper-bound build shape used to compute the AS storage size at
    /// creation. Must describe the same shape (or an upper bound thereof)
    /// passed to the per-build
    /// [`PartitionedAccelerationStructureBuildIndirectInfo::input`].
    pub build_sizes_input: PartitionedAccelerationStructureBuildSizesDescriptor,
}

impl<L> CreatePartitionedTlasDescriptor<L> {
    /// Takes a closure and maps the label of the descriptor into another.
    pub fn map_label<K>(&self, fun: impl FnOnce(&L) -> K) -> CreatePartitionedTlasDescriptor<K> {
        CreatePartitionedTlasDescriptor {
            label: fun(&self.label),
            max_instances: self.max_instances,
            flags: self.flags,
            update_mode: self.update_mode,
            build_sizes_input: self.build_sizes_input,
        }
    }
}

bitflags::bitflags!(
    /// Flags for acceleration structures
    #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
    #[cfg_attr(feature = "serde", serde(transparent))]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct AccelerationStructureFlags: u8 {
        /// Allow for incremental updates (no change in size), currently this is unimplemented
        /// and will build as normal (this is fine, update vs build should be unnoticeable)
        const ALLOW_UPDATE = 1 << 0;
        /// Allow the acceleration structure to be compacted in a copy operation
        /// (`Blas::prepare_for_compaction`, `CommandEncoder::compact_blas`).
        const ALLOW_COMPACTION = 1 << 1;
        /// Optimize for fast ray tracing performance, recommended if the geometry is unlikely
        /// to change (e.g. in a game: non-interactive scene geometry)
        const PREFER_FAST_TRACE = 1 << 2;
        /// Optimize for fast build time, recommended if geometry is likely to change frequently
        /// (e.g. in a game: player model).
        const PREFER_FAST_BUILD = 1 << 3;
        /// Optimize for low memory footprint (both while building and in the output BLAS).
        const LOW_MEMORY = 1 << 4;
        /// Use `BlasTriangleGeometry::transform_buffer` when building a BLAS (only allowed in
        /// BLAS creation)
        const USE_TRANSFORM = 1 << 5;
        /// Allow retrieval of the vertices of the triangle hit by a ray.
        const ALLOW_RAY_HIT_VERTEX_RETURN = 1 << 6;
    }
);

bitflags::bitflags!(
    /// Flags for acceleration structure geometries
    #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
    #[cfg_attr(feature = "serde", serde(transparent))]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct AccelerationStructureGeometryFlags: u8 {
        /// Is OPAQUE (is there no alpha test) recommended as currently in naga there is no
        /// candidate intersections yet so currently BLASes without this flag will not have hits.
        /// Not enabling this makes the BLAS unable to be interacted with in WGSL.
        const OPAQUE = 1 << 0;
        /// NO_DUPLICATE_ANY_HIT_INVOCATION, not useful unless using hal with wgpu, ray-tracing
        /// pipelines are not supported in wgpu so any-hit shaders do not exist. For when any-hit
        /// shaders are implemented (or experienced users who combine this with an underlying library:
        /// for any primitive (triangle or AABB) multiple any-hit shaders sometimes may be invoked
        /// (especially in AABBs like a sphere), if this flag in present only one hit on a primitive may
        /// invoke an any-hit shader.
        const NO_DUPLICATE_ANY_HIT_INVOCATION = 1 << 1;
    }
);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// What a copy between acceleration structures should do
pub enum AccelerationStructureCopy {
    /// Directly duplicate an acceleration structure to another
    Clone,
    /// Duplicate and compact an acceleration structure
    Compact,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// What type the data of an acceleration structure is
pub enum AccelerationStructureType {
    /// The types of the acceleration structure are triangles
    Triangles,
    /// The types of the acceleration structure are axis aligned bounding boxes
    AABBs,
    /// The types of the acceleration structure are instances
    Instances,
}

/// Alignment requirement for transform buffers used in acceleration structure builds
pub const TRANSFORM_BUFFER_ALIGNMENT: crate::BufferAddress = 16;

/// Alignment requirement for instance buffers used in acceleration structure builds (`build_acceleration_structures_unsafe_tlas`)
pub const INSTANCE_BUFFER_ALIGNMENT: crate::BufferAddress = 16;

// ============================================================================
// VK_NV_cluster_acceleration_structure types
// ============================================================================
//
// These describe a GPU-driven build of bottom-level acceleration structures
// composed of pre-built cluster acceleration structures (CLAS). The build is
// indirect: per-op input args, op count, and CLAS device addresses all live in
// device memory so a compute shader can emit them earlier in the same frame.
//
// Mirrors VkClusterAccelerationStructureInputInfoNV (op classification) and
// VkClusterAccelerationStructureCommandsInfoNV (per-build buffer slots). The
// safe wrapper for these lives in `wgpu::CommandEncoder` and takes typed
// references to wgpu Buffers; it converts to the vk:: descriptor internally
// and registers the right BufferUses with the tracker.

/// Build operation type for a cluster acceleration structure build.
///
/// Selects which kind of clusters or BLASes the indirect build produces.
/// Maps 1:1 to `VkClusterAccelerationStructureOpTypeNV`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ClusterAccelerationStructureOpType {
    /// Build per-instance BLASes from a list of CLAS device addresses.
    ///
    /// This is the Mega-Geometry "per-frame BLAS rebuild" path: each
    /// per-op input is a list of pre-built CLAS device addresses, and the
    /// output is a BLAS whose contents are those clusters aggregated.
    /// Pairs with [`ClusterAccelerationStructureOpInput::ClustersBottomLevel`].
    BuildClustersBottomLevel,
    /// Build per-cluster CLASes from triangle geometry.
    ///
    /// This is the upload-time CLAS construction path: each per-op input
    /// describes one cluster's triangles (via the
    /// `VkClusterAccelerationStructureBuildTriangleClusterInfoNV` struct
    /// in the user-supplied `src_infos_array`), and the output is a
    /// freshly-built CLAS whose device address is written to
    /// `dst_addresses_array`. Pairs with
    /// [`ClusterAccelerationStructureOpInput::TriangleCluster`].
    BuildTriangleCluster,
}

/// How destination addresses are supplied to a cluster AS build.
///
/// Maps 1:1 to `VkClusterAccelerationStructureOpModeNV`. The size-query
/// mode (`COMPUTE_SIZES`) is intentionally not exposed — wgpu's
/// `get_cluster_acceleration_structure_build_sizes` covers that path.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ClusterAccelerationStructureOpMode {
    /// Driver sub-allocates inside `dst_implicit_data` and writes each
    /// per-op output's device address into `dst_addresses_array`.
    ImplicitDestinations,
    /// Caller pre-populates `dst_addresses_array` with the target device
    /// addresses; the driver writes each per-op output AT the supplied
    /// address. Used by aurora to pin BLAS addresses per TLAS slot so
    /// the TLAS instance pointer stays stable across frames.
    ExplicitDestinations,
}

/// Per-op input shape for `BuildClustersBottomLevel`.
///
/// Mirrors `VkClusterAccelerationStructureClustersBottomLevelInputNV`. These
/// are *upper-bound* counts used by `Device::get_cluster_build_sizes` to
/// allocate scratch and storage; the actual per-build inputs are uploaded
/// into the args buffer at indirect-build time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ClusterAccelerationStructureClustersBottomLevelInput {
    /// Maximum total number of clusters across all per-op outputs in the build.
    pub max_total_cluster_count: u32,
    /// Maximum number of clusters in any single per-op output.
    pub max_cluster_count_per_acceleration_structure: u32,
}

/// Per-op input shape for `BuildTriangleCluster`.
///
/// Mirrors `VkClusterAccelerationStructureTriangleClusterInputNV`. These
/// are *upper-bound* counts used by `Device::get_cluster_build_sizes` to
/// allocate scratch and storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ClusterAccelerationStructureTriangleClusterInput {
    /// Vertex format encoded as a raw `VkFormat` enum value. Aurora uses
    /// `VK_FORMAT_R32G32B32_SFLOAT` (= 106). Mirrored verbatim so callers
    /// don't need to depend on ash.
    pub vertex_format: u32,
    /// Maximum geometry-index value any single triangle uses (4-bit slot,
    /// max 15 per NV spec).
    pub max_geometry_index_value: u32,
    /// Maximum number of unique geometries referenced across the build
    /// (NV spec caps at 10).
    pub max_cluster_unique_geometry_count: u32,
    /// Maximum triangle count in any single output cluster.
    pub max_cluster_triangle_count: u32,
    /// Maximum vertex count in any single output cluster.
    pub max_cluster_vertex_count: u32,
    /// Total triangle count across all output clusters.
    pub max_total_triangle_count: u32,
    /// Total vertex count across all output clusters.
    pub max_total_vertex_count: u32,
    /// Minimum position truncate bit count (passed through; 0 = no truncation).
    pub min_position_truncate_bit_count: u32,
}

/// Op-type-specific input data for a cluster AS build.
///
/// Variant must match the `op_type` selected in
/// [`ClusterAccelerationStructureBuildSizesDescriptor`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ClusterAccelerationStructureOpInput {
    /// Input for [`ClusterAccelerationStructureOpType::BuildClustersBottomLevel`].
    ClustersBottomLevel(ClusterAccelerationStructureClustersBottomLevelInput),
    /// Input for [`ClusterAccelerationStructureOpType::BuildTriangleCluster`].
    TriangleCluster(ClusterAccelerationStructureTriangleClusterInput),
}

/// Descriptor passed to `Device::get_cluster_build_sizes` and shared by the
/// command-time `build_cluster_acceleration_structures_indirect` call so the
/// driver knows the upper-bound shape of the build.
///
/// Mirrors `VkClusterAccelerationStructureInputInfoNV`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ClusterAccelerationStructureBuildSizesDescriptor {
    /// Upper bound on the per-op outputs the build will produce.
    pub max_acceleration_structure_count: u32,
    /// Build performance / memory hints.
    ///
    /// `AccelerationStructureFlags::PREFER_FAST_TRACE` matches what Aurora
    /// uses; `ALLOW_UPDATE` is meaningless for cluster builds and ignored.
    pub flags: AccelerationStructureFlags,
    /// Which kind of cluster build this is.
    pub op_type: ClusterAccelerationStructureOpType,
    /// How destination memory is supplied. See variant docs.
    pub op_mode: ClusterAccelerationStructureOpMode,
    /// Op-type-specific input upper bounds.
    pub op_input: ClusterAccelerationStructureOpInput,
}

/// A strided device-address region for a cluster_AS indirect-build input
/// or output, generic over the buffer reference type so this struct can
/// flow through wgpu API → wgpu-core → wgpu-hal without rebuilding.
///
/// Mirrors `VkStridedDeviceAddressRegionKHR`. The `buffer` reference
/// identifies the wgpu/hal Buffer; `offset` is added to the buffer's base
/// device address. `stride` is the per-element byte stride (e.g. 8 for an
/// array of u64 device addresses, or `size_of::<BuildClustersBottomLevelInfoNV>()`
/// for per-op input args). `size` is the total covered byte size.
#[derive(Clone, Debug)]
pub struct ClusterAccelerationStructureStridedBufferRegion<B> {
    /// Buffer that backs the region.
    pub buffer: B,
    /// Byte offset into the buffer at which the region starts.
    pub offset: u64,
    /// Per-element byte stride.
    pub stride: u64,
    /// Total byte size of the region. Must be at least
    /// `stride * (entry_count - 1) + element_size`.
    pub size: u64,
}

/// Argument bundle for the safe cluster_AS indirect build.
///
/// Generic over the buffer reference type — `&wgpu::Buffer` in the public
/// safe API, `&dyn hal::DynBuffer` once erased for dispatch, and the
/// backend's concrete buffer type inside the hal Vulkan impl.
///
/// Mirrors `VkClusterAccelerationStructureCommandsInfoNV` but exposes
/// typed buffer references that wgpu-core can register with the tracker
/// before dispatch.
#[derive(Clone, Debug)]
pub struct ClusterAccelerationStructureBuildIndirectInfo<'a, B> {
    /// Upper-bound build shape (must match the descriptor that was passed
    /// to `get_cluster_build_sizes` to allocate `dst_implicit_data` /
    /// scratch). Borrowed so we don't redundantly clone the inner enum
    /// payload on every frame.
    pub input: &'a ClusterAccelerationStructureBuildSizesDescriptor,
    /// Output AS storage when `op_mode = ImplicitDestinations`. The driver
    /// sub-allocates each per-op output inside this buffer.
    pub dst_implicit_data: B,
    /// Byte offset into `dst_implicit_data` for the storage base.
    pub dst_implicit_data_offset: u64,
    /// Indirect-build scratch buffer. Must satisfy
    /// `ClusterAccelerationStructureBuildSizes::build_scratch_size`.
    pub scratch_data: B,
    /// Byte offset into `scratch_data`.
    pub scratch_data_offset: u64,
    /// Output addresses array: `op_count` entries of u64 BLAS device
    /// addresses, written by the driver. None disables the write.
    pub dst_addresses_array: Option<ClusterAccelerationStructureStridedBufferRegion<B>>,
    /// Output sizes array: `op_count` entries of u32 byte sizes, written
    /// by the driver. None disables the write.
    pub dst_sizes_array: Option<ClusterAccelerationStructureStridedBufferRegion<B>>,
    /// Per-op input arg structs (e.g. `VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV`),
    /// read by the driver at build time.
    pub src_infos_array: ClusterAccelerationStructureStridedBufferRegion<B>,
    /// Buffer holding a single u32 indirect op count, read by the driver
    /// at build time.
    pub src_infos_count: B,
    /// Byte offset into `src_infos_count` (must be 4-byte aligned).
    pub src_infos_count_offset: u64,
}

impl<'a, B> ClusterAccelerationStructureBuildIndirectInfo<'a, B> {
    /// Re-map every buffer reference through `f`, preserving offsets /
    /// strides / sizes. Used to lower from `&wgpu::Buffer` through
    /// `&dyn hal::DynBuffer` to the backend-concrete buffer type.
    pub fn map_buffers<B2>(
        self,
        mut f: impl FnMut(B) -> B2,
    ) -> ClusterAccelerationStructureBuildIndirectInfo<'a, B2> {
        ClusterAccelerationStructureBuildIndirectInfo {
            input: self.input,
            dst_implicit_data: f(self.dst_implicit_data),
            dst_implicit_data_offset: self.dst_implicit_data_offset,
            scratch_data: f(self.scratch_data),
            scratch_data_offset: self.scratch_data_offset,
            dst_addresses_array: self.dst_addresses_array.map(|r| {
                ClusterAccelerationStructureStridedBufferRegion {
                    buffer: f(r.buffer),
                    offset: r.offset,
                    stride: r.stride,
                    size: r.size,
                }
            }),
            dst_sizes_array: self.dst_sizes_array.map(|r| {
                ClusterAccelerationStructureStridedBufferRegion {
                    buffer: f(r.buffer),
                    offset: r.offset,
                    stride: r.stride,
                    size: r.size,
                }
            }),
            src_infos_array: ClusterAccelerationStructureStridedBufferRegion {
                buffer: f(self.src_infos_array.buffer),
                offset: self.src_infos_array.offset,
                stride: self.src_infos_array.stride,
                size: self.src_infos_array.size,
            },
            src_infos_count: f(self.src_infos_count),
            src_infos_count_offset: self.src_infos_count_offset,
        }
    }
}

/// Sizes returned by `Device::get_cluster_build_sizes`.
///
/// Same shape as `VkAccelerationStructureBuildSizesInfoKHR` but exposed in
/// wgpu's namespace so callers don't need to depend on ash directly. The
/// values are upper bounds; the actual per-build sizes depend on the
/// per-op inputs supplied at command time.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ClusterAccelerationStructureBuildSizes {
    /// Bytes required to hold all per-op output cluster ASes in
    /// `dst_implicit_data`. Pass at least this many bytes when allocating
    /// the destination buffer.
    pub acceleration_structure_size: u64,
    /// Bytes required for the indirect-build scratch buffer.
    pub build_scratch_size: u64,
    /// Update scratch size — present for spec parity but zero for cluster
    /// builds, which don't support in-place update.
    pub update_scratch_size: u64,
}

// ============================================================================
// VK_NV_partitioned_acceleration_structure types
// ============================================================================
//
// Pairs with VK_NV_cluster_acceleration_structure above. The partitioned
// TLAS is the load-bearing companion: ray queries against a regular KHR
// TLAS that instances cluster-built BLASes silently miss; the partitioned
// TLAS's traversal knows how to follow the cluster-AS CLAS references.

/// Descriptor passed to `Device::get_partitioned_acceleration_structure_build_sizes`
/// and shared by the command-time
/// `build_partitioned_acceleration_structures` call so the driver knows
/// the upper-bound shape of the build.
///
/// Mirrors `VkPartitionedAccelerationStructureInstancesInputNV`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PartitionedAccelerationStructureBuildSizesDescriptor {
    /// Build performance / memory hints. Only the
    /// `PREFER_FAST_TRACE` / `PREFER_FAST_BUILD` / `ALLOW_UPDATE` bits
    /// have semantic meaning here; others are ignored.
    pub flags: AccelerationStructureFlags,
    /// Total number of instances across all partitions.
    pub instance_count: u32,
    /// Maximum number of instances in any single partition.
    pub max_instance_per_partition_count: u32,
    /// Number of partitions in this AS.
    pub partition_count: u32,
    /// Maximum number of instances that can live in the global partition.
    pub max_instance_in_global_partition_count: u32,
}

/// Sizes returned by `Device::get_partitioned_acceleration_structure_build_sizes`.
///
/// Same shape as the cluster_AS version above. The values bound the
/// destination AS storage and the build scratch needed for the build
/// shape described by the descriptor.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PartitionedAccelerationStructureBuildSizes {
    /// Bytes required for the destination partitioned-AS storage.
    pub acceleration_structure_size: u64,
    /// Bytes required for the build scratch buffer.
    pub build_scratch_size: u64,
    /// Bytes required for in-place update scratch. Zero if the build
    /// `flags` don't request `ALLOW_UPDATE`.
    pub update_scratch_size: u64,
}

/// Argument bundle for the safe partitioned-AS build, generic over the
/// buffer reference type — `&wgpu::Buffer` in the public API, lowered to
/// the backend's concrete buffer type for the hal call.
///
/// Mirrors `VkBuildPartitionedAccelerationStructureInfoNV` but the AS
/// destination is identified via a wgpu Tlas (resolved separately) so the
/// tracker knows about it.
#[derive(Clone, Debug)]
pub struct PartitionedAccelerationStructureBuildIndirectInfo<'a, B> {
    /// Upper-bound build shape.
    pub input: &'a PartitionedAccelerationStructureBuildSizesDescriptor,
    /// Per-instance WRITE_INSTANCE / UPDATE_INSTANCE op records.
    pub src_infos: B,
    /// Byte offset into `src_infos`.
    pub src_infos_offset: u64,
    /// Single-u32 indirect op-count buffer.
    pub src_infos_count: B,
    /// Byte offset into `src_infos_count`.
    pub src_infos_count_offset: u64,
    /// Build scratch.
    pub scratch_data: B,
    /// Byte offset into `scratch_data`.
    pub scratch_data_offset: u64,
}

impl<'a, B> PartitionedAccelerationStructureBuildIndirectInfo<'a, B> {
    /// Re-map every buffer reference through `f`.
    pub fn map_buffers<B2>(
        self,
        mut f: impl FnMut(B) -> B2,
    ) -> PartitionedAccelerationStructureBuildIndirectInfo<'a, B2> {
        PartitionedAccelerationStructureBuildIndirectInfo {
            input: self.input,
            src_infos: f(self.src_infos),
            src_infos_offset: self.src_infos_offset,
            src_infos_count: f(self.src_infos_count),
            src_infos_count_offset: self.src_infos_count_offset,
            scratch_data: f(self.scratch_data),
            scratch_data_offset: self.scratch_data_offset,
        }
    }
}
