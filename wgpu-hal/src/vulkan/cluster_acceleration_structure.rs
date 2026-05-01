//! `VK_NV_cluster_acceleration_structure` extension wrapper.
//!
//! This is the bottom-of-the-stack acceleration structure used by RTX
//! MegaGeometry: it lets a BLAS reference a list of pre-built clusters (CLAS)
//! by device address rather than holding triangles directly. Combined with
//! `VK_NV_partitioned_acceleration_structure`, it's the basis of the
//! Nanite-style traced-geometry pipeline NVIDIA showed in Zorah.
//!
//! The bindings come from the [`ash`] dep, which on this branch is patched to
//! the upstream `ash` master HEAD (PR #1028). The published `ash 0.38` release
//! does not yet expose this extension.
//!
//! Gated on the `experimental-cluster-acceleration-structure` feature.

use ash::{nv, vk};

/// Loaded `VK_NV_cluster_acceleration_structure` entry points and the device
/// they were loaded against.
///
/// Construct with [`Self::load`]; call sites use [`Self::cmd_build_indirect`]
/// and [`Self::get_build_sizes`].
pub struct Functions {
    device: nv::cluster_acceleration_structure::Device,
}

impl Functions {
    /// Load the extension's function pointers for a given ash device.
    ///
    /// The caller must have already enabled
    /// `VK_NV_cluster_acceleration_structure` (and its dependency,
    /// `VK_KHR_acceleration_structure`) when creating the device, plus the
    /// `clusterAccelerationStructure` feature on
    /// [`vk::PhysicalDeviceClusterAccelerationStructureFeaturesNV`]. Otherwise
    /// the loaded function pointers will dispatch to a panicking stub.
    pub fn load(instance: &ash::Instance, device: &ash::Device) -> Self {
        Self {
            device: nv::cluster_acceleration_structure::Device::load(instance, device),
        }
    }

    /// `vkGetClusterAccelerationStructureBuildSizesNV`.
    ///
    /// Returns the device-local memory required to hold a built cluster AS
    /// (and its build scratch) given a description of the inputs.
    pub unsafe fn get_build_sizes(
        &self,
        info: &vk::ClusterAccelerationStructureInputInfoNV<'_>,
    ) -> vk::AccelerationStructureBuildSizesInfoKHR<'static> {
        let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            (self.device.fp().get_cluster_acceleration_structure_build_sizes_nv)(
                self.device.device(),
                info,
                &mut size_info,
            );
        }
        size_info
    }

    /// `vkCmdBuildClusterAccelerationStructureIndirectNV`.
    ///
    /// Records a fully GPU-driven build into the given command buffer; the
    /// build descriptors live behind the device addresses inside `info` and
    /// are produced earlier in the same frame by other compute work.
    pub unsafe fn cmd_build_indirect(
        &self,
        command_buffer: vk::CommandBuffer,
        info: &vk::ClusterAccelerationStructureCommandsInfoNV<'_>,
    ) {
        unsafe {
            (self.device.fp().cmd_build_cluster_acceleration_structure_indirect_nv)(
                command_buffer,
                info,
            );
        }
    }
}

/// Extension name string. Pass to the device-extension list when opening an
/// adapter to enable this extension.
pub const NAME: &core::ffi::CStr = vk::NV_CLUSTER_ACCELERATION_STRUCTURE_NAME;
