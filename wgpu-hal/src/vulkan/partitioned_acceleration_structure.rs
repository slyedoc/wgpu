//! `VK_NV_partitioned_acceleration_structure` extension wrapper.
//!
//! Pairs with `VK_NV_cluster_acceleration_structure` (this fork's
//! [`super::cluster_acceleration_structure`] module). The cluster_AS extension
//! produces BLASes whose internal layout is *not* traversable by the standard
//! `VK_KHR_acceleration_structure` TLAS path -- ray queries against a regular
//! KHR TLAS that instances cluster-built BLASes silently miss. The partitioned
//! TLAS is the load-bearing companion: its traversal knows how to follow the
//! cluster-AS internal CLAS references.
//!
//! The bindings come from the [`ash`] dep, which on this branch is patched to
//! the upstream `ash` master HEAD (PR #1028). The published `ash 0.38` release
//! does not yet expose this extension.
//!
//! Gated on the `experimental-partitioned-acceleration-structure` feature.

use ash::{nv, vk};

/// Loaded `VK_NV_partitioned_acceleration_structure` entry points and the
/// device they were loaded against.
///
/// Construct with [`Self::load`]; call sites use [`Self::cmd_build`] and
/// [`Self::get_build_sizes`].
pub struct Functions {
    device: nv::partitioned_acceleration_structure::Device,
}

impl Functions {
    /// Load the extension's function pointers for a given ash device.
    ///
    /// The caller must have already enabled
    /// `VK_NV_partitioned_acceleration_structure` (and its dependency,
    /// `VK_KHR_acceleration_structure`) when creating the device, plus the
    /// `partitionedAccelerationStructure` feature on
    /// [`vk::PhysicalDevicePartitionedAccelerationStructureFeaturesNV`].
    /// Otherwise the loaded function pointers will dispatch to a panicking
    /// stub.
    pub fn load(instance: &ash::Instance, device: &ash::Device) -> Self {
        Self {
            device: nv::partitioned_acceleration_structure::Device::load(instance, device),
        }
    }

    /// `vkGetPartitionedAccelerationStructuresBuildSizesNV`.
    ///
    /// Returns the device-local memory required to hold a built partitioned
    /// AS (and its build scratch) given a description of the inputs.
    pub unsafe fn get_build_sizes(
        &self,
        info: &vk::PartitionedAccelerationStructureInstancesInputNV<'_>,
    ) -> vk::AccelerationStructureBuildSizesInfoKHR<'static> {
        let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            (self
                .device
                .fp()
                .get_partitioned_acceleration_structures_build_sizes_nv)(
                self.device.device(),
                info,
                &mut size_info,
            );
        }
        size_info
    }

    /// `vkCmdBuildPartitionedAccelerationStructuresNV`.
    ///
    /// Records a partitioned-AS build into the given command buffer. The
    /// build's per-instance + per-partition data is read from the
    /// `srcInfos` device-address buffer; counts come from `srcInfosCount`
    /// (a pointer to a u32 holding the actual count). See the Vulkan spec
    /// for `VkBuildPartitionedAccelerationStructureInfoNV`.
    pub unsafe fn cmd_build(
        &self,
        command_buffer: vk::CommandBuffer,
        info: &vk::BuildPartitionedAccelerationStructureInfoNV<'_>,
    ) {
        unsafe {
            (self
                .device
                .fp()
                .cmd_build_partitioned_acceleration_structures_nv)(command_buffer, info);
        }
    }
}

/// Extension name string. Pass to the device-extension list when opening an
/// adapter to enable this extension.
pub const NAME: &core::ffi::CStr = vk::NV_PARTITIONED_ACCELERATION_STRUCTURE_NAME;
