use crate::*;

/// Type of a ray tracing shader group.
pub type RayTracingShaderGroupType = wgt::RayTracingShaderGroupType;

/// Describes a single shader group in a ray tracing pipeline.
pub type RayTracingShaderGroupDescriptor = wgt::RayTracingShaderGroupDescriptor;

/// Region of a shader binding table buffer.
pub type ShaderBindingTableRegion = wgt::ShaderBindingTableRegion;

/// Handle to a ray tracing pipeline.
///
/// A `RayTracingPipeline` object represents a ray tracing pipeline with multiple shader stages
/// and shader groups that define the shader binding table layout.
/// It can be created with [`Device::create_ray_tracing_pipeline`].
#[derive(Debug, Clone)]
pub struct RayTracingPipeline {
    pub(crate) inner: dispatch::DispatchRayTracingPipeline,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(RayTracingPipeline: Send, Sync);

crate::cmp::impl_eq_ord_hash_proxy!(RayTracingPipeline => .inner);

impl RayTracingPipeline {
    #[cfg(custom)]
    /// Returns custom implementation of RayTracingPipeline (if custom backend and is internally T)
    pub fn as_custom<T: custom::RayTracingPipelineInterface>(&self) -> Option<&T> {
        self.inner.as_custom()
    }
}

/// Describes a programmable stage in a ray tracing pipeline.
#[derive(Clone, Debug)]
pub struct RayTracingPipelineStageDescriptor<'a> {
    /// The compiled shader module for this stage.
    pub module: &'a ShaderModule,
    /// The name of the entry point in the compiled shader.
    pub entry_point: Option<&'a str>,
    /// Advanced compilation options.
    pub compilation_options: PipelineCompilationOptions<'a>,
}

/// Describes a ray tracing pipeline.
///
/// For use with [`Device::create_ray_tracing_pipeline`].
#[derive(Clone, Debug)]
pub struct RayTracingPipelineDescriptor<'a> {
    /// Debug label of the pipeline. This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
    /// The layout of bind groups for this pipeline.
    pub layout: Option<&'a PipelineLayout>,
    /// All shader stages used by this pipeline.
    pub stages: &'a [RayTracingPipelineStageDescriptor<'a>],
    /// Shader group definitions mapping stages into SBT records.
    pub groups: &'a [wgt::RayTracingShaderGroupDescriptor],
    /// Maximum ray recursion depth.
    pub max_pipeline_ray_recursion_depth: u32,
    /// The pipeline cache to use when creating this pipeline.
    pub cache: Option<&'a PipelineCache>,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(RayTracingPipelineDescriptor<'_>: Send, Sync);
