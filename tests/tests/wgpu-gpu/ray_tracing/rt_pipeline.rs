use crate::ray_tracing::acceleration_structure_limits;
use wgpu::{
    include_wgsl, Backends, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType,
    PipelineLayoutDescriptor, ShaderStages,
};
use wgpu_macros::gpu_test;
use wgpu_test::{FailureCase, GpuTestInitializer};
use wgpu_test::{GpuTestConfiguration, TestParameters, TestingContext};

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.push(CREATE_RT_PIPELINE);
}

#[gpu_test]
static CREATE_RT_PIPELINE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .limits(acceleration_structure_limits())
            .features(
                wgpu::Features::EXPERIMENTAL_RAY_QUERY
                    | wgpu::Features::EXPERIMENTAL_RAY_TRACING_PIPELINE,
            )
            .skip(FailureCase::backend(Backends::GL))
            .skip(FailureCase::backend(Backends::DX12))
            .skip(FailureCase::backend(Backends::METAL)),
    )
    .run_sync(create_rt_pipeline);

fn create_rt_pipeline(ctx: TestingContext) {
    let shader = ctx
        .device
        .create_shader_module(include_wgsl!("rt_pipeline.wgsl"));

    let bind_group_layout = ctx
        .device
        .create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("RT BGL"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::all(),
                ty: BindingType::AccelerationStructure { vertex_return: false },
                count: None,
            }],
        });

    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("RT Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

    let _pipeline = ctx
        .device
        .create_ray_tracing_pipeline(&wgpu::RayTracingPipelineDescriptor {
            label: Some("RT Pipeline"),
            layout: Some(&pipeline_layout),
            stages: &[
                wgpu::RayTracingPipelineStageDescriptor {
                    module: &shader,
                    entry_point: Some("ray_gen_main"),
                    compilation_options: Default::default(),
                },
                wgpu::RayTracingPipelineStageDescriptor {
                    module: &shader,
                    entry_point: Some("miss"),
                    compilation_options: Default::default(),
                },
                wgpu::RayTracingPipelineStageDescriptor {
                    module: &shader,
                    entry_point: Some("any_hit_main"),
                    compilation_options: Default::default(),
                },
                wgpu::RayTracingPipelineStageDescriptor {
                    module: &shader,
                    entry_point: Some("closest_hit_main"),
                    compilation_options: Default::default(),
                },
            ],
            groups: &[
                // Raygen group
                wgpu::RayTracingShaderGroupDescriptor {
                    group_type: wgpu::RayTracingShaderGroupType::General,
                    general_stage_index: Some(0),
                    closest_hit_stage_index: None,
                    any_hit_stage_index: None,
                    intersection_stage_index: None,
                },
                // Miss group
                wgpu::RayTracingShaderGroupDescriptor {
                    group_type: wgpu::RayTracingShaderGroupType::General,
                    general_stage_index: Some(1),
                    closest_hit_stage_index: None,
                    any_hit_stage_index: None,
                    intersection_stage_index: None,
                },
                // Hit group (closest-hit + any-hit)
                wgpu::RayTracingShaderGroupDescriptor {
                    group_type: wgpu::RayTracingShaderGroupType::TrianglesHitGroup,
                    general_stage_index: None,
                    closest_hit_stage_index: Some(3),
                    any_hit_stage_index: Some(2),
                    intersection_stage_index: None,
                },
            ],
            max_pipeline_ray_recursion_depth: 1,
            cache: None,
        });

    // If we reach here, pipeline creation succeeded on the GPU.
    // Dispatching trace_rays requires command encoder integration (future work).
}
