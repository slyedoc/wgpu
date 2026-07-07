//! Code for ray tracing pipelines

use crate::back::spv::{selection::Selection, Block, BlockContext, Instruction, WriterFlags};

impl BlockContext<'_> {
    pub(in super::super) fn write_ray_tracing_pipeline_function(
        &mut self,
        function: &crate::RayPipelineFunction,
        block: &mut Block,
    ) {
        match *function {
            crate::RayPipelineFunction::TraceRay {
                acceleration_structure,
                descriptor,
                payload,
                sbt_record_offset,
                sbt_record_stride,
                miss_index,
            } => {
                let payload_id = self.payload_access_id(payload);
                let desc_id = self.cached[descriptor];
                let acc_struct_id = self.get_handle_id(acceleration_structure);
                // Default any omitted SBT operand to 0 (the plain `traceRay` form).
                let zero = self.writer.get_constant_scalar(crate::Literal::U32(0));
                let sbt_record_offset_id = sbt_record_offset.map_or(zero, |h| self.cached[h]);
                let sbt_record_stride_id = sbt_record_stride.map_or(zero, |h| self.cached[h]);
                let miss_index_id = miss_index.map_or(zero, |h| self.cached[h]);

                // Emitted INLINE at every call site (exactly like `hitObjectTraceRay`
                // below), never via a shared per-payload helper function: an
                // `OpTraceRayKHR` inside a callee reached from two live call sites is
                // a shape no other toolchain produces, and NVIDIA's RT compiler
                // miscompiles it (two-site closest-hit = black scene or hang; one
                // site in a loop was fine — the trivial single-call inline).
                let super::ExtractedRayDesc {
                    ray_flags_id,
                    cull_mask_id,
                    tmin_id,
                    tmax_id,
                    ray_origin_id,
                    ray_dir_id,
                    valid_id,
                } = self.writer.write_extract_ray_desc(
                    block,
                    desc_id,
                    self.writer.trace_ray_argument_validation,
                );

                let trace = Instruction::trace_ray(
                    acc_struct_id,
                    ray_flags_id,
                    cull_mask_id,
                    sbt_record_offset_id,
                    sbt_record_stride_id,
                    miss_index_id,
                    ray_origin_id,
                    tmin_id,
                    ray_dir_id,
                    tmax_id,
                    payload_id,
                );
                match valid_id {
                    Some(all_valid_id) => {
                        if self.writer.flags.contains(WriterFlags::PRINT_ON_TRACE_RAYS_FAIL) {
                            let bool_type_id = self.writer.get_bool_type_id();
                            let not_valid_id = self.gen_id();
                            block.body.push(Instruction::unary(
                                spirv::Op::LogicalNot,
                                bool_type_id,
                                not_valid_id,
                                all_valid_id,
                            ));
                            let mut printf = Selection::start(block, ());
                            printf.if_true(self, not_valid_id, ());
                            self.writer.write_debug_printf(
                                printf.block(),
                                "Naga ignored invalid arguments to traceRay with flags: %u t_min: %f t_max: %f origin: %v4f dir: %v4f",
                                &[ray_flags_id, tmin_id, tmax_id, ray_origin_id, ray_dir_id],
                            );
                            printf.finish(self, ());
                        }
                        let mut trace_sel = Selection::start(block, ());
                        trace_sel.if_true(self, all_valid_id, ());
                        trace_sel.block().body.push(trace);
                        trace_sel.finish(self, ());
                    }
                    None => block.body.push(trace),
                }
            }
            crate::RayPipelineFunction::HitObjectTraceRay {
                hit_object,
                acceleration_structure,
                descriptor,
                payload,
                sbt_record_offset,
                sbt_record_stride,
                miss_index,
            } => {
                self.writer.require_shader_invocation_reorder();
                let hit_object_id = self.hit_object_access_id(hit_object);
                let payload_id = self.payload_access_id(payload);
                let acc_struct_id = self.get_handle_id(acceleration_structure);
                let desc_id = self.cached[descriptor];

                // SER records the hit inline (no validation branch — the
                // hitObject path is opt-in and the SBT is built by the host).
                let super::ExtractedRayDesc {
                    ray_flags_id,
                    cull_mask_id,
                    tmin_id,
                    tmax_id,
                    ray_origin_id,
                    ray_dir_id,
                    valid_id: _,
                } = self.writer.write_extract_ray_desc(block, desc_id, false);
                let zero = self.writer.get_constant_scalar(crate::Literal::U32(0));
                let sbt_record_offset_id = sbt_record_offset.map_or(zero, |h| self.cached[h]);
                let sbt_record_stride_id = sbt_record_stride.map_or(zero, |h| self.cached[h]);
                let miss_index_id = miss_index.map_or(zero, |h| self.cached[h]);
                block.body.push(Instruction::hit_object_trace_ray(
                    hit_object_id,
                    acc_struct_id,
                    ray_flags_id,
                    cull_mask_id,
                    sbt_record_offset_id,
                    sbt_record_stride_id,
                    miss_index_id,
                    ray_origin_id,
                    tmin_id,
                    ray_dir_id,
                    tmax_id,
                    payload_id,
                ));
            }
            crate::RayPipelineFunction::ReorderThread {
                hit_object,
                hint,
                hint_bits,
            } => {
                self.writer.require_shader_invocation_reorder();
                let hit_object_id = self.hit_object_access_id(hit_object);
                let hint_id = hint.map(|h| self.cached[h]);
                let hint_bits_id = hint_bits.map(|h| self.cached[h]);
                block.body.push(Instruction::reorder_thread_with_hit_object(
                    hit_object_id,
                    hint_id,
                    hint_bits_id,
                ));
            }
            crate::RayPipelineFunction::HitObjectExecuteShader {
                hit_object,
                payload,
            } => {
                self.writer.require_shader_invocation_reorder();
                let hit_object_id = self.hit_object_access_id(hit_object);
                let payload_id = self.payload_access_id(payload);
                block.body.push(Instruction::hit_object_execute_shader(
                    hit_object_id,
                    payload_id,
                ));
            }
        }
    }

    /// Resolve a `hit_object` / `payload` pointer expression to its SPIR-V id.
    /// The payload is a global (`ray_payload`); the hit object is typically a
    /// function-local `var`, so handle both — a global resolves to its
    /// `access_id`, anything else to its cached pointer id.
    fn payload_access_id(&self, expr: crate::Handle<crate::Expression>) -> spirv::Word {
        self.ser_pointer_id(expr)
    }

    /// Emit an `OpHitObjectGet*NV` / `OpHitObjectIs*NV` query (an
    /// [`Expression::HitObjectGet`](crate::Expression::HitObjectGet)), returning
    /// the result id. All map to a single-operand instruction over the hit-object
    /// pointer; `result_type_id` is the type the typifier assigned.
    pub(in super::super) fn write_hit_object_get(
        &mut self,
        hit_object: crate::Handle<crate::Expression>,
        query: crate::HitObjectQuery,
        result_type_id: spirv::Word,
        block: &mut Block,
    ) -> spirv::Word {
        use crate::HitObjectQuery as Q;
        use spirv::Op;

        self.writer.require_shader_invocation_reorder();
        let op = match query {
            Q::IsHit => Op::HitObjectIsHitNV,
            Q::IsMiss => Op::HitObjectIsMissNV,
            Q::IsEmpty => Op::HitObjectIsEmptyNV,
            Q::SbtRecordIndex => Op::HitObjectGetShaderBindingTableRecordIndexNV,
            Q::InstanceId => Op::HitObjectGetInstanceIdNV,
            Q::InstanceCustomIndex => Op::HitObjectGetInstanceCustomIndexNV,
            Q::PrimitiveIndex => Op::HitObjectGetPrimitiveIndexNV,
            Q::GeometryIndex => Op::HitObjectGetGeometryIndexNV,
            Q::ClusterId => {
                // The cluster-id query additionally needs the cluster-AS feature.
                self.writer
                    .require_any(
                        "HitObjectGetClusterId",
                        &[spirv::Capability::RayTracingClusterAccelerationStructureNV],
                    )
                    .ok();
                self.writer
                    .use_extension("SPV_NV_cluster_acceleration_structure");
                Op::HitObjectGetClusterIdNV
            }
            Q::HitKind => Op::HitObjectGetHitKindNV,
            Q::RayTMin => Op::HitObjectGetRayTMinNV,
            Q::RayTMax => Op::HitObjectGetRayTMaxNV,
            Q::WorldRayOrigin => Op::HitObjectGetWorldRayOriginNV,
            Q::WorldRayDirection => Op::HitObjectGetWorldRayDirectionNV,
            Q::ObjectRayOrigin => Op::HitObjectGetObjectRayOriginNV,
            Q::ObjectRayDirection => Op::HitObjectGetObjectRayDirectionNV,
        };
        let hit_object_id = self.hit_object_access_id(hit_object);
        let id = self.gen_id();
        block.body.push(Instruction::hit_object_get(
            op,
            result_type_id,
            id,
            hit_object_id,
        ));
        id
    }

    fn hit_object_access_id(&self, expr: crate::Handle<crate::Expression>) -> spirv::Word {
        self.ser_pointer_id(expr)
    }

    fn ser_pointer_id(&self, expr: crate::Handle<crate::Expression>) -> spirv::Word {
        // hit_object / payload are pointer expressions. Globals and locals are
        // pre-emitted (not in `cached`), so resolve them the way the general
        // pointer path does; fall back to `cached` for anything value-like.
        match self.ir_function.expressions[expr] {
            crate::Expression::GlobalVariable(gv) => self.writer.global_variables[gv].access_id,
            crate::Expression::LocalVariable(var) => self.function.variables[&var].id,
            crate::Expression::FunctionArgument(index) => self.function.parameter_id(index),
            _ => self.cached[expr],
        }
    }
}
