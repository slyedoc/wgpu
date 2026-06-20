//! Code for ray tracing pipelines

use crate::back::spv::{
    Block, BlockContext, Instruction, LocalType, LookupRaytracingFunction, Writer, WriterFlags,
};

impl Writer {
    fn write_trace_ray(
        &mut self,
        ir_module: &crate::Module,
        payload: crate::Handle<crate::GlobalVariable>,
    ) -> spirv::Word {
        if let Some(&word) = self
            .ray_tracing_functions
            .get(&LookupRaytracingFunction::TraceRay { payload })
        {
            return word;
        }

        let acceleration_structure_type_id =
            self.get_localtype_id(LocalType::AccelerationStructure);

        let ray_desc_type_id = self.get_handle_type_id(
            ir_module
                .special_types
                .ray_desc
                .expect("ray desc should be set if `traceRays` is called"),
        );

        let u32_type_id = self.get_u32_type_id();
        let (func_id, mut function, arg_ids) = self.write_function_signature(
            &[
                acceleration_structure_type_id,
                ray_desc_type_id,
                // sbt_record_offset, sbt_record_stride, miss_index (all u32). Passed
                // per-call so one cached helper serves every traceRay for this payload.
                u32_type_id,
                u32_type_id,
                u32_type_id,
            ],
            self.void_type,
        );

        let acceleration_structure_id = arg_ids[0];
        let desc_id = arg_ids[1];
        let sbt_record_offset_id = arg_ids[2];
        let sbt_record_stride_id = arg_ids[3];
        let miss_index_id = arg_ids[4];
        let payload_id = self.global_variables[payload].access_id;

        let label_id = self.id_gen.next();
        let mut block = Block::new(label_id);

        let super::ExtractedRayDesc {
            ray_flags_id,
            cull_mask_id,
            tmin_id,
            tmax_id,
            ray_origin_id,
            ray_dir_id,
            valid_id,
        } = self.write_extract_ray_desc(&mut block, desc_id, self.trace_ray_argument_validation);

        let merge_label_id = self.id_gen.next();
        let merge_block = Block::new(merge_label_id);

        // NOTE: this block will be unreachable if trace ray validation is disabled.
        let invalid_label_id = self.id_gen.next();
        let mut invalid_block = Block::new(invalid_label_id);

        let valid_label_id = self.id_gen.next();
        let mut valid_block = Block::new(valid_label_id);

        match valid_id {
            Some(all_valid_id) => {
                block.body.push(Instruction::selection_merge(
                    merge_label_id,
                    spirv::SelectionControl::NONE,
                ));
                function.consume(
                    block,
                    Instruction::branch_conditional(all_valid_id, valid_label_id, invalid_label_id),
                );
            }
            None => {
                function.consume(block, Instruction::branch(valid_label_id));
            }
        }

        valid_block.body.push(Instruction::trace_ray(
            acceleration_structure_id,
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

        function.consume(valid_block, Instruction::branch(merge_label_id));

        if self.flags.contains(WriterFlags::PRINT_ON_TRACE_RAYS_FAIL) {
            self.write_debug_printf(
                &mut invalid_block,
                "Naga ignored invalid arguments to traceRay with flags: %u t_min: %f t_max: %f origin: %v4f dir: %v4f",
                &[
                    ray_flags_id,
                    tmin_id,
                    tmax_id,
                    ray_origin_id,
                    ray_dir_id,
                ],
            );
        }

        function.consume(invalid_block, Instruction::branch(merge_label_id));

        function.consume(merge_block, Instruction::return_void());

        function.to_words(&mut self.logical_layout.function_definitions);

        self.ray_tracing_functions
            .insert(LookupRaytracingFunction::TraceRay { payload }, func_id);

        func_id
    }
}

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
                // Checked for when validating the module in `validate_block_impl`.
                let crate::Expression::GlobalVariable(payload) =
                    self.ir_function.expressions[payload]
                else {
                    unreachable!()
                };

                let desc_id = self.cached[descriptor];
                let acc_struct_id = self.get_handle_id(acceleration_structure);
                // Default any omitted SBT operand to 0 (the plain `traceRay` form).
                let zero = self.writer.get_constant_scalar(crate::Literal::U32(0));
                let sbt_record_offset_id = sbt_record_offset.map_or(zero, |h| self.cached[h]);
                let sbt_record_stride_id = sbt_record_stride.map_or(zero, |h| self.cached[h]);
                let miss_index_id = miss_index.map_or(zero, |h| self.cached[h]);

                let func = self.writer.write_trace_ray(self.ir_module, payload);

                let func_id = self.gen_id();
                block.body.push(Instruction::function_call(
                    self.writer.void_type,
                    func_id,
                    func,
                    &[
                        acc_struct_id,
                        desc_id,
                        sbt_record_offset_id,
                        sbt_record_stride_id,
                        miss_index_id,
                    ],
                ));
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
