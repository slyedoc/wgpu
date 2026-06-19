use crate::*;

/// Handle to a binding group layout.
///
/// A `BindGroupLayout` is a handle to the GPU-side layout of a binding group. It can be used to
/// create a [`BindGroupDescriptor`] object, which in turn can be used to create a [`BindGroup`]
/// object with [`Device::create_bind_group`]. A series of `BindGroupLayout`s can also be used to
/// create a [`PipelineLayoutDescriptor`], which can be used to create a [`PipelineLayout`].
///
/// It can be created with [`Device::create_bind_group_layout`].
///
/// Corresponds to [WebGPU `GPUBindGroupLayout`](
/// https://gpuweb.github.io/gpuweb/#gpubindgrouplayout).
#[derive(Debug, Clone)]
pub struct BindGroupLayout {
    pub(crate) inner: dispatch::DispatchBindGroupLayout,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(BindGroupLayout: Send, Sync);

crate::cmp::impl_eq_ord_hash_proxy!(BindGroupLayout => .inner);

impl BindGroupLayout {
    /// Returns the inner hal BindGroupLayout using a callback. The hal layout will be `None` if the
    /// backend type argument does not match with this wgpu BindGroupLayout.
    ///
    /// # Safety
    ///
    /// - The raw handle obtained from the hal BindGroupLayout must not be manually destroyed
    #[cfg(wgpu_core)]
    pub unsafe fn as_hal<A: hal::Api>(
        &self,
    ) -> Option<impl core::ops::Deref<Target = A::BindGroupLayout>> {
        let layout = self.inner.as_core_opt()?;
        unsafe { layout.context.bind_group_layout_as_hal::<A>(layout) }
    }

    #[cfg(custom)]
    /// Returns custom implementation of BindGroupLayout (if custom backend and is internally T)
    pub fn as_custom<T: custom::BindGroupLayoutInterface>(&self) -> Option<&T> {
        self.inner.as_custom()
    }
}

/// Describes a [`BindGroupLayout`].
///
/// For use with [`Device::create_bind_group_layout`].
///
/// Corresponds to [WebGPU `GPUBindGroupLayoutDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpubindgrouplayoutdescriptor).
#[derive(Clone, Debug)]
pub struct BindGroupLayoutDescriptor<'a> {
    /// Debug label of the bind group layout. This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,

    /// Array of entries in this BindGroupLayout
    pub entries: &'a [BindGroupLayoutEntry],
}
static_assertions::assert_impl_all!(BindGroupLayoutDescriptor<'_>: Send, Sync);
