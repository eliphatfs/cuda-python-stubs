from typing import *
import cuda as cuda
import cuda.bindings.driver
import enum
from _typeshed import Incomplete
from typing import Any, Callable, ClassVar

CUDART_VERSION: int
CUDA_EGL_MAX_PLANES: int
CUDA_IPC_HANDLE_SIZE: int

def __reduce_cython__(self):
    """
    VdpOutputSurface.__reduce_cython__(self)
    """


def __setstate_cython__(self, __pyx_state):
    """
    VdpOutputSurface.__setstate_cython__(self, __pyx_state)
    """

__test__: dict
cudaArrayColorAttachment: int
cudaArrayCubemap: int
cudaArrayDefault: int
cudaArrayDeferredMapping: int

def cudaArrayGetInfo(array):
    """
    cudaArrayGetInfo(array)
     Gets info about the specified cudaArray.

        Returns in `*desc`, `*extent` and `*flags` respectively, the type,
        shape and flags of `array`.

        Any of `*desc`, `*extent` and `*flags` may be specified as NULL.

        Parameters
        ----------
        array : :py:obj:`~.cudaArray_t`
            The :py:obj:`~.cudaArray` to get info for

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`
        desc : :py:obj:`~.cudaChannelFormatDesc`
            Returned array type
        extent : :py:obj:`~.cudaExtent`
            Returned array shape. 2D arrays will have depth of zero
        flags : unsigned int
            Returned array flags

        See Also
        --------
        :py:obj:`~.cuArrayGetDescriptor`, :py:obj:`~.cuArray3DGetDescriptor`
    """


def cudaArrayGetMemoryRequirements(array, device):
    """
    cudaArrayGetMemoryRequirements(array, int device)
     Returns the memory requirements of a CUDA array.

        Returns the memory requirements of a CUDA array in `memoryRequirements`
        If the CUDA array is not allocated with flag
        :py:obj:`~.cudaArrayDeferredMapping` :py:obj:`~.cudaErrorInvalidValue`
        will be returned.

        The returned value in :py:obj:`~.cudaArrayMemoryRequirements.size`
        represents the total size of the CUDA array. The returned value in
        :py:obj:`~.cudaArrayMemoryRequirements.alignment` represents the
        alignment necessary for mapping the CUDA array.

        Parameters
        ----------
        array : :py:obj:`~.cudaArray_t`
            CUDA array to get the memory requirements of
        device : int
            Device to get the memory requirements for

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess` :py:obj:`~.cudaErrorInvalidValue`
        memoryRequirements : :py:obj:`~.cudaArrayMemoryRequirements`
            Pointer to :py:obj:`~.cudaArrayMemoryRequirements`

        See Also
        --------
        :py:obj:`~.cudaMipmappedArrayGetMemoryRequirements`
    """


def cudaArrayGetPlane(hArray, planeIdx):
    """
    cudaArrayGetPlane(hArray, unsigned int planeIdx)
     Gets a CUDA array plane from a CUDA array.

        Returns in `pPlaneArray` a CUDA array that represents a single format
        plane of the CUDA array `hArray`.

        If `planeIdx` is greater than the maximum number of planes in this
        array or if the array does not have a multi-planar format e.g:
        :py:obj:`~.cudaChannelFormatKindNV12`, then
        :py:obj:`~.cudaErrorInvalidValue` is returned.

        Note that if the `hArray` has format
        :py:obj:`~.cudaChannelFormatKindNV12`, then passing in 0 for `planeIdx`
        returns a CUDA array of the same size as `hArray` but with one 8-bit
        channel and :py:obj:`~.cudaChannelFormatKindUnsigned` as its format
        kind. If 1 is passed for `planeIdx`, then the returned CUDA array has
        half the height and width of `hArray` with two 8-bit channels and
        :py:obj:`~.cudaChannelFormatKindUnsigned` as its format kind.

        Parameters
        ----------
        hArray : :py:obj:`~.cudaArray_t`
            CUDA array
        planeIdx : unsigned int
            Plane index

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue` :py:obj:`~.cudaErrorInvalidResourceHandle`
        pPlaneArray : :py:obj:`~.cudaArray_t`
            Returned CUDA array referenced by the `planeIdx`

        See Also
        --------
        :py:obj:`~.cuArrayGetPlane`
    """


def cudaArrayGetSparseProperties(array):
    """
    cudaArrayGetSparseProperties(array)
     Returns the layout properties of a sparse CUDA array.

        Returns the layout properties of a sparse CUDA array in
        `sparseProperties`. If the CUDA array is not allocated with flag
        :py:obj:`~.cudaArraySparse` :py:obj:`~.cudaErrorInvalidValue` will be
        returned.

        If the returned value in :py:obj:`~.cudaArraySparseProperties.flags`
        contains :py:obj:`~.cudaArraySparsePropertiesSingleMipTail`, then
        :py:obj:`~.cudaArraySparseProperties.miptailSize` represents the total
        size of the array. Otherwise, it will be zero. Also, the returned value
        in :py:obj:`~.cudaArraySparseProperties.miptailFirstLevel` is always
        zero. Note that the `array` must have been allocated using
        :py:obj:`~.cudaMallocArray` or :py:obj:`~.cudaMalloc3DArray`. For CUDA
        arrays obtained using :py:obj:`~.cudaMipmappedArrayGetLevel`,
        :py:obj:`~.cudaErrorInvalidValue` will be returned. Instead,
        :py:obj:`~.cudaMipmappedArrayGetSparseProperties` must be used to
        obtain the sparse properties of the entire CUDA mipmapped array to
        which `array` belongs to.

        Parameters
        ----------
        array : :py:obj:`~.cudaArray_t`
            The CUDA array to get the sparse properties of

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess` :py:obj:`~.cudaErrorInvalidValue`
        sparseProperties : :py:obj:`~.cudaArraySparseProperties`
            Pointer to return the :py:obj:`~.cudaArraySparseProperties`

        See Also
        --------
        :py:obj:`~.cudaMipmappedArrayGetSparseProperties`, :py:obj:`~.cuMemMapArrayAsync`
    """

cudaArrayLayered: int
cudaArraySparse: int
cudaArraySparsePropertiesSingleMipTail: int
cudaArraySurfaceLoadStore: int
cudaArrayTextureGather: int

def cudaChooseDevice(prop: 'Optional[cudaDeviceProp]'):
    """
    cudaChooseDevice(cudaDeviceProp prop: Optional[cudaDeviceProp])
     Select compute-device which best matches criteria.

        Returns in `*device` the device which has properties that best match
        `*prop`.

        Parameters
        ----------
        prop : :py:obj:`~.cudaDeviceProp`
            Desired device properties

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`
        device : int
            Device with best match

        See Also
        --------
        :py:obj:`~.cudaGetDeviceCount`, :py:obj:`~.cudaGetDevice`, :py:obj:`~.cudaSetDevice`, :py:obj:`~.cudaGetDeviceProperties`, :py:obj:`~.cudaInitDevice`
    """

cudaCooperativeLaunchMultiDeviceNoPostSync: int
cudaCooperativeLaunchMultiDeviceNoPreSync: int
cudaCpuDeviceId: int

def cudaCreateChannelDesc(x, y, z, w, f: 'cudaChannelFormatKind'):
    """
    cudaCreateChannelDesc(int x, int y, int z, int w, f: cudaChannelFormatKind)
     Returns a channel descriptor using the specified format.

        Returns a channel descriptor with format `f` and number of bits of each
        component `x`, `y`, `z`, and `w`. The :py:obj:`~.cudaChannelFormatDesc`
        is defined as:

        **View CUDA Toolkit Documentation for a C++ code example**

        where :py:obj:`~.cudaChannelFormatKind` is one of
        :py:obj:`~.cudaChannelFormatKindSigned`,
        :py:obj:`~.cudaChannelFormatKindUnsigned`, or
        :py:obj:`~.cudaChannelFormatKindFloat`.

        Parameters
        ----------
        x : int
            X component
        y : int
            Y component
        z : int
            Z component
        w : int
            W component
        f : :py:obj:`~.cudaChannelFormatKind`
            Channel format

        Returns
        -------
        cudaError_t.cudaSuccess
            cudaError_t.cudaSuccess
        :py:obj:`~.cudaChannelFormatDesc`
            Channel descriptor with format `f`

        See Also
        --------
        cudaCreateChannelDesc (C++ API), :py:obj:`~.cudaGetChannelDesc`, :py:obj:`~.cudaCreateTextureObject`, :py:obj:`~.cudaCreateSurfaceObject`
    """


def cudaCreateSurfaceObject(pResDesc: 'Optional[cudaResourceDesc]'):
    """
    cudaCreateSurfaceObject(cudaResourceDesc pResDesc: Optional[cudaResourceDesc])
     Creates a surface object.

        Creates a surface object and returns it in `pSurfObject`. `pResDesc`
        describes the data to perform surface load/stores on.
        :py:obj:`~.cudaResourceDesc.resType` must be
        :py:obj:`~.cudaResourceTypeArray` and
        :py:obj:`~.cudaResourceDesc`::res::array::array must be set to a valid
        CUDA array handle.

        Surface objects are only supported on devices of compute capability 3.0
        or higher. Additionally, a surface object is an opaque value, and, as
        such, should only be accessed through CUDA API calls.

        Parameters
        ----------
        pResDesc : :py:obj:`~.cudaResourceDesc`
            Resource descriptor

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidChannelDescriptor`, :py:obj:`~.cudaErrorInvalidResourceHandle`
        pSurfObject : :py:obj:`~.cudaSurfaceObject_t`
            Surface object to create

        See Also
        --------
        :py:obj:`~.cudaDestroySurfaceObject`, :py:obj:`~.cuSurfObjectCreate`
    """


def cudaCreateTextureObject(pResDesc: 'Optional[cudaResourceDesc]', pTexDesc: 'Optional[cudaTextureDesc]', pResViewDesc: 'Optional[cudaResourceViewDesc]'):
    """
    cudaCreateTextureObject(cudaResourceDesc pResDesc: Optional[cudaResourceDesc], cudaTextureDesc pTexDesc: Optional[cudaTextureDesc], cudaResourceViewDesc pResViewDesc: Optional[cudaResourceViewDesc])
     Creates a texture object.

        Creates a texture object and returns it in `pTexObject`. `pResDesc`
        describes the data to texture from. `pTexDesc` describes how the data
        should be sampled. `pResViewDesc` is an optional argument that
        specifies an alternate format for the data described by `pResDesc`, and
        also describes the subresource region to restrict access to when
        texturing. `pResViewDesc` can only be specified if the type of resource
        is a CUDA array or a CUDA mipmapped array not in a block compressed
        format.

        Texture objects are only supported on devices of compute capability 3.0
        or higher. Additionally, a texture object is an opaque value, and, as
        such, should only be accessed through CUDA API calls.

        The :py:obj:`~.cudaResourceDesc` structure is defined as:

        **View CUDA Toolkit Documentation for a C++ code example**

        where:

        - :py:obj:`~.cudaResourceDesc.resType` specifies the type of resource
          to texture from. CUresourceType is defined as:

        - **View CUDA Toolkit Documentation for a C++ code example**

        If :py:obj:`~.cudaResourceDesc.resType` is set to
        :py:obj:`~.cudaResourceTypeArray`,
        :py:obj:`~.cudaResourceDesc`::res::array::array must be set to a valid
        CUDA array handle.

        If :py:obj:`~.cudaResourceDesc.resType` is set to
        :py:obj:`~.cudaResourceTypeMipmappedArray`,
        :py:obj:`~.cudaResourceDesc`::res::mipmap::mipmap must be set to a
        valid CUDA mipmapped array handle and
        :py:obj:`~.cudaTextureDesc.normalizedCoords` must be set to true.

        If :py:obj:`~.cudaResourceDesc.resType` is set to
        :py:obj:`~.cudaResourceTypeLinear`,
        :py:obj:`~.cudaResourceDesc`::res::linear::devPtr must be set to a
        valid device pointer, that is aligned to
        :py:obj:`~.cudaDeviceProp.textureAlignment`.
        :py:obj:`~.cudaResourceDesc`::res::linear::desc describes the format
        and the number of components per array element.
        :py:obj:`~.cudaResourceDesc`::res::linear::sizeInBytes specifies the
        size of the array in bytes. The total number of elements in the linear
        address range cannot exceed
        :py:obj:`~.cudaDeviceProp.maxTexture1DLinear`. The number of elements
        is computed as (sizeInBytes / sizeof(desc)).

        If :py:obj:`~.cudaResourceDesc.resType` is set to
        :py:obj:`~.cudaResourceTypePitch2D`,
        :py:obj:`~.cudaResourceDesc`::res::pitch2D::devPtr must be set to a
        valid device pointer, that is aligned to
        :py:obj:`~.cudaDeviceProp.textureAlignment`.
        :py:obj:`~.cudaResourceDesc`::res::pitch2D::desc describes the format
        and the number of components per array element.
        :py:obj:`~.cudaResourceDesc`::res::pitch2D::width and
        :py:obj:`~.cudaResourceDesc`::res::pitch2D::height specify the width
        and height of the array in elements, and cannot exceed
        :py:obj:`~.cudaDeviceProp.maxTexture2DLinear`[0] and
        :py:obj:`~.cudaDeviceProp.maxTexture2DLinear`[1] respectively.
        :py:obj:`~.cudaResourceDesc`::res::pitch2D::pitchInBytes specifies the
        pitch between two rows in bytes and has to be aligned to
        :py:obj:`~.cudaDeviceProp.texturePitchAlignment`. Pitch cannot exceed
        :py:obj:`~.cudaDeviceProp.maxTexture2DLinear`[2].

        The :py:obj:`~.cudaTextureDesc` struct is defined as

        **View CUDA Toolkit Documentation for a C++ code example**

        where

        - :py:obj:`~.cudaTextureDesc.addressMode` specifies the addressing mode
          for each dimension of the texture data.
          :py:obj:`~.cudaTextureAddressMode` is defined as:

        - **View CUDA Toolkit Documentation for a C++ code example**

        - This is ignored if :py:obj:`~.cudaResourceDesc.resType` is
          :py:obj:`~.cudaResourceTypeLinear`. Also, if
          :py:obj:`~.cudaTextureDesc.normalizedCoords` is set to zero,
          :py:obj:`~.cudaAddressModeWrap` and :py:obj:`~.cudaAddressModeMirror`
          won't be supported and will be switched to
          :py:obj:`~.cudaAddressModeClamp`.

        - :py:obj:`~.cudaTextureDesc.filterMode` specifies the filtering mode
          to be used when fetching from the texture.
          :py:obj:`~.cudaTextureFilterMode` is defined as:

        - **View CUDA Toolkit Documentation for a C++ code example**

        - This is ignored if :py:obj:`~.cudaResourceDesc.resType` is
          :py:obj:`~.cudaResourceTypeLinear`.

        - :py:obj:`~.cudaTextureDesc.readMode` specifies whether integer data
          should be converted to floating point or not.
          :py:obj:`~.cudaTextureReadMode` is defined as:

        - **View CUDA Toolkit Documentation for a C++ code example**

        - Note that this applies only to 8-bit and 16-bit integer formats.
          32-bit integer format would not be promoted, regardless of whether or
          not this :py:obj:`~.cudaTextureDesc.readMode` is set
          :py:obj:`~.cudaReadModeNormalizedFloat` is specified.

        - :py:obj:`~.cudaTextureDesc.sRGB` specifies whether sRGB to linear
          conversion should be performed during texture fetch.

        - :py:obj:`~.cudaTextureDesc.borderColor` specifies the float values of
          color. where: :py:obj:`~.cudaTextureDesc.borderColor`[0] contains
          value of 'R', :py:obj:`~.cudaTextureDesc.borderColor`[1] contains
          value of 'G', :py:obj:`~.cudaTextureDesc.borderColor`[2] contains
          value of 'B', :py:obj:`~.cudaTextureDesc.borderColor`[3] contains
          value of 'A' Note that application using integer border color values
          will need to <reinterpret_cast> these values to float. The values are
          set only when the addressing mode specified by
          :py:obj:`~.cudaTextureDesc.addressMode` is cudaAddressModeBorder.

        - :py:obj:`~.cudaTextureDesc.normalizedCoords` specifies whether the
          texture coordinates will be normalized or not.

        - :py:obj:`~.cudaTextureDesc.maxAnisotropy` specifies the maximum
          anistropy ratio to be used when doing anisotropic filtering. This
          value will be clamped to the range [1,16].

        - :py:obj:`~.cudaTextureDesc.mipmapFilterMode` specifies the filter
          mode when the calculated mipmap level lies between two defined mipmap
          levels.

        - :py:obj:`~.cudaTextureDesc.mipmapLevelBias` specifies the offset to
          be applied to the calculated mipmap level.

        - :py:obj:`~.cudaTextureDesc.minMipmapLevelClamp` specifies the lower
          end of the mipmap level range to clamp access to.

        - :py:obj:`~.cudaTextureDesc.maxMipmapLevelClamp` specifies the upper
          end of the mipmap level range to clamp access to.

        - :py:obj:`~.cudaTextureDesc.disableTrilinearOptimization` specifies
          whether the trilinear filtering optimizations will be disabled.

        - :py:obj:`~.cudaTextureDesc.seamlessCubemap` specifies whether
          seamless cube map filtering is enabled. This flag can only be
          specified if the underlying resource is a CUDA array or a CUDA
          mipmapped array that was created with the flag
          :py:obj:`~.cudaArrayCubemap`. When seamless cube map filtering is
          enabled, texture address modes specified by
          :py:obj:`~.cudaTextureDesc.addressMode` are ignored. Instead, if the
          :py:obj:`~.cudaTextureDesc.filterMode` is set to
          :py:obj:`~.cudaFilterModePoint` the address mode
          :py:obj:`~.cudaAddressModeClamp` will be applied for all dimensions.
          If the :py:obj:`~.cudaTextureDesc.filterMode` is set to
          :py:obj:`~.cudaFilterModeLinear` seamless cube map filtering will be
          performed when sampling along the cube face borders.

        The :py:obj:`~.cudaResourceViewDesc` struct is defined as

        **View CUDA Toolkit Documentation for a C++ code example**

        where:

        - :py:obj:`~.cudaResourceViewDesc.format` specifies how the data
          contained in the CUDA array or CUDA mipmapped array should be
          interpreted. Note that this can incur a change in size of the texture
          data. If the resource view format is a block compressed format, then
          the underlying CUDA array or CUDA mipmapped array has to have a
          32-bit unsigned integer format with 2 or 4 channels, depending on the
          block compressed format. For ex., BC1 and BC4 require the underlying
          CUDA array to have a 32-bit unsigned int with 2 channels. The other
          BC formats require the underlying resource to have the same 32-bit
          unsigned int format but with 4 channels.

        - :py:obj:`~.cudaResourceViewDesc.width` specifies the new width of the
          texture data. If the resource view format is a block compressed
          format, this value has to be 4 times the original width of the
          resource. For non block compressed formats, this value has to be
          equal to that of the original resource.

        - :py:obj:`~.cudaResourceViewDesc.height` specifies the new height of
          the texture data. If the resource view format is a block compressed
          format, this value has to be 4 times the original height of the
          resource. For non block compressed formats, this value has to be
          equal to that of the original resource.

        - :py:obj:`~.cudaResourceViewDesc.depth` specifies the new depth of the
          texture data. This value has to be equal to that of the original
          resource.

        - :py:obj:`~.cudaResourceViewDesc.firstMipmapLevel` specifies the most
          detailed mipmap level. This will be the new mipmap level zero. For
          non-mipmapped resources, this value has to be
          zero.:py:obj:`~.cudaTextureDesc.minMipmapLevelClamp` and
          :py:obj:`~.cudaTextureDesc.maxMipmapLevelClamp` will be relative to
          this value. For ex., if the firstMipmapLevel is set to 2, and a
          minMipmapLevelClamp of 1.2 is specified, then the actual minimum
          mipmap level clamp will be 3.2.

        - :py:obj:`~.cudaResourceViewDesc.lastMipmapLevel` specifies the least
          detailed mipmap level. For non-mipmapped resources, this value has to
          be zero.

        - :py:obj:`~.cudaResourceViewDesc.firstLayer` specifies the first layer
          index for layered textures. This will be the new layer zero. For non-
          layered resources, this value has to be zero.

        - :py:obj:`~.cudaResourceViewDesc.lastLayer` specifies the last layer
          index for layered textures. For non-layered resources, this value has
          to be zero.

        Parameters
        ----------
        pResDesc : :py:obj:`~.cudaResourceDesc`
            Resource descriptor
        pTexDesc : :py:obj:`~.cudaTextureDesc`
            Texture descriptor
        pResViewDesc : :py:obj:`~.cudaResourceViewDesc`
            Resource view descriptor

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`
        pTexObject : :py:obj:`~.cudaTextureObject_t`
            Texture object to create

        See Also
        --------
        :py:obj:`~.cudaDestroyTextureObject`, :py:obj:`~.cuTexObjectCreate`
    """


def cudaCtxResetPersistingL2Cache():
    """
    cudaCtxResetPersistingL2Cache()
     Resets all persisting lines in cache to normal status.

        Resets all persisting lines in cache to normal status. Takes effect on
        function return.

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`,

        See Also
        --------
        :py:obj:`~.cudaAccessPolicyWindow`
    """


def cudaDestroyExternalMemory(extMem):
    """
    cudaDestroyExternalMemory(extMem)
     Destroys an external memory object.

        Destroys the specified external memory object. Any existing buffers and
        CUDA mipmapped arrays mapped onto this object must no longer be used
        and must be explicitly freed using :py:obj:`~.cudaFree` and
        :py:obj:`~.cudaFreeMipmappedArray` respectively.

        Parameters
        ----------
        extMem : :py:obj:`~.cudaExternalMemory_t`
            External memory object to be destroyed

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidResourceHandle`

        See Also
        --------
        :py:obj:`~.cudaImportExternalMemory`, :py:obj:`~.cudaExternalMemoryGetMappedBuffer`, :py:obj:`~.cudaExternalMemoryGetMappedMipmappedArray`
    """


def cudaDestroyExternalSemaphore(extSem):
    """
    cudaDestroyExternalSemaphore(extSem)
     Destroys an external semaphore.

        Destroys an external semaphore object and releases any references to
        the underlying resource. Any outstanding signals or waits must have
        completed before the semaphore is destroyed.

        Parameters
        ----------
        extSem : :py:obj:`~.cudaExternalSemaphore_t`
            External semaphore to be destroyed

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidResourceHandle`

        See Also
        --------
        :py:obj:`~.cudaImportExternalSemaphore`, :py:obj:`~.cudaSignalExternalSemaphoresAsync`, :py:obj:`~.cudaWaitExternalSemaphoresAsync`
    """


def cudaDestroySurfaceObject(surfObject):
    """
    cudaDestroySurfaceObject(surfObject)
     Destroys a surface object.

        Destroys the surface object specified by `surfObject`.

        Parameters
        ----------
        surfObject : :py:obj:`~.cudaSurfaceObject_t`
            Surface object to destroy

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`

        See Also
        --------
        :py:obj:`~.cudaCreateSurfaceObject`, :py:obj:`~.cuSurfObjectDestroy`
    """


def cudaDestroyTextureObject(texObject):
    """
    cudaDestroyTextureObject(texObject)
     Destroys a texture object.

        Destroys the texture object specified by `texObject`.

        Parameters
        ----------
        texObject : :py:obj:`~.cudaTextureObject_t`
            Texture object to destroy

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`

        See Also
        --------
        :py:obj:`~.cudaCreateTextureObject`, :py:obj:`~.cuTexObjectDestroy`
    """

cudaDeviceBlockingSync: int

def cudaDeviceCanAccessPeer(device, peerDevice):
    """
    cudaDeviceCanAccessPeer(int device, int peerDevice)
     Queries if a device may directly access a peer device's memory.

        Returns in `*canAccessPeer` a value of 1 if device `device` is capable
        of directly accessing memory from `peerDevice` and 0 otherwise. If
        direct access of `peerDevice` from `device` is possible, then access
        may be enabled by calling :py:obj:`~.cudaDeviceEnablePeerAccess()`.

        Parameters
        ----------
        device : int
            Device from which allocations on `peerDevice` are to be directly
            accessed.
        peerDevice : int
            Device on which the allocations to be directly accessed by `device`
            reside.

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidDevice`
        canAccessPeer : int
            Returned access capability

        See Also
        --------
        :py:obj:`~.cudaDeviceEnablePeerAccess`, :py:obj:`~.cudaDeviceDisablePeerAccess`, :py:obj:`~.cuDeviceCanAccessPeer`
    """


def cudaDeviceDisablePeerAccess(peerDevice):
    """
    cudaDeviceDisablePeerAccess(int peerDevice)
     Disables direct access to memory allocations on a peer device.

        Returns :py:obj:`~.cudaErrorPeerAccessNotEnabled` if direct access to
        memory on `peerDevice` has not yet been enabled from the current
        device.

        Parameters
        ----------
        peerDevice : int
            Peer device to disable direct access to

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorPeerAccessNotEnabled`, :py:obj:`~.cudaErrorInvalidDevice`

        See Also
        --------
        :py:obj:`~.cudaDeviceCanAccessPeer`, :py:obj:`~.cudaDeviceEnablePeerAccess`, :py:obj:`~.cuCtxDisablePeerAccess`
    """


def cudaDeviceEnablePeerAccess(peerDevice, flags):
    """
    cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags)
     Enables direct access to memory allocations on a peer device.

        On success, all allocations from `peerDevice` will immediately be
        accessible by the current device. They will remain accessible until
        access is explicitly disabled using
        :py:obj:`~.cudaDeviceDisablePeerAccess()` or either device is reset
        using :py:obj:`~.cudaDeviceReset()`.

        Note that access granted by this call is unidirectional and that in
        order to access memory on the current device from `peerDevice`, a
        separate symmetric call to :py:obj:`~.cudaDeviceEnablePeerAccess()` is
        required.

        Note that there are both device-wide and system-wide limitations per
        system configuration, as noted in the CUDA Programming Guide under the
        section "Peer-to-Peer Memory Access".

        Returns :py:obj:`~.cudaErrorInvalidDevice` if
        :py:obj:`~.cudaDeviceCanAccessPeer()` indicates that the current device
        cannot directly access memory from `peerDevice`.

        Returns :py:obj:`~.cudaErrorPeerAccessAlreadyEnabled` if direct access
        of `peerDevice` from the current device has already been enabled.

        Returns :py:obj:`~.cudaErrorInvalidValue` if `flags` is not 0.

        Parameters
        ----------
        peerDevice : int
            Peer device to enable direct access to from the current device
        flags : unsigned int
            Reserved for future use and must be set to 0

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidDevice`, :py:obj:`~.cudaErrorPeerAccessAlreadyEnabled`, :py:obj:`~.cudaErrorInvalidValue`

        See Also
        --------
        :py:obj:`~.cudaDeviceCanAccessPeer`, :py:obj:`~.cudaDeviceDisablePeerAccess`, :py:obj:`~.cuCtxEnablePeerAccess`
    """


def cudaDeviceFlushGPUDirectRDMAWrites(target: 'cudaFlushGPUDirectRDMAWritesTarget', scope: 'cudaFlushGPUDirectRDMAWritesScope'):
    """
    cudaDeviceFlushGPUDirectRDMAWrites(target: cudaFlushGPUDirectRDMAWritesTarget, scope: cudaFlushGPUDirectRDMAWritesScope)
     Blocks until remote writes are visible to the specified scope.

        Blocks until remote writes to the target context via mappings created
        through GPUDirect RDMA APIs, like nvidia_p2p_get_pages (see
        https://docs.nvidia.com/cuda/gpudirect-rdma for more information), are
        visible to the specified scope.

        If the scope equals or lies within the scope indicated by
        :py:obj:`~.cudaDevAttrGPUDirectRDMAWritesOrdering`, the call will be a
        no-op and can be safely omitted for performance. This can be determined
        by comparing the numerical values between the two enums, with smaller
        scopes having smaller values.

        Users may query support for this API via
        :py:obj:`~.cudaDevAttrGPUDirectRDMAFlushWritesOptions`.

        Parameters
        ----------
        target : :py:obj:`~.cudaFlushGPUDirectRDMAWritesTarget`
            The target of the operation, see cudaFlushGPUDirectRDMAWritesTarget
        scope : :py:obj:`~.cudaFlushGPUDirectRDMAWritesScope`
            The scope of the operation, see cudaFlushGPUDirectRDMAWritesScope

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorNotSupported`,

        See Also
        --------
        :py:obj:`~.cuFlushGPUDirectRDMAWrites`
    """


def cudaDeviceGetAttribute(attr: 'cudaDeviceAttr', device):
    """
    cudaDeviceGetAttribute(attr: cudaDeviceAttr, int device)
     Returns information about the device.

        Returns in `*value` the integer value of the attribute `attr` on device
        `device`. The supported attributes are:

        - :py:obj:`~.cudaDevAttrMaxThreadsPerBlock`: Maximum number of threads
          per block

        - :py:obj:`~.cudaDevAttrMaxBlockDimX`: Maximum x-dimension of a block

        - :py:obj:`~.cudaDevAttrMaxBlockDimY`: Maximum y-dimension of a block

        - :py:obj:`~.cudaDevAttrMaxBlockDimZ`: Maximum z-dimension of a block

        - :py:obj:`~.cudaDevAttrMaxGridDimX`: Maximum x-dimension of a grid

        - :py:obj:`~.cudaDevAttrMaxGridDimY`: Maximum y-dimension of a grid

        - :py:obj:`~.cudaDevAttrMaxGridDimZ`: Maximum z-dimension of a grid

        - :py:obj:`~.cudaDevAttrMaxSharedMemoryPerBlock`: Maximum amount of
          shared memory available to a thread block in bytes

        - :py:obj:`~.cudaDevAttrTotalConstantMemory`: Memory available on
          device for constant variables in a CUDA C kernel in bytes

        - :py:obj:`~.cudaDevAttrWarpSize`: Warp size in threads

        - :py:obj:`~.cudaDevAttrMaxPitch`: Maximum pitch in bytes allowed by
          the memory copy functions that involve memory regions allocated
          through :py:obj:`~.cudaMallocPitch()`

        - :py:obj:`~.cudaDevAttrMaxTexture1DWidth`: Maximum 1D texture width

        - :py:obj:`~.cudaDevAttrMaxTexture1DLinearWidth`: Maximum width for a
          1D texture bound to linear memory

        - :py:obj:`~.cudaDevAttrMaxTexture1DMipmappedWidth`: Maximum mipmapped
          1D texture width

        - :py:obj:`~.cudaDevAttrMaxTexture2DWidth`: Maximum 2D texture width

        - :py:obj:`~.cudaDevAttrMaxTexture2DHeight`: Maximum 2D texture height

        - :py:obj:`~.cudaDevAttrMaxTexture2DLinearWidth`: Maximum width for a
          2D texture bound to linear memory

        - :py:obj:`~.cudaDevAttrMaxTexture2DLinearHeight`: Maximum height for a
          2D texture bound to linear memory

        - :py:obj:`~.cudaDevAttrMaxTexture2DLinearPitch`: Maximum pitch in
          bytes for a 2D texture bound to linear memory

        - :py:obj:`~.cudaDevAttrMaxTexture2DMipmappedWidth`: Maximum mipmapped
          2D texture width

        - :py:obj:`~.cudaDevAttrMaxTexture2DMipmappedHeight`: Maximum mipmapped
          2D texture height

        - :py:obj:`~.cudaDevAttrMaxTexture3DWidth`: Maximum 3D texture width

        - :py:obj:`~.cudaDevAttrMaxTexture3DHeight`: Maximum 3D texture height

        - :py:obj:`~.cudaDevAttrMaxTexture3DDepth`: Maximum 3D texture depth

        - :py:obj:`~.cudaDevAttrMaxTexture3DWidthAlt`: Alternate maximum 3D
          texture width, 0 if no alternate maximum 3D texture size is supported

        - :py:obj:`~.cudaDevAttrMaxTexture3DHeightAlt`: Alternate maximum 3D
          texture height, 0 if no alternate maximum 3D texture size is
          supported

        - :py:obj:`~.cudaDevAttrMaxTexture3DDepthAlt`: Alternate maximum 3D
          texture depth, 0 if no alternate maximum 3D texture size is supported

        - :py:obj:`~.cudaDevAttrMaxTextureCubemapWidth`: Maximum cubemap
          texture width or height

        - :py:obj:`~.cudaDevAttrMaxTexture1DLayeredWidth`: Maximum 1D layered
          texture width

        - :py:obj:`~.cudaDevAttrMaxTexture1DLayeredLayers`: Maximum layers in a
          1D layered texture

        - :py:obj:`~.cudaDevAttrMaxTexture2DLayeredWidth`: Maximum 2D layered
          texture width

        - :py:obj:`~.cudaDevAttrMaxTexture2DLayeredHeight`: Maximum 2D layered
          texture height

        - :py:obj:`~.cudaDevAttrMaxTexture2DLayeredLayers`: Maximum layers in a
          2D layered texture

        - :py:obj:`~.cudaDevAttrMaxTextureCubemapLayeredWidth`: Maximum cubemap
          layered texture width or height

        - :py:obj:`~.cudaDevAttrMaxTextureCubemapLayeredLayers`: Maximum layers
          in a cubemap layered texture

        - :py:obj:`~.cudaDevAttrMaxSurface1DWidth`: Maximum 1D surface width

        - :py:obj:`~.cudaDevAttrMaxSurface2DWidth`: Maximum 2D surface width

        - :py:obj:`~.cudaDevAttrMaxSurface2DHeight`: Maximum 2D surface height

        - :py:obj:`~.cudaDevAttrMaxSurface3DWidth`: Maximum 3D surface width

        - :py:obj:`~.cudaDevAttrMaxSurface3DHeight`: Maximum 3D surface height

        - :py:obj:`~.cudaDevAttrMaxSurface3DDepth`: Maximum 3D surface depth

        - :py:obj:`~.cudaDevAttrMaxSurface1DLayeredWidth`: Maximum 1D layered
          surface width

        - :py:obj:`~.cudaDevAttrMaxSurface1DLayeredLayers`: Maximum layers in a
          1D layered surface

        - :py:obj:`~.cudaDevAttrMaxSurface2DLayeredWidth`: Maximum 2D layered
          surface width

        - :py:obj:`~.cudaDevAttrMaxSurface2DLayeredHeight`: Maximum 2D layered
          surface height

        - :py:obj:`~.cudaDevAttrMaxSurface2DLayeredLayers`: Maximum layers in a
          2D layered surface

        - :py:obj:`~.cudaDevAttrMaxSurfaceCubemapWidth`: Maximum cubemap
          surface width

        - :py:obj:`~.cudaDevAttrMaxSurfaceCubemapLayeredWidth`: Maximum cubemap
          layered surface width

        - :py:obj:`~.cudaDevAttrMaxSurfaceCubemapLayeredLayers`: Maximum layers
          in a cubemap layered surface

        - :py:obj:`~.cudaDevAttrMaxRegistersPerBlock`: Maximum number of 32-bit
          registers available to a thread block

        - :py:obj:`~.cudaDevAttrClockRate`: Peak clock frequency in kilohertz

        - :py:obj:`~.cudaDevAttrTextureAlignment`: Alignment requirement;
          texture base addresses aligned to :py:obj:`~.textureAlign` bytes do
          not need an offset applied to texture fetches

        - :py:obj:`~.cudaDevAttrTexturePitchAlignment`: Pitch alignment
          requirement for 2D texture references bound to pitched memory

        - :py:obj:`~.cudaDevAttrGpuOverlap`: 1 if the device can concurrently
          copy memory between host and device while executing a kernel, or 0 if
          not

        - :py:obj:`~.cudaDevAttrMultiProcessorCount`: Number of multiprocessors
          on the device

        - :py:obj:`~.cudaDevAttrKernelExecTimeout`: 1 if there is a run time
          limit for kernels executed on the device, or 0 if not

        - :py:obj:`~.cudaDevAttrIntegrated`: 1 if the device is integrated with
          the memory subsystem, or 0 if not

        - :py:obj:`~.cudaDevAttrCanMapHostMemory`: 1 if the device can map host
          memory into the CUDA address space, or 0 if not

        - :py:obj:`~.cudaDevAttrComputeMode`: Compute mode is the compute mode
          that the device is currently in. Available modes are as follows:

          - :py:obj:`~.cudaComputeModeDefault`: Default mode - Device is not
            restricted and multiple threads can use :py:obj:`~.cudaSetDevice()`
            with this device.

          - :py:obj:`~.cudaComputeModeProhibited`: Compute-prohibited mode - No
            threads can use :py:obj:`~.cudaSetDevice()` with this device.

          - :py:obj:`~.cudaComputeModeExclusiveProcess`: Compute-exclusive-
            process mode - Many threads in one process will be able to use
            :py:obj:`~.cudaSetDevice()` with this device.

        - :py:obj:`~.cudaDevAttrConcurrentKernels`: 1 if the device supports
          executing multiple kernels within the same context simultaneously, or
          0 if not. It is not guaranteed that multiple kernels will be resident
          on the device concurrently so this feature should not be relied upon
          for correctness.

        - :py:obj:`~.cudaDevAttrEccEnabled`: 1 if error correction is enabled
          on the device, 0 if error correction is disabled or not supported by
          the device

        - :py:obj:`~.cudaDevAttrPciBusId`: PCI bus identifier of the device

        - :py:obj:`~.cudaDevAttrPciDeviceId`: PCI device (also known as slot)
          identifier of the device

        - :py:obj:`~.cudaDevAttrTccDriver`: 1 if the device is using a TCC
          driver. TCC is only available on Tesla hardware running Windows Vista
          or later.

        - :py:obj:`~.cudaDevAttrMemoryClockRate`: Peak memory clock frequency
          in kilohertz

        - :py:obj:`~.cudaDevAttrGlobalMemoryBusWidth`: Global memory bus width
          in bits

        - :py:obj:`~.cudaDevAttrL2CacheSize`: Size of L2 cache in bytes. 0 if
          the device doesn't have L2 cache.

        - :py:obj:`~.cudaDevAttrMaxThreadsPerMultiProcessor`: Maximum resident
          threads per multiprocessor

        - :py:obj:`~.cudaDevAttrUnifiedAddressing`: 1 if the device shares a
          unified address space with the host, or 0 if not

        - :py:obj:`~.cudaDevAttrComputeCapabilityMajor`: Major compute
          capability version number

        - :py:obj:`~.cudaDevAttrComputeCapabilityMinor`: Minor compute
          capability version number

        - :py:obj:`~.cudaDevAttrStreamPrioritiesSupported`: 1 if the device
          supports stream priorities, or 0 if not

        - :py:obj:`~.cudaDevAttrGlobalL1CacheSupported`: 1 if device supports
          caching globals in L1 cache, 0 if not

        - :py:obj:`~.cudaDevAttrLocalL1CacheSupported`: 1 if device supports
          caching locals in L1 cache, 0 if not

        - :py:obj:`~.cudaDevAttrMaxSharedMemoryPerMultiprocessor`: Maximum
          amount of shared memory available to a multiprocessor in bytes; this
          amount is shared by all thread blocks simultaneously resident on a
          multiprocessor

        - :py:obj:`~.cudaDevAttrMaxRegistersPerMultiprocessor`: Maximum number
          of 32-bit registers available to a multiprocessor; this number is
          shared by all thread blocks simultaneously resident on a
          multiprocessor

        - :py:obj:`~.cudaDevAttrManagedMemory`: 1 if device supports allocating
          managed memory, 0 if not

        - :py:obj:`~.cudaDevAttrIsMultiGpuBoard`: 1 if device is on a multi-GPU
          board, 0 if not

        - :py:obj:`~.cudaDevAttrMultiGpuBoardGroupID`: Unique identifier for a
          group of devices on the same multi-GPU board

        - :py:obj:`~.cudaDevAttrHostNativeAtomicSupported`: 1 if the link
          between the device and the host supports native atomic operations

        - :py:obj:`~.cudaDevAttrSingleToDoublePrecisionPerfRatio`: Ratio of
          single precision performance (in floating-point operations per
          second) to double precision performance

        - :py:obj:`~.cudaDevAttrPageableMemoryAccess`: 1 if the device supports
          coherently accessing pageable memory without calling cudaHostRegister
          on it, and 0 otherwise

        - :py:obj:`~.cudaDevAttrConcurrentManagedAccess`: 1 if the device can
          coherently access managed memory concurrently with the CPU, and 0
          otherwise

        - :py:obj:`~.cudaDevAttrComputePreemptionSupported`: 1 if the device
          supports Compute Preemption, 0 if not

        - :py:obj:`~.cudaDevAttrCanUseHostPointerForRegisteredMem`: 1 if the
          device can access host registered memory at the same virtual address
          as the CPU, and 0 otherwise

        - :py:obj:`~.cudaDevAttrCooperativeLaunch`: 1 if the device supports
          launching cooperative kernels via
          :py:obj:`~.cudaLaunchCooperativeKernel`, and 0 otherwise

        - :py:obj:`~.cudaDevAttrCooperativeMultiDeviceLaunch`: 1 if the device
          supports launching cooperative kernels via
          :py:obj:`~.cudaLaunchCooperativeKernelMultiDevice`, and 0 otherwise

        - :py:obj:`~.cudaDevAttrCanFlushRemoteWrites`: 1 if the device supports
          flushing of outstanding remote writes, and 0 otherwise

        - :py:obj:`~.cudaDevAttrHostRegisterSupported`: 1 if the device
          supports host memory registration via :py:obj:`~.cudaHostRegister`,
          and 0 otherwise

        - :py:obj:`~.cudaDevAttrPageableMemoryAccessUsesHostPageTables`: 1 if
          the device accesses pageable memory via the host's page tables, and 0
          otherwise

        - :py:obj:`~.cudaDevAttrDirectManagedMemAccessFromHost`: 1 if the host
          can directly access managed memory on the device without migration,
          and 0 otherwise

        - :py:obj:`~.cudaDevAttrMaxSharedMemoryPerBlockOptin`: Maximum per
          block shared memory size on the device. This value can be opted into
          when using :py:obj:`~.cudaFuncSetAttribute`

        - :py:obj:`~.cudaDevAttrMaxBlocksPerMultiprocessor`: Maximum number of
          thread blocks that can reside on a multiprocessor

        - :py:obj:`~.cudaDevAttrMaxPersistingL2CacheSize`: Maximum L2
          persisting lines capacity setting in bytes

        - :py:obj:`~.cudaDevAttrMaxAccessPolicyWindowSize`: Maximum value of
          :py:obj:`~.cudaAccessPolicyWindow.num_bytes`

        - :py:obj:`~.cudaDevAttrReservedSharedMemoryPerBlock`: Shared memory
          reserved by CUDA driver per block in bytes

        - :py:obj:`~.cudaDevAttrSparseCudaArraySupported`: 1 if the device
          supports sparse CUDA arrays and sparse CUDA mipmapped arrays.

        - :py:obj:`~.cudaDevAttrHostRegisterReadOnlySupported`: Device supports
          using the :py:obj:`~.cudaHostRegister` flag cudaHostRegisterReadOnly
          to register memory that must be mapped as read-only to the GPU

        - :py:obj:`~.cudaDevAttrMemoryPoolsSupported`: 1 if the device supports
          using the cudaMallocAsync and cudaMemPool family of APIs, and 0
          otherwise

        - :py:obj:`~.cudaDevAttrGPUDirectRDMASupported`: 1 if the device
          supports GPUDirect RDMA APIs, and 0 otherwise

        - :py:obj:`~.cudaDevAttrGPUDirectRDMAFlushWritesOptions`: bitmask to be
          interpreted according to the
          :py:obj:`~.cudaFlushGPUDirectRDMAWritesOptions` enum

        - :py:obj:`~.cudaDevAttrGPUDirectRDMAWritesOrdering`: see the
          :py:obj:`~.cudaGPUDirectRDMAWritesOrdering` enum for numerical values

        - :py:obj:`~.cudaDevAttrMemoryPoolSupportedHandleTypes`: Bitmask of
          handle types supported with mempool based IPC

        - :py:obj:`~.cudaDevAttrDeferredMappingCudaArraySupported` : 1 if the
          device supports deferred mapping CUDA arrays and CUDA mipmapped
          arrays.

        - :py:obj:`~.cudaDevAttrIpcEventSupport`: 1 if the device supports IPC
          Events.

        - :py:obj:`~.cudaDevAttrNumaConfig`: NUMA configuration of a device:
          value is of type :py:obj:`~.cudaDeviceNumaConfig` enum

        - :py:obj:`~.cudaDevAttrNumaId`: NUMA node ID of the GPU memory

        - :py:obj:`~.cudaDevAttrGpuPciDeviceId`: The combined 16-bit PCI device
          ID and 16-bit PCI vendor ID.

        - :py:obj:`~.cudaDevAttrGpuPciSubsystemId`: The combined 16-bit PCI
          subsystem ID and 16-bit PCI vendor subsystem ID.

        Parameters
        ----------
        attr : :py:obj:`~.cudaDeviceAttr`
            Device attribute to query
        device : int
            Device number to query

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidDevice`, :py:obj:`~.cudaErrorInvalidValue`
        value : int
            Returned device attribute value

        See Also
        --------
        :py:obj:`~.cudaGetDeviceCount`, :py:obj:`~.cudaGetDevice`, :py:obj:`~.cudaSetDevice`, :py:obj:`~.cudaChooseDevice`, :py:obj:`~.cudaGetDeviceProperties`, :py:obj:`~.cudaInitDevice`, :py:obj:`~.cuDeviceGetAttribute`
    """


def cudaDeviceGetByPCIBusId(pciBusId):
    """
    cudaDeviceGetByPCIBusId(char *pciBusId)
     Returns a handle to a compute device.

        Returns in `*device` a device ordinal given a PCI bus ID string.

        where `domain`, `bus`, `device`, and `function` are all hexadecimal
        values

        Parameters
        ----------
        pciBusId : bytes
            String in one of the following forms:

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidDevice`
        device : int
            Returned device ordinal

        See Also
        --------
        :py:obj:`~.cudaDeviceGetPCIBusId`, :py:obj:`~.cuDeviceGetByPCIBusId`
    """


def cudaDeviceGetCacheConfig():
    """
    cudaDeviceGetCacheConfig()
     Returns the preferred cache configuration for the current device.

        On devices where the L1 cache and shared memory use the same hardware
        resources, this returns through `pCacheConfig` the preferred cache
        configuration for the current device. This is only a preference. The
        runtime will use the requested configuration if possible, but it is
        free to choose a different configuration if required to execute
        functions.

        This will return a `pCacheConfig` of
        :py:obj:`~.cudaFuncCachePreferNone` on devices where the size of the L1
        cache and shared memory are fixed.

        The supported cache configurations are:

        - :py:obj:`~.cudaFuncCachePreferNone`: no preference for shared memory
          or L1 (default)

        - :py:obj:`~.cudaFuncCachePreferShared`: prefer larger shared memory
          and smaller L1 cache

        - :py:obj:`~.cudaFuncCachePreferL1`: prefer larger L1 cache and smaller
          shared memory

        - :py:obj:`~.cudaFuncCachePreferEqual`: prefer equal size L1 cache and
          shared memory

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`
        pCacheConfig : :py:obj:`~.cudaFuncCache`
            Returned cache configuration

        See Also
        --------
        :py:obj:`~.cudaDeviceSetCacheConfig`, :py:obj:`~.cudaFuncSetCacheConfig (C API)`, cudaFuncSetCacheConfig (C++ API), :py:obj:`~.cuCtxGetCacheConfig`
    """


def cudaDeviceGetDefaultMemPool(device):
    """
    cudaDeviceGetDefaultMemPool(int device)
     Returns the default mempool of a device.

        The default mempool of a device contains device memory from that
        device.

        Parameters
        ----------
        device : int
            None

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidDevice`, :py:obj:`~.cudaErrorInvalidValue` :py:obj:`~.cudaErrorNotSupported`
        memPool : :py:obj:`~.cudaMemPool_t`
            None

        See Also
        --------
        :py:obj:`~.cuDeviceGetDefaultMemPool`, :py:obj:`~.cudaMallocAsync`, :py:obj:`~.cudaMemPoolTrimTo`, :py:obj:`~.cudaMemPoolGetAttribute`, :py:obj:`~.cudaDeviceSetMemPool`, :py:obj:`~.cudaMemPoolSetAttribute`, :py:obj:`~.cudaMemPoolSetAccess`
    """


def cudaDeviceGetGraphMemAttribute(device, attr: 'cudaGraphMemAttributeType'):
    """
    cudaDeviceGetGraphMemAttribute(int device, attr: cudaGraphMemAttributeType)
     Query asynchronous allocation attributes related to graphs.

        Valid attributes are:

        - :py:obj:`~.cudaGraphMemAttrUsedMemCurrent`: Amount of memory, in
          bytes, currently associated with graphs

        - :py:obj:`~.cudaGraphMemAttrUsedMemHigh`: High watermark of memory, in
          bytes, associated with graphs since the last time it was reset. High
          watermark can only be reset to zero.

        - :py:obj:`~.cudaGraphMemAttrReservedMemCurrent`: Amount of memory, in
          bytes, currently allocated for use by the CUDA graphs asynchronous
          allocator.

        - :py:obj:`~.cudaGraphMemAttrReservedMemHigh`: High watermark of
          memory, in bytes, currently allocated for use by the CUDA graphs
          asynchronous allocator.

        Parameters
        ----------
        device : int
            Specifies the scope of the query
        attr : :py:obj:`~.cudaGraphMemAttributeType`
            attribute to get

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidDevice`
        value : Any
            retrieved value

        See Also
        --------
        :py:obj:`~.cudaDeviceSetGraphMemAttribute`, :py:obj:`~.cudaGraphAddMemAllocNode`, :py:obj:`~.cudaGraphAddMemFreeNode`, :py:obj:`~.cudaDeviceGraphMemTrim`, :py:obj:`~.cudaMallocAsync`, :py:obj:`~.cudaFreeAsync`
    """


def cudaDeviceGetLimit(limit: 'cudaLimit'):
    """
    cudaDeviceGetLimit(limit: cudaLimit)
     Return resource limits.

        Returns in `*pValue` the current size of `limit`. The following
        :py:obj:`~.cudaLimit` values are supported.

        - :py:obj:`~.cudaLimitStackSize` is the stack size in bytes of each GPU
          thread.

        - :py:obj:`~.cudaLimitPrintfFifoSize` is the size in bytes of the
          shared FIFO used by the :py:obj:`~.printf()` device system call.

        - :py:obj:`~.cudaLimitMallocHeapSize` is the size in bytes of the heap
          used by the :py:obj:`~.malloc()` and :py:obj:`~.free()` device system
          calls.

        - :py:obj:`~.cudaLimitDevRuntimeSyncDepth` is the maximum grid depth at
          which a thread can isssue the device runtime call
          :py:obj:`~.cudaDeviceSynchronize()` to wait on child grid launches to
          complete. This functionality is removed for devices of compute
          capability >= 9.0, and hence will return error
          :py:obj:`~.cudaErrorUnsupportedLimit` on such devices.

        - :py:obj:`~.cudaLimitDevRuntimePendingLaunchCount` is the maximum
          number of outstanding device runtime launches.

        - :py:obj:`~.cudaLimitMaxL2FetchGranularity` is the L2 cache fetch
          granularity.

        - :py:obj:`~.cudaLimitPersistingL2CacheSize` is the persisting L2 cache
          size in bytes.

        Parameters
        ----------
        limit : :py:obj:`~.cudaLimit`
            Limit to query

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorUnsupportedLimit`, :py:obj:`~.cudaErrorInvalidValue`
        pValue : int
            Returned size of the limit

        See Also
        --------
        :py:obj:`~.cudaDeviceSetLimit`, :py:obj:`~.cuCtxGetLimit`
    """


def cudaDeviceGetMemPool(device):
    """
    cudaDeviceGetMemPool(int device)
     Gets the current mempool for a device.

        Returns the last pool provided to :py:obj:`~.cudaDeviceSetMemPool` for
        this device or the device's default memory pool if
        :py:obj:`~.cudaDeviceSetMemPool` has never been called. By default the
        current mempool is the default mempool for a device, otherwise the
        returned pool must have been set with :py:obj:`~.cuDeviceSetMemPool` or
        :py:obj:`~.cudaDeviceSetMemPool`.

        Parameters
        ----------
        device : int
            None

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue` :py:obj:`~.cudaErrorNotSupported`
        memPool : :py:obj:`~.cudaMemPool_t`
            None

        See Also
        --------
        :py:obj:`~.cuDeviceGetMemPool`, :py:obj:`~.cudaDeviceGetDefaultMemPool`, :py:obj:`~.cudaDeviceSetMemPool`
    """


def cudaDeviceGetNvSciSyncAttributes(nvSciSyncAttrList, device, flags):
    """
    cudaDeviceGetNvSciSyncAttributes(nvSciSyncAttrList, int device, int flags)
     Return NvSciSync attributes that this device can support.

        Returns in `nvSciSyncAttrList`, the properties of NvSciSync that this
        CUDA device, `dev` can support. The returned `nvSciSyncAttrList` can be
        used to create an NvSciSync that matches this device's capabilities.

        If NvSciSyncAttrKey_RequiredPerm field in `nvSciSyncAttrList` is
        already set this API will return :py:obj:`~.cudaErrorInvalidValue`.

        The applications should set `nvSciSyncAttrList` to a valid
        NvSciSyncAttrList failing which this API will return
        :py:obj:`~.cudaErrorInvalidHandle`.

        The `flags` controls how applications intends to use the NvSciSync
        created from the `nvSciSyncAttrList`. The valid flags are:

        - :py:obj:`~.cudaNvSciSyncAttrSignal`, specifies that the applications
          intends to signal an NvSciSync on this CUDA device.

        - :py:obj:`~.cudaNvSciSyncAttrWait`, specifies that the applications
          intends to wait on an NvSciSync on this CUDA device.

        At least one of these flags must be set, failing which the API returns
        :py:obj:`~.cudaErrorInvalidValue`. Both the flags are orthogonal to one
        another: a developer may set both these flags that allows to set both
        wait and signal specific attributes in the same `nvSciSyncAttrList`.

        Note that this API updates the input `nvSciSyncAttrList` with values
        equivalent to the following public attribute key-values:
        NvSciSyncAttrKey_RequiredPerm is set to

        - NvSciSyncAccessPerm_SignalOnly if :py:obj:`~.cudaNvSciSyncAttrSignal`
          is set in `flags`.

        - NvSciSyncAccessPerm_WaitOnly if :py:obj:`~.cudaNvSciSyncAttrWait` is
          set in `flags`.

        - NvSciSyncAccessPerm_WaitSignal if both
          :py:obj:`~.cudaNvSciSyncAttrWait` and
          :py:obj:`~.cudaNvSciSyncAttrSignal` are set in `flags`.
          NvSciSyncAttrKey_PrimitiveInfo is set to

        - NvSciSyncAttrValPrimitiveType_SysmemSemaphore on any valid `device`.

        - NvSciSyncAttrValPrimitiveType_Syncpoint if `device` is a Tegra
          device.

        - NvSciSyncAttrValPrimitiveType_SysmemSemaphorePayload64b if `device`
          is GA10X+. NvSciSyncAttrKey_GpuId is set to the same UUID that is
          returned in `None` from :py:obj:`~.cudaDeviceGetProperties` for this
          `device`.

        :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorDeviceUninitialized`,
        :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidHandle`,
        :py:obj:`~.cudaErrorInvalidDevice`, :py:obj:`~.cudaErrorNotSupported`,
        :py:obj:`~.cudaErrorMemoryAllocation`

        Parameters
        ----------
        nvSciSyncAttrList : Any
            Return NvSciSync attributes supported.
        device : int
            Valid Cuda Device to get NvSciSync attributes for.
        flags : int
            flags describing NvSciSync usage.

        Returns
        -------
        cudaError_t


        See Also
        --------
        :py:obj:`~.cudaImportExternalSemaphore`, :py:obj:`~.cudaDestroyExternalSemaphore`, :py:obj:`~.cudaSignalExternalSemaphoresAsync`, :py:obj:`~.cudaWaitExternalSemaphoresAsync`
    """


def cudaDeviceGetP2PAttribute(attr: 'cudaDeviceP2PAttr', srcDevice, dstDevice):
    """
    cudaDeviceGetP2PAttribute(attr: cudaDeviceP2PAttr, int srcDevice, int dstDevice)
     Queries attributes of the link between two devices.

        Returns in `*value` the value of the requested attribute `attrib` of
        the link between `srcDevice` and `dstDevice`. The supported attributes
        are:

        - :py:obj:`~.cudaDevP2PAttrPerformanceRank`: A relative value
          indicating the performance of the link between two devices. Lower
          value means better performance (0 being the value used for most
          performant link).

        - :py:obj:`~.cudaDevP2PAttrAccessSupported`: 1 if peer access is
          enabled.

        - :py:obj:`~.cudaDevP2PAttrNativeAtomicSupported`: 1 if native atomic
          operations over the link are supported.

        - :py:obj:`~.cudaDevP2PAttrCudaArrayAccessSupported`: 1 if accessing
          CUDA arrays over the link is supported.

        Returns :py:obj:`~.cudaErrorInvalidDevice` if `srcDevice` or
        `dstDevice` are not valid or if they represent the same device.

        Returns :py:obj:`~.cudaErrorInvalidValue` if `attrib` is not valid or
        if `value` is a null pointer.

        Parameters
        ----------
        attrib : :py:obj:`~.cudaDeviceP2PAttr`
            The requested attribute of the link between `srcDevice` and
            `dstDevice`.
        srcDevice : int
            The source device of the target link.
        dstDevice : int
            The destination device of the target link.

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidDevice`, :py:obj:`~.cudaErrorInvalidValue`
        value : int
            Returned value of the requested attribute

        See Also
        --------
        :py:obj:`~.cudaDeviceEnablePeerAccess`, :py:obj:`~.cudaDeviceDisablePeerAccess`, :py:obj:`~.cudaDeviceCanAccessPeer`, :py:obj:`~.cuDeviceGetP2PAttribute`
    """


def cudaDeviceGetPCIBusId(length, device):
    """
    cudaDeviceGetPCIBusId(int length, int device)
     Returns a PCI Bus Id string for the device.

        Returns an ASCII string identifying the device `dev` in the NULL-
        terminated string pointed to by `pciBusId`. `length` specifies the
        maximum length of the string that may be returned.

        where `domain`, `bus`, `device`, and `function` are all hexadecimal
        values. pciBusId should be large enough to store 13 characters
        including the NULL-terminator.

        Parameters
        ----------
        length : int
            Maximum length of string to store in `name`
        device : int
            Device to get identifier string for

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidDevice`
        pciBusId : bytes
            Returned identifier string for the device in the following format

        See Also
        --------
        :py:obj:`~.cudaDeviceGetByPCIBusId`, :py:obj:`~.cuDeviceGetPCIBusId`
    """


def cudaDeviceGetSharedMemConfig():
    """
    cudaDeviceGetSharedMemConfig()
     Returns the shared memory configuration for the current device.

        [Deprecated]

        This function will return in `pConfig` the current size of shared
        memory banks on the current device. On devices with configurable shared
        memory banks, :py:obj:`~.cudaDeviceSetSharedMemConfig` can be used to
        change this setting, so that all subsequent kernel launches will by
        default use the new bank size. When
        :py:obj:`~.cudaDeviceGetSharedMemConfig` is called on devices without
        configurable shared memory, it will return the fixed bank size of the
        hardware.

        The returned bank configurations can be either:

        - :py:obj:`~.cudaSharedMemBankSizeFourByte` - shared memory bank width
          is four bytes.

        - :py:obj:`~.cudaSharedMemBankSizeEightByte` - shared memory bank width
          is eight bytes.

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`
        pConfig : :py:obj:`~.cudaSharedMemConfig`
            Returned cache configuration

        See Also
        --------
        :py:obj:`~.cudaDeviceSetCacheConfig`, :py:obj:`~.cudaDeviceGetCacheConfig`, :py:obj:`~.cudaDeviceSetSharedMemConfig`, :py:obj:`~.cudaFuncSetCacheConfig`, :py:obj:`~.cuCtxGetSharedMemConfig`
    """


def cudaDeviceGetStreamPriorityRange():
    """
    cudaDeviceGetStreamPriorityRange()
     Returns numerical values that correspond to the least and greatest stream priorities.

        Returns in `*leastPriority` and `*greatestPriority` the numerical
        values that correspond to the least and greatest stream priorities
        respectively. Stream priorities follow a convention where lower numbers
        imply greater priorities. The range of meaningful stream priorities is
        given by [`*greatestPriority`, `*leastPriority`]. If the user attempts
        to create a stream with a priority value that is outside the the
        meaningful range as specified by this API, the priority is
        automatically clamped down or up to either `*leastPriority` or
        `*greatestPriority` respectively. See
        :py:obj:`~.cudaStreamCreateWithPriority` for details on creating a
        priority stream. A NULL may be passed in for `*leastPriority` or
        `*greatestPriority` if the value is not desired.

        This function will return '0' in both `*leastPriority` and
        `*greatestPriority` if the current context's device does not support
        stream priorities (see :py:obj:`~.cudaDeviceGetAttribute`).

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`
        leastPriority : int
            Pointer to an int in which the numerical value for least stream
            priority is returned
        greatestPriority : int
            Pointer to an int in which the numerical value for greatest stream
            priority is returned

        See Also
        --------
        :py:obj:`~.cudaStreamCreateWithPriority`, :py:obj:`~.cudaStreamGetPriority`, :py:obj:`~.cuCtxGetStreamPriorityRange`
    """


def cudaDeviceGetTexture1DLinearMaxWidth(fmtDesc: 'Optional[cudaChannelFormatDesc]', device):
    """
    cudaDeviceGetTexture1DLinearMaxWidth(cudaChannelFormatDesc fmtDesc: Optional[cudaChannelFormatDesc], int device)
     Returns the maximum number of elements allocatable in a 1D linear texture for a given element size.

        Returns in `maxWidthInElements` the maximum number of elements
        allocatable in a 1D linear texture for given format descriptor
        `fmtDesc`.

        Parameters
        ----------
        fmtDesc : :py:obj:`~.cudaChannelFormatDesc`
            Texture format description.
        None : int
            None

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorUnsupportedLimit`, :py:obj:`~.cudaErrorInvalidValue`
        maxWidthInElements : int
            Returns maximum number of texture elements allocatable for given
            `fmtDesc`.

        See Also
        --------
        :py:obj:`~.cuDeviceGetTexture1DLinearMaxWidth`
    """


def cudaDeviceGraphMemTrim(device):
    """
    cudaDeviceGraphMemTrim(int device)
     Free unused memory that was cached on the specified device for use with graphs back to the OS.

        Blocks which are not in use by a graph that is either currently
        executing or scheduled to execute are freed back to the operating
        system.

        Parameters
        ----------
        device : int
            The device for which cached memory should be freed.

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`

        See Also
        --------
        :py:obj:`~.cudaGraphAddMemAllocNode`, :py:obj:`~.cudaGraphAddMemFreeNode`, :py:obj:`~.cudaDeviceGetGraphMemAttribute`, :py:obj:`~.cudaDeviceSetGraphMemAttribute`, :py:obj:`~.cudaMallocAsync`, :py:obj:`~.cudaFreeAsync`
    """

cudaDeviceLmemResizeToMax: int
cudaDeviceMapHost: int
cudaDeviceMask: int

def cudaDeviceRegisterAsyncNotification(device, callbackFunc, userData):
    """
    cudaDeviceRegisterAsyncNotification(int device, callbackFunc, userData)
     Registers a callback function to receive async notifications.

        Registers `callbackFunc` to receive async notifications.

        The `userData` parameter is passed to the callback function at async
        notification time. Likewise, `callback` is also passed to the callback
        function to distinguish between multiple registered callbacks.

        The callback function being registered should be designed to return
        quickly (~10ms). Any long running tasks should be queued for execution
        on an application thread.

        Callbacks may not call cudaDeviceRegisterAsyncNotification or
        cudaDeviceUnregisterAsyncNotification. Doing so will result in
        :py:obj:`~.cudaErrorNotPermitted`. Async notification callbacks execute
        in an undefined order and may be serialized.

        Returns in `*callback` a handle representing the registered callback
        instance.

        Parameters
        ----------
        device : int
            The device on which to register the callback
        callbackFunc : :py:obj:`~.cudaAsyncCallback`
            The function to register as a callback
        userData : Any
            A generic pointer to user data. This is passed into the callback
            function.

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess` :py:obj:`~.cudaErrorNotSupported` :py:obj:`~.cudaErrorInvalidDevice` :py:obj:`~.cudaErrorInvalidValue` :py:obj:`~.cudaErrorNotPermitted` :py:obj:`~.cudaErrorUnknown`
        callback : :py:obj:`~.cudaAsyncCallbackHandle_t`
            A handle representing the registered callback instance

        See Also
        --------
        :py:obj:`~.cudaDeviceUnregisterAsyncNotification`
    """


def cudaDeviceReset():
    """
    cudaDeviceReset()
     Destroy all allocations and reset all state on the current device in the current process.

        Explicitly destroys and cleans up all resources associated with the
        current device in the current process. It is the caller's
        responsibility to ensure that the resources are not accessed or passed
        in subsequent API calls and doing so will result in undefined behavior.
        These resources include CUDA types :py:obj:`~.cudaStream_t`,
        :py:obj:`~.cudaEvent_t`, :py:obj:`~.cudaArray_t`,
        :py:obj:`~.cudaMipmappedArray_t`, :py:obj:`~.cudaPitchedPtr`,
        :py:obj:`~.cudaTextureObject_t`, :py:obj:`~.cudaSurfaceObject_t`,
        :py:obj:`~.textureReference`, :py:obj:`~.surfaceReference`,
        :py:obj:`~.cudaExternalMemory_t`, :py:obj:`~.cudaExternalSemaphore_t`
        and :py:obj:`~.cudaGraphicsResource_t`. These resources also include
        memory allocations by :py:obj:`~.cudaMalloc`,
        :py:obj:`~.cudaMallocHost`, :py:obj:`~.cudaMallocManaged` and
        :py:obj:`~.cudaMallocPitch`. Any subsequent API call to this device
        will reinitialize the device.

        Note that this function will reset the device immediately. It is the
        caller's responsibility to ensure that the device is not being accessed
        by any other host threads from the process when this function is
        called.

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`

        See Also
        --------
        :py:obj:`~.cudaDeviceSynchronize`

        Notes
        -----
        :py:obj:`~.cudaDeviceReset()` will not destroy memory allocations by :py:obj:`~.cudaMallocAsync()` and :py:obj:`~.cudaMallocFromPoolAsync()`. These memory allocations need to be destroyed explicitly.

        If a non-primary :py:obj:`~.CUcontext` is current to the thread, :py:obj:`~.cudaDeviceReset()` will destroy only the internal CUDA RT state for that :py:obj:`~.CUcontext`.
    """

cudaDeviceScheduleAuto: int
cudaDeviceScheduleBlockingSync: int
cudaDeviceScheduleMask: int
cudaDeviceScheduleSpin: int
cudaDeviceScheduleYield: int

def cudaDeviceSetCacheConfig(cacheConfig: 'cudaFuncCache'):
    """
    cudaDeviceSetCacheConfig(cacheConfig: cudaFuncCache)
     Sets the preferred cache configuration for the current device.

        On devices where the L1 cache and shared memory use the same hardware
        resources, this sets through `cacheConfig` the preferred cache
        configuration for the current device. This is only a preference. The
        runtime will use the requested configuration if possible, but it is
        free to choose a different configuration if required to execute the
        function. Any function preference set via
        :py:obj:`~.cudaFuncSetCacheConfig (C API)` or cudaFuncSetCacheConfig
        (C++ API) will be preferred over this device-wide setting. Setting the
        device-wide cache configuration to :py:obj:`~.cudaFuncCachePreferNone`
        will cause subsequent kernel launches to prefer to not change the cache
        configuration unless required to launch the kernel.

        This setting does nothing on devices where the size of the L1 cache and
        shared memory are fixed.

        Launching a kernel with a different preference than the most recent
        preference setting may insert a device-side synchronization point.

        The supported cache configurations are:

        - :py:obj:`~.cudaFuncCachePreferNone`: no preference for shared memory
          or L1 (default)

        - :py:obj:`~.cudaFuncCachePreferShared`: prefer larger shared memory
          and smaller L1 cache

        - :py:obj:`~.cudaFuncCachePreferL1`: prefer larger L1 cache and smaller
          shared memory

        - :py:obj:`~.cudaFuncCachePreferEqual`: prefer equal size L1 cache and
          shared memory

        Parameters
        ----------
        cacheConfig : :py:obj:`~.cudaFuncCache`
            Requested cache configuration

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`

        See Also
        --------
        :py:obj:`~.cudaDeviceGetCacheConfig`, :py:obj:`~.cudaFuncSetCacheConfig (C API)`, cudaFuncSetCacheConfig (C++ API), :py:obj:`~.cuCtxSetCacheConfig`
    """


def cudaDeviceSetGraphMemAttribute(device, attr: 'cudaGraphMemAttributeType', value):
    """
    cudaDeviceSetGraphMemAttribute(int device, attr: cudaGraphMemAttributeType, value)
     Set asynchronous allocation attributes related to graphs.

        Valid attributes are:

        - :py:obj:`~.cudaGraphMemAttrUsedMemHigh`: High watermark of memory, in
          bytes, associated with graphs since the last time it was reset. High
          watermark can only be reset to zero.

        - :py:obj:`~.cudaGraphMemAttrReservedMemHigh`: High watermark of
          memory, in bytes, currently allocated for use by the CUDA graphs
          asynchronous allocator.

        Parameters
        ----------
        device : int
            Specifies the scope of the query
        attr : :py:obj:`~.cudaGraphMemAttributeType`
            attribute to get
        value : Any
            pointer to value to set

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidDevice`

        See Also
        --------
        :py:obj:`~.cudaDeviceGetGraphMemAttribute`, :py:obj:`~.cudaGraphAddMemAllocNode`, :py:obj:`~.cudaGraphAddMemFreeNode`, :py:obj:`~.cudaDeviceGraphMemTrim`, :py:obj:`~.cudaMallocAsync`, :py:obj:`~.cudaFreeAsync`
    """


def cudaDeviceSetLimit(limit: 'cudaLimit', value):
    """
    cudaDeviceSetLimit(limit: cudaLimit, size_t value)
     Set resource limits.

        Setting `limit` to `value` is a request by the application to update
        the current limit maintained by the device. The driver is free to
        modify the requested value to meet h/w requirements (this could be
        clamping to minimum or maximum values, rounding up to nearest element
        size, etc). The application can use :py:obj:`~.cudaDeviceGetLimit()` to
        find out exactly what the limit has been set to.

        Setting each :py:obj:`~.cudaLimit` has its own specific restrictions,
        so each is discussed here.

        - :py:obj:`~.cudaLimitStackSize` controls the stack size in bytes of
          each GPU thread.

        - :py:obj:`~.cudaLimitPrintfFifoSize` controls the size in bytes of the
          shared FIFO used by the :py:obj:`~.printf()` device system call.
          Setting :py:obj:`~.cudaLimitPrintfFifoSize` must not be performed
          after launching any kernel that uses the :py:obj:`~.printf()` device
          system call - in such case :py:obj:`~.cudaErrorInvalidValue` will be
          returned.

        - :py:obj:`~.cudaLimitMallocHeapSize` controls the size in bytes of the
          heap used by the :py:obj:`~.malloc()` and :py:obj:`~.free()` device
          system calls. Setting :py:obj:`~.cudaLimitMallocHeapSize` must not be
          performed after launching any kernel that uses the
          :py:obj:`~.malloc()` or :py:obj:`~.free()` device system calls - in
          such case :py:obj:`~.cudaErrorInvalidValue` will be returned.

        - :py:obj:`~.cudaLimitDevRuntimeSyncDepth` controls the maximum nesting
          depth of a grid at which a thread can safely call
          :py:obj:`~.cudaDeviceSynchronize()`. Setting this limit must be
          performed before any launch of a kernel that uses the device runtime
          and calls :py:obj:`~.cudaDeviceSynchronize()` above the default sync
          depth, two levels of grids. Calls to
          :py:obj:`~.cudaDeviceSynchronize()` will fail with error code
          :py:obj:`~.cudaErrorSyncDepthExceeded` if the limitation is violated.
          This limit can be set smaller than the default or up the maximum
          launch depth of 24. When setting this limit, keep in mind that
          additional levels of sync depth require the runtime to reserve large
          amounts of device memory which can no longer be used for user
          allocations. If these reservations of device memory fail,
          :py:obj:`~.cudaDeviceSetLimit` will return
          :py:obj:`~.cudaErrorMemoryAllocation`, and the limit can be reset to
          a lower value. This limit is only applicable to devices of compute
          capability < 9.0. Attempting to set this limit on devices of other
          compute capability will results in error
          :py:obj:`~.cudaErrorUnsupportedLimit` being returned.

        - :py:obj:`~.cudaLimitDevRuntimePendingLaunchCount` controls the
          maximum number of outstanding device runtime launches that can be
          made from the current device. A grid is outstanding from the point of
          launch up until the grid is known to have been completed. Device
          runtime launches which violate this limitation fail and return
          :py:obj:`~.cudaErrorLaunchPendingCountExceeded` when
          :py:obj:`~.cudaGetLastError()` is called after launch. If more
          pending launches than the default (2048 launches) are needed for a
          module using the device runtime, this limit can be increased. Keep in
          mind that being able to sustain additional pending launches will
          require the runtime to reserve larger amounts of device memory
          upfront which can no longer be used for allocations. If these
          reservations fail, :py:obj:`~.cudaDeviceSetLimit` will return
          :py:obj:`~.cudaErrorMemoryAllocation`, and the limit can be reset to
          a lower value. This limit is only applicable to devices of compute
          capability 3.5 and higher. Attempting to set this limit on devices of
          compute capability less than 3.5 will result in the error
          :py:obj:`~.cudaErrorUnsupportedLimit` being returned.

        - :py:obj:`~.cudaLimitMaxL2FetchGranularity` controls the L2 cache
          fetch granularity. Values can range from 0B to 128B. This is purely a
          performance hint and it can be ignored or clamped depending on the
          platform.

        - :py:obj:`~.cudaLimitPersistingL2CacheSize` controls size in bytes
          available for persisting L2 cache. This is purely a performance hint
          and it can be ignored or clamped depending on the platform.

        Parameters
        ----------
        limit : :py:obj:`~.cudaLimit`
            Limit to set
        value : size_t
            Size of limit

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorUnsupportedLimit`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorMemoryAllocation`

        See Also
        --------
        :py:obj:`~.cudaDeviceGetLimit`, :py:obj:`~.cuCtxSetLimit`
    """


def cudaDeviceSetMemPool(device, memPool):
    """
    cudaDeviceSetMemPool(int device, memPool)
     Sets the current memory pool of a device.

        The memory pool must be local to the specified device. Unless a mempool
        is specified in the :py:obj:`~.cudaMallocAsync` call,
        :py:obj:`~.cudaMallocAsync` allocates from the current mempool of the
        provided stream's device. By default, a device's current memory pool is
        its default memory pool.

        Parameters
        ----------
        device : int
            None
        memPool : :py:obj:`~.CUmemoryPool` or :py:obj:`~.cudaMemPool_t`
            None

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue` :py:obj:`~.cudaErrorInvalidDevice` :py:obj:`~.cudaErrorNotSupported`

        See Also
        --------
        :py:obj:`~.cuDeviceSetMemPool`, :py:obj:`~.cudaDeviceGetMemPool`, :py:obj:`~.cudaDeviceGetDefaultMemPool`, :py:obj:`~.cudaMemPoolCreate`, :py:obj:`~.cudaMemPoolDestroy`, :py:obj:`~.cudaMallocFromPoolAsync`

        Notes
        -----
        Use :py:obj:`~.cudaMallocFromPoolAsync` to specify asynchronous allocations from a device different than the one the stream runs on.
    """


def cudaDeviceSetSharedMemConfig(config: 'cudaSharedMemConfig'):
    """
    cudaDeviceSetSharedMemConfig(config: cudaSharedMemConfig)
     Sets the shared memory configuration for the current device.

        [Deprecated]

        On devices with configurable shared memory banks, this function will
        set the shared memory bank size which is used for all subsequent kernel
        launches. Any per-function setting of shared memory set via
        :py:obj:`~.cudaFuncSetSharedMemConfig` will override the device wide
        setting.

        Changing the shared memory configuration between launches may introduce
        a device side synchronization point.

        Changing the shared memory bank size will not increase shared memory
        usage or affect occupancy of kernels, but may have major effects on
        performance. Larger bank sizes will allow for greater potential
        bandwidth to shared memory, but will change what kinds of accesses to
        shared memory will result in bank conflicts.

        This function will do nothing on devices with fixed shared memory bank
        size.

        The supported bank configurations are:

        - :py:obj:`~.cudaSharedMemBankSizeDefault`: set bank width the device
          default (currently, four bytes)

        - :py:obj:`~.cudaSharedMemBankSizeFourByte`: set shared memory bank
          width to be four bytes natively.

        - :py:obj:`~.cudaSharedMemBankSizeEightByte`: set shared memory bank
          width to be eight bytes natively.

        Parameters
        ----------
        config : :py:obj:`~.cudaSharedMemConfig`
            Requested cache configuration

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`

        See Also
        --------
        :py:obj:`~.cudaDeviceSetCacheConfig`, :py:obj:`~.cudaDeviceGetCacheConfig`, :py:obj:`~.cudaDeviceGetSharedMemConfig`, :py:obj:`~.cudaFuncSetCacheConfig`, :py:obj:`~.cuCtxSetSharedMemConfig`
    """

cudaDeviceSyncMemops: int

def cudaDeviceSynchronize():
    """
    cudaDeviceSynchronize()
     Wait for compute device to finish.

        Blocks until the device has completed all preceding requested tasks.
        :py:obj:`~.cudaDeviceSynchronize()` returns an error if one of the
        preceding tasks has failed. If the
        :py:obj:`~.cudaDeviceScheduleBlockingSync` flag was set for this
        device, the host thread will block until the device has finished its
        work.

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`

        See Also
        --------
        :py:obj:`~.cudaDeviceReset`, :py:obj:`~.cuCtxSynchronize`
    """


def cudaDeviceUnregisterAsyncNotification(device, callback):
    """
    cudaDeviceUnregisterAsyncNotification(int device, callback)
     Unregisters an async notification callback.

        Unregisters `callback` so that the corresponding callback function will
        stop receiving async notifications.

        Parameters
        ----------
        device : int
            The device from which to remove `callback`.
        callback : :py:obj:`~.cudaAsyncCallbackHandle_t`
            The callback instance to unregister from receiving async
            notifications.

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess` :py:obj:`~.cudaErrorNotSupported` :py:obj:`~.cudaErrorInvalidDevice` :py:obj:`~.cudaErrorInvalidValue` :py:obj:`~.cudaErrorNotPermitted` :py:obj:`~.cudaErrorUnknown`

        See Also
        --------
        :py:obj:`~.cudaDeviceRegisterAsyncNotification`
    """


def cudaDriverGetVersion():
    """
    cudaDriverGetVersion()
     Returns the latest version of CUDA supported by the driver.

        Returns in `*driverVersion` the latest version of CUDA supported by the
        driver. The version is returned as (1000 * major + 10 * minor). For
        example, CUDA 9.2 would be represented by 9020. If no driver is
        installed, then 0 is returned as the driver version.

        This function automatically returns :py:obj:`~.cudaErrorInvalidValue`
        if `driverVersion` is NULL.

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`
        driverVersion : int
            Returns the CUDA driver version.

        See Also
        --------
        :py:obj:`~.cudaRuntimeGetVersion`, :py:obj:`~.cuDriverGetVersion`
    """


def cudaEGLStreamConsumerAcquireFrame(conn, pCudaResource, pStream, timeout):
    """
    cudaEGLStreamConsumerAcquireFrame(conn, pCudaResource, pStream, unsigned int timeout)
     Acquire an image frame from the EGLStream with CUDA as a consumer.

        Acquire an image frame from EGLStreamKHR.
        :py:obj:`~.cudaGraphicsResourceGetMappedEglFrame` can be called on
        `pCudaResource` to get :py:obj:`~.cudaEglFrame`.

        Parameters
        ----------
        conn : :py:obj:`~.cudaEglStreamConnection`
            Connection on which to acquire
        pCudaResource : :py:obj:`~.cudaGraphicsResource_t`
            CUDA resource on which the EGLStream frame will be mapped for use.
        pStream : :py:obj:`~.cudaStream_t`
            CUDA stream for synchronization and any data migrations implied by
            :py:obj:`~.cudaEglResourceLocationFlags`.
        timeout : unsigned int
            Desired timeout in usec.

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorUnknown`, :py:obj:`~.cudaErrorLaunchTimeout`

        See Also
        --------
        :py:obj:`~.cudaEGLStreamConsumerConnect`, :py:obj:`~.cudaEGLStreamConsumerDisconnect`, :py:obj:`~.cudaEGLStreamConsumerReleaseFrame`, :py:obj:`~.cuEGLStreamConsumerAcquireFrame`
    """


def cudaEGLStreamConsumerConnect(eglStream):
    """
    cudaEGLStreamConsumerConnect(eglStream)
     Connect CUDA to EGLStream as a consumer.

        Connect CUDA as a consumer to EGLStreamKHR specified by `eglStream`.

        The EGLStreamKHR is an EGL object that transfers a sequence of image
        frames from one API to another.

        Parameters
        ----------
        eglStream : :py:obj:`~.EGLStreamKHR`
            EGLStreamKHR handle

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorUnknown`
        conn : :py:obj:`~.cudaEglStreamConnection`
            Pointer to the returned connection handle

        See Also
        --------
        :py:obj:`~.cudaEGLStreamConsumerDisconnect`, :py:obj:`~.cudaEGLStreamConsumerAcquireFrame`, :py:obj:`~.cudaEGLStreamConsumerReleaseFrame`, :py:obj:`~.cuEGLStreamConsumerConnect`
    """


def cudaEGLStreamConsumerConnectWithFlags(eglStream, flags):
    """
    cudaEGLStreamConsumerConnectWithFlags(eglStream, unsigned int flags)
     Connect CUDA to EGLStream as a consumer with given flags.

        Connect CUDA as a consumer to EGLStreamKHR specified by `stream` with
        specified `flags` defined by :py:obj:`~.cudaEglResourceLocationFlags`.

        The flags specify whether the consumer wants to access frames from
        system memory or video memory. Default is
        :py:obj:`~.cudaEglResourceLocationVidmem`.

        Parameters
        ----------
        eglStream : :py:obj:`~.EGLStreamKHR`
            EGLStreamKHR handle
        flags : unsigned int
            Flags denote intended location - system or video.

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorUnknown`
        conn : :py:obj:`~.cudaEglStreamConnection`
            Pointer to the returned connection handle

        See Also
        --------
        :py:obj:`~.cudaEGLStreamConsumerDisconnect`, :py:obj:`~.cudaEGLStreamConsumerAcquireFrame`, :py:obj:`~.cudaEGLStreamConsumerReleaseFrame`, :py:obj:`~.cuEGLStreamConsumerConnectWithFlags`
    """


def cudaEGLStreamConsumerDisconnect(conn):
    """
    cudaEGLStreamConsumerDisconnect(conn)
     Disconnect CUDA as a consumer to EGLStream .

        Disconnect CUDA as a consumer to EGLStreamKHR.

        Parameters
        ----------
        conn : :py:obj:`~.cudaEglStreamConnection`
            Conection to disconnect.

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorUnknown`

        See Also
        --------
        :py:obj:`~.cudaEGLStreamConsumerConnect`, :py:obj:`~.cudaEGLStreamConsumerAcquireFrame`, :py:obj:`~.cudaEGLStreamConsumerReleaseFrame`, :py:obj:`~.cuEGLStreamConsumerDisconnect`
    """


def cudaEGLStreamConsumerReleaseFrame(conn, pCudaResource, pStream):
    """
    cudaEGLStreamConsumerReleaseFrame(conn, pCudaResource, pStream)
     Releases the last frame acquired from the EGLStream.

        Release the acquired image frame specified by `pCudaResource` to
        EGLStreamKHR.

        Parameters
        ----------
        conn : :py:obj:`~.cudaEglStreamConnection`
            Connection on which to release
        pCudaResource : :py:obj:`~.cudaGraphicsResource_t`
            CUDA resource whose corresponding frame is to be released
        pStream : :py:obj:`~.cudaStream_t`
            CUDA stream on which release will be done.

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorUnknown`

        See Also
        --------
        :py:obj:`~.cudaEGLStreamConsumerConnect`, :py:obj:`~.cudaEGLStreamConsumerDisconnect`, :py:obj:`~.cudaEGLStreamConsumerAcquireFrame`, :py:obj:`~.cuEGLStreamConsumerReleaseFrame`
    """


def cudaEGLStreamProducerConnect(eglStream, width, height):
    """
    cudaEGLStreamProducerConnect(eglStream, width, height)
     Connect CUDA to EGLStream as a producer.

        Connect CUDA as a producer to EGLStreamKHR specified by `stream`.

        The EGLStreamKHR is an EGL object that transfers a sequence of image
        frames from one API to another.

        Parameters
        ----------
        eglStream : :py:obj:`~.EGLStreamKHR`
            EGLStreamKHR handle
        width : :py:obj:`~.EGLint`
            width of the image to be submitted to the stream
        height : :py:obj:`~.EGLint`
            height of the image to be submitted to the stream

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorUnknown`
        conn : :py:obj:`~.cudaEglStreamConnection`
            Pointer to the returned connection handle

        See Also
        --------
        :py:obj:`~.cudaEGLStreamProducerDisconnect`, :py:obj:`~.cudaEGLStreamProducerPresentFrame`, :py:obj:`~.cudaEGLStreamProducerReturnFrame`, :py:obj:`~.cuEGLStreamProducerConnect`
    """


def cudaEGLStreamProducerDisconnect(conn):
    """
    cudaEGLStreamProducerDisconnect(conn)
     Disconnect CUDA as a producer to EGLStream .

        Disconnect CUDA as a producer to EGLStreamKHR.

        Parameters
        ----------
        conn : :py:obj:`~.cudaEglStreamConnection`
            Conection to disconnect.

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorUnknown`

        See Also
        --------
        :py:obj:`~.cudaEGLStreamProducerConnect`, :py:obj:`~.cudaEGLStreamProducerPresentFrame`, :py:obj:`~.cudaEGLStreamProducerReturnFrame`, :py:obj:`~.cuEGLStreamProducerDisconnect`
    """


def cudaEGLStreamProducerPresentFrame(conn, eglframe: 'cudaEglFrame', pStream):
    """
    cudaEGLStreamProducerPresentFrame(conn, cudaEglFrame eglframe: cudaEglFrame, pStream)
     Present a CUDA eglFrame to the EGLStream with CUDA as a producer.

        The :py:obj:`~.cudaEglFrame` is defined as:

        **View CUDA Toolkit Documentation for a C++ code example**

        For :py:obj:`~.cudaEglFrame` of type :py:obj:`~.cudaEglFrameTypePitch`,
        the application may present sub-region of a memory allocation. In that
        case, :py:obj:`~.cudaPitchedPtr.ptr` will specify the start address of
        the sub-region in the allocation and :py:obj:`~.cudaEglPlaneDesc` will
        specify the dimensions of the sub-region.

        Parameters
        ----------
        conn : :py:obj:`~.cudaEglStreamConnection`
            Connection on which to present the CUDA array
        eglframe : :py:obj:`~.cudaEglFrame`
            CUDA Eglstream Proucer Frame handle to be sent to the consumer over
            EglStream.
        pStream : :py:obj:`~.cudaStream_t`
            CUDA stream on which to present the frame.

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorUnknown`

        See Also
        --------
        :py:obj:`~.cudaEGLStreamProducerConnect`, :py:obj:`~.cudaEGLStreamProducerDisconnect`, :py:obj:`~.cudaEGLStreamProducerReturnFrame`, :py:obj:`~.cuEGLStreamProducerPresentFrame`
    """


def cudaEGLStreamProducerReturnFrame(conn, eglframe: 'Optional[cudaEglFrame]', pStream):
    """
    cudaEGLStreamProducerReturnFrame(conn, cudaEglFrame eglframe: Optional[cudaEglFrame], pStream)
     Return the CUDA eglFrame to the EGLStream last released by the consumer.

        This API can potentially return cudaErrorLaunchTimeout if the consumer
        has not returned a frame to EGL stream. If timeout is returned the
        application can retry.

        Parameters
        ----------
        conn : :py:obj:`~.cudaEglStreamConnection`
            Connection on which to present the CUDA array
        eglframe : :py:obj:`~.cudaEglFrame`
            CUDA Eglstream Proucer Frame handle returned from the consumer over
            EglStream.
        pStream : :py:obj:`~.cudaStream_t`
            CUDA stream on which to return the frame.

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorLaunchTimeout`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorUnknown`

        See Also
        --------
        :py:obj:`~.cudaEGLStreamProducerConnect`, :py:obj:`~.cudaEGLStreamProducerDisconnect`, :py:obj:`~.cudaEGLStreamProducerPresentFrame`, :py:obj:`~.cuEGLStreamProducerReturnFrame`
    """

cudaEventBlockingSync: int

def cudaEventCreate():
    """
    cudaEventCreate()
     Creates an event object.

        Creates an event object for the current device using
        :py:obj:`~.cudaEventDefault`.

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorLaunchFailure`, :py:obj:`~.cudaErrorMemoryAllocation`
        event : :py:obj:`~.cudaEvent_t`
            Newly created event

        See Also
        --------
        cudaEventCreate (C++ API), :py:obj:`~.cudaEventCreateWithFlags`, :py:obj:`~.cudaEventRecord`, :py:obj:`~.cudaEventQuery`, :py:obj:`~.cudaEventSynchronize`, :py:obj:`~.cudaEventDestroy`, :py:obj:`~.cudaEventElapsedTime`, :py:obj:`~.cudaStreamWaitEvent`, :py:obj:`~.cuEventCreate`
    """


def cudaEventCreateFromEGLSync(eglSync, flags):
    """
    cudaEventCreateFromEGLSync(eglSync, unsigned int flags)
     Creates an event from EGLSync object.

        Creates an event *phEvent from an EGLSyncKHR eglSync with the flages
        specified via `flags`. Valid flags include:

        - :py:obj:`~.cudaEventDefault`: Default event creation flag.

        - :py:obj:`~.cudaEventBlockingSync`: Specifies that the created event
          should use blocking synchronization. A CPU thread that uses
          :py:obj:`~.cudaEventSynchronize()` to wait on an event created with
          this flag will block until the event has actually been completed.

        :py:obj:`~.cudaEventRecord` and TimingData are not supported for events
        created from EGLSync.

        The EGLSyncKHR is an opaque handle to an EGL sync object. typedef void*
        EGLSyncKHR

        Parameters
        ----------
        eglSync : :py:obj:`~.EGLSyncKHR`
            Opaque handle to EGLSync object
        flags : unsigned int
            Event creation flags

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInitializationError`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorLaunchFailure`, :py:obj:`~.cudaErrorMemoryAllocation`
        phEvent : :py:obj:`~.cudaEvent_t`
            Returns newly created event

        See Also
        --------
        :py:obj:`~.cudaEventQuery`, :py:obj:`~.cudaEventSynchronize`, :py:obj:`~.cudaEventDestroy`
    """


def cudaEventCreateWithFlags(flags):
    """
    cudaEventCreateWithFlags(unsigned int flags)
     Creates an event object with the specified flags.

        Creates an event object for the current device with the specified
        flags. Valid flags include:

        - :py:obj:`~.cudaEventDefault`: Default event creation flag.

        - :py:obj:`~.cudaEventBlockingSync`: Specifies that event should use
          blocking synchronization. A host thread that uses
          :py:obj:`~.cudaEventSynchronize()` to wait on an event created with
          this flag will block until the event actually completes.

        - :py:obj:`~.cudaEventDisableTiming`: Specifies that the created event
          does not need to record timing data. Events created with this flag
          specified and the :py:obj:`~.cudaEventBlockingSync` flag not
          specified will provide the best performance when used with
          :py:obj:`~.cudaStreamWaitEvent()` and :py:obj:`~.cudaEventQuery()`.

        - :py:obj:`~.cudaEventInterprocess`: Specifies that the created event
          may be used as an interprocess event by
          :py:obj:`~.cudaIpcGetEventHandle()`.
          :py:obj:`~.cudaEventInterprocess` must be specified along with
          :py:obj:`~.cudaEventDisableTiming`.

        Parameters
        ----------
        flags : unsigned int
            Flags for new event

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorLaunchFailure`, :py:obj:`~.cudaErrorMemoryAllocation`
        event : :py:obj:`~.cudaEvent_t`
            Newly created event

        See Also
        --------
        :py:obj:`~.cudaEventCreate (C API)`, :py:obj:`~.cudaEventSynchronize`, :py:obj:`~.cudaEventDestroy`, :py:obj:`~.cudaEventElapsedTime`, :py:obj:`~.cudaStreamWaitEvent`, :py:obj:`~.cuEventCreate`
    """

cudaEventDefault: int

def cudaEventDestroy(event):
    """
    cudaEventDestroy(event)
     Destroys an event object.

        Destroys the event specified by `event`.

        An event may be destroyed before it is complete (i.e., while
        :py:obj:`~.cudaEventQuery()` would return
        :py:obj:`~.cudaErrorNotReady`). In this case, the call does not block
        on completion of the event, and any associated resources will
        automatically be released asynchronously at completion.

        Parameters
        ----------
        event : :py:obj:`~.CUevent` or :py:obj:`~.cudaEvent_t`
            Event to destroy

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidResourceHandle`, :py:obj:`~.cudaErrorLaunchFailure`

        See Also
        --------
        :py:obj:`~.cudaEventCreate (C API)`, :py:obj:`~.cudaEventCreateWithFlags`, :py:obj:`~.cudaEventQuery`, :py:obj:`~.cudaEventSynchronize`, :py:obj:`~.cudaEventRecord`, :py:obj:`~.cudaEventElapsedTime`, :py:obj:`~.cuEventDestroy`
    """

cudaEventDisableTiming: int

def cudaEventElapsedTime(start, end):
    """
    cudaEventElapsedTime(start, end)
     Computes the elapsed time between events.

        Computes the elapsed time between two events (in milliseconds with a
        resolution of around 0.5 microseconds).

        If either event was last recorded in a non-NULL stream, the resulting
        time may be greater than expected (even if both used the same stream
        handle). This happens because the :py:obj:`~.cudaEventRecord()`
        operation takes place asynchronously and there is no guarantee that the
        measured latency is actually just between the two events. Any number of
        other different stream operations could execute in between the two
        measured events, thus altering the timing in a significant way.

        If :py:obj:`~.cudaEventRecord()` has not been called on either event,
        then :py:obj:`~.cudaErrorInvalidResourceHandle` is returned. If
        :py:obj:`~.cudaEventRecord()` has been called on both events but one or
        both of them has not yet been completed (that is,
        :py:obj:`~.cudaEventQuery()` would return :py:obj:`~.cudaErrorNotReady`
        on at least one of the events), :py:obj:`~.cudaErrorNotReady` is
        returned. If either event was created with the
        :py:obj:`~.cudaEventDisableTiming` flag, then this function will return
        :py:obj:`~.cudaErrorInvalidResourceHandle`.

        Parameters
        ----------
        start : :py:obj:`~.CUevent` or :py:obj:`~.cudaEvent_t`
            Starting event
        end : :py:obj:`~.CUevent` or :py:obj:`~.cudaEvent_t`
            Ending event

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorNotReady`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidResourceHandle`, :py:obj:`~.cudaErrorLaunchFailure`, :py:obj:`~.cudaErrorUnknown`
        ms : float
            Time between `start` and `end` in ms

        See Also
        --------
        :py:obj:`~.cudaEventCreate (C API)`, :py:obj:`~.cudaEventCreateWithFlags`, :py:obj:`~.cudaEventQuery`, :py:obj:`~.cudaEventSynchronize`, :py:obj:`~.cudaEventDestroy`, :py:obj:`~.cudaEventRecord`, :py:obj:`~.cuEventElapsedTime`
    """


def cudaEventElapsedTime_v2(start, end):
    """
    cudaEventElapsedTime_v2(start, end)
     Computes the elapsed time between events.

        Computes the elapsed time between two events (in milliseconds with a
        resolution of around 0.5 microseconds). Note this API is not guaranteed
        to return the latest errors for pending work. As such this API is
        intended to serve as a elapsed time calculation only and polling for
        completion on the events to be compared should be done with
        :py:obj:`~.cudaEventQuery` instead.

        If either event was last recorded in a non-NULL stream, the resulting
        time may be greater than expected (even if both used the same stream
        handle). This happens because the :py:obj:`~.cudaEventRecord()`
        operation takes place asynchronously and there is no guarantee that the
        measured latency is actually just between the two events. Any number of
        other different stream operations could execute in between the two
        measured events, thus altering the timing in a significant way.

        If :py:obj:`~.cudaEventRecord()` has not been called on either event,
        then :py:obj:`~.cudaErrorInvalidResourceHandle` is returned. If
        :py:obj:`~.cudaEventRecord()` has been called on both events but one or
        both of them has not yet been completed (that is,
        :py:obj:`~.cudaEventQuery()` would return :py:obj:`~.cudaErrorNotReady`
        on at least one of the events), :py:obj:`~.cudaErrorNotReady` is
        returned. If either event was created with the
        :py:obj:`~.cudaEventDisableTiming` flag, then this function will return
        :py:obj:`~.cudaErrorInvalidResourceHandle`.

        Parameters
        ----------
        start : :py:obj:`~.CUevent` or :py:obj:`~.cudaEvent_t`
            Starting event
        end : :py:obj:`~.CUevent` or :py:obj:`~.cudaEvent_t`
            Ending event

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorNotReady`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidResourceHandle`, :py:obj:`~.cudaErrorLaunchFailure`, :py:obj:`~.cudaErrorUnknown`
        ms : float
            Time between `start` and `end` in ms

        See Also
        --------
        :py:obj:`~.cudaEventCreate (C API)`, :py:obj:`~.cudaEventCreateWithFlags`, :py:obj:`~.cudaEventQuery`, :py:obj:`~.cudaEventSynchronize`, :py:obj:`~.cudaEventDestroy`, :py:obj:`~.cudaEventRecord`, :py:obj:`~.cuEventElapsedTime`
    """

cudaEventInterprocess: int

def cudaEventQuery(event):
    """
    cudaEventQuery(event)
     Queries an event's status.

        Queries the status of all work currently captured by `event`. See
        :py:obj:`~.cudaEventRecord()` for details on what is captured by an
        event.

        Returns :py:obj:`~.cudaSuccess` if all captured work has been
        completed, or :py:obj:`~.cudaErrorNotReady` if any captured work is
        incomplete.

        For the purposes of Unified Memory, a return value of
        :py:obj:`~.cudaSuccess` is equivalent to having called
        :py:obj:`~.cudaEventSynchronize()`.

        Parameters
        ----------
        event : :py:obj:`~.CUevent` or :py:obj:`~.cudaEvent_t`
            Event to query

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorNotReady`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidResourceHandle`, :py:obj:`~.cudaErrorLaunchFailure`

        See Also
        --------
        :py:obj:`~.cudaEventCreate (C API)`, :py:obj:`~.cudaEventCreateWithFlags`, :py:obj:`~.cudaEventRecord`, :py:obj:`~.cudaEventSynchronize`, :py:obj:`~.cudaEventDestroy`, :py:obj:`~.cudaEventElapsedTime`, :py:obj:`~.cuEventQuery`
    """


def cudaEventRecord(event, stream):
    """
    cudaEventRecord(event, stream)
     Records an event.

        Captures in `event` the contents of `stream` at the time of this call.
        `event` and `stream` must be on the same CUDA context. Calls such as
        :py:obj:`~.cudaEventQuery()` or :py:obj:`~.cudaStreamWaitEvent()` will
        then examine or wait for completion of the work that was captured. Uses
        of `stream` after this call do not modify `event`. See note on default
        stream behavior for what is captured in the default case.

        :py:obj:`~.cudaEventRecord()` can be called multiple times on the same
        event and will overwrite the previously captured state. Other APIs such
        as :py:obj:`~.cudaStreamWaitEvent()` use the most recently captured
        state at the time of the API call, and are not affected by later calls
        to :py:obj:`~.cudaEventRecord()`. Before the first call to
        :py:obj:`~.cudaEventRecord()`, an event represents an empty set of
        work, so for example :py:obj:`~.cudaEventQuery()` would return
        :py:obj:`~.cudaSuccess`.

        Parameters
        ----------
        event : :py:obj:`~.CUevent` or :py:obj:`~.cudaEvent_t`
            Event to record
        stream : :py:obj:`~.CUstream` or :py:obj:`~.cudaStream_t`
            Stream in which to record event

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidResourceHandle`, :py:obj:`~.cudaErrorLaunchFailure`

        See Also
        --------
        :py:obj:`~.cudaEventCreate (C API)`, :py:obj:`~.cudaEventCreateWithFlags`, :py:obj:`~.cudaEventQuery`, :py:obj:`~.cudaEventSynchronize`, :py:obj:`~.cudaEventDestroy`, :py:obj:`~.cudaEventElapsedTime`, :py:obj:`~.cudaStreamWaitEvent`, :py:obj:`~.cudaEventRecordWithFlags`, :py:obj:`~.cuEventRecord`
    """

cudaEventRecordDefault: int
cudaEventRecordExternal: int

def cudaEventRecordWithFlags(event, stream, flags):
    """
    cudaEventRecordWithFlags(event, stream, unsigned int flags)
     Records an event.

        Captures in `event` the contents of `stream` at the time of this call.
        `event` and `stream` must be on the same CUDA context. Calls such as
        :py:obj:`~.cudaEventQuery()` or :py:obj:`~.cudaStreamWaitEvent()` will
        then examine or wait for completion of the work that was captured. Uses
        of `stream` after this call do not modify `event`. See note on default
        stream behavior for what is captured in the default case.

        :py:obj:`~.cudaEventRecordWithFlags()` can be called multiple times on
        the same event and will overwrite the previously captured state. Other
        APIs such as :py:obj:`~.cudaStreamWaitEvent()` use the most recently
        captured state at the time of the API call, and are not affected by
        later calls to :py:obj:`~.cudaEventRecordWithFlags()`. Before the first
        call to :py:obj:`~.cudaEventRecordWithFlags()`, an event represents an
        empty set of work, so for example :py:obj:`~.cudaEventQuery()` would
        return :py:obj:`~.cudaSuccess`.

        flags include:

        - :py:obj:`~.cudaEventRecordDefault`: Default event creation flag.

        - :py:obj:`~.cudaEventRecordExternal`: Event is captured in the graph
          as an external event node when performing stream capture.

        Parameters
        ----------
        event : :py:obj:`~.CUevent` or :py:obj:`~.cudaEvent_t`
            Event to record
        stream : :py:obj:`~.CUstream` or :py:obj:`~.cudaStream_t`
            Stream in which to record event
        flags : unsigned int
            Parameters for the operation(See above)

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidResourceHandle`, :py:obj:`~.cudaErrorLaunchFailure`

        See Also
        --------
        :py:obj:`~.cudaEventCreate (C API)`, :py:obj:`~.cudaEventCreateWithFlags`, :py:obj:`~.cudaEventQuery`, :py:obj:`~.cudaEventSynchronize`, :py:obj:`~.cudaEventDestroy`, :py:obj:`~.cudaEventElapsedTime`, :py:obj:`~.cudaStreamWaitEvent`, :py:obj:`~.cudaEventRecord`, :py:obj:`~.cuEventRecord`,
    """


def cudaEventSynchronize(event):
    """
    cudaEventSynchronize(event)
     Waits for an event to complete.

        Waits until the completion of all work currently captured in `event`.
        See :py:obj:`~.cudaEventRecord()` for details on what is captured by an
        event.

        Waiting for an event that was created with the
        :py:obj:`~.cudaEventBlockingSync` flag will cause the calling CPU
        thread to block until the event has been completed by the device. If
        the :py:obj:`~.cudaEventBlockingSync` flag has not been set, then the
        CPU thread will busy-wait until the event has been completed by the
        device.

        Parameters
        ----------
        event : :py:obj:`~.CUevent` or :py:obj:`~.cudaEvent_t`
            Event to wait for

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidResourceHandle`, :py:obj:`~.cudaErrorLaunchFailure`

        See Also
        --------
        :py:obj:`~.cudaEventCreate (C API)`, :py:obj:`~.cudaEventCreateWithFlags`, :py:obj:`~.cudaEventRecord`, :py:obj:`~.cudaEventQuery`, :py:obj:`~.cudaEventDestroy`, :py:obj:`~.cudaEventElapsedTime`, :py:obj:`~.cuEventSynchronize`
    """

cudaEventWaitDefault: int
cudaEventWaitExternal: int
cudaExternalMemoryDedicated: int

def cudaExternalMemoryGetMappedBuffer(extMem, bufferDesc: 'Optional[cudaExternalMemoryBufferDesc]'):
    """
    cudaExternalMemoryGetMappedBuffer(extMem, cudaExternalMemoryBufferDesc bufferDesc: Optional[cudaExternalMemoryBufferDesc])
     Maps a buffer onto an imported memory object.

        Maps a buffer onto an imported memory object and returns a device
        pointer in `devPtr`.

        The properties of the buffer being mapped must be described in
        `bufferDesc`. The :py:obj:`~.cudaExternalMemoryBufferDesc` structure is
        defined as follows:

        **View CUDA Toolkit Documentation for a C++ code example**

        where :py:obj:`~.cudaExternalMemoryBufferDesc.offset` is the offset in
        the memory object where the buffer's base address is.
        :py:obj:`~.cudaExternalMemoryBufferDesc.size` is the size of the
        buffer. :py:obj:`~.cudaExternalMemoryBufferDesc.flags` must be zero.

        The offset and size have to be suitably aligned to match the
        requirements of the external API. Mapping two buffers whose ranges
        overlap may or may not result in the same virtual address being
        returned for the overlapped portion. In such cases, the application
        must ensure that all accesses to that region from the GPU are volatile.
        Otherwise writes made via one address are not guaranteed to be visible
        via the other address, even if they're issued by the same thread. It is
        recommended that applications map the combined range instead of mapping
        separate buffers and then apply the appropriate offsets to the returned
        pointer to derive the individual buffers.

        The returned pointer `devPtr` must be freed using :py:obj:`~.cudaFree`.

        Parameters
        ----------
        extMem : :py:obj:`~.cudaExternalMemory_t`
            Handle to external memory object
        bufferDesc : :py:obj:`~.cudaExternalMemoryBufferDesc`
            Buffer descriptor

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidResourceHandle`
        devPtr : Any
            Returned device pointer to buffer

        See Also
        --------
        :py:obj:`~.cudaImportExternalMemory`, :py:obj:`~.cudaDestroyExternalMemory`, :py:obj:`~.cudaExternalMemoryGetMappedMipmappedArray`
    """


def cudaExternalMemoryGetMappedMipmappedArray(extMem, mipmapDesc: 'Optional[cudaExternalMemoryMipmappedArrayDesc]'):
    """
    cudaExternalMemoryGetMappedMipmappedArray(extMem, cudaExternalMemoryMipmappedArrayDesc mipmapDesc: Optional[cudaExternalMemoryMipmappedArrayDesc])
     Maps a CUDA mipmapped array onto an external memory object.

        Maps a CUDA mipmapped array onto an external object and returns a
        handle to it in `mipmap`.

        The properties of the CUDA mipmapped array being mapped must be
        described in `mipmapDesc`. The structure
        :py:obj:`~.cudaExternalMemoryMipmappedArrayDesc` is defined as follows:

        **View CUDA Toolkit Documentation for a C++ code example**

        where :py:obj:`~.cudaExternalMemoryMipmappedArrayDesc.offset` is the
        offset in the memory object where the base level of the mipmap chain
        is. :py:obj:`~.cudaExternalMemoryMipmappedArrayDesc.formatDesc`
        describes the format of the data.
        :py:obj:`~.cudaExternalMemoryMipmappedArrayDesc.extent` specifies the
        dimensions of the base level of the mipmap chain.
        :py:obj:`~.cudaExternalMemoryMipmappedArrayDesc.flags` are flags
        associated with CUDA mipmapped arrays. For further details, please
        refer to the documentation for :py:obj:`~.cudaMalloc3DArray`. Note that
        if the mipmapped array is bound as a color target in the graphics API,
        then the flag :py:obj:`~.cudaArrayColorAttachment` must be specified in
        :py:obj:`~.cudaExternalMemoryMipmappedArrayDesc.flags`.
        :py:obj:`~.cudaExternalMemoryMipmappedArrayDesc.numLevels` specifies
        the total number of levels in the mipmap chain.

        The returned CUDA mipmapped array must be freed using
        :py:obj:`~.cudaFreeMipmappedArray`.

        Parameters
        ----------
        extMem : :py:obj:`~.cudaExternalMemory_t`
            Handle to external memory object
        mipmapDesc : :py:obj:`~.cudaExternalMemoryMipmappedArrayDesc`
            CUDA array descriptor

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidResourceHandle`
        mipmap : :py:obj:`~.cudaMipmappedArray_t`
            Returned CUDA mipmapped array

        See Also
        --------
        :py:obj:`~.cudaImportExternalMemory`, :py:obj:`~.cudaDestroyExternalMemory`, :py:obj:`~.cudaExternalMemoryGetMappedBuffer`

        Notes
        -----
        If :py:obj:`~.cudaExternalMemoryHandleDesc.type` is :py:obj:`~.cudaExternalMemoryHandleTypeNvSciBuf`, then :py:obj:`~.cudaExternalMemoryMipmappedArrayDesc.numLevels` must not be greater than 1.
    """

cudaExternalSemaphoreSignalSkipNvSciBufMemSync: int
cudaExternalSemaphoreWaitSkipNvSciBufMemSync: int

def cudaFree(devPtr):
    """
    cudaFree(devPtr)
     Frees memory on the device.

        Frees the memory space pointed to by `devPtr`, which must have been
        returned by a previous call to one of the following memory allocation
        APIs - :py:obj:`~.cudaMalloc()`, :py:obj:`~.cudaMallocPitch()`,
        :py:obj:`~.cudaMallocManaged()`, :py:obj:`~.cudaMallocAsync()`,
        :py:obj:`~.cudaMallocFromPoolAsync()`.

        Note - This API will not perform any implicit synchronization when the
        pointer was allocated with :py:obj:`~.cudaMallocAsync` or
        :py:obj:`~.cudaMallocFromPoolAsync`. Callers must ensure that all
        accesses to these pointer have completed before invoking
        :py:obj:`~.cudaFree`. For best performance and memory reuse, users
        should use :py:obj:`~.cudaFreeAsync` to free memory allocated via the
        stream ordered memory allocator. For all other pointers, this API may
        perform implicit synchronization.

        If :py:obj:`~.cudaFree`(`devPtr`) has already been called before, an
        error is returned. If `devPtr` is 0, no operation is performed.
        :py:obj:`~.cudaFree()` returns :py:obj:`~.cudaErrorValue` in case of
        failure.

        The device version of :py:obj:`~.cudaFree` cannot be used with a
        `*devPtr` allocated using the host API, and vice versa.

        Parameters
        ----------
        devPtr : Any
            Device pointer to memory to free

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`

        See Also
        --------
        :py:obj:`~.cudaMalloc`, :py:obj:`~.cudaMallocPitch`, :py:obj:`~.cudaMallocManaged`, :py:obj:`~.cudaMallocArray`, :py:obj:`~.cudaFreeArray`, :py:obj:`~.cudaMallocAsync`, :py:obj:`~.cudaMallocFromPoolAsync` :py:obj:`~.cudaMallocHost (C API)`, :py:obj:`~.cudaFreeHost`, :py:obj:`~.cudaMalloc3D`, :py:obj:`~.cudaMalloc3DArray`, :py:obj:`~.cudaFreeAsync` :py:obj:`~.cudaHostAlloc`, :py:obj:`~.cuMemFree`
    """


def cudaFreeArray(array):
    """
    cudaFreeArray(array)
     Frees an array on the device.

        Frees the CUDA array `array`, which must have been returned by a
        previous call to :py:obj:`~.cudaMallocArray()`. If `devPtr` is 0, no
        operation is performed.

        Parameters
        ----------
        array : :py:obj:`~.cudaArray_t`
            Pointer to array to free

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`

        See Also
        --------
        :py:obj:`~.cudaMalloc`, :py:obj:`~.cudaMallocPitch`, :py:obj:`~.cudaFree`, :py:obj:`~.cudaMallocArray`, :py:obj:`~.cudaMallocHost (C API)`, :py:obj:`~.cudaFreeHost`, :py:obj:`~.cudaHostAlloc`, :py:obj:`~.cuArrayDestroy`
    """


def cudaFreeAsync(devPtr, hStream):
    """
    cudaFreeAsync(devPtr, hStream)
     Frees memory with stream ordered semantics.

        Inserts a free operation into `hStream`. The allocation must not be
        accessed after stream execution reaches the free. After this API
        returns, accessing the memory from any subsequent work launched on the
        GPU or querying its pointer attributes results in undefined behavior.

        Parameters
        ----------
        dptr : Any
            memory to free
        hStream : :py:obj:`~.CUstream` or :py:obj:`~.cudaStream_t`
            The stream establishing the stream ordering promise

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorNotSupported`

        See Also
        --------
        :py:obj:`~.cuMemFreeAsync`, :py:obj:`~.cudaMallocAsync`

        Notes
        -----
        During stream capture, this function results in the creation of a free node and must therefore be passed the address of a graph allocation.
    """


def cudaFreeHost(ptr):
    """
    cudaFreeHost(ptr)
     Frees page-locked memory.

        Frees the memory space pointed to by `hostPtr`, which must have been
        returned by a previous call to :py:obj:`~.cudaMallocHost()` or
        :py:obj:`~.cudaHostAlloc()`.

        Parameters
        ----------
        ptr : Any
            Pointer to memory to free

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`

        See Also
        --------
        :py:obj:`~.cudaMalloc`, :py:obj:`~.cudaMallocPitch`, :py:obj:`~.cudaFree`, :py:obj:`~.cudaMallocArray`, :py:obj:`~.cudaFreeArray`, :py:obj:`~.cudaMallocHost (C API)`, :py:obj:`~.cudaMalloc3D`, :py:obj:`~.cudaMalloc3DArray`, :py:obj:`~.cudaHostAlloc`, :py:obj:`~.cuMemFreeHost`
    """


def cudaFreeMipmappedArray(mipmappedArray):
    """
    cudaFreeMipmappedArray(mipmappedArray)
     Frees a mipmapped array on the device.

        Frees the CUDA mipmapped array `mipmappedArray`, which must have been
        returned by a previous call to :py:obj:`~.cudaMallocMipmappedArray()`.
        If `devPtr` is 0, no operation is performed.

        Parameters
        ----------
        mipmappedArray : :py:obj:`~.cudaMipmappedArray_t`
            Pointer to mipmapped array to free

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`

        See Also
        --------
        :py:obj:`~.cudaMalloc`, :py:obj:`~.cudaMallocPitch`, :py:obj:`~.cudaFree`, :py:obj:`~.cudaMallocArray`, :py:obj:`~.cudaMallocHost (C API)`, :py:obj:`~.cudaFreeHost`, :py:obj:`~.cudaHostAlloc`, :py:obj:`~.cuMipmappedArrayDestroy`
    """


def cudaFuncGetAttributes(func):
    """
    cudaFuncGetAttributes(func)
     Find out attributes for a given function.

        This function obtains the attributes of a function specified via
        `func`. `func` is a device function symbol and must be declared as a
        `None` function. The fetched attributes are placed in `attr`. If the
        specified function does not exist, then it is assumed to be a
        :py:obj:`~.cudaKernel_t` and used as is. For templated functions, pass
        the function symbol as follows:
        func_name<template_arg_0,...,template_arg_N>

        Note that some function attributes such as
        :py:obj:`~.maxThreadsPerBlock` may vary based on the device that is
        currently being used.

        Parameters
        ----------
        func : Any
            Device function symbol

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidDeviceFunction`2
        attr : :py:obj:`~.cudaFuncAttributes`
            Return pointer to function's attributes

        See Also
        --------
        :py:obj:`~.cudaFuncSetCacheConfig (C API)`, cudaFuncGetAttributes (C++ API), :py:obj:`~.cudaLaunchKernel (C API)`, :py:obj:`~.cuFuncGetAttribute`
    """


def cudaFuncSetAttribute(func, attr: 'cudaFuncAttribute', value):
    """
    cudaFuncSetAttribute(func, attr: cudaFuncAttribute, int value)
     Set attributes for a given function.

        This function sets the attributes of a function specified via `func`.
        The parameter `func` must be a pointer to a function that executes on
        the device. The parameter specified by `func` must be declared as a
        `None` function. The enumeration defined by `attr` is set to the value
        defined by `value`. If the specified function does not exist, then it
        is assumed to be a :py:obj:`~.cudaKernel_t` and used as is. If the
        specified attribute cannot be written, or if the value is incorrect,
        then :py:obj:`~.cudaErrorInvalidValue` is returned.

        Valid values for `attr` are:

        - :py:obj:`~.cudaFuncAttributeMaxDynamicSharedMemorySize` - The
          requested maximum size in bytes of dynamically-allocated shared
          memory. The sum of this value and the function attribute
          :py:obj:`~.sharedSizeBytes` cannot exceed the device attribute
          :py:obj:`~.cudaDevAttrMaxSharedMemoryPerBlockOptin`. The maximal size
          of requestable dynamic shared memory may differ by GPU architecture.

        - :py:obj:`~.cudaFuncAttributePreferredSharedMemoryCarveout` - On
          devices where the L1 cache and shared memory use the same hardware
          resources, this sets the shared memory carveout preference, in
          percent of the total shared memory. See
          :py:obj:`~.cudaDevAttrMaxSharedMemoryPerMultiprocessor`. This is only
          a hint, and the driver can choose a different ratio if required to
          execute the function.

        - :py:obj:`~.cudaFuncAttributeRequiredClusterWidth`: The required
          cluster width in blocks. The width, height, and depth values must
          either all be 0 or all be positive. The validity of the cluster
          dimensions is checked at launch time. If the value is set during
          compile time, it cannot be set at runtime. Setting it at runtime will
          return cudaErrorNotPermitted.

        - :py:obj:`~.cudaFuncAttributeRequiredClusterHeight`: The required
          cluster height in blocks. The width, height, and depth values must
          either all be 0 or all be positive. The validity of the cluster
          dimensions is checked at launch time. If the value is set during
          compile time, it cannot be set at runtime. Setting it at runtime will
          return cudaErrorNotPermitted.

        - :py:obj:`~.cudaFuncAttributeRequiredClusterDepth`: The required
          cluster depth in blocks. The width, height, and depth values must
          either all be 0 or all be positive. The validity of the cluster
          dimensions is checked at launch time. If the value is set during
          compile time, it cannot be set at runtime. Setting it at runtime will
          return cudaErrorNotPermitted.

        - :py:obj:`~.cudaFuncAttributeNonPortableClusterSizeAllowed`: Indicates
          whether the function can be launched with non-portable cluster size.
          1 is allowed, 0 is disallowed.

        - :py:obj:`~.cudaFuncAttributeClusterSchedulingPolicyPreference`: The
          block scheduling policy of a function. The value type is
          cudaClusterSchedulingPolicy.

        cudaLaunchKernel (C++ API), cudaFuncSetCacheConfig (C++ API),
        :py:obj:`~.cudaFuncGetAttributes (C API)`,

        Parameters
        ----------
        func : Any
            Function to get attributes of
        attr : :py:obj:`~.cudaFuncAttribute`
            Attribute to set
        value : int
            Value to set

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidDeviceFunction`, :py:obj:`~.cudaErrorInvalidValue`
    """


def cudaFuncSetCacheConfig(func, cacheConfig: 'cudaFuncCache'):
    """
    cudaFuncSetCacheConfig(func, cacheConfig: cudaFuncCache)
     Sets the preferred cache configuration for a device function.

        On devices where the L1 cache and shared memory use the same hardware
        resources, this sets through `cacheConfig` the preferred cache
        configuration for the function specified via `func`. This is only a
        preference. The runtime will use the requested configuration if
        possible, but it is free to choose a different configuration if
        required to execute `func`.

        `func` is a device function symbol and must be declared as a `None`
        function. If the specified function does not exist, then
        :py:obj:`~.cudaErrorInvalidDeviceFunction` is returned. For templated
        functions, pass the function symbol as follows:
        func_name<template_arg_0,...,template_arg_N>

        This setting does nothing on devices where the size of the L1 cache and
        shared memory are fixed.

        Launching a kernel with a different preference than the most recent
        preference setting may insert a device-side synchronization point.

        The supported cache configurations are:

        - :py:obj:`~.cudaFuncCachePreferNone`: no preference for shared memory
          or L1 (default)

        - :py:obj:`~.cudaFuncCachePreferShared`: prefer larger shared memory
          and smaller L1 cache

        - :py:obj:`~.cudaFuncCachePreferL1`: prefer larger L1 cache and smaller
          shared memory

        - :py:obj:`~.cudaFuncCachePreferEqual`: prefer equal size L1 cache and
          shared memory

        Parameters
        ----------
        func : Any
            Device function symbol
        cacheConfig : :py:obj:`~.cudaFuncCache`
            Requested cache configuration

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidDeviceFunction`2

        See Also
        --------
        cudaFuncSetCacheConfig (C++ API), :py:obj:`~.cudaFuncGetAttributes (C API)`, :py:obj:`~.cudaLaunchKernel (C API)`, :py:obj:`~.cuFuncSetCacheConfig`

        Notes
        -----
        This API does not accept a :py:obj:`~.cudaKernel_t` casted as void*. If cache config modification is required for a :py:obj:`~.cudaKernel_t` (or a global function), it can be replaced with a call to :py:obj:`~.cudaFuncSetAttributes` with the attribute :py:obj:`~.cudaFuncAttributePreferredSharedMemoryCarveout` to specify a more granular L1 cache and shared memory split configuration.
    """


def cudaFuncSetSharedMemConfig(func, config: 'cudaSharedMemConfig'):
    """
    cudaFuncSetSharedMemConfig(func, config: cudaSharedMemConfig)
     Sets the shared memory configuration for a device function.

        [Deprecated]

        On devices with configurable shared memory banks, this function will
        force all subsequent launches of the specified device function to have
        the given shared memory bank size configuration. On any given launch of
        the function, the shared memory configuration of the device will be
        temporarily changed if needed to suit the function's preferred
        configuration. Changes in shared memory configuration between
        subsequent launches of functions, may introduce a device side
        synchronization point.

        Any per-function setting of shared memory bank size set via
        :py:obj:`~.cudaFuncSetSharedMemConfig` will override the device wide
        setting set by :py:obj:`~.cudaDeviceSetSharedMemConfig`.

        Changing the shared memory bank size will not increase shared memory
        usage or affect occupancy of kernels, but may have major effects on
        performance. Larger bank sizes will allow for greater potential
        bandwidth to shared memory, but will change what kinds of accesses to
        shared memory will result in bank conflicts.

        This function will do nothing on devices with fixed shared memory bank
        size.

        For templated functions, pass the function symbol as follows:
        func_name<template_arg_0,...,template_arg_N>

        The supported bank configurations are:

        - :py:obj:`~.cudaSharedMemBankSizeDefault`: use the device's shared
          memory configuration when launching this function.

        - :py:obj:`~.cudaSharedMemBankSizeFourByte`: set shared memory bank
          width to be four bytes natively when launching this function.

        - :py:obj:`~.cudaSharedMemBankSizeEightByte`: set shared memory bank
          width to be eight bytes natively when launching this function.

        Parameters
        ----------
        func : Any
            Device function symbol
        config : :py:obj:`~.cudaSharedMemConfig`
            Requested shared memory configuration

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidDeviceFunction`, :py:obj:`~.cudaErrorInvalidValue`,2

        See Also
        --------
        :py:obj:`~.cudaDeviceSetSharedMemConfig`, :py:obj:`~.cudaDeviceGetSharedMemConfig`, :py:obj:`~.cudaDeviceSetCacheConfig`, :py:obj:`~.cudaDeviceGetCacheConfig`, :py:obj:`~.cudaFuncSetCacheConfig`, :py:obj:`~.cuFuncSetSharedMemConfig`
    """


def cudaGLGetDevices(cudaDeviceCount, deviceList: 'cudaGLDeviceList'):
    """
    cudaGLGetDevices(unsigned int cudaDeviceCount, deviceList: cudaGLDeviceList)
     Gets the CUDA devices associated with the current OpenGL context.

        Returns in `*pCudaDeviceCount` the number of CUDA-compatible devices
        corresponding to the current OpenGL context. Also returns in
        `*pCudaDevices` at most `cudaDeviceCount` of the CUDA-compatible
        devices corresponding to the current OpenGL context. If any of the GPUs
        being used by the current OpenGL context are not CUDA capable then the
        call will return cudaErrorNoDevice.

        Parameters
        ----------
        cudaDeviceCount : unsigned int
            The size of the output device array `pCudaDevices`
        deviceList : cudaGLDeviceList
            The set of devices to return. This set may be cudaGLDeviceListAll
            for all devices, cudaGLDeviceListCurrentFrame for the devices used
            to render the current frame (in SLI), or cudaGLDeviceListNextFrame
            for the devices used to render the next frame (in SLI).

        Returns
        -------
        cudaError_t
            cudaSuccess
            cudaErrorNoDevice
            cudaErrorInvalidGraphicsContext
            cudaErrorUnknown
        pCudaDeviceCount : unsigned int
            Returned number of CUDA devices corresponding to the current OpenGL
            context
        pCudaDevices : List[int]
            Returned CUDA devices corresponding to the current OpenGL context

        See Also
        --------
        ~.cudaGraphicsUnregisterResource
        ~.cudaGraphicsMapResources
        ~.cudaGraphicsSubResourceGetMappedArray
        ~.cudaGraphicsResourceGetMappedPointer
        ~.cuGLGetDevices

        Notes
        -----
        This function is not supported on Mac OS X.
    """


def cudaGetChannelDesc(array):
    """
    cudaGetChannelDesc(array)
     Get the channel descriptor of an array.

        Returns in `*desc` the channel descriptor of the CUDA array `array`.

        Parameters
        ----------
        array : :py:obj:`~.cudaArray_const_t`
            Memory array on device

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`
        desc : :py:obj:`~.cudaChannelFormatDesc`
            Channel format

        See Also
        --------
        :py:obj:`~.cudaCreateChannelDesc (C API)`, :py:obj:`~.cudaCreateTextureObject`, :py:obj:`~.cudaCreateSurfaceObject`
    """


def cudaGetDevice():
    """
    cudaGetDevice()
     Returns which device is currently being used.

        Returns in `*device` the current device for the calling host thread.

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorDeviceUnavailable`,
        device : int
            Returns the device on which the active host thread executes the
            device code.

        See Also
        --------
        :py:obj:`~.cudaGetDeviceCount`, :py:obj:`~.cudaSetDevice`, :py:obj:`~.cudaGetDeviceProperties`, :py:obj:`~.cudaChooseDevice`, :py:obj:`~.cuCtxGetCurrent`
    """


def cudaGetDeviceCount():
    """
    cudaGetDeviceCount()
     Returns the number of compute-capable devices.

        Returns in `*count` the number of devices with compute capability
        greater or equal to 2.0 that are available for execution.

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`
        count : int
            Returns the number of devices with compute capability greater or
            equal to 2.0

        See Also
        --------
        :py:obj:`~.cudaGetDevice`, :py:obj:`~.cudaSetDevice`, :py:obj:`~.cudaGetDeviceProperties`, :py:obj:`~.cudaChooseDevice`, :py:obj:`~.cudaInitDevice`, :py:obj:`~.cuDeviceGetCount`
    """


def cudaGetDeviceFlags():
    """
    cudaGetDeviceFlags()
     Gets the flags for the current device.

        Returns in `flags` the flags for the current device. If there is a
        current device for the calling thread, the flags for the device are
        returned. If there is no current device, the flags for the first device
        are returned, which may be the default flags. Compare to the behavior
        of :py:obj:`~.cudaSetDeviceFlags`.

        Typically, the flags returned should match the behavior that will be
        seen if the calling thread uses a device after this call, without any
        change to the flags or current device inbetween by this or another
        thread. Note that if the device is not initialized, it is possible for
        another thread to change the flags for the current device before it is
        initialized. Additionally, when using exclusive mode, if this thread
        has not requested a specific device, it may use a device other than the
        first device, contrary to the assumption made by this function.

        If a context has been created via the driver API and is current to the
        calling thread, the flags for that context are always returned.

        Flags returned by this function may specifically include
        :py:obj:`~.cudaDeviceMapHost` even though it is not accepted by
        :py:obj:`~.cudaSetDeviceFlags` because it is implicit in runtime API
        flags. The reason for this is that the current context may have been
        created via the driver API in which case the flag is not implicit and
        may be unset.

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidDevice`
        flags : unsigned int
            Pointer to store the device flags

        See Also
        --------
        :py:obj:`~.cudaGetDevice`, :py:obj:`~.cudaGetDeviceProperties`, :py:obj:`~.cudaSetDevice`, :py:obj:`~.cudaSetDeviceFlags`, :py:obj:`~.cudaInitDevice`, :py:obj:`~.cuCtxGetFlags`, :py:obj:`~.cuDevicePrimaryCtxGetState`
    """


def cudaGetDeviceProperties(device):
    """
    cudaGetDeviceProperties(int device)
     Returns information about the compute-device.

        Returns in `*prop` the properties of device `dev`. The
        :py:obj:`~.cudaDeviceProp` structure is defined as:

        **View CUDA Toolkit Documentation for a C++ code example**

        where:

        - :py:obj:`~.name[256]` is an ASCII string identifying the device.

        - :py:obj:`~.uuid` is a 16-byte unique identifier.

        - :py:obj:`~.totalGlobalMem` is the total amount of global memory
          available on the device in bytes.

        - :py:obj:`~.sharedMemPerBlock` is the maximum amount of shared memory
          available to a thread block in bytes.

        - :py:obj:`~.regsPerBlock` is the maximum number of 32-bit registers
          available to a thread block.

        - :py:obj:`~.warpSize` is the warp size in threads.

        - :py:obj:`~.memPitch` is the maximum pitch in bytes allowed by the
          memory copy functions that involve memory regions allocated through
          :py:obj:`~.cudaMallocPitch()`.

        - :py:obj:`~.maxThreadsPerBlock` is the maximum number of threads per
          block.

        - :py:obj:`~.maxThreadsDim[3]` contains the maximum size of each
          dimension of a block.

        - :py:obj:`~.maxGridSize[3]` contains the maximum size of each
          dimension of a grid.

        - :py:obj:`~.clockRate` is the clock frequency in kilohertz.

        - :py:obj:`~.totalConstMem` is the total amount of constant memory
          available on the device in bytes.

        - :py:obj:`~.major`, :py:obj:`~.minor` are the major and minor revision
          numbers defining the device's compute capability.

        - :py:obj:`~.textureAlignment` is the alignment requirement; texture
          base addresses that are aligned to :py:obj:`~.textureAlignment` bytes
          do not need an offset applied to texture fetches.

        - :py:obj:`~.texturePitchAlignment` is the pitch alignment requirement
          for 2D texture references that are bound to pitched memory.

        - :py:obj:`~.deviceOverlap` is 1 if the device can concurrently copy
          memory between host and device while executing a kernel, or 0 if not.
          Deprecated, use instead asyncEngineCount.

        - :py:obj:`~.multiProcessorCount` is the number of multiprocessors on
          the device.

        - :py:obj:`~.kernelExecTimeoutEnabled` is 1 if there is a run time
          limit for kernels executed on the device, or 0 if not.

        - :py:obj:`~.integrated` is 1 if the device is an integrated
          (motherboard) GPU and 0 if it is a discrete (card) component.

        - :py:obj:`~.canMapHostMemory` is 1 if the device can map host memory
          into the CUDA address space for use with
          :py:obj:`~.cudaHostAlloc()`/:py:obj:`~.cudaHostGetDevicePointer()`,
          or 0 if not.

        - :py:obj:`~.computeMode` is the compute mode that the device is
          currently in. Available modes are as follows:

          - cudaComputeModeDefault: Default mode - Device is not restricted and
            multiple threads can use :py:obj:`~.cudaSetDevice()` with this
            device.

          - cudaComputeModeProhibited: Compute-prohibited mode - No threads can
            use :py:obj:`~.cudaSetDevice()` with this device.

          - cudaComputeModeExclusiveProcess: Compute-exclusive-process mode -
            Many threads in one process will be able to use
            :py:obj:`~.cudaSetDevice()` with this device.   When an occupied
            exclusive mode device is chosen with :py:obj:`~.cudaSetDevice`, all
            subsequent non-device management runtime functions will return
            :py:obj:`~.cudaErrorDevicesUnavailable`.

        - :py:obj:`~.maxTexture1D` is the maximum 1D texture size.

        - :py:obj:`~.maxTexture1DMipmap` is the maximum 1D mipmapped texture
          texture size.

        - :py:obj:`~.maxTexture1DLinear` is the maximum 1D texture size for
          textures bound to linear memory.

        - :py:obj:`~.maxTexture2D[2]` contains the maximum 2D texture
          dimensions.

        - :py:obj:`~.maxTexture2DMipmap[2]` contains the maximum 2D mipmapped
          texture dimensions.

        - :py:obj:`~.maxTexture2DLinear[3]` contains the maximum 2D texture
          dimensions for 2D textures bound to pitch linear memory.

        - :py:obj:`~.maxTexture2DGather[2]` contains the maximum 2D texture
          dimensions if texture gather operations have to be performed.

        - :py:obj:`~.maxTexture3D[3]` contains the maximum 3D texture
          dimensions.

        - :py:obj:`~.maxTexture3DAlt[3]` contains the maximum alternate 3D
          texture dimensions.

        - :py:obj:`~.maxTextureCubemap` is the maximum cubemap texture width or
          height.

        - :py:obj:`~.maxTexture1DLayered[2]` contains the maximum 1D layered
          texture dimensions.

        - :py:obj:`~.maxTexture2DLayered[3]` contains the maximum 2D layered
          texture dimensions.

        - :py:obj:`~.maxTextureCubemapLayered[2]` contains the maximum cubemap
          layered texture dimensions.

        - :py:obj:`~.maxSurface1D` is the maximum 1D surface size.

        - :py:obj:`~.maxSurface2D[2]` contains the maximum 2D surface
          dimensions.

        - :py:obj:`~.maxSurface3D[3]` contains the maximum 3D surface
          dimensions.

        - :py:obj:`~.maxSurface1DLayered[2]` contains the maximum 1D layered
          surface dimensions.

        - :py:obj:`~.maxSurface2DLayered[3]` contains the maximum 2D layered
          surface dimensions.

        - :py:obj:`~.maxSurfaceCubemap` is the maximum cubemap surface width or
          height.

        - :py:obj:`~.maxSurfaceCubemapLayered[2]` contains the maximum cubemap
          layered surface dimensions.

        - :py:obj:`~.surfaceAlignment` specifies the alignment requirements for
          surfaces.

        - :py:obj:`~.concurrentKernels` is 1 if the device supports executing
          multiple kernels within the same context simultaneously, or 0 if not.
          It is not guaranteed that multiple kernels will be resident on the
          device concurrently so this feature should not be relied upon for
          correctness.

        - :py:obj:`~.ECCEnabled` is 1 if the device has ECC support turned on,
          or 0 if not.

        - :py:obj:`~.pciBusID` is the PCI bus identifier of the device.

        - :py:obj:`~.pciDeviceID` is the PCI device (sometimes called slot)
          identifier of the device.

        - :py:obj:`~.pciDomainID` is the PCI domain identifier of the device.

        - :py:obj:`~.tccDriver` is 1 if the device is using a TCC driver or 0
          if not.

        - :py:obj:`~.asyncEngineCount` is 1 when the device can concurrently
          copy memory between host and device while executing a kernel. It is 2
          when the device can concurrently copy memory between host and device
          in both directions and execute a kernel at the same time. It is 0 if
          neither of these is supported.

        - :py:obj:`~.unifiedAddressing` is 1 if the device shares a unified
          address space with the host and 0 otherwise.

        - :py:obj:`~.memoryClockRate` is the peak memory clock frequency in
          kilohertz.

        - :py:obj:`~.memoryBusWidth` is the memory bus width   in bits.

        - :py:obj:`~.l2CacheSize` is L2 cache size in bytes.

        - :py:obj:`~.persistingL2CacheMaxSize` is L2 cache's maximum persisting
          lines size in bytes.

        - :py:obj:`~.maxThreadsPerMultiProcessor`   is the number of maximum
          resident threads per multiprocessor.

        - :py:obj:`~.streamPrioritiesSupported` is 1 if the device supports
          stream priorities, or 0 if it is not supported.

        - :py:obj:`~.globalL1CacheSupported` is 1 if the device supports
          caching of globals in L1 cache, or 0 if it is not supported.

        - :py:obj:`~.localL1CacheSupported` is 1 if the device supports caching
          of locals in L1 cache, or 0 if it is not supported.

        - :py:obj:`~.sharedMemPerMultiprocessor` is the maximum amount of
          shared memory available to a multiprocessor in bytes; this amount is
          shared by all thread blocks simultaneously resident on a
          multiprocessor.

        - :py:obj:`~.regsPerMultiprocessor` is the maximum number of 32-bit
          registers available to a multiprocessor; this number is shared by all
          thread blocks simultaneously resident on a multiprocessor.

        - :py:obj:`~.managedMemory` is 1 if the device supports allocating
          managed memory on this system, or 0 if it is not supported.

        - :py:obj:`~.isMultiGpuBoard` is 1 if the device is on a multi-GPU
          board (e.g. Gemini cards), and 0 if not;

        - :py:obj:`~.multiGpuBoardGroupID` is a unique identifier for a group
          of devices associated with the same board. Devices on the same multi-
          GPU board will share the same identifier.

        - :py:obj:`~.hostNativeAtomicSupported` is 1 if the link between the
          device and the host supports native atomic operations, or 0 if it is
          not supported.

        - :py:obj:`~.singleToDoublePrecisionPerfRatio`   is the ratio of single
          precision performance (in floating-point operations per second) to
          double precision performance.

        - :py:obj:`~.pageableMemoryAccess` is 1 if the device supports
          coherently accessing pageable memory without calling cudaHostRegister
          on it, and 0 otherwise.

        - :py:obj:`~.concurrentManagedAccess` is 1 if the device can coherently
          access managed memory concurrently with the CPU, and 0 otherwise.

        - :py:obj:`~.computePreemptionSupported` is 1 if the device supports
          Compute Preemption, and 0 otherwise.

        - :py:obj:`~.canUseHostPointerForRegisteredMem` is 1 if the device can
          access host registered memory at the same virtual address as the CPU,
          and 0 otherwise.

        - :py:obj:`~.cooperativeLaunch` is 1 if the device supports launching
          cooperative kernels via :py:obj:`~.cudaLaunchCooperativeKernel`, and
          0 otherwise.

        - :py:obj:`~.cooperativeMultiDeviceLaunch` is 1 if the device supports
          launching cooperative kernels via
          :py:obj:`~.cudaLaunchCooperativeKernelMultiDevice`, and 0 otherwise.

        - :py:obj:`~.sharedMemPerBlockOptin` is the per device maximum shared
          memory per block usable by special opt in

        - :py:obj:`~.pageableMemoryAccessUsesHostPageTables` is 1 if the device
          accesses pageable memory via the host's page tables, and 0 otherwise.

        - :py:obj:`~.directManagedMemAccessFromHost` is 1 if the host can
          directly access managed memory on the device without migration, and 0
          otherwise.

        - :py:obj:`~.maxBlocksPerMultiProcessor` is the maximum number of
          thread blocks that can reside on a multiprocessor.

        - :py:obj:`~.accessPolicyMaxWindowSize` is the maximum value of
          :py:obj:`~.cudaAccessPolicyWindow.num_bytes`.

        - :py:obj:`~.reservedSharedMemPerBlock` is the shared memory reserved
          by CUDA driver per block in bytes

        - :py:obj:`~.hostRegisterSupported` is 1 if the device supports host
          memory registration via :py:obj:`~.cudaHostRegister`, and 0
          otherwise.

        - :py:obj:`~.sparseCudaArraySupported` is 1 if the device supports
          sparse CUDA arrays and sparse CUDA mipmapped arrays, 0 otherwise

        - :py:obj:`~.hostRegisterReadOnlySupported` is 1 if the device supports
          using the :py:obj:`~.cudaHostRegister` flag cudaHostRegisterReadOnly
          to register memory that must be mapped as read-only to the GPU

        - :py:obj:`~.timelineSemaphoreInteropSupported` is 1 if external
          timeline semaphore interop is supported on the device, 0 otherwise

        - :py:obj:`~.memoryPoolsSupported` is 1 if the device supports using
          the cudaMallocAsync and cudaMemPool family of APIs, 0 otherwise

        - :py:obj:`~.gpuDirectRDMASupported` is 1 if the device supports
          GPUDirect RDMA APIs, 0 otherwise

        - :py:obj:`~.gpuDirectRDMAFlushWritesOptions` is a bitmask to be
          interpreted according to the
          :py:obj:`~.cudaFlushGPUDirectRDMAWritesOptions` enum

        - :py:obj:`~.gpuDirectRDMAWritesOrdering` See the
          :py:obj:`~.cudaGPUDirectRDMAWritesOrdering` enum for numerical values

        - :py:obj:`~.memoryPoolSupportedHandleTypes` is a bitmask of handle
          types supported with mempool-based IPC

        - :py:obj:`~.deferredMappingCudaArraySupported` is 1 if the device
          supports deferred mapping CUDA arrays and CUDA mipmapped arrays

        - :py:obj:`~.ipcEventSupported` is 1 if the device supports IPC Events,
          and 0 otherwise

        - :py:obj:`~.unifiedFunctionPointers` is 1 if the device support
          unified pointers, and 0 otherwise

        Parameters
        ----------
        device : int
            None

        Returns
        -------
        cudaError_t

        prop : :py:obj:`~.cudaDeviceProp`
            None
    """


def cudaGetDriverEntryPoint(symbol, flags):
    """
    cudaGetDriverEntryPoint(char *symbol, unsigned long long flags)
     Returns the requested driver API function pointer.

        Returns in `**funcPtr` the address of the CUDA driver function for the
        requested flags.

        For a requested driver symbol, if the CUDA version in which the driver
        symbol was introduced is less than or equal to the CUDA runtime
        version, the API will return the function pointer to the corresponding
        versioned driver function.

        The pointer returned by the API should be cast to a function pointer
        matching the requested driver function's definition in the API header
        file. The function pointer typedef can be picked up from the
        corresponding typedefs header file. For example, cudaTypedefs.h
        consists of function pointer typedefs for driver APIs defined in
        cuda.h.

        The API will return :py:obj:`~.cudaSuccess` and set the returned
        `funcPtr` if the requested driver function is valid and supported on
        the platform.

        The API will return :py:obj:`~.cudaSuccess` and set the returned
        `funcPtr` to NULL if the requested driver function is not supported on
        the platform, no ABI compatible driver function exists for the CUDA
        runtime version or if the driver symbol is invalid.

        It will also set the optional `driverStatus` to one of the values in
        :py:obj:`~.cudaDriverEntryPointQueryResult` with the following
        meanings:

        - :py:obj:`~.cudaDriverEntryPointSuccess` - The requested symbol was
          succesfully found based on input arguments and `pfn` is valid

        - :py:obj:`~.cudaDriverEntryPointSymbolNotFound` - The requested symbol
          was not found

        - :py:obj:`~.cudaDriverEntryPointVersionNotSufficent` - The requested
          symbol was found but is not supported by the current runtime version
          (CUDART_VERSION)

        The requested flags can be:

        - :py:obj:`~.cudaEnableDefault`: This is the default mode. This is
          equivalent to :py:obj:`~.cudaEnablePerThreadDefaultStream` if the
          code is compiled with --default-stream per-thread compilation flag or
          the macro CUDA_API_PER_THREAD_DEFAULT_STREAM is defined;
          :py:obj:`~.cudaEnableLegacyStream` otherwise.

        - :py:obj:`~.cudaEnableLegacyStream`: This will enable the search for
          all driver symbols that match the requested driver symbol name except
          the corresponding per-thread versions.

        - :py:obj:`~.cudaEnablePerThreadDefaultStream`: This will enable the
          search for all driver symbols that match the requested driver symbol
          name including the per-thread versions. If a per-thread version is
          not found, the API will return the legacy version of the driver
          function.

        Parameters
        ----------
        symbol : bytes
            The base name of the driver API function to look for. As an
            example, for the driver API :py:obj:`~.cuMemAlloc_v2`, `symbol`
            would be cuMemAlloc. Note that the API will use the CUDA runtime
            version to return the address to the most recent ABI compatible
            driver symbol, :py:obj:`~.cuMemAlloc` or :py:obj:`~.cuMemAlloc_v2`.
        flags : unsigned long long
            Flags to specify search options.

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorNotSupported`
        funcPtr : Any
            Location to return the function pointer to the requested driver
            function
        driverStatus : :py:obj:`~.cudaDriverEntryPointQueryResult`
            Optional location to store the status of finding the symbol from
            the driver. See :py:obj:`~.cudaDriverEntryPointQueryResult` for
            possible values.

        See Also
        --------
        :py:obj:`~.cuGetProcAddress`
    """


def cudaGetDriverEntryPointByVersion(symbol, cudaVersion, flags):
    """
    cudaGetDriverEntryPointByVersion(char *symbol, unsigned int cudaVersion, unsigned long long flags)
     Returns the requested driver API function pointer by CUDA version.

        Returns in `**funcPtr` the address of the CUDA driver function for the
        requested flags and CUDA driver version.

        The CUDA version is specified as (1000 * major + 10 * minor), so CUDA
        11.2 should be specified as 11020. For a requested driver symbol, if
        the specified CUDA version is greater than or equal to the CUDA version
        in which the driver symbol was introduced, this API will return the
        function pointer to the corresponding versioned function.

        The pointer returned by the API should be cast to a function pointer
        matching the requested driver function's definition in the API header
        file. The function pointer typedef can be picked up from the
        corresponding typedefs header file. For example, cudaTypedefs.h
        consists of function pointer typedefs for driver APIs defined in
        cuda.h.

        For the case where the CUDA version requested is greater than the CUDA
        Toolkit installed, there may not be an appropriate function pointer
        typedef in the corresponding header file and may need a custom typedef
        to match the driver function signature returned. This can be done by
        getting the typedefs from a later toolkit or creating appropriately
        matching custom function typedefs.

        The API will return :py:obj:`~.cudaSuccess` and set the returned
        `funcPtr` if the requested driver function is valid and supported on
        the platform.

        The API will return :py:obj:`~.cudaSuccess` and set the returned
        `funcPtr` to NULL if the requested driver function is not supported on
        the platform, no ABI compatible driver function exists for the
        requested version or if the driver symbol is invalid.

        It will also set the optional `driverStatus` to one of the values in
        :py:obj:`~.cudaDriverEntryPointQueryResult` with the following
        meanings:

        - :py:obj:`~.cudaDriverEntryPointSuccess` - The requested symbol was
          succesfully found based on input arguments and `pfn` is valid

        - :py:obj:`~.cudaDriverEntryPointSymbolNotFound` - The requested symbol
          was not found

        - :py:obj:`~.cudaDriverEntryPointVersionNotSufficent` - The requested
          symbol was found but is not supported by the specified version
          `cudaVersion`

        The requested flags can be:

        - :py:obj:`~.cudaEnableDefault`: This is the default mode. This is
          equivalent to :py:obj:`~.cudaEnablePerThreadDefaultStream` if the
          code is compiled with --default-stream per-thread compilation flag or
          the macro CUDA_API_PER_THREAD_DEFAULT_STREAM is defined;
          :py:obj:`~.cudaEnableLegacyStream` otherwise.

        - :py:obj:`~.cudaEnableLegacyStream`: This will enable the search for
          all driver symbols that match the requested driver symbol name except
          the corresponding per-thread versions.

        - :py:obj:`~.cudaEnablePerThreadDefaultStream`: This will enable the
          search for all driver symbols that match the requested driver symbol
          name including the per-thread versions. If a per-thread version is
          not found, the API will return the legacy version of the driver
          function.

        Parameters
        ----------
        symbol : bytes
            The base name of the driver API function to look for. As an
            example, for the driver API :py:obj:`~.cuMemAlloc_v2`, `symbol`
            would be cuMemAlloc.
        cudaVersion : unsigned int
            The CUDA version to look for the requested driver symbol
        flags : unsigned long long
            Flags to specify search options.

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorNotSupported`
        funcPtr : Any
            Location to return the function pointer to the requested driver
            function
        driverStatus : :py:obj:`~.cudaDriverEntryPointQueryResult`
            Optional location to store the status of finding the symbol from
            the driver. See :py:obj:`~.cudaDriverEntryPointQueryResult` for
            possible values.

        See Also
        --------
        :py:obj:`~.cuGetProcAddress`
    """


def cudaGetErrorName(error: 'cudaError_t'):
    """
    cudaGetErrorName(error: cudaError_t)
     Returns the string representation of an error code enum name.

        Returns a string containing the name of an error code in the enum. If
        the error code is not recognized, "unrecognized error code" is
        returned.

        Parameters
        ----------
        error : :py:obj:`~.cudaError_t`
            Error code to convert to string

        Returns
        -------
        cudaError_t.cudaSuccess
            cudaError_t.cudaSuccess
        bytes
            `char*` pointer to a NULL-terminated string

        See Also
        --------
        :py:obj:`~.cudaGetErrorString`, :py:obj:`~.cudaGetLastError`, :py:obj:`~.cudaPeekAtLastError`, :py:obj:`~.cudaError`, :py:obj:`~.cuGetErrorName`
    """


def cudaGetErrorString(error: 'cudaError_t'):
    """
    cudaGetErrorString(error: cudaError_t)
     Returns the description string for an error code.

        Returns the description string for an error code. If the error code is
        not recognized, "unrecognized error code" is returned.

        Parameters
        ----------
        error : :py:obj:`~.cudaError_t`
            Error code to convert to string

        Returns
        -------
        cudaError_t.cudaSuccess
            cudaError_t.cudaSuccess
        bytes
            `char*` pointer to a NULL-terminated string

        See Also
        --------
        :py:obj:`~.cudaGetErrorName`, :py:obj:`~.cudaGetLastError`, :py:obj:`~.cudaPeekAtLastError`, :py:obj:`~.cudaError`, :py:obj:`~.cuGetErrorString`
    """


def cudaGetExportTable(pExportTableId: 'Optional[cudaUUID_t]'):
    """
    cudaGetExportTable(cudaUUID_t pExportTableId: Optional[cudaUUID_t])
    """


def cudaGetKernel(entryFuncAddr):
    """
    cudaGetKernel(entryFuncAddr)
     Get pointer to device kernel that matches entry function `entryFuncAddr`.

        Returns in `kernelPtr` the device kernel corresponding to the entry
        function `entryFuncAddr`.

        Note that it is possible that there are multiple symbols belonging to
        different translation units with the same `entryFuncAddr` registered
        with this CUDA Runtime and so the order which the translation units are
        loaded and registered with the CUDA Runtime can lead to differing
        return pointers in `kernelPtr` . Suggested methods of ensuring
        uniqueness are to limit visibility of global device functions by using
        static or hidden visibility attribute in the respective translation
        units.

        Parameters
        ----------
        entryFuncAddr : Any
            Address of device entry function to search kernel for

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`
        kernelPtr : :py:obj:`~.cudaKernel_t`
            Returns the device kernel

        See Also
        --------
        cudaGetKernel (C++ API)
    """


def cudaGetLastError():
    """
    cudaGetLastError()
     Returns the last error from a runtime call.

        Returns the last error that has been produced by any of the runtime
        calls in the same instance of the CUDA Runtime library in the host
        thread and resets it to :py:obj:`~.cudaSuccess`.

        Note: Multiple instances of the CUDA Runtime library can be present in
        an application when using a library that statically links the CUDA
        Runtime.

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorMissingConfiguration`, :py:obj:`~.cudaErrorMemoryAllocation`, :py:obj:`~.cudaErrorInitializationError`, :py:obj:`~.cudaErrorLaunchFailure`, :py:obj:`~.cudaErrorLaunchTimeout`, :py:obj:`~.cudaErrorLaunchOutOfResources`, :py:obj:`~.cudaErrorInvalidDeviceFunction`, :py:obj:`~.cudaErrorInvalidConfiguration`, :py:obj:`~.cudaErrorInvalidDevice`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidPitchValue`, :py:obj:`~.cudaErrorInvalidSymbol`, :py:obj:`~.cudaErrorUnmapBufferObjectFailed`, :py:obj:`~.cudaErrorInvalidDevicePointer`, :py:obj:`~.cudaErrorInvalidTexture`, :py:obj:`~.cudaErrorInvalidTextureBinding`, :py:obj:`~.cudaErrorInvalidChannelDescriptor`, :py:obj:`~.cudaErrorInvalidMemcpyDirection`, :py:obj:`~.cudaErrorInvalidFilterSetting`, :py:obj:`~.cudaErrorInvalidNormSetting`, :py:obj:`~.cudaErrorUnknown`, :py:obj:`~.cudaErrorInvalidResourceHandle`, :py:obj:`~.cudaErrorInsufficientDriver`, :py:obj:`~.cudaErrorNoDevice`, :py:obj:`~.cudaErrorSetOnActiveProcess`, :py:obj:`~.cudaErrorStartupFailure`, :py:obj:`~.cudaErrorInvalidPtx`, :py:obj:`~.cudaErrorUnsupportedPtxVersion`, :py:obj:`~.cudaErrorNoKernelImageForDevice`, :py:obj:`~.cudaErrorJitCompilerNotFound`, :py:obj:`~.cudaErrorJitCompilationDisabled`

        See Also
        --------
        :py:obj:`~.cudaPeekAtLastError`, :py:obj:`~.cudaGetErrorName`, :py:obj:`~.cudaGetErrorString`, :py:obj:`~.cudaError`
    """


def cudaGetMipmappedArrayLevel(mipmappedArray, level):
    """
    cudaGetMipmappedArrayLevel(mipmappedArray, unsigned int level)
     Gets a mipmap level of a CUDA mipmapped array.

        Returns in `*levelArray` a CUDA array that represents a single mipmap
        level of the CUDA mipmapped array `mipmappedArray`.

        If `level` is greater than the maximum number of levels in this
        mipmapped array, :py:obj:`~.cudaErrorInvalidValue` is returned.

        If `mipmappedArray` is NULL, :py:obj:`~.cudaErrorInvalidResourceHandle`
        is returned.

        Parameters
        ----------
        mipmappedArray : :py:obj:`~.cudaMipmappedArray_const_t`
            CUDA mipmapped array
        level : unsigned int
            Mipmap level

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue` :py:obj:`~.cudaErrorInvalidResourceHandle`
        levelArray : :py:obj:`~.cudaArray_t`
            Returned mipmap level CUDA array

        See Also
        --------
        :py:obj:`~.cudaMalloc3D`, :py:obj:`~.cudaMalloc`, :py:obj:`~.cudaMallocPitch`, :py:obj:`~.cudaFree`, :py:obj:`~.cudaFreeArray`, :py:obj:`~.cudaMallocHost (C API)`, :py:obj:`~.cudaFreeHost`, :py:obj:`~.cudaHostAlloc`, :py:obj:`~.make_cudaExtent`, :py:obj:`~.cuMipmappedArrayGetLevel`
    """


def cudaGetSurfaceObjectResourceDesc(surfObject):
    """
    cudaGetSurfaceObjectResourceDesc(surfObject)
     Returns a surface object's resource descriptor Returns the resource descriptor for the surface object specified by `surfObject`.

        Parameters
        ----------
        surfObject : :py:obj:`~.cudaSurfaceObject_t`
            Surface object

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`
        pResDesc : :py:obj:`~.cudaResourceDesc`
            Resource descriptor

        See Also
        --------
        :py:obj:`~.cudaCreateSurfaceObject`, :py:obj:`~.cuSurfObjectGetResourceDesc`
    """


def cudaGetTextureObjectResourceDesc(texObject):
    """
    cudaGetTextureObjectResourceDesc(texObject)
     Returns a texture object's resource descriptor.

        Returns the resource descriptor for the texture object specified by
        `texObject`.

        Parameters
        ----------
        texObject : :py:obj:`~.cudaTextureObject_t`
            Texture object

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`
        pResDesc : :py:obj:`~.cudaResourceDesc`
            Resource descriptor

        See Also
        --------
        :py:obj:`~.cudaCreateTextureObject`, :py:obj:`~.cuTexObjectGetResourceDesc`
    """


def cudaGetTextureObjectResourceViewDesc(texObject):
    """
    cudaGetTextureObjectResourceViewDesc(texObject)
     Returns a texture object's resource view descriptor.

        Returns the resource view descriptor for the texture object specified
        by `texObject`. If no resource view was specified,
        :py:obj:`~.cudaErrorInvalidValue` is returned.

        Parameters
        ----------
        texObject : :py:obj:`~.cudaTextureObject_t`
            Texture object

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`
        pResViewDesc : :py:obj:`~.cudaResourceViewDesc`
            Resource view descriptor

        See Also
        --------
        :py:obj:`~.cudaCreateTextureObject`, :py:obj:`~.cuTexObjectGetResourceViewDesc`
    """


def cudaGetTextureObjectTextureDesc(texObject):
    """
    cudaGetTextureObjectTextureDesc(texObject)
     Returns a texture object's texture descriptor.

        Returns the texture descriptor for the texture object specified by
        `texObject`.

        Parameters
        ----------
        texObject : :py:obj:`~.cudaTextureObject_t`
            Texture object

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`
        pTexDesc : :py:obj:`~.cudaTextureDesc`
            Texture descriptor

        See Also
        --------
        :py:obj:`~.cudaCreateTextureObject`, :py:obj:`~.cuTexObjectGetTextureDesc`
    """


def cudaGraphAddChildGraphNode(graph, pDependencies: 'Optional[Tuple[cudaGraphNode_t] | List[cudaGraphNode_t]]', numDependencies, childGraph):
    """
    cudaGraphAddChildGraphNode(graph, pDependencies: Optional[Tuple[cudaGraphNode_t] | List[cudaGraphNode_t]], size_t numDependencies, childGraph)
     Creates a child graph node and adds it to a graph.

        Creates a new node which executes an embedded graph, and adds it to
        `graph` with `numDependencies` dependencies specified via
        `pDependencies`. It is possible for `numDependencies` to be 0, in which
        case the node will be placed at the root of the graph. `pDependencies`
        may not have any duplicate entries. A handle to the new node will be
        returned in `pGraphNode`.

        If `hGraph` contains allocation or free nodes, this call will return an
        error.

        The node executes an embedded child graph. The child graph is cloned in
        this call.

        Parameters
        ----------
        graph : :py:obj:`~.CUgraph` or :py:obj:`~.cudaGraph_t`
            Graph to which to add the node
        pDependencies : List[:py:obj:`~.cudaGraphNode_t`]
            Dependencies of the node
        numDependencies : size_t
            Number of dependencies
        childGraph : :py:obj:`~.CUgraph` or :py:obj:`~.cudaGraph_t`
            The graph to clone into this node

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`
        pGraphNode : :py:obj:`~.cudaGraphNode_t`
            Returns newly created node

        See Also
        --------
        :py:obj:`~.cudaGraphAddNode`, :py:obj:`~.cudaGraphChildGraphNodeGetGraph`, :py:obj:`~.cudaGraphCreate`, :py:obj:`~.cudaGraphDestroyNode`, :py:obj:`~.cudaGraphAddEmptyNode`, :py:obj:`~.cudaGraphAddKernelNode`, :py:obj:`~.cudaGraphAddHostNode`, :py:obj:`~.cudaGraphAddMemcpyNode`, :py:obj:`~.cudaGraphAddMemsetNode`, :py:obj:`~.cudaGraphClone`
    """


def cudaGraphAddDependencies(graph, from_: 'Optional[Tuple[cudaGraphNode_t] | List[cudaGraphNode_t]]', to: 'Optional[Tuple[cudaGraphNode_t] | List[cudaGraphNode_t]]', numDependencies):
    """
    cudaGraphAddDependencies(graph, from_: Optional[Tuple[cudaGraphNode_t] | List[cudaGraphNode_t]], to: Optional[Tuple[cudaGraphNode_t] | List[cudaGraphNode_t]], size_t numDependencies)
     Adds dependency edges to a graph.

        The number of dependencies to be added is defined by `numDependencies`
        Elements in `pFrom` and `pTo` at corresponding indices define a
        dependency. Each node in `pFrom` and `pTo` must belong to `graph`.

        If `numDependencies` is 0, elements in `pFrom` and `pTo` will be
        ignored. Specifying an existing dependency will return an error.

        Parameters
        ----------
        graph : :py:obj:`~.CUgraph` or :py:obj:`~.cudaGraph_t`
            Graph to which dependencies are added
        from : List[:py:obj:`~.cudaGraphNode_t`]
            Array of nodes that provide the dependencies
        to : List[:py:obj:`~.cudaGraphNode_t`]
            Array of dependent nodes
        numDependencies : size_t
            Number of dependencies to be added

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`

        See Also
        --------
        :py:obj:`~.cudaGraphRemoveDependencies`, :py:obj:`~.cudaGraphGetEdges`, :py:obj:`~.cudaGraphNodeGetDependencies`, :py:obj:`~.cudaGraphNodeGetDependentNodes`
    """


def cudaGraphAddDependencies_v2(graph, from_: 'Optional[Tuple[cudaGraphNode_t] | List[cudaGraphNode_t]]', to: 'Optional[Tuple[cudaGraphNode_t] | List[cudaGraphNode_t]]', edgeData: 'Optional[Tuple[cudaGraphEdgeData] | List[cudaGraphEdgeData]]', numDependencies):
    """
    cudaGraphAddDependencies_v2(graph, from_: Optional[Tuple[cudaGraphNode_t] | List[cudaGraphNode_t]], to: Optional[Tuple[cudaGraphNode_t] | List[cudaGraphNode_t]], edgeData: Optional[Tuple[cudaGraphEdgeData] | List[cudaGraphEdgeData]], size_t numDependencies)
     Adds dependency edges to a graph. (12.3+)

        The number of dependencies to be added is defined by `numDependencies`
        Elements in `pFrom` and `pTo` at corresponding indices define a
        dependency. Each node in `pFrom` and `pTo` must belong to `graph`.

        If `numDependencies` is 0, elements in `pFrom` and `pTo` will be
        ignored. Specifying an existing dependency will return an error.

        Parameters
        ----------
        graph : :py:obj:`~.CUgraph` or :py:obj:`~.cudaGraph_t`
            Graph to which dependencies are added
        from : List[:py:obj:`~.cudaGraphNode_t`]
            Array of nodes that provide the dependencies
        to : List[:py:obj:`~.cudaGraphNode_t`]
            Array of dependent nodes
        edgeData : List[:py:obj:`~.cudaGraphEdgeData`]
            Optional array of edge data. If NULL, default (zeroed) edge data is
            assumed.
        numDependencies : size_t
            Number of dependencies to be added

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`

        See Also
        --------
        :py:obj:`~.cudaGraphRemoveDependencies`, :py:obj:`~.cudaGraphGetEdges`, :py:obj:`~.cudaGraphNodeGetDependencies`, :py:obj:`~.cudaGraphNodeGetDependentNodes`
    """


def cudaGraphAddEmptyNode(graph, pDependencies: 'Optional[Tuple[cudaGraphNode_t] | List[cudaGraphNode_t]]', numDependencies):
    """
    cudaGraphAddEmptyNode(graph, pDependencies: Optional[Tuple[cudaGraphNode_t] | List[cudaGraphNode_t]], size_t numDependencies)
     Creates an empty node and adds it to a graph.

        Creates a new node which performs no operation, and adds it to `graph`
        with `numDependencies` dependencies specified via `pDependencies`. It
        is possible for `numDependencies` to be 0, in which case the node will
        be placed at the root of the graph. `pDependencies` may not have any
        duplicate entries. A handle to the new node will be returned in
        `pGraphNode`.

        An empty node performs no operation during execution, but can be used
        for transitive ordering. For example, a phased execution graph with 2
        groups of n nodes with a barrier between them can be represented using
        an empty node and 2*n dependency edges, rather than no empty node and
        n^2 dependency edges.

        Parameters
        ----------
        graph : :py:obj:`~.CUgraph` or :py:obj:`~.cudaGraph_t`
            Graph to which to add the node
        pDependencies : List[:py:obj:`~.cudaGraphNode_t`]
            Dependencies of the node
        numDependencies : size_t
            Number of dependencies

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`
        pGraphNode : :py:obj:`~.cudaGraphNode_t`
            Returns newly created node

        See Also
        --------
        :py:obj:`~.cudaGraphAddNode`, :py:obj:`~.cudaGraphCreate`, :py:obj:`~.cudaGraphDestroyNode`, :py:obj:`~.cudaGraphAddChildGraphNode`, :py:obj:`~.cudaGraphAddKernelNode`, :py:obj:`~.cudaGraphAddHostNode`, :py:obj:`~.cudaGraphAddMemcpyNode`, :py:obj:`~.cudaGraphAddMemsetNode`
    """


def cudaGraphAddEventRecordNode(graph, pDependencies: 'Optional[Tuple[cudaGraphNode_t] | List[cudaGraphNode_t]]', numDependencies, event):
    """
    cudaGraphAddEventRecordNode(graph, pDependencies: Optional[Tuple[cudaGraphNode_t] | List[cudaGraphNode_t]], size_t numDependencies, event)
     Creates an event record node and adds it to a graph.

        Creates a new event record node and adds it to `hGraph` with
        `numDependencies` dependencies specified via `dependencies` and event
        specified in `event`. It is possible for `numDependencies` to be 0, in
        which case the node will be placed at the root of the graph.
        `dependencies` may not have any duplicate entries. A handle to the new
        node will be returned in `phGraphNode`.

        Each launch of the graph will record `event` to capture execution of
        the node's dependencies.

        These nodes may not be used in loops or conditionals.

        Parameters
        ----------
        hGraph : :py:obj:`~.CUgraph` or :py:obj:`~.cudaGraph_t`
            Graph to which to add the node
        dependencies : List[:py:obj:`~.cudaGraphNode_t`]
            Dependencies of the node
        numDependencies : size_t
            Number of dependencies
        event : :py:obj:`~.CUevent` or :py:obj:`~.cudaEvent_t`
            Event for the node

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`
        phGraphNode : :py:obj:`~.cudaGraphNode_t`
            Returns newly created node

        See Also
        --------
        :py:obj:`~.cudaGraphAddNode`, :py:obj:`~.cudaGraphAddEventWaitNode`, :py:obj:`~.cudaEventRecordWithFlags`, :py:obj:`~.cudaStreamWaitEvent`, :py:obj:`~.cudaGraphCreate`, :py:obj:`~.cudaGraphDestroyNode`, :py:obj:`~.cudaGraphAddChildGraphNode`, :py:obj:`~.cudaGraphAddEmptyNode`, :py:obj:`~.cudaGraphAddKernelNode`, :py:obj:`~.cudaGraphAddMemcpyNode`, :py:obj:`~.cudaGraphAddMemsetNode`
    """


def cudaGraphAddEventWaitNode(graph, pDependencies: 'Optional[Tuple[cudaGraphNode_t] | List[cudaGraphNode_t]]', numDependencies, event):
    """
    cudaGraphAddEventWaitNode(graph, pDependencies: Optional[Tuple[cudaGraphNode_t] | List[cudaGraphNode_t]], size_t numDependencies, event)
     Creates an event wait node and adds it to a graph.

        Creates a new event wait node and adds it to `hGraph` with
        `numDependencies` dependencies specified via `dependencies` and event
        specified in `event`. It is possible for `numDependencies` to be 0, in
        which case the node will be placed at the root of the graph.
        `dependencies` may not have any duplicate entries. A handle to the new
        node will be returned in `phGraphNode`.

        The graph node will wait for all work captured in `event`. See
        :py:obj:`~.cuEventRecord()` for details on what is captured by an
        event. The synchronization will be performed efficiently on the device
        when applicable. `event` may be from a different context or device than
        the launch stream.

        These nodes may not be used in loops or conditionals.

        Parameters
        ----------
        hGraph : :py:obj:`~.CUgraph` or :py:obj:`~.cudaGraph_t`
            Graph to which to add the node
        dependencies : List[:py:obj:`~.cudaGraphNode_t`]
            Dependencies of the node
        numDependencies : size_t
            Number of dependencies
        event : :py:obj:`~.CUevent` or :py:obj:`~.cudaEvent_t`
            Event for the node

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`
        phGraphNode : :py:obj:`~.cudaGraphNode_t`
            Returns newly created node

        See Also
        --------
        :py:obj:`~.cudaGraphAddNode`, :py:obj:`~.cudaGraphAddEventRecordNode`, :py:obj:`~.cudaEventRecordWithFlags`, :py:obj:`~.cudaStreamWaitEvent`, :py:obj:`~.cudaGraphCreate`, :py:obj:`~.cudaGraphDestroyNode`, :py:obj:`~.cudaGraphAddChildGraphNode`, :py:obj:`~.cudaGraphAddEmptyNode`, :py:obj:`~.cudaGraphAddKernelNode`, :py:obj:`~.cudaGraphAddMemcpyNode`, :py:obj:`~.cudaGraphAddMemsetNode`
    """


def cudaGraphAddExternalSemaphoresSignalNode(graph, pDependencies: 'Optional[Tuple[cudaGraphNode_t] | List[cudaGraphNode_t]]', numDependencies, nodeParams: 'Optional[cudaExternalSemaphoreSignalNodeParams]'):
    """
    cudaGraphAddExternalSemaphoresSignalNode(graph, pDependencies: Optional[Tuple[cudaGraphNode_t] | List[cudaGraphNode_t]], size_t numDependencies, cudaExternalSemaphoreSignalNodeParams nodeParams: Optional[cudaExternalSemaphoreSignalNodeParams])
     Creates an external semaphore signal node and adds it to a graph.

        Creates a new external semaphore signal node and adds it to `graph`
        with `numDependencies` dependencies specified via `dependencies` and
        arguments specified in `nodeParams`. It is possible for
        `numDependencies` to be 0, in which case the node will be placed at the
        root of the graph. `dependencies` may not have any duplicate entries. A
        handle to the new node will be returned in `pGraphNode`.

        Performs a signal operation on a set of externally allocated semaphore
        objects when the node is launched. The operation(s) will occur after
        all of the node's dependencies have completed.

        Parameters
        ----------
        graph : :py:obj:`~.CUgraph` or :py:obj:`~.cudaGraph_t`
            Graph to which to add the node
        pDependencies : List[:py:obj:`~.cudaGraphNode_t`]
            Dependencies of the node
        numDependencies : size_t
            Number of dependencies
        nodeParams : :py:obj:`~.cudaExternalSemaphoreSignalNodeParams`
            Parameters for the node

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`
        pGraphNode : :py:obj:`~.cudaGraphNode_t`
            Returns newly created node

        See Also
        --------
        :py:obj:`~.cudaGraphAddNode`, :py:obj:`~.cudaGraphExternalSemaphoresSignalNodeGetParams`, :py:obj:`~.cudaGraphExternalSemaphoresSignalNodeSetParams`, :py:obj:`~.cudaGraphExecExternalSemaphoresSignalNodeSetParams`, :py:obj:`~.cudaGraphAddExternalSemaphoresWaitNode`, :py:obj:`~.cudaImportExternalSemaphore`, :py:obj:`~.cudaSignalExternalSemaphoresAsync`, :py:obj:`~.cudaWaitExternalSemaphoresAsync`, :py:obj:`~.cudaGraphCreate`, :py:obj:`~.cudaGraphDestroyNode`, :py:obj:`~.cudaGraphAddEventRecordNode`, :py:obj:`~.cudaGraphAddEventWaitNode`, :py:obj:`~.cudaGraphAddChildGraphNode`, :py:obj:`~.cudaGraphAddEmptyNode`, :py:obj:`~.cudaGraphAddKernelNode`, :py:obj:`~.cudaGraphAddMemcpyNode`, :py:obj:`~.cudaGraphAddMemsetNode`
    """


def cudaGraphAddExternalSemaphoresWaitNode(graph, pDependencies: 'Optional[Tuple[cudaGraphNode_t] | List[cudaGraphNode_t]]', numDependencies, nodeParams: 'Optional[cudaExternalSemaphoreWaitNodeParams]'):
    """
    cudaGraphAddExternalSemaphoresWaitNode(graph, pDependencies: Optional[Tuple[cudaGraphNode_t] | List[cudaGraphNode_t]], size_t numDependencies, cudaExternalSemaphoreWaitNodeParams nodeParams: Optional[cudaExternalSemaphoreWaitNodeParams])
     Creates an external semaphore wait node and adds it to a graph.

        Creates a new external semaphore wait node and adds it to `graph` with
        `numDependencies` dependencies specified via `dependencies` and
        arguments specified in `nodeParams`. It is possible for
        `numDependencies` to be 0, in which case the node will be placed at the
        root of the graph. `dependencies` may not have any duplicate entries. A
        handle to the new node will be returned in `pGraphNode`.

        Performs a wait operation on a set of externally allocated semaphore
        objects when the node is launched. The node's dependencies will not be
        launched until the wait operation has completed.

        Parameters
        ----------
        graph : :py:obj:`~.CUgraph` or :py:obj:`~.cudaGraph_t`
            Graph to which to add the node
        pDependencies : List[:py:obj:`~.cudaGraphNode_t`]
            Dependencies of the node
        numDependencies : size_t
            Number of dependencies
        nodeParams : :py:obj:`~.cudaExternalSemaphoreWaitNodeParams`
            Parameters for the node

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`
        pGraphNode : :py:obj:`~.cudaGraphNode_t`
            Returns newly created node

        See Also
        --------
        :py:obj:`~.cudaGraphAddNode`, :py:obj:`~.cudaGraphExternalSemaphoresWaitNodeGetParams`, :py:obj:`~.cudaGraphExternalSemaphoresWaitNodeSetParams`, :py:obj:`~.cudaGraphExecExternalSemaphoresWaitNodeSetParams`, :py:obj:`~.cudaGraphAddExternalSemaphoresSignalNode`, :py:obj:`~.cudaImportExternalSemaphore`, :py:obj:`~.cudaSignalExternalSemaphoresAsync`, :py:obj:`~.cudaWaitExternalSemaphoresAsync`, :py:obj:`~.cudaGraphCreate`, :py:obj:`~.cudaGraphDestroyNode`, :py:obj:`~.cudaGraphAddEventRecordNode`, :py:obj:`~.cudaGraphAddEventWaitNode`, :py:obj:`~.cudaGraphAddChildGraphNode`, :py:obj:`~.cudaGraphAddEmptyNode`, :py:obj:`~.cudaGraphAddKernelNode`, :py:obj:`~.cudaGraphAddMemcpyNode`, :py:obj:`~.cudaGraphAddMemsetNode`
    """


def cudaGraphAddHostNode(graph, pDependencies: 'Optional[Tuple[cudaGraphNode_t] | List[cudaGraphNode_t]]', numDependencies, pNodeParams: 'Optional[cudaHostNodeParams]'):
    """
    cudaGraphAddHostNode(graph, pDependencies: Optional[Tuple[cudaGraphNode_t] | List[cudaGraphNode_t]], size_t numDependencies, cudaHostNodeParams pNodeParams: Optional[cudaHostNodeParams])
     Creates a host execution node and adds it to a graph.

        Creates a new CPU execution node and adds it to `graph` with
        `numDependencies` dependencies specified via `pDependencies` and
        arguments specified in `pNodeParams`. It is possible for
        `numDependencies` to be 0, in which case the node will be placed at the
        root of the graph. `pDependencies` may not have any duplicate entries.
        A handle to the new node will be returned in `pGraphNode`.

        When the graph is launched, the node will invoke the specified CPU
        function. Host nodes are not supported under MPS with pre-Volta GPUs.

        Parameters
        ----------
        graph : :py:obj:`~.CUgraph` or :py:obj:`~.cudaGraph_t`
            Graph to which to add the node
        pDependencies : List[:py:obj:`~.cudaGraphNode_t`]
            Dependencies of the node
        numDependencies : size_t
            Number of dependencies
        pNodeParams : :py:obj:`~.cudaHostNodeParams`
            Parameters for the host node

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorNotSupported`, :py:obj:`~.cudaErrorInvalidValue`
        pGraphNode : :py:obj:`~.cudaGraphNode_t`
            Returns newly created node

        See Also
        --------
        :py:obj:`~.cudaGraphAddNode`, :py:obj:`~.cudaLaunchHostFunc`, :py:obj:`~.cudaGraphHostNodeGetParams`, :py:obj:`~.cudaGraphHostNodeSetParams`, :py:obj:`~.cudaGraphCreate`, :py:obj:`~.cudaGraphDestroyNode`, :py:obj:`~.cudaGraphAddChildGraphNode`, :py:obj:`~.cudaGraphAddEmptyNode`, :py:obj:`~.cudaGraphAddKernelNode`, :py:obj:`~.cudaGraphAddMemcpyNode`, :py:obj:`~.cudaGraphAddMemsetNode`
    """


def cudaGraphAddKernelNode(graph, pDependencies: 'Optional[Tuple[cudaGraphNode_t] | List[cudaGraphNode_t]]', numDependencies, pNodeParams: 'Optional[cudaKernelNodeParams]'):
    """
    cudaGraphAddKernelNode(graph, pDependencies: Optional[Tuple[cudaGraphNode_t] | List[cudaGraphNode_t]], size_t numDependencies, cudaKernelNodeParams pNodeParams: Optional[cudaKernelNodeParams])
     Creates a kernel execution node and adds it to a graph.

        Creates a new kernel execution node and adds it to `graph` with
        `numDependencies` dependencies specified via `pDependencies` and
        arguments specified in `pNodeParams`. It is possible for
        `numDependencies` to be 0, in which case the node will be placed at the
        root of the graph. `pDependencies` may not have any duplicate entries.
        A handle to the new node will be returned in `pGraphNode`.

        The :py:obj:`~.cudaKernelNodeParams` structure is defined as:

        **View CUDA Toolkit Documentation for a C++ code example**

        When the graph is launched, the node will invoke kernel `func` on a
        (`gridDim.x` x `gridDim.y` x `gridDim.z`) grid of blocks. Each block
        contains (`blockDim.x` x `blockDim.y` x `blockDim.z`) threads.

        `sharedMem` sets the amount of dynamic shared memory that will be
        available to each thread block.

        Kernel parameters to `func` can be specified in one of two ways:

        1) Kernel parameters can be specified via `kernelParams`. If the kernel
        has N parameters, then `kernelParams` needs to be an array of N
        pointers. Each pointer, from `kernelParams`[0] to `kernelParams`[N-1],
        points to the region of memory from which the actual parameter will be
        copied. The number of kernel parameters and their offsets and sizes do
        not need to be specified as that information is retrieved directly from
        the kernel's image.

        2) Kernel parameters can also be packaged by the application into a
        single buffer that is passed in via `extra`. This places the burden on
        the application of knowing each kernel parameter's size and
        alignment/padding within the buffer. The `extra` parameter exists to
        allow this function to take additional less commonly used arguments.
        `extra` specifies a list of names of extra settings and their
        corresponding values. Each extra setting name is immediately followed
        by the corresponding value. The list must be terminated with either
        NULL or CU_LAUNCH_PARAM_END.

        - :py:obj:`~.CU_LAUNCH_PARAM_END`, which indicates the end of the
          `extra` array;

        - :py:obj:`~.CU_LAUNCH_PARAM_BUFFER_POINTER`, which specifies that the
          next value in `extra` will be a pointer to a buffer containing all
          the kernel parameters for launching kernel `func`;

        - :py:obj:`~.CU_LAUNCH_PARAM_BUFFER_SIZE`, which specifies that the
          next value in `extra` will be a pointer to a size_t containing the
          size of the buffer specified with
          :py:obj:`~.CU_LAUNCH_PARAM_BUFFER_POINTER`;

        The error :py:obj:`~.cudaErrorInvalidValue` will be returned if kernel
        parameters are specified with both `kernelParams` and `extra` (i.e.
        both `kernelParams` and `extra` are non-NULL).

        The `kernelParams` or `extra` array, as well as the argument values it
        points to, are copied during this call.

        Parameters
        ----------
        graph : :py:obj:`~.CUgraph` or :py:obj:`~.cudaGraph_t`
            Graph to which to add the node
        pDependencies : List[:py:obj:`~.cudaGraphNode_t`]
            Dependencies of the node
        numDependencies : size_t
            Number of dependencies
        pNodeParams : :py:obj:`~.cudaKernelNodeParams`
            Parameters for the GPU execution node

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidDeviceFunction`
        pGraphNode : :py:obj:`~.cudaGraphNode_t`
            Returns newly created node

        See Also
        --------
        :py:obj:`~.cudaGraphAddNode`, :py:obj:`~.cudaLaunchKernel`, :py:obj:`~.cudaGraphKernelNodeGetParams`, :py:obj:`~.cudaGraphKernelNodeSetParams`, :py:obj:`~.cudaGraphCreate`, :py:obj:`~.cudaGraphDestroyNode`, :py:obj:`~.cudaGraphAddChildGraphNode`, :py:obj:`~.cudaGraphAddEmptyNode`, :py:obj:`~.cudaGraphAddHostNode`, :py:obj:`~.cudaGraphAddMemcpyNode`, :py:obj:`~.cudaGraphAddMemsetNode`

        Notes
        -----
        Kernels launched using graphs must not use texture and surface references. Reading or writing through any texture or surface reference is undefined behavior. This restriction does not apply to texture and surface objects.
    """


def cudaGraphAddMemAllocNode(graph, pDependencies: 'Optional[Tuple[cudaGraphNode_t] | List[cudaGraphNode_t]]', numDependencies, nodeParams: 'Optional[cudaMemAllocNodeParams]'):
    """
    cudaGraphAddMemAllocNode(graph, pDependencies: Optional[Tuple[cudaGraphNode_t] | List[cudaGraphNode_t]], size_t numDependencies, cudaMemAllocNodeParams nodeParams: Optional[cudaMemAllocNodeParams])
     Creates an allocation node and adds it to a graph.

        Creates a new allocation node and adds it to `graph` with
        `numDependencies` dependencies specified via `pDependencies` and
        arguments specified in `nodeParams`. It is possible for
        `numDependencies` to be 0, in which case the node will be placed at the
        root of the graph. `pDependencies` may not have any duplicate entries.
        A handle to the new node will be returned in `pGraphNode`.

        When :py:obj:`~.cudaGraphAddMemAllocNode` creates an allocation node,
        it returns the address of the allocation in `nodeParams.dptr`. The
        allocation's address remains fixed across instantiations and launches.

        If the allocation is freed in the same graph, by creating a free node
        using :py:obj:`~.cudaGraphAddMemFreeNode`, the allocation can be
        accessed by nodes ordered after the allocation node but before the free
        node. These allocations cannot be freed outside the owning graph, and
        they can only be freed once in the owning graph.

        If the allocation is not freed in the same graph, then it can be
        accessed not only by nodes in the graph which are ordered after the
        allocation node, but also by stream operations ordered after the
        graph's execution but before the allocation is freed.

        Allocations which are not freed in the same graph can be freed by:

        - passing the allocation to :py:obj:`~.cudaMemFreeAsync` or
          :py:obj:`~.cudaMemFree`;

        - launching a graph with a free node for that allocation; or

        - specifying :py:obj:`~.cudaGraphInstantiateFlagAutoFreeOnLaunch`
          during instantiation, which makes each launch behave as though it
          called :py:obj:`~.cudaMemFreeAsync` for every unfreed allocation.

        It is not possible to free an allocation in both the owning graph and
        another graph. If the allocation is freed in the same graph, a free
        node cannot be added to another graph. If the allocation is freed in
        another graph, a free node can no longer be added to the owning graph.

        The following restrictions apply to graphs which contain allocation
        and/or memory free nodes:

        - Nodes and edges of the graph cannot be deleted.

        - The graph cannot be used in a child node.

        - Only one instantiation of the graph may exist at any point in time.

        - The graph cannot be cloned.

        Parameters
        ----------
        graph : :py:obj:`~.CUgraph` or :py:obj:`~.cudaGraph_t`
            Graph to which to add the node
        pDependencies : List[:py:obj:`~.cudaGraphNode_t`]
            Dependencies of the node
        numDependencies : size_t
            Number of dependencies
        nodeParams : :py:obj:`~.cudaMemAllocNodeParams`
            Parameters for the node

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorCudartUnloading`, :py:obj:`~.cudaErrorInitializationError`, :py:obj:`~.cudaErrorNotSupported`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorOutOfMemory`
        pGraphNode : :py:obj:`~.cudaGraphNode_t`
            Returns newly created node

        See Also
        --------
        :py:obj:`~.cudaGraphAddNode`, :py:obj:`~.cudaGraphAddMemFreeNode`, :py:obj:`~.cudaGraphMemAllocNodeGetParams`, :py:obj:`~.cudaDeviceGraphMemTrim`, :py:obj:`~.cudaDeviceGetGraphMemAttribute`, :py:obj:`~.cudaDeviceSetGraphMemAttribute`, :py:obj:`~.cudaMallocAsync`, :py:obj:`~.cudaFreeAsync`, :py:obj:`~.cudaGraphCreate`, :py:obj:`~.cudaGraphDestroyNode`, :py:obj:`~.cudaGraphAddChildGraphNode`, :py:obj:`~.cudaGraphAddEmptyNode`, :py:obj:`~.cudaGraphAddEventRecordNode`, :py:obj:`~.cudaGraphAddEventWaitNode`, :py:obj:`~.cudaGraphAddExternalSemaphoresSignalNode`, :py:obj:`~.cudaGraphAddExternalSemaphoresWaitNode`, :py:obj:`~.cudaGraphAddKernelNode`, :py:obj:`~.cudaGraphAddMemcpyNode`, :py:obj:`~.cudaGraphAddMemsetNode`
    """


def cudaGraphAddMemFreeNode(graph, pDependencies: 'Optional[Tuple[cudaGraphNode_t] | List[cudaGraphNode_t]]', numDependencies, dptr):
    """
    cudaGraphAddMemFreeNode(graph, pDependencies: Optional[Tuple[cudaGraphNode_t] | List[cudaGraphNode_t]], size_t numDependencies, dptr)
     Creates a memory free node and adds it to a graph.

        Creates a new memory free node and adds it to `graph` with
        `numDependencies` dependencies specified via `pDependencies` and
        address specified in `dptr`. It is possible for `numDependencies` to be
        0, in which case the node will be placed at the root of the graph.
        `pDependencies` may not have any duplicate entries. A handle to the new
        node will be returned in `pGraphNode`.

        :py:obj:`~.cudaGraphAddMemFreeNode` will return
        :py:obj:`~.cudaErrorInvalidValue` if the user attempts to free:

        - an allocation twice in the same graph.

        - an address that was not returned by an allocation node.

        - an invalid address.

        The following restrictions apply to graphs which contain allocation
        and/or memory free nodes:

        - Nodes and edges of the graph cannot be deleted.

        - The graph cannot be used in a child node.

        - Only one instantiation of the graph may exist at any point in time.

        - The graph cannot be cloned.

        Parameters
        ----------
        graph : :py:obj:`~.CUgraph` or :py:obj:`~.cudaGraph_t`
            Graph to which to add the node
        pDependencies : List[:py:obj:`~.cudaGraphNode_t`]
            Dependencies of the node
        numDependencies : size_t
            Number of dependencies
        dptr : Any
            Address of memory to free

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorCudartUnloading`, :py:obj:`~.cudaErrorInitializationError`, :py:obj:`~.cudaErrorNotSupported`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorOutOfMemory`
        pGraphNode : :py:obj:`~.cudaGraphNode_t`
            Returns newly created node

        See Also
        --------
        :py:obj:`~.cudaGraphAddNode`, :py:obj:`~.cudaGraphAddMemAllocNode`, :py:obj:`~.cudaGraphMemFreeNodeGetParams`, :py:obj:`~.cudaDeviceGraphMemTrim`, :py:obj:`~.cudaDeviceGetGraphMemAttribute`, :py:obj:`~.cudaDeviceSetGraphMemAttribute`, :py:obj:`~.cudaMallocAsync`, :py:obj:`~.cudaFreeAsync`, :py:obj:`~.cudaGraphCreate`, :py:obj:`~.cudaGraphDestroyNode`, :py:obj:`~.cudaGraphAddChildGraphNode`, :py:obj:`~.cudaGraphAddEmptyNode`, :py:obj:`~.cudaGraphAddEventRecordNode`, :py:obj:`~.cudaGraphAddEventWaitNode`, :py:obj:`~.cudaGraphAddExternalSemaphoresSignalNode`, :py:obj:`~.cudaGraphAddExternalSemaphoresWaitNode`, :py:obj:`~.cudaGraphAddKernelNode`, :py:obj:`~.cudaGraphAddMemcpyNode`, :py:obj:`~.cudaGraphAddMemsetNode`
    """


def cudaGraphAddMemcpyNode(graph, pDependencies: 'Optional[Tuple[cudaGraphNode_t] | List[cudaGraphNode_t]]', numDependencies, pCopyParams: 'Optional[cudaMemcpy3DParms]'):
    """
    cudaGraphAddMemcpyNode(graph, pDependencies: Optional[Tuple[cudaGraphNode_t] | List[cudaGraphNode_t]], size_t numDependencies, cudaMemcpy3DParms pCopyParams: Optional[cudaMemcpy3DParms])
     Creates a memcpy node and adds it to a graph.

        Creates a new memcpy node and adds it to `graph` with `numDependencies`
        dependencies specified via `pDependencies`. It is possible for
        `numDependencies` to be 0, in which case the node will be placed at the
        root of the graph. `pDependencies` may not have any duplicate entries.
        A handle to the new node will be returned in `pGraphNode`.

        When the graph is launched, the node will perform the memcpy described
        by `pCopyParams`. See :py:obj:`~.cudaMemcpy3D()` for a description of
        the structure and its restrictions.

        Memcpy nodes have some additional restrictions with regards to managed
        memory, if the system contains at least one device which has a zero
        value for the device attribute
        :py:obj:`~.cudaDevAttrConcurrentManagedAccess`.

        Parameters
        ----------
        graph : :py:obj:`~.CUgraph` or :py:obj:`~.cudaGraph_t`
            Graph to which to add the node
        pDependencies : List[:py:obj:`~.cudaGraphNode_t`]
            Dependencies of the node
        numDependencies : size_t
            Number of dependencies
        pCopyParams : :py:obj:`~.cudaMemcpy3DParms`
            Parameters for the memory copy

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`
        pGraphNode : :py:obj:`~.cudaGraphNode_t`
            Returns newly created node

        See Also
        --------
        :py:obj:`~.cudaGraphAddNode`, :py:obj:`~.cudaMemcpy3D`, :py:obj:`~.cudaGraphAddMemcpyNodeToSymbol`, :py:obj:`~.cudaGraphAddMemcpyNodeFromSymbol`, :py:obj:`~.cudaGraphAddMemcpyNode1D`, :py:obj:`~.cudaGraphMemcpyNodeGetParams`, :py:obj:`~.cudaGraphMemcpyNodeSetParams`, :py:obj:`~.cudaGraphCreate`, :py:obj:`~.cudaGraphDestroyNode`, :py:obj:`~.cudaGraphAddChildGraphNode`, :py:obj:`~.cudaGraphAddEmptyNode`, :py:obj:`~.cudaGraphAddKernelNode`, :py:obj:`~.cudaGraphAddHostNode`, :py:obj:`~.cudaGraphAddMemsetNode`
    """


def cudaGraphAddMemcpyNode1D(graph, pDependencies: 'Optional[Tuple[cudaGraphNode_t] | List[cudaGraphNode_t]]', numDependencies, dst, src, count, kind: 'cudaMemcpyKind'):
    """
    cudaGraphAddMemcpyNode1D(graph, pDependencies: Optional[Tuple[cudaGraphNode_t] | List[cudaGraphNode_t]], size_t numDependencies, dst, src, size_t count, kind: cudaMemcpyKind)
     Creates a 1D memcpy node and adds it to a graph.

        Creates a new 1D memcpy node and adds it to `graph` with
        `numDependencies` dependencies specified via `pDependencies`. It is
        possible for `numDependencies` to be 0, in which case the node will be
        placed at the root of the graph. `pDependencies` may not have any
        duplicate entries. A handle to the new node will be returned in
        `pGraphNode`.

        When the graph is launched, the node will copy `count` bytes from the
        memory area pointed to by `src` to the memory area pointed to by `dst`,
        where `kind` specifies the direction of the copy, and must be one of
        :py:obj:`~.cudaMemcpyHostToHost`, :py:obj:`~.cudaMemcpyHostToDevice`,
        :py:obj:`~.cudaMemcpyDeviceToHost`,
        :py:obj:`~.cudaMemcpyDeviceToDevice`, or :py:obj:`~.cudaMemcpyDefault`.
        Passing :py:obj:`~.cudaMemcpyDefault` is recommended, in which case the
        type of transfer is inferred from the pointer values. However,
        :py:obj:`~.cudaMemcpyDefault` is only allowed on systems that support
        unified virtual addressing. Launching a memcpy node with dst and src
        pointers that do not match the direction of the copy results in an
        undefined behavior.

        Memcpy nodes have some additional restrictions with regards to managed
        memory, if the system contains at least one device which has a zero
        value for the device attribute
        :py:obj:`~.cudaDevAttrConcurrentManagedAccess`.

        Parameters
        ----------
        graph : :py:obj:`~.CUgraph` or :py:obj:`~.cudaGraph_t`
            Graph to which to add the node
        pDependencies : List[:py:obj:`~.cudaGraphNode_t`]
            Dependencies of the node
        numDependencies : size_t
            Number of dependencies
        dst : Any
            Destination memory address
        src : Any
            Source memory address
        count : size_t
            Size in bytes to copy
        kind : :py:obj:`~.cudaMemcpyKind`
            Type of transfer

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`
        pGraphNode : :py:obj:`~.cudaGraphNode_t`
            Returns newly created node

        See Also
        --------
        :py:obj:`~.cudaMemcpy`, :py:obj:`~.cudaGraphAddMemcpyNode`, :py:obj:`~.cudaGraphMemcpyNodeGetParams`, :py:obj:`~.cudaGraphMemcpyNodeSetParams`, :py:obj:`~.cudaGraphMemcpyNodeSetParams1D`, :py:obj:`~.cudaGraphCreate`, :py:obj:`~.cudaGraphDestroyNode`, :py:obj:`~.cudaGraphAddChildGraphNode`, :py:obj:`~.cudaGraphAddEmptyNode`, :py:obj:`~.cudaGraphAddKernelNode`, :py:obj:`~.cudaGraphAddHostNode`, :py:obj:`~.cudaGraphAddMemsetNode`
    """


def cudaGraphAddMemsetNode(graph, pDependencies: 'Optional[Tuple[cudaGraphNode_t] | List[cudaGraphNode_t]]', numDependencies, pMemsetParams: 'Optional[cudaMemsetParams]'):
    """
    cudaGraphAddMemsetNode(graph, pDependencies: Optional[Tuple[cudaGraphNode_t] | List[cudaGraphNode_t]], size_t numDependencies, cudaMemsetParams pMemsetParams: Optional[cudaMemsetParams])
     Creates a memset node and adds it to a graph.

        Creates a new memset node and adds it to `graph` with `numDependencies`
        dependencies specified via `pDependencies`. It is possible for
        `numDependencies` to be 0, in which case the node will be placed at the
        root of the graph. `pDependencies` may not have any duplicate entries.
        A handle to the new node will be returned in `pGraphNode`.

        The element size must be 1, 2, or 4 bytes. When the graph is launched,
        the node will perform the memset described by `pMemsetParams`.

        Parameters
        ----------
        graph : :py:obj:`~.CUgraph` or :py:obj:`~.cudaGraph_t`
            Graph to which to add the node
        pDependencies : List[:py:obj:`~.cudaGraphNode_t`]
            Dependencies of the node
        numDependencies : size_t
            Number of dependencies
        pMemsetParams : :py:obj:`~.cudaMemsetParams`
            Parameters for the memory set

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidDevice`
        pGraphNode : :py:obj:`~.cudaGraphNode_t`
            Returns newly created node

        See Also
        --------
        :py:obj:`~.cudaGraphAddNode`, :py:obj:`~.cudaMemset2D`, :py:obj:`~.cudaGraphMemsetNodeGetParams`, :py:obj:`~.cudaGraphMemsetNodeSetParams`, :py:obj:`~.cudaGraphCreate`, :py:obj:`~.cudaGraphDestroyNode`, :py:obj:`~.cudaGraphAddChildGraphNode`, :py:obj:`~.cudaGraphAddEmptyNode`, :py:obj:`~.cudaGraphAddKernelNode`, :py:obj:`~.cudaGraphAddHostNode`, :py:obj:`~.cudaGraphAddMemcpyNode`
    """


def cudaGraphAddNode(graph, pDependencies: 'Optional[Tuple[cudaGraphNode_t] | List[cudaGraphNode_t]]', numDependencies, nodeParams: 'Optional[cudaGraphNodeParams]'):
    """
    cudaGraphAddNode(graph, pDependencies: Optional[Tuple[cudaGraphNode_t] | List[cudaGraphNode_t]], size_t numDependencies, cudaGraphNodeParams nodeParams: Optional[cudaGraphNodeParams])
     Adds a node of arbitrary type to a graph.

        Creates a new node in `graph` described by `nodeParams` with
        `numDependencies` dependencies specified via `pDependencies`.
        `numDependencies` may be 0. `pDependencies` may be null if
        `numDependencies` is 0. `pDependencies` may not have any duplicate
        entries.

        `nodeParams` is a tagged union. The node type should be specified in
        the `typename` field, and type-specific parameters in the corresponding
        union member. All unused bytes - that is, `reserved0` and all bytes
        past the utilized union member - must be set to zero. It is recommended
        to use brace initialization or memset to ensure all bytes are
        initialized.

        Note that for some node types, `nodeParams` may contain "out
        parameters" which are modified during the call, such as
        `nodeParams->alloc.dptr`.

        A handle to the new node will be returned in `phGraphNode`.

        Parameters
        ----------
        graph : :py:obj:`~.CUgraph` or :py:obj:`~.cudaGraph_t`
            Graph to which to add the node
        pDependencies : List[:py:obj:`~.cudaGraphNode_t`]
            Dependencies of the node
        numDependencies : size_t
            Number of dependencies
        nodeParams : :py:obj:`~.cudaGraphNodeParams`
            Specification of the node

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidDeviceFunction`, :py:obj:`~.cudaErrorNotSupported`
        pGraphNode : :py:obj:`~.cudaGraphNode_t`
            Returns newly created node

        See Also
        --------
        :py:obj:`~.cudaGraphCreate`, :py:obj:`~.cudaGraphNodeSetParams`, :py:obj:`~.cudaGraphExecNodeSetParams`
    """


def cudaGraphAddNode_v2(graph, pDependencies: 'Optional[Tuple[cudaGraphNode_t] | List[cudaGraphNode_t]]', dependencyData: 'Optional[Tuple[cudaGraphEdgeData] | List[cudaGraphEdgeData]]', numDependencies, nodeParams: 'Optional[cudaGraphNodeParams]'):
    """
    cudaGraphAddNode_v2(graph, pDependencies: Optional[Tuple[cudaGraphNode_t] | List[cudaGraphNode_t]], dependencyData: Optional[Tuple[cudaGraphEdgeData] | List[cudaGraphEdgeData]], size_t numDependencies, cudaGraphNodeParams nodeParams: Optional[cudaGraphNodeParams])
     Adds a node of arbitrary type to a graph (12.3+)

        Creates a new node in `graph` described by `nodeParams` with
        `numDependencies` dependencies specified via `pDependencies`.
        `numDependencies` may be 0. `pDependencies` may be null if
        `numDependencies` is 0. `pDependencies` may not have any duplicate
        entries.

        `nodeParams` is a tagged union. The node type should be specified in
        the `typename` field, and type-specific parameters in the corresponding
        union member. All unused bytes - that is, `reserved0` and all bytes
        past the utilized union member - must be set to zero. It is recommended
        to use brace initialization or memset to ensure all bytes are
        initialized.

        Note that for some node types, `nodeParams` may contain "out
        parameters" which are modified during the call, such as
        `nodeParams->alloc.dptr`.

        A handle to the new node will be returned in `phGraphNode`.

        Parameters
        ----------
        graph : :py:obj:`~.CUgraph` or :py:obj:`~.cudaGraph_t`
            Graph to which to add the node
        pDependencies : List[:py:obj:`~.cudaGraphNode_t`]
            Dependencies of the node
        dependencyData : List[:py:obj:`~.cudaGraphEdgeData`]
            Optional edge data for the dependencies. If NULL, the data is
            assumed to be default (zeroed) for all dependencies.
        numDependencies : size_t
            Number of dependencies
        nodeParams : :py:obj:`~.cudaGraphNodeParams`
            Specification of the node

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidDeviceFunction`, :py:obj:`~.cudaErrorNotSupported`
        pGraphNode : :py:obj:`~.cudaGraphNode_t`
            Returns newly created node

        See Also
        --------
        :py:obj:`~.cudaGraphCreate`, :py:obj:`~.cudaGraphNodeSetParams`, :py:obj:`~.cudaGraphExecNodeSetParams`
    """


def cudaGraphChildGraphNodeGetGraph(node):
    """
    cudaGraphChildGraphNodeGetGraph(node)
     Gets a handle to the embedded graph of a child graph node.

        Gets a handle to the embedded graph in a child graph node. This call
        does not clone the graph. Changes to the graph will be reflected in the
        node, and the node retains ownership of the graph.

        Allocation and free nodes cannot be added to the returned graph.
        Attempting to do so will return an error.

        Parameters
        ----------
        node : :py:obj:`~.CUgraphNode` or :py:obj:`~.cudaGraphNode_t`
            Node to get the embedded graph for

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`
        pGraph : :py:obj:`~.cudaGraph_t`
            Location to store a handle to the graph

        See Also
        --------
        :py:obj:`~.cudaGraphAddChildGraphNode`, :py:obj:`~.cudaGraphNodeFindInClone`
    """


def cudaGraphClone(originalGraph):
    """
    cudaGraphClone(originalGraph)
     Clones a graph.

        This function creates a copy of `originalGraph` and returns it in
        `pGraphClone`. All parameters are copied into the cloned graph. The
        original graph may be modified after this call without affecting the
        clone.

        Child graph nodes in the original graph are recursively copied into the
        clone.

        Parameters
        ----------
        originalGraph : :py:obj:`~.CUgraph` or :py:obj:`~.cudaGraph_t`
            Graph to clone

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorMemoryAllocation`
        pGraphClone : :py:obj:`~.cudaGraph_t`
            Returns newly created cloned graph

        See Also
        --------
        :py:obj:`~.cudaGraphCreate`, :py:obj:`~.cudaGraphNodeFindInClone`
    """


def cudaGraphConditionalHandleCreate(graph, defaultLaunchValue, flags):
    """
    cudaGraphConditionalHandleCreate(graph, unsigned int defaultLaunchValue, unsigned int flags)
     Create a conditional handle.

        Creates a conditional handle associated with `hGraph`.

        The conditional handle must be associated with a conditional node in
        this graph or one of its children.

        Handles not associated with a conditional node may cause graph
        instantiation to fail.

        Parameters
        ----------
        hGraph : :py:obj:`~.CUgraph` or :py:obj:`~.cudaGraph_t`
            Graph which will contain the conditional node using this handle.
        defaultLaunchValue : unsigned int
            Optional initial value for the conditional variable. Applied at the
            beginning of each graph execution if cudaGraphCondAssignDefault is
            set in `flags`.
        flags : unsigned int
            Currently must be cudaGraphCondAssignDefault or 0.

        Returns
        -------
        cudaError_t
            :py:obj:`~.CUDA_SUCCESS`, :py:obj:`~.CUDA_ERROR_INVALID_VALUE`, :py:obj:`~.CUDA_ERROR_NOT_SUPPORTED`
        pHandle_out : :py:obj:`~.cudaGraphConditionalHandle`
            Pointer used to return the handle to the caller.

        See Also
        --------
        :py:obj:`~.cuGraphAddNode`,
    """


def cudaGraphCreate(flags):
    """
    cudaGraphCreate(unsigned int flags)
     Creates a graph.

        Creates an empty graph, which is returned via `pGraph`.

        Parameters
        ----------
        flags : unsigned int
            Graph creation flags, must be 0

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorMemoryAllocation`
        pGraph : :py:obj:`~.cudaGraph_t`
            Returns newly created graph

        See Also
        --------
        :py:obj:`~.cudaGraphAddChildGraphNode`, :py:obj:`~.cudaGraphAddEmptyNode`, :py:obj:`~.cudaGraphAddKernelNode`, :py:obj:`~.cudaGraphAddHostNode`, :py:obj:`~.cudaGraphAddMemcpyNode`, :py:obj:`~.cudaGraphAddMemsetNode`, :py:obj:`~.cudaGraphInstantiate`, :py:obj:`~.cudaGraphDestroy`, :py:obj:`~.cudaGraphGetNodes`, :py:obj:`~.cudaGraphGetRootNodes`, :py:obj:`~.cudaGraphGetEdges`, :py:obj:`~.cudaGraphClone`
    """


def cudaGraphDebugDotPrint(graph, path, flags):
    """
    cudaGraphDebugDotPrint(graph, char *path, unsigned int flags)
     Write a DOT file describing graph structure.

        Using the provided `graph`, write to `path` a DOT formatted description
        of the graph. By default this includes the graph topology, node types,
        node id, kernel names and memcpy direction. `flags` can be specified to
        write more detailed information about each node type such as parameter
        values, kernel attributes, node and function handles.

        Parameters
        ----------
        graph : :py:obj:`~.CUgraph` or :py:obj:`~.cudaGraph_t`
            The graph to create a DOT file from
        path : bytes
            The path to write the DOT file to
        flags : unsigned int
            Flags from cudaGraphDebugDotFlags for specifying which additional
            node information to write

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorOperatingSystem`
    """


def cudaGraphDestroy(graph):
    """
    cudaGraphDestroy(graph)
     Destroys a graph.

        Destroys the graph specified by `graph`, as well as all of its nodes.

        Parameters
        ----------
        graph : :py:obj:`~.CUgraph` or :py:obj:`~.cudaGraph_t`
            Graph to destroy

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`

        See Also
        --------
        :py:obj:`~.cudaGraphCreate`
    """


def cudaGraphDestroyNode(node):
    """
    cudaGraphDestroyNode(node)
     Remove a node from the graph.

        Removes `node` from its graph. This operation also severs any
        dependencies of other nodes on `node` and vice versa.

        Dependencies cannot be removed from graphs which contain allocation or
        free nodes. Any attempt to do so will return an error.

        Parameters
        ----------
        node : :py:obj:`~.CUgraphNode` or :py:obj:`~.cudaGraphNode_t`
            Node to remove

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`

        See Also
        --------
        :py:obj:`~.cudaGraphAddChildGraphNode`, :py:obj:`~.cudaGraphAddEmptyNode`, :py:obj:`~.cudaGraphAddKernelNode`, :py:obj:`~.cudaGraphAddHostNode`, :py:obj:`~.cudaGraphAddMemcpyNode`, :py:obj:`~.cudaGraphAddMemsetNode`
    """


def cudaGraphEventRecordNodeGetEvent(node):
    """
    cudaGraphEventRecordNodeGetEvent(node)
     Returns the event associated with an event record node.

        Returns the event of event record node `hNode` in `event_out`.

        Parameters
        ----------
        hNode : :py:obj:`~.CUgraphNode` or :py:obj:`~.cudaGraphNode_t`
            Node to get the event for

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`
        event_out : :py:obj:`~.cudaEvent_t`
            Pointer to return the event

        See Also
        --------
        :py:obj:`~.cudaGraphAddEventRecordNode`, :py:obj:`~.cudaGraphEventRecordNodeSetEvent`, :py:obj:`~.cudaGraphEventWaitNodeGetEvent`, :py:obj:`~.cudaEventRecordWithFlags`, :py:obj:`~.cudaStreamWaitEvent`
    """


def cudaGraphEventRecordNodeSetEvent(node, event):
    """
    cudaGraphEventRecordNodeSetEvent(node, event)
     Sets an event record node's event.

        Sets the event of event record node `hNode` to `event`.

        Parameters
        ----------
        hNode : :py:obj:`~.CUgraphNode` or :py:obj:`~.cudaGraphNode_t`
            Node to set the event for
        event : :py:obj:`~.CUevent` or :py:obj:`~.cudaEvent_t`
            Event to use

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`

        See Also
        --------
        :py:obj:`~.cudaGraphNodeSetParams`, :py:obj:`~.cudaGraphAddEventRecordNode`, :py:obj:`~.cudaGraphEventRecordNodeGetEvent`, :py:obj:`~.cudaGraphEventWaitNodeSetEvent`, :py:obj:`~.cudaEventRecordWithFlags`, :py:obj:`~.cudaStreamWaitEvent`
    """


def cudaGraphEventWaitNodeGetEvent(node):
    """
    cudaGraphEventWaitNodeGetEvent(node)
     Returns the event associated with an event wait node.

        Returns the event of event wait node `hNode` in `event_out`.

        Parameters
        ----------
        hNode : :py:obj:`~.CUgraphNode` or :py:obj:`~.cudaGraphNode_t`
            Node to get the event for

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`
        event_out : :py:obj:`~.cudaEvent_t`
            Pointer to return the event

        See Also
        --------
        :py:obj:`~.cudaGraphAddEventWaitNode`, :py:obj:`~.cudaGraphEventWaitNodeSetEvent`, :py:obj:`~.cudaGraphEventRecordNodeGetEvent`, :py:obj:`~.cudaEventRecordWithFlags`, :py:obj:`~.cudaStreamWaitEvent`
    """


def cudaGraphEventWaitNodeSetEvent(node, event):
    """
    cudaGraphEventWaitNodeSetEvent(node, event)
     Sets an event wait node's event.

        Sets the event of event wait node `hNode` to `event`.

        Parameters
        ----------
        hNode : :py:obj:`~.CUgraphNode` or :py:obj:`~.cudaGraphNode_t`
            Node to set the event for
        event : :py:obj:`~.CUevent` or :py:obj:`~.cudaEvent_t`
            Event to use

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`

        See Also
        --------
        :py:obj:`~.cudaGraphNodeSetParams`, :py:obj:`~.cudaGraphAddEventWaitNode`, :py:obj:`~.cudaGraphEventWaitNodeGetEvent`, :py:obj:`~.cudaGraphEventRecordNodeSetEvent`, :py:obj:`~.cudaEventRecordWithFlags`, :py:obj:`~.cudaStreamWaitEvent`
    """


def cudaGraphExecChildGraphNodeSetParams(hGraphExec, node, childGraph):
    """
    cudaGraphExecChildGraphNodeSetParams(hGraphExec, node, childGraph)
     Updates node parameters in the child graph node in the given graphExec.

        Updates the work represented by `node` in `hGraphExec` as though the
        nodes contained in `node's` graph had the parameters contained in
        `childGraph's` nodes at instantiation. `node` must remain in the graph
        which was used to instantiate `hGraphExec`. Changed edges to and from
        `node` are ignored.

        The modifications only affect future launches of `hGraphExec`. Already
        enqueued or running launches of `hGraphExec` are not affected by this
        call. `node` is also not modified by this call.

        The topology of `childGraph`, as well as the node insertion order, must
        match that of the graph contained in `node`. See
        :py:obj:`~.cudaGraphExecUpdate()` for a list of restrictions on what
        can be updated in an instantiated graph. The update is recursive, so
        child graph nodes contained within the top level child graph will also
        be updated.

        Parameters
        ----------
        hGraphExec : :py:obj:`~.CUgraphExec` or :py:obj:`~.cudaGraphExec_t`
            The executable graph in which to set the specified node
        node : :py:obj:`~.CUgraphNode` or :py:obj:`~.cudaGraphNode_t`
            Host node from the graph which was used to instantiate graphExec
        childGraph : :py:obj:`~.CUgraph` or :py:obj:`~.cudaGraph_t`
            The graph supplying the updated parameters

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`,

        See Also
        --------
        :py:obj:`~.cudaGraphExecNodeSetParams`, :py:obj:`~.cudaGraphAddChildGraphNode`, :py:obj:`~.cudaGraphChildGraphNodeGetGraph`, :py:obj:`~.cudaGraphExecKernelNodeSetParams`, :py:obj:`~.cudaGraphExecMemcpyNodeSetParams`, :py:obj:`~.cudaGraphExecMemsetNodeSetParams`, :py:obj:`~.cudaGraphExecHostNodeSetParams`, :py:obj:`~.cudaGraphExecEventRecordNodeSetEvent`, :py:obj:`~.cudaGraphExecEventWaitNodeSetEvent`, :py:obj:`~.cudaGraphExecExternalSemaphoresSignalNodeSetParams`, :py:obj:`~.cudaGraphExecExternalSemaphoresWaitNodeSetParams`, :py:obj:`~.cudaGraphExecUpdate`, :py:obj:`~.cudaGraphInstantiate`
    """


def cudaGraphExecDestroy(graphExec):
    """
    cudaGraphExecDestroy(graphExec)
     Destroys an executable graph.

        Destroys the executable graph specified by `graphExec`.

        Parameters
        ----------
        graphExec : :py:obj:`~.CUgraphExec` or :py:obj:`~.cudaGraphExec_t`
            Executable graph to destroy

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`

        See Also
        --------
        :py:obj:`~.cudaGraphInstantiate`, :py:obj:`~.cudaGraphUpload`, :py:obj:`~.cudaGraphLaunch`
    """


def cudaGraphExecEventRecordNodeSetEvent(hGraphExec, hNode, event):
    """
    cudaGraphExecEventRecordNodeSetEvent(hGraphExec, hNode, event)
     Sets the event for an event record node in the given graphExec.

        Sets the event of an event record node in an executable graph
        `hGraphExec`. The node is identified by the corresponding node `hNode`
        in the non-executable graph, from which the executable graph was
        instantiated.

        The modifications only affect future launches of `hGraphExec`. Already
        enqueued or running launches of `hGraphExec` are not affected by this
        call. `hNode` is also not modified by this call.

        Parameters
        ----------
        hGraphExec : :py:obj:`~.CUgraphExec` or :py:obj:`~.cudaGraphExec_t`
            The executable graph in which to set the specified node
        hNode : :py:obj:`~.CUgraphNode` or :py:obj:`~.cudaGraphNode_t`
            Event record node from the graph from which graphExec was
            instantiated
        event : :py:obj:`~.CUevent` or :py:obj:`~.cudaEvent_t`
            Updated event to use

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`,

        See Also
        --------
        :py:obj:`~.cudaGraphExecNodeSetParams`, :py:obj:`~.cudaGraphAddEventRecordNode`, :py:obj:`~.cudaGraphEventRecordNodeGetEvent`, :py:obj:`~.cudaGraphEventWaitNodeSetEvent`, :py:obj:`~.cudaEventRecordWithFlags`, :py:obj:`~.cudaStreamWaitEvent`, :py:obj:`~.cudaGraphExecKernelNodeSetParams`, :py:obj:`~.cudaGraphExecMemcpyNodeSetParams`, :py:obj:`~.cudaGraphExecMemsetNodeSetParams`, :py:obj:`~.cudaGraphExecHostNodeSetParams`, :py:obj:`~.cudaGraphExecChildGraphNodeSetParams`, :py:obj:`~.cudaGraphExecEventWaitNodeSetEvent`, :py:obj:`~.cudaGraphExecExternalSemaphoresSignalNodeSetParams`, :py:obj:`~.cudaGraphExecExternalSemaphoresWaitNodeSetParams`, :py:obj:`~.cudaGraphExecUpdate`, :py:obj:`~.cudaGraphInstantiate`
    """


def cudaGraphExecEventWaitNodeSetEvent(hGraphExec, hNode, event):
    """
    cudaGraphExecEventWaitNodeSetEvent(hGraphExec, hNode, event)
     Sets the event for an event wait node in the given graphExec.

        Sets the event of an event wait node in an executable graph
        `hGraphExec`. The node is identified by the corresponding node `hNode`
        in the non-executable graph, from which the executable graph was
        instantiated.

        The modifications only affect future launches of `hGraphExec`. Already
        enqueued or running launches of `hGraphExec` are not affected by this
        call. `hNode` is also not modified by this call.

        Parameters
        ----------
        hGraphExec : :py:obj:`~.CUgraphExec` or :py:obj:`~.cudaGraphExec_t`
            The executable graph in which to set the specified node
        hNode : :py:obj:`~.CUgraphNode` or :py:obj:`~.cudaGraphNode_t`
            Event wait node from the graph from which graphExec was
            instantiated
        event : :py:obj:`~.CUevent` or :py:obj:`~.cudaEvent_t`
            Updated event to use

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`,

        See Also
        --------
        :py:obj:`~.cudaGraphExecNodeSetParams`, :py:obj:`~.cudaGraphAddEventWaitNode`, :py:obj:`~.cudaGraphEventWaitNodeGetEvent`, :py:obj:`~.cudaGraphEventRecordNodeSetEvent`, :py:obj:`~.cudaEventRecordWithFlags`, :py:obj:`~.cudaStreamWaitEvent`, :py:obj:`~.cudaGraphExecKernelNodeSetParams`, :py:obj:`~.cudaGraphExecMemcpyNodeSetParams`, :py:obj:`~.cudaGraphExecMemsetNodeSetParams`, :py:obj:`~.cudaGraphExecHostNodeSetParams`, :py:obj:`~.cudaGraphExecChildGraphNodeSetParams`, :py:obj:`~.cudaGraphExecEventRecordNodeSetEvent`, :py:obj:`~.cudaGraphExecExternalSemaphoresSignalNodeSetParams`, :py:obj:`~.cudaGraphExecExternalSemaphoresWaitNodeSetParams`, :py:obj:`~.cudaGraphExecUpdate`, :py:obj:`~.cudaGraphInstantiate`
    """


def cudaGraphExecExternalSemaphoresSignalNodeSetParams(hGraphExec, hNode, nodeParams: 'Optional[cudaExternalSemaphoreSignalNodeParams]'):
    """
    cudaGraphExecExternalSemaphoresSignalNodeSetParams(hGraphExec, hNode, cudaExternalSemaphoreSignalNodeParams nodeParams: Optional[cudaExternalSemaphoreSignalNodeParams])
     Sets the parameters for an external semaphore signal node in the given graphExec.

        Sets the parameters of an external semaphore signal node in an
        executable graph `hGraphExec`. The node is identified by the
        corresponding node `hNode` in the non-executable graph, from which the
        executable graph was instantiated.

        `hNode` must not have been removed from the original graph.

        The modifications only affect future launches of `hGraphExec`. Already
        enqueued or running launches of `hGraphExec` are not affected by this
        call. `hNode` is also not modified by this call.

        Changing `nodeParams->numExtSems` is not supported.

        Parameters
        ----------
        hGraphExec : :py:obj:`~.CUgraphExec` or :py:obj:`~.cudaGraphExec_t`
            The executable graph in which to set the specified node
        hNode : :py:obj:`~.CUgraphNode` or :py:obj:`~.cudaGraphNode_t`
            semaphore signal node from the graph from which graphExec was
            instantiated
        nodeParams : :py:obj:`~.cudaExternalSemaphoreSignalNodeParams`
            Updated Parameters to set

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`,

        See Also
        --------
        :py:obj:`~.cudaGraphExecNodeSetParams`, :py:obj:`~.cudaGraphAddExternalSemaphoresSignalNode`, :py:obj:`~.cudaImportExternalSemaphore`, :py:obj:`~.cudaSignalExternalSemaphoresAsync`, :py:obj:`~.cudaWaitExternalSemaphoresAsync`, :py:obj:`~.cudaGraphExecKernelNodeSetParams`, :py:obj:`~.cudaGraphExecMemcpyNodeSetParams`, :py:obj:`~.cudaGraphExecMemsetNodeSetParams`, :py:obj:`~.cudaGraphExecHostNodeSetParams`, :py:obj:`~.cudaGraphExecChildGraphNodeSetParams`, :py:obj:`~.cudaGraphExecEventRecordNodeSetEvent`, :py:obj:`~.cudaGraphExecEventWaitNodeSetEvent`, :py:obj:`~.cudaGraphExecExternalSemaphoresWaitNodeSetParams`, :py:obj:`~.cudaGraphExecUpdate`, :py:obj:`~.cudaGraphInstantiate`
    """


def cudaGraphExecExternalSemaphoresWaitNodeSetParams(hGraphExec, hNode, nodeParams: 'Optional[cudaExternalSemaphoreWaitNodeParams]'):
    """
    cudaGraphExecExternalSemaphoresWaitNodeSetParams(hGraphExec, hNode, cudaExternalSemaphoreWaitNodeParams nodeParams: Optional[cudaExternalSemaphoreWaitNodeParams])
     Sets the parameters for an external semaphore wait node in the given graphExec.

        Sets the parameters of an external semaphore wait node in an executable
        graph `hGraphExec`. The node is identified by the corresponding node
        `hNode` in the non-executable graph, from which the executable graph
        was instantiated.

        `hNode` must not have been removed from the original graph.

        The modifications only affect future launches of `hGraphExec`. Already
        enqueued or running launches of `hGraphExec` are not affected by this
        call. `hNode` is also not modified by this call.

        Changing `nodeParams->numExtSems` is not supported.

        Parameters
        ----------
        hGraphExec : :py:obj:`~.CUgraphExec` or :py:obj:`~.cudaGraphExec_t`
            The executable graph in which to set the specified node
        hNode : :py:obj:`~.CUgraphNode` or :py:obj:`~.cudaGraphNode_t`
            semaphore wait node from the graph from which graphExec was
            instantiated
        nodeParams : :py:obj:`~.cudaExternalSemaphoreWaitNodeParams`
            Updated Parameters to set

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`,

        See Also
        --------
        :py:obj:`~.cudaGraphExecNodeSetParams`, :py:obj:`~.cudaGraphAddExternalSemaphoresWaitNode`, :py:obj:`~.cudaImportExternalSemaphore`, :py:obj:`~.cudaSignalExternalSemaphoresAsync`, :py:obj:`~.cudaWaitExternalSemaphoresAsync`, :py:obj:`~.cudaGraphExecKernelNodeSetParams`, :py:obj:`~.cudaGraphExecMemcpyNodeSetParams`, :py:obj:`~.cudaGraphExecMemsetNodeSetParams`, :py:obj:`~.cudaGraphExecHostNodeSetParams`, :py:obj:`~.cudaGraphExecChildGraphNodeSetParams`, :py:obj:`~.cudaGraphExecEventRecordNodeSetEvent`, :py:obj:`~.cudaGraphExecEventWaitNodeSetEvent`, :py:obj:`~.cudaGraphExecExternalSemaphoresSignalNodeSetParams`, :py:obj:`~.cudaGraphExecUpdate`, :py:obj:`~.cudaGraphInstantiate`
    """


def cudaGraphExecGetFlags(graphExec):
    """
    cudaGraphExecGetFlags(graphExec)
     Query the instantiation flags of an executable graph.

        Returns the flags that were passed to instantiation for the given
        executable graph. :py:obj:`~.cudaGraphInstantiateFlagUpload` will not
        be returned by this API as it does not affect the resulting executable
        graph.

        Parameters
        ----------
        graphExec : :py:obj:`~.CUgraphExec` or :py:obj:`~.cudaGraphExec_t`
            The executable graph to query

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`
        flags : unsigned long long
            Returns the instantiation flags

        See Also
        --------
        :py:obj:`~.cudaGraphInstantiate`, :py:obj:`~.cudaGraphInstantiateWithFlags`, :py:obj:`~.cudaGraphInstantiateWithParams`
    """


def cudaGraphExecHostNodeSetParams(hGraphExec, node, pNodeParams: 'Optional[cudaHostNodeParams]'):
    """
    cudaGraphExecHostNodeSetParams(hGraphExec, node, cudaHostNodeParams pNodeParams: Optional[cudaHostNodeParams])
     Sets the parameters for a host node in the given graphExec.

        Updates the work represented by `node` in `hGraphExec` as though `node`
        had contained `pNodeParams` at instantiation. `node` must remain in the
        graph which was used to instantiate `hGraphExec`. Changed edges to and
        from `node` are ignored.

        The modifications only affect future launches of `hGraphExec`. Already
        enqueued or running launches of `hGraphExec` are not affected by this
        call. `node` is also not modified by this call.

        Parameters
        ----------
        hGraphExec : :py:obj:`~.CUgraphExec` or :py:obj:`~.cudaGraphExec_t`
            The executable graph in which to set the specified node
        node : :py:obj:`~.CUgraphNode` or :py:obj:`~.cudaGraphNode_t`
            Host node from the graph which was used to instantiate graphExec
        pNodeParams : :py:obj:`~.cudaHostNodeParams`
            Updated Parameters to set

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`,

        See Also
        --------
        :py:obj:`~.cudaGraphExecNodeSetParams`, :py:obj:`~.cudaGraphAddHostNode`, :py:obj:`~.cudaGraphHostNodeSetParams`, :py:obj:`~.cudaGraphExecKernelNodeSetParams`, :py:obj:`~.cudaGraphExecMemcpyNodeSetParams`, :py:obj:`~.cudaGraphExecMemsetNodeSetParams`, :py:obj:`~.cudaGraphExecChildGraphNodeSetParams`, :py:obj:`~.cudaGraphExecEventRecordNodeSetEvent`, :py:obj:`~.cudaGraphExecEventWaitNodeSetEvent`, :py:obj:`~.cudaGraphExecExternalSemaphoresSignalNodeSetParams`, :py:obj:`~.cudaGraphExecExternalSemaphoresWaitNodeSetParams`, :py:obj:`~.cudaGraphExecUpdate`, :py:obj:`~.cudaGraphInstantiate`
    """


def cudaGraphExecKernelNodeSetParams(hGraphExec, node, pNodeParams: 'Optional[cudaKernelNodeParams]'):
    """
    cudaGraphExecKernelNodeSetParams(hGraphExec, node, cudaKernelNodeParams pNodeParams: Optional[cudaKernelNodeParams])
     Sets the parameters for a kernel node in the given graphExec.

        Sets the parameters of a kernel node in an executable graph
        `hGraphExec`. The node is identified by the corresponding node `node`
        in the non-executable graph, from which the executable graph was
        instantiated.

        `node` must not have been removed from the original graph. All
        `nodeParams` fields may change, but the following restrictions apply to
        `func` updates:

        - The owning device of the function cannot change.

        - A node whose function originally did not use CUDA dynamic parallelism
          cannot be updated to a function which uses CDP

        - A node whose function originally did not make device-side update
          calls cannot be updated to a function which makes device-side update
          calls.

        - If `hGraphExec` was not instantiated for device launch, a node whose
          function originally did not use device-side
          :py:obj:`~.cudaGraphLaunch()` cannot be updated to a function which
          uses device-side :py:obj:`~.cudaGraphLaunch()` unless the node
          resides on the same device as nodes which contained such calls at
          instantiate-time. If no such calls were present at instantiation,
          these updates cannot be performed at all.

        The modifications only affect future launches of `hGraphExec`. Already
        enqueued or running launches of `hGraphExec` are not affected by this
        call. `node` is also not modified by this call.

        If `node` is a device-updatable kernel node, the next upload/launch of
        `hGraphExec` will overwrite any previous device-side updates.
        Additionally, applying host updates to a device-updatable kernel node
        while it is being updated from the device will result in undefined
        behavior.

        Parameters
        ----------
        hGraphExec : :py:obj:`~.CUgraphExec` or :py:obj:`~.cudaGraphExec_t`
            The executable graph in which to set the specified node
        node : :py:obj:`~.CUgraphNode` or :py:obj:`~.cudaGraphNode_t`
            kernel node from the graph from which graphExec was instantiated
        pNodeParams : :py:obj:`~.cudaKernelNodeParams`
            Updated Parameters to set

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`,

        See Also
        --------
        :py:obj:`~.cudaGraphExecNodeSetParams`, :py:obj:`~.cudaGraphAddKernelNode`, :py:obj:`~.cudaGraphKernelNodeSetParams`, :py:obj:`~.cudaGraphExecMemcpyNodeSetParams`, :py:obj:`~.cudaGraphExecMemsetNodeSetParams`, :py:obj:`~.cudaGraphExecHostNodeSetParams`, :py:obj:`~.cudaGraphExecChildGraphNodeSetParams`, :py:obj:`~.cudaGraphExecEventRecordNodeSetEvent`, :py:obj:`~.cudaGraphExecEventWaitNodeSetEvent`, :py:obj:`~.cudaGraphExecExternalSemaphoresSignalNodeSetParams`, :py:obj:`~.cudaGraphExecExternalSemaphoresWaitNodeSetParams`, :py:obj:`~.cudaGraphExecUpdate`, :py:obj:`~.cudaGraphInstantiate`
    """


def cudaGraphExecMemcpyNodeSetParams(hGraphExec, node, pNodeParams: 'Optional[cudaMemcpy3DParms]'):
    """
    cudaGraphExecMemcpyNodeSetParams(hGraphExec, node, cudaMemcpy3DParms pNodeParams: Optional[cudaMemcpy3DParms])
     Sets the parameters for a memcpy node in the given graphExec.

        Updates the work represented by `node` in `hGraphExec` as though `node`
        had contained `pNodeParams` at instantiation. `node` must remain in the
        graph which was used to instantiate `hGraphExec`. Changed edges to and
        from `node` are ignored.

        The source and destination memory in `pNodeParams` must be allocated
        from the same contexts as the original source and destination memory.
        Both the instantiation-time memory operands and the memory operands in
        `pNodeParams` must be 1-dimensional. Zero-length operations are not
        supported.

        The modifications only affect future launches of `hGraphExec`. Already
        enqueued or running launches of `hGraphExec` are not affected by this
        call. `node` is also not modified by this call.

        Returns :py:obj:`~.cudaErrorInvalidValue` if the memory operands'
        mappings changed or either the original or new memory operands are
        multidimensional.

        Parameters
        ----------
        hGraphExec : :py:obj:`~.CUgraphExec` or :py:obj:`~.cudaGraphExec_t`
            The executable graph in which to set the specified node
        node : :py:obj:`~.CUgraphNode` or :py:obj:`~.cudaGraphNode_t`
            Memcpy node from the graph which was used to instantiate graphExec
        pNodeParams : :py:obj:`~.cudaMemcpy3DParms`
            Updated Parameters to set

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`,

        See Also
        --------
        :py:obj:`~.cudaGraphExecNodeSetParams`, :py:obj:`~.cudaGraphAddMemcpyNode`, :py:obj:`~.cudaGraphMemcpyNodeSetParams`, :py:obj:`~.cudaGraphExecMemcpyNodeSetParamsToSymbol`, :py:obj:`~.cudaGraphExecMemcpyNodeSetParamsFromSymbol`, :py:obj:`~.cudaGraphExecMemcpyNodeSetParams1D`, :py:obj:`~.cudaGraphExecKernelNodeSetParams`, :py:obj:`~.cudaGraphExecMemsetNodeSetParams`, :py:obj:`~.cudaGraphExecHostNodeSetParams`, :py:obj:`~.cudaGraphExecChildGraphNodeSetParams`, :py:obj:`~.cudaGraphExecEventRecordNodeSetEvent`, :py:obj:`~.cudaGraphExecEventWaitNodeSetEvent`, :py:obj:`~.cudaGraphExecExternalSemaphoresSignalNodeSetParams`, :py:obj:`~.cudaGraphExecExternalSemaphoresWaitNodeSetParams`, :py:obj:`~.cudaGraphExecUpdate`, :py:obj:`~.cudaGraphInstantiate`
    """


def cudaGraphExecMemcpyNodeSetParams1D(hGraphExec, node, dst, src, count, kind: 'cudaMemcpyKind'):
    """
    cudaGraphExecMemcpyNodeSetParams1D(hGraphExec, node, dst, src, size_t count, kind: cudaMemcpyKind)
     Sets the parameters for a memcpy node in the given graphExec to perform a 1-dimensional copy.

        Updates the work represented by `node` in `hGraphExec` as though `node`
        had contained the given params at instantiation. `node` must remain in
        the graph which was used to instantiate `hGraphExec`. Changed edges to
        and from `node` are ignored.

        `src` and `dst` must be allocated from the same contexts as the
        original source and destination memory. The instantiation-time memory
        operands must be 1-dimensional. Zero-length operations are not
        supported.

        The modifications only affect future launches of `hGraphExec`. Already
        enqueued or running launches of `hGraphExec` are not affected by this
        call. `node` is also not modified by this call.

        Returns :py:obj:`~.cudaErrorInvalidValue` if the memory operands'
        mappings changed or the original memory operands are multidimensional.

        Parameters
        ----------
        hGraphExec : :py:obj:`~.CUgraphExec` or :py:obj:`~.cudaGraphExec_t`
            The executable graph in which to set the specified node
        node : :py:obj:`~.CUgraphNode` or :py:obj:`~.cudaGraphNode_t`
            Memcpy node from the graph which was used to instantiate graphExec
        dst : Any
            Destination memory address
        src : Any
            Source memory address
        count : size_t
            Size in bytes to copy
        kind : :py:obj:`~.cudaMemcpyKind`
            Type of transfer

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`

        See Also
        --------
        :py:obj:`~.cudaGraphAddMemcpyNode`, :py:obj:`~.cudaGraphAddMemcpyNode1D`, :py:obj:`~.cudaGraphMemcpyNodeSetParams`, :py:obj:`~.cudaGraphMemcpyNodeSetParams1D`, :py:obj:`~.cudaGraphExecMemcpyNodeSetParams`, :py:obj:`~.cudaGraphExecKernelNodeSetParams`, :py:obj:`~.cudaGraphExecMemsetNodeSetParams`, :py:obj:`~.cudaGraphExecHostNodeSetParams`, :py:obj:`~.cudaGraphExecChildGraphNodeSetParams`, :py:obj:`~.cudaGraphExecEventRecordNodeSetEvent`, :py:obj:`~.cudaGraphExecEventWaitNodeSetEvent`, :py:obj:`~.cudaGraphExecExternalSemaphoresSignalNodeSetParams`, :py:obj:`~.cudaGraphExecExternalSemaphoresWaitNodeSetParams`, :py:obj:`~.cudaGraphExecUpdate`, :py:obj:`~.cudaGraphInstantiate`
    """


def cudaGraphExecMemsetNodeSetParams(hGraphExec, node, pNodeParams: 'Optional[cudaMemsetParams]'):
    """
    cudaGraphExecMemsetNodeSetParams(hGraphExec, node, cudaMemsetParams pNodeParams: Optional[cudaMemsetParams])
     Sets the parameters for a memset node in the given graphExec.

        Updates the work represented by `node` in `hGraphExec` as though `node`
        had contained `pNodeParams` at instantiation. `node` must remain in the
        graph which was used to instantiate `hGraphExec`. Changed edges to and
        from `node` are ignored.

        Zero sized operations are not supported.

        The new destination pointer in `pNodeParams` must be to the same kind
        of allocation as the original destination pointer and have the same
        context association and device mapping as the original destination
        pointer.

        Both the value and pointer address may be updated.   Changing other
        aspects of the memset (width, height, element size or pitch) may cause
        the update to be rejected. Specifically, for 2d memsets, all dimension
        changes are rejected. For 1d memsets, changes in height are explicitly
        rejected and other changes are oportunistically allowed if the
        resulting work maps onto the work resources already allocated for the
        node.

        The modifications only affect future launches of `hGraphExec`. Already
        enqueued or running launches of `hGraphExec` are not affected by this
        call. `node` is also not modified by this call.

        Parameters
        ----------
        hGraphExec : :py:obj:`~.CUgraphExec` or :py:obj:`~.cudaGraphExec_t`
            The executable graph in which to set the specified node
        node : :py:obj:`~.CUgraphNode` or :py:obj:`~.cudaGraphNode_t`
            Memset node from the graph which was used to instantiate graphExec
        pNodeParams : :py:obj:`~.cudaMemsetParams`
            Updated Parameters to set

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`,

        See Also
        --------
        :py:obj:`~.cudaGraphExecNodeSetParams`, :py:obj:`~.cudaGraphAddMemsetNode`, :py:obj:`~.cudaGraphMemsetNodeSetParams`, :py:obj:`~.cudaGraphExecKernelNodeSetParams`, :py:obj:`~.cudaGraphExecMemcpyNodeSetParams`, :py:obj:`~.cudaGraphExecHostNodeSetParams`, :py:obj:`~.cudaGraphExecChildGraphNodeSetParams`, :py:obj:`~.cudaGraphExecEventRecordNodeSetEvent`, :py:obj:`~.cudaGraphExecEventWaitNodeSetEvent`, :py:obj:`~.cudaGraphExecExternalSemaphoresSignalNodeSetParams`, :py:obj:`~.cudaGraphExecExternalSemaphoresWaitNodeSetParams`, :py:obj:`~.cudaGraphExecUpdate`, :py:obj:`~.cudaGraphInstantiate`
    """


def cudaGraphExecNodeSetParams(graphExec, node, nodeParams: 'Optional[cudaGraphNodeParams]'):
    """
    cudaGraphExecNodeSetParams(graphExec, node, cudaGraphNodeParams nodeParams: Optional[cudaGraphNodeParams])
     Update's a graph node's parameters in an instantiated graph.

        Sets the parameters of a node in an executable graph `graphExec`. The
        node is identified by the corresponding node `node` in the non-
        executable graph from which the executable graph was instantiated.
        `node` must not have been removed from the original graph.

        The modifications only affect future launches of `graphExec`. Already
        enqueued or running launches of `graphExec` are not affected by this
        call. `node` is also not modified by this call.

        Allowed changes to parameters on executable graphs are as follows:

        **View CUDA Toolkit Documentation for a table example**

        Parameters
        ----------
        graphExec : :py:obj:`~.CUgraphExec` or :py:obj:`~.cudaGraphExec_t`
            The executable graph in which to update the specified node
        node : :py:obj:`~.CUgraphNode` or :py:obj:`~.cudaGraphNode_t`
            Corresponding node from the graph from which graphExec was
            instantiated
        nodeParams : :py:obj:`~.cudaGraphNodeParams`
            Updated Parameters to set

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidDeviceFunction`, :py:obj:`~.cudaErrorNotSupported`

        See Also
        --------
        :py:obj:`~.cudaGraphAddNode`, :py:obj:`~.cudaGraphNodeSetParams` :py:obj:`~.cudaGraphExecUpdate`, :py:obj:`~.cudaGraphInstantiate`
    """


def cudaGraphExecUpdate(hGraphExec, hGraph):
    """
    cudaGraphExecUpdate(hGraphExec, hGraph)
     Check whether an executable graph can be updated with a graph and perform the update if possible.

        Updates the node parameters in the instantiated graph specified by
        `hGraphExec` with the node parameters in a topologically identical
        graph specified by `hGraph`.

        Limitations:

        - Kernel nodes:

          - The owning context of the function cannot change.

          - A node whose function originally did not use CUDA dynamic
            parallelism cannot be updated to a function which uses CDP.

          - A node whose function originally did not make device-side update
            calls cannot be updated to a function which makes device-side
            update calls.

          - A cooperative node cannot be updated to a non-cooperative node, and
            vice-versa.

          - If the graph was instantiated with
            cudaGraphInstantiateFlagUseNodePriority, the priority attribute
            cannot change. Equality is checked on the originally requested
            priority values, before they are clamped to the device's supported
            range.

          - If `hGraphExec` was not instantiated for device launch, a node
            whose function originally did not use device-side
            :py:obj:`~.cudaGraphLaunch()` cannot be updated to a function which
            uses device-side :py:obj:`~.cudaGraphLaunch()` unless the node
            resides on the same device as nodes which contained such calls at
            instantiate-time. If no such calls were present at instantiation,
            these updates cannot be performed at all.

          - Neither `hGraph` nor `hGraphExec` may contain device-updatable
            kernel nodes.

        - Memset and memcpy nodes:

          - The CUDA device(s) to which the operand(s) was allocated/mapped
            cannot change.

          - The source/destination memory must be allocated from the same
            contexts as the original source/destination memory.

          - For 2d memsets, only address and assinged value may be updated.

          - For 1d memsets, updating dimensions is also allowed, but may fail
            if the resulting operation doesn't map onto the work resources
            already allocated for the node.

        - Additional memcpy node restrictions:

          - Changing either the source or destination memory type(i.e.
            CU_MEMORYTYPE_DEVICE, CU_MEMORYTYPE_ARRAY, etc.) is not supported.

        - Conditional nodes:

          - Changing node parameters is not supported.

          - Changeing parameters of nodes within the conditional body graph is
            subject to the rules above.

          - Conditional handle flags and default values are updated as part of
            the graph update.

        Note: The API may add further restrictions in future releases. The
        return code should always be checked.

        cudaGraphExecUpdate sets the result member of `resultInfo` to
        cudaGraphExecUpdateErrorTopologyChanged under the following conditions:

        - The count of nodes directly in `hGraphExec` and `hGraph` differ, in
          which case resultInfo->errorNode is set to NULL.

        - `hGraph` has more exit nodes than `hGraph`, in which case
          resultInfo->errorNode is set to one of the exit nodes in hGraph.

        - A node in `hGraph` has a different number of dependencies than the
          node from `hGraphExec` it is paired with, in which case
          resultInfo->errorNode is set to the node from `hGraph`.

        - A node in `hGraph` has a dependency that does not match with the
          corresponding dependency of the paired node from `hGraphExec`.
          resultInfo->errorNode will be set to the node from `hGraph`.
          resultInfo->errorFromNode will be set to the mismatched dependency.
          The dependencies are paired based on edge order and a dependency does
          not match when the nodes are already paired based on other edges
          examined in the graph.

        cudaGraphExecUpdate sets `the` result member of `resultInfo` to:

        - cudaGraphExecUpdateError if passed an invalid value.

        - cudaGraphExecUpdateErrorTopologyChanged if the graph topology changed

        - cudaGraphExecUpdateErrorNodeTypeChanged if the type of a node
          changed, in which case `hErrorNode_out` is set to the node from
          `hGraph`.

        - cudaGraphExecUpdateErrorFunctionChanged if the function of a kernel
          node changed (CUDA driver < 11.2)

        - cudaGraphExecUpdateErrorUnsupportedFunctionChange if the func field
          of a kernel changed in an unsupported way(see note above), in which
          case `hErrorNode_out` is set to the node from `hGraph`

        - cudaGraphExecUpdateErrorParametersChanged if any parameters to a node
          changed in a way that is not supported, in which case
          `hErrorNode_out` is set to the node from `hGraph`

        - cudaGraphExecUpdateErrorAttributesChanged if any attributes of a node
          changed in a way that is not supported, in which case
          `hErrorNode_out` is set to the node from `hGraph`

        - cudaGraphExecUpdateErrorNotSupported if something about a node is
          unsupported, like the node's type or configuration, in which case
          `hErrorNode_out` is set to the node from `hGraph`

        If the update fails for a reason not listed above, the result member of
        `resultInfo` will be set to cudaGraphExecUpdateError. If the update
        succeeds, the result member will be set to cudaGraphExecUpdateSuccess.

        cudaGraphExecUpdate returns cudaSuccess when the updated was performed
        successfully. It returns cudaErrorGraphExecUpdateFailure if the graph
        update was not performed because it included changes which violated
        constraints specific to instantiated graph update.

        Parameters
        ----------
        hGraphExec : :py:obj:`~.CUgraphExec` or :py:obj:`~.cudaGraphExec_t`
            The instantiated graph to be updated
        hGraph : :py:obj:`~.CUgraph` or :py:obj:`~.cudaGraph_t`
            The graph containing the updated parameters

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorGraphExecUpdateFailure`,
        resultInfo : :py:obj:`~.cudaGraphExecUpdateResultInfo`
            the error info structure

        See Also
        --------
        :py:obj:`~.cudaGraphInstantiate`
    """


def cudaGraphExternalSemaphoresSignalNodeGetParams(hNode):
    """
    cudaGraphExternalSemaphoresSignalNodeGetParams(hNode)
     Returns an external semaphore signal node's parameters.

        Returns the parameters of an external semaphore signal node `hNode` in
        `params_out`. The `extSemArray` and `paramsArray` returned in
        `params_out`, are owned by the node. This memory remains valid until
        the node is destroyed or its parameters are modified, and should not be
        modified directly. Use
        :py:obj:`~.cudaGraphExternalSemaphoresSignalNodeSetParams` to update
        the parameters of this node.

        Parameters
        ----------
        hNode : :py:obj:`~.CUgraphNode` or :py:obj:`~.cudaGraphNode_t`
            Node to get the parameters for

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`
        params_out : :py:obj:`~.cudaExternalSemaphoreSignalNodeParams`
            Pointer to return the parameters

        See Also
        --------
        :py:obj:`~.cudaLaunchKernel`, :py:obj:`~.cudaGraphAddExternalSemaphoresSignalNode`, :py:obj:`~.cudaGraphExternalSemaphoresSignalNodeSetParams`, :py:obj:`~.cudaGraphAddExternalSemaphoresWaitNode`, :py:obj:`~.cudaSignalExternalSemaphoresAsync`, :py:obj:`~.cudaWaitExternalSemaphoresAsync`
    """


def cudaGraphExternalSemaphoresSignalNodeSetParams(hNode, nodeParams: 'Optional[cudaExternalSemaphoreSignalNodeParams]'):
    """
    cudaGraphExternalSemaphoresSignalNodeSetParams(hNode, cudaExternalSemaphoreSignalNodeParams nodeParams: Optional[cudaExternalSemaphoreSignalNodeParams])
     Sets an external semaphore signal node's parameters.

        Sets the parameters of an external semaphore signal node `hNode` to
        `nodeParams`.

        Parameters
        ----------
        hNode : :py:obj:`~.CUgraphNode` or :py:obj:`~.cudaGraphNode_t`
            Node to set the parameters for
        nodeParams : :py:obj:`~.cudaExternalSemaphoreSignalNodeParams`
            Parameters to copy

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`

        See Also
        --------
        :py:obj:`~.cudaGraphNodeSetParams`, :py:obj:`~.cudaGraphAddExternalSemaphoresSignalNode`, :py:obj:`~.cudaGraphExternalSemaphoresSignalNodeSetParams`, :py:obj:`~.cudaGraphAddExternalSemaphoresWaitNode`, :py:obj:`~.cudaSignalExternalSemaphoresAsync`, :py:obj:`~.cudaWaitExternalSemaphoresAsync`
    """


def cudaGraphExternalSemaphoresWaitNodeGetParams(hNode):
    """
    cudaGraphExternalSemaphoresWaitNodeGetParams(hNode)
     Returns an external semaphore wait node's parameters.

        Returns the parameters of an external semaphore wait node `hNode` in
        `params_out`. The `extSemArray` and `paramsArray` returned in
        `params_out`, are owned by the node. This memory remains valid until
        the node is destroyed or its parameters are modified, and should not be
        modified directly. Use
        :py:obj:`~.cudaGraphExternalSemaphoresSignalNodeSetParams` to update
        the parameters of this node.

        Parameters
        ----------
        hNode : :py:obj:`~.CUgraphNode` or :py:obj:`~.cudaGraphNode_t`
            Node to get the parameters for

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`
        params_out : :py:obj:`~.cudaExternalSemaphoreWaitNodeParams`
            Pointer to return the parameters

        See Also
        --------
        :py:obj:`~.cudaLaunchKernel`, :py:obj:`~.cudaGraphAddExternalSemaphoresWaitNode`, :py:obj:`~.cudaGraphExternalSemaphoresWaitNodeSetParams`, :py:obj:`~.cudaGraphAddExternalSemaphoresWaitNode`, :py:obj:`~.cudaSignalExternalSemaphoresAsync`, :py:obj:`~.cudaWaitExternalSemaphoresAsync`
    """


def cudaGraphExternalSemaphoresWaitNodeSetParams(hNode, nodeParams: 'Optional[cudaExternalSemaphoreWaitNodeParams]'):
    """
    cudaGraphExternalSemaphoresWaitNodeSetParams(hNode, cudaExternalSemaphoreWaitNodeParams nodeParams: Optional[cudaExternalSemaphoreWaitNodeParams])
     Sets an external semaphore wait node's parameters.

        Sets the parameters of an external semaphore wait node `hNode` to
        `nodeParams`.

        Parameters
        ----------
        hNode : :py:obj:`~.CUgraphNode` or :py:obj:`~.cudaGraphNode_t`
            Node to set the parameters for
        nodeParams : :py:obj:`~.cudaExternalSemaphoreWaitNodeParams`
            Parameters to copy

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`

        See Also
        --------
        :py:obj:`~.cudaGraphNodeSetParams`, :py:obj:`~.cudaGraphAddExternalSemaphoresWaitNode`, :py:obj:`~.cudaGraphExternalSemaphoresWaitNodeSetParams`, :py:obj:`~.cudaGraphAddExternalSemaphoresWaitNode`, :py:obj:`~.cudaSignalExternalSemaphoresAsync`, :py:obj:`~.cudaWaitExternalSemaphoresAsync`
    """


def cudaGraphGetEdges(graph, numEdges=0):
    """
    cudaGraphGetEdges(graph, size_t numEdges=0)
     Returns a graph's dependency edges.

        Returns a list of `graph's` dependency edges. Edges are returned via
        corresponding indices in `from` and `to`; that is, the node in `to`[i]
        has a dependency on the node in `from`[i]. `from` and `to` may both be
        NULL, in which case this function only returns the number of edges in
        `numEdges`. Otherwise, `numEdges` entries will be filled in. If
        `numEdges` is higher than the actual number of edges, the remaining
        entries in `from` and `to` will be set to NULL, and the number of edges
        actually returned will be written to `numEdges`.

        Parameters
        ----------
        graph : :py:obj:`~.CUgraph` or :py:obj:`~.cudaGraph_t`
            Graph to get the edges from
        numEdges : int
            See description

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`
        from : List[:py:obj:`~.cudaGraphNode_t`]
            Location to return edge endpoints
        to : List[:py:obj:`~.cudaGraphNode_t`]
            Location to return edge endpoints
        numEdges : int
            See description

        See Also
        --------
        :py:obj:`~.cudaGraphGetNodes`, :py:obj:`~.cudaGraphGetRootNodes`, :py:obj:`~.cudaGraphAddDependencies`, :py:obj:`~.cudaGraphRemoveDependencies`, :py:obj:`~.cudaGraphNodeGetDependencies`, :py:obj:`~.cudaGraphNodeGetDependentNodes`
    """


def cudaGraphGetEdges_v2(graph, numEdges=0):
    """
    cudaGraphGetEdges_v2(graph, size_t numEdges=0)
     Returns a graph's dependency edges (12.3+)

        Returns a list of `graph's` dependency edges. Edges are returned via
        corresponding indices in `from`, `to` and `edgeData`; that is, the node
        in `to`[i] has a dependency on the node in `from`[i] with data
        `edgeData`[i]. `from` and `to` may both be NULL, in which case this
        function only returns the number of edges in `numEdges`. Otherwise,
        `numEdges` entries will be filled in. If `numEdges` is higher than the
        actual number of edges, the remaining entries in `from` and `to` will
        be set to NULL, and the number of edges actually returned will be
        written to `numEdges`. `edgeData` may alone be NULL, in which case the
        edges must all have default (zeroed) edge data. Attempting a losst
        query via NULL `edgeData` will result in
        :py:obj:`~.cudaErrorLossyQuery`. If `edgeData` is non-NULL then `from`
        and `to` must be as well.

        Parameters
        ----------
        graph : :py:obj:`~.CUgraph` or :py:obj:`~.cudaGraph_t`
            Graph to get the edges from
        numEdges : int
            See description

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorLossyQuery`, :py:obj:`~.cudaErrorInvalidValue`
        from : List[:py:obj:`~.cudaGraphNode_t`]
            Location to return edge endpoints
        to : List[:py:obj:`~.cudaGraphNode_t`]
            Location to return edge endpoints
        edgeData : List[:py:obj:`~.cudaGraphEdgeData`]
            Optional location to return edge data
        numEdges : int
            See description

        See Also
        --------
        :py:obj:`~.cudaGraphGetNodes`, :py:obj:`~.cudaGraphGetRootNodes`, :py:obj:`~.cudaGraphAddDependencies`, :py:obj:`~.cudaGraphRemoveDependencies`, :py:obj:`~.cudaGraphNodeGetDependencies`, :py:obj:`~.cudaGraphNodeGetDependentNodes`
    """


def cudaGraphGetNodes(graph, numNodes=0):
    """
    cudaGraphGetNodes(graph, size_t numNodes=0)
     Returns a graph's nodes.

        Returns a list of `graph's` nodes. `nodes` may be NULL, in which case
        this function will return the number of nodes in `numNodes`. Otherwise,
        `numNodes` entries will be filled in. If `numNodes` is higher than the
        actual number of nodes, the remaining entries in `nodes` will be set to
        NULL, and the number of nodes actually obtained will be returned in
        `numNodes`.

        Parameters
        ----------
        graph : :py:obj:`~.CUgraph` or :py:obj:`~.cudaGraph_t`
            Graph to query
        numNodes : int
            See description

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`
        nodes : List[:py:obj:`~.cudaGraphNode_t`]
            Pointer to return the nodes
        numNodes : int
            See description

        See Also
        --------
        :py:obj:`~.cudaGraphCreate`, :py:obj:`~.cudaGraphGetRootNodes`, :py:obj:`~.cudaGraphGetEdges`, :py:obj:`~.cudaGraphNodeGetType`, :py:obj:`~.cudaGraphNodeGetDependencies`, :py:obj:`~.cudaGraphNodeGetDependentNodes`
    """


def cudaGraphGetRootNodes(graph, pNumRootNodes=0):
    """
    cudaGraphGetRootNodes(graph, size_t pNumRootNodes=0)
     Returns a graph's root nodes.

        Returns a list of `graph's` root nodes. `pRootNodes` may be NULL, in
        which case this function will return the number of root nodes in
        `pNumRootNodes`. Otherwise, `pNumRootNodes` entries will be filled in.
        If `pNumRootNodes` is higher than the actual number of root nodes, the
        remaining entries in `pRootNodes` will be set to NULL, and the number
        of nodes actually obtained will be returned in `pNumRootNodes`.

        Parameters
        ----------
        graph : :py:obj:`~.CUgraph` or :py:obj:`~.cudaGraph_t`
            Graph to query
        pNumRootNodes : int
            See description

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`
        pRootNodes : List[:py:obj:`~.cudaGraphNode_t`]
            Pointer to return the root nodes
        pNumRootNodes : int
            See description

        See Also
        --------
        :py:obj:`~.cudaGraphCreate`, :py:obj:`~.cudaGraphGetNodes`, :py:obj:`~.cudaGraphGetEdges`, :py:obj:`~.cudaGraphNodeGetType`, :py:obj:`~.cudaGraphNodeGetDependencies`, :py:obj:`~.cudaGraphNodeGetDependentNodes`
    """


def cudaGraphHostNodeGetParams(node):
    """
    cudaGraphHostNodeGetParams(node)
     Returns a host node's parameters.

        Returns the parameters of host node `node` in `pNodeParams`.

        Parameters
        ----------
        node : :py:obj:`~.CUgraphNode` or :py:obj:`~.cudaGraphNode_t`
            Node to get the parameters for

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`
        pNodeParams : :py:obj:`~.cudaHostNodeParams`
            Pointer to return the parameters

        See Also
        --------
        :py:obj:`~.cudaLaunchHostFunc`, :py:obj:`~.cudaGraphAddHostNode`, :py:obj:`~.cudaGraphHostNodeSetParams`
    """


def cudaGraphHostNodeSetParams(node, pNodeParams: 'Optional[cudaHostNodeParams]'):
    """
    cudaGraphHostNodeSetParams(node, cudaHostNodeParams pNodeParams: Optional[cudaHostNodeParams])
     Sets a host node's parameters.

        Sets the parameters of host node `node` to `nodeParams`.

        Parameters
        ----------
        node : :py:obj:`~.CUgraphNode` or :py:obj:`~.cudaGraphNode_t`
            Node to set the parameters for
        pNodeParams : :py:obj:`~.cudaHostNodeParams`
            Parameters to copy

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`

        See Also
        --------
        :py:obj:`~.cudaGraphNodeSetParams`, :py:obj:`~.cudaLaunchHostFunc`, :py:obj:`~.cudaGraphAddHostNode`, :py:obj:`~.cudaGraphHostNodeGetParams`
    """


def cudaGraphInstantiate(graph, flags):
    """
    cudaGraphInstantiate(graph, unsigned long long flags)
     Creates an executable graph from a graph.

        Instantiates `graph` as an executable graph. The graph is validated for
        any structural constraints or intra-node constraints which were not
        previously validated. If instantiation is successful, a handle to the
        instantiated graph is returned in `pGraphExec`.

        The `flags` parameter controls the behavior of instantiation and
        subsequent graph launches. Valid flags are:

        - :py:obj:`~.cudaGraphInstantiateFlagAutoFreeOnLaunch`, which
          configures a graph containing memory allocation nodes to
          automatically free any unfreed memory allocations before the graph is
          relaunched.

        - :py:obj:`~.cudaGraphInstantiateFlagDeviceLaunch`, which configures
          the graph for launch from the device. If this flag is passed, the
          executable graph handle returned can be used to launch the graph from
          both the host and device. This flag cannot be used in conjunction
          with :py:obj:`~.cudaGraphInstantiateFlagAutoFreeOnLaunch`.

        - :py:obj:`~.cudaGraphInstantiateFlagUseNodePriority`, which causes the
          graph to use the priorities from the per-node attributes rather than
          the priority of the launch stream during execution. Note that
          priorities are only available on kernel nodes, and are copied from
          stream priority during stream capture.

        If `graph` contains any allocation or free nodes, there can be at most
        one executable graph in existence for that graph at a time. An attempt
        to instantiate a second executable graph before destroying the first
        with :py:obj:`~.cudaGraphExecDestroy` will result in an error. The same
        also applies if `graph` contains any device-updatable kernel nodes.

        Graphs instantiated for launch on the device have additional
        restrictions which do not apply to host graphs:

        - The graph's nodes must reside on a single device.

        - The graph can only contain kernel nodes, memcpy nodes, memset nodes,
          and child graph nodes.

        - The graph cannot be empty and must contain at least one kernel,
          memcpy, or memset node. Operation-specific restrictions are outlined
          below.

        - Kernel nodes:

          - Use of CUDA Dynamic Parallelism is not permitted.

          - Cooperative launches are permitted as long as MPS is not in use.

        - Memcpy nodes:

          - Only copies involving device memory and/or pinned device-mapped
            host memory are permitted.

          - Copies involving CUDA arrays are not permitted.

          - Both operands must be accessible from the current device, and the
            current device must match the device of other nodes in the graph.

        If `graph` is not instantiated for launch on the device but contains
        kernels which call device-side :py:obj:`~.cudaGraphLaunch()` from
        multiple devices, this will result in an error.

        Parameters
        ----------
        graph : :py:obj:`~.CUgraph` or :py:obj:`~.cudaGraph_t`
            Graph to instantiate
        flags : unsigned long long
            Flags to control instantiation. See
            :py:obj:`~.CUgraphInstantiate_flags`.

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`
        pGraphExec : :py:obj:`~.cudaGraphExec_t`
            Returns instantiated graph

        See Also
        --------
        :py:obj:`~.cudaGraphInstantiateWithFlags`, :py:obj:`~.cudaGraphCreate`, :py:obj:`~.cudaGraphUpload`, :py:obj:`~.cudaGraphLaunch`, :py:obj:`~.cudaGraphExecDestroy`
    """


def cudaGraphInstantiateWithFlags(graph, flags):
    """
    cudaGraphInstantiateWithFlags(graph, unsigned long long flags)
     Creates an executable graph from a graph.

        Instantiates `graph` as an executable graph. The graph is validated for
        any structural constraints or intra-node constraints which were not
        previously validated. If instantiation is successful, a handle to the
        instantiated graph is returned in `pGraphExec`.

        The `flags` parameter controls the behavior of instantiation and
        subsequent graph launches. Valid flags are:

        - :py:obj:`~.cudaGraphInstantiateFlagAutoFreeOnLaunch`, which
          configures a graph containing memory allocation nodes to
          automatically free any unfreed memory allocations before the graph is
          relaunched.

        - :py:obj:`~.cudaGraphInstantiateFlagDeviceLaunch`, which configures
          the graph for launch from the device. If this flag is passed, the
          executable graph handle returned can be used to launch the graph from
          both the host and device. This flag can only be used on platforms
          which support unified addressing. This flag cannot be used in
          conjunction with
          :py:obj:`~.cudaGraphInstantiateFlagAutoFreeOnLaunch`.

        - :py:obj:`~.cudaGraphInstantiateFlagUseNodePriority`, which causes the
          graph to use the priorities from the per-node attributes rather than
          the priority of the launch stream during execution. Note that
          priorities are only available on kernel nodes, and are copied from
          stream priority during stream capture.

        If `graph` contains any allocation or free nodes, there can be at most
        one executable graph in existence for that graph at a time. An attempt
        to instantiate a second executable graph before destroying the first
        with :py:obj:`~.cudaGraphExecDestroy` will result in an error. The same
        also applies if `graph` contains any device-updatable kernel nodes.

        If `graph` contains kernels which call device-side
        :py:obj:`~.cudaGraphLaunch()` from multiple devices, this will result
        in an error.

        Graphs instantiated for launch on the device have additional
        restrictions which do not apply to host graphs:

        - The graph's nodes must reside on a single device.

        - The graph can only contain kernel nodes, memcpy nodes, memset nodes,
          and child graph nodes.

        - The graph cannot be empty and must contain at least one kernel,
          memcpy, or memset node. Operation-specific restrictions are outlined
          below.

        - Kernel nodes:

          - Use of CUDA Dynamic Parallelism is not permitted.

          - Cooperative launches are permitted as long as MPS is not in use.

        - Memcpy nodes:

          - Only copies involving device memory and/or pinned device-mapped
            host memory are permitted.

          - Copies involving CUDA arrays are not permitted.

          - Both operands must be accessible from the current device, and the
            current device must match the device of other nodes in the graph.

        Parameters
        ----------
        graph : :py:obj:`~.CUgraph` or :py:obj:`~.cudaGraph_t`
            Graph to instantiate
        flags : unsigned long long
            Flags to control instantiation. See
            :py:obj:`~.CUgraphInstantiate_flags`.

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`
        pGraphExec : :py:obj:`~.cudaGraphExec_t`
            Returns instantiated graph

        See Also
        --------
        :py:obj:`~.cudaGraphInstantiate`, :py:obj:`~.cudaGraphCreate`, :py:obj:`~.cudaGraphUpload`, :py:obj:`~.cudaGraphLaunch`, :py:obj:`~.cudaGraphExecDestroy`
    """


def cudaGraphInstantiateWithParams(graph, instantiateParams: 'Optional[cudaGraphInstantiateParams]'):
    """
    cudaGraphInstantiateWithParams(graph, cudaGraphInstantiateParams instantiateParams: Optional[cudaGraphInstantiateParams])
     Creates an executable graph from a graph.

        Instantiates `graph` as an executable graph according to the
        `instantiateParams` structure. The graph is validated for any
        structural constraints or intra-node constraints which were not
        previously validated. If instantiation is successful, a handle to the
        instantiated graph is returned in `pGraphExec`.

        `instantiateParams` controls the behavior of instantiation and
        subsequent graph launches, as well as returning more detailed
        information in the event of an error.
        :py:obj:`~.cudaGraphInstantiateParams` is defined as:

        **View CUDA Toolkit Documentation for a C++ code example**

        The `flags` field controls the behavior of instantiation and subsequent
        graph launches. Valid flags are:

        - :py:obj:`~.cudaGraphInstantiateFlagAutoFreeOnLaunch`, which
          configures a graph containing memory allocation nodes to
          automatically free any unfreed memory allocations before the graph is
          relaunched.

        - :py:obj:`~.cudaGraphInstantiateFlagUpload`, which will perform an
          upload of the graph into `uploadStream` once the graph has been
          instantiated.

        - :py:obj:`~.cudaGraphInstantiateFlagDeviceLaunch`, which configures
          the graph for launch from the device. If this flag is passed, the
          executable graph handle returned can be used to launch the graph from
          both the host and device. This flag can only be used on platforms
          which support unified addressing. This flag cannot be used in
          conjunction with
          :py:obj:`~.cudaGraphInstantiateFlagAutoFreeOnLaunch`.

        - :py:obj:`~.cudaGraphInstantiateFlagUseNodePriority`, which causes the
          graph to use the priorities from the per-node attributes rather than
          the priority of the launch stream during execution. Note that
          priorities are only available on kernel nodes, and are copied from
          stream priority during stream capture.

        If `graph` contains any allocation or free nodes, there can be at most
        one executable graph in existence for that graph at a time. An attempt
        to instantiate a second executable graph before destroying the first
        with :py:obj:`~.cudaGraphExecDestroy` will result in an error. The same
        also applies if `graph` contains any device-updatable kernel nodes.

        If `graph` contains kernels which call device-side
        :py:obj:`~.cudaGraphLaunch()` from multiple devices, this will result
        in an error.

        Graphs instantiated for launch on the device have additional
        restrictions which do not apply to host graphs:

        - The graph's nodes must reside on a single device.

        - The graph can only contain kernel nodes, memcpy nodes, memset nodes,
          and child graph nodes.

        - The graph cannot be empty and must contain at least one kernel,
          memcpy, or memset node. Operation-specific restrictions are outlined
          below.

        - Kernel nodes:

          - Use of CUDA Dynamic Parallelism is not permitted.

          - Cooperative launches are permitted as long as MPS is not in use.

        - Memcpy nodes:

          - Only copies involving device memory and/or pinned device-mapped
            host memory are permitted.

          - Copies involving CUDA arrays are not permitted.

          - Both operands must be accessible from the current device, and the
            current device must match the device of other nodes in the graph.

        In the event of an error, the `result_out` and `errNode_out` fields
        will contain more information about the nature of the error. Possible
        error reporting includes:

        - :py:obj:`~.cudaGraphInstantiateError`, if passed an invalid value or
          if an unexpected error occurred which is described by the return
          value of the function. `errNode_out` will be set to NULL.

        - :py:obj:`~.cudaGraphInstantiateInvalidStructure`, if the graph
          structure is invalid. `errNode_out` will be set to one of the
          offending nodes.

        - :py:obj:`~.cudaGraphInstantiateNodeOperationNotSupported`, if the
          graph is instantiated for device launch but contains a node of an
          unsupported node type, or a node which performs unsupported
          operations, such as use of CUDA dynamic parallelism within a kernel
          node. `errNode_out` will be set to this node.

        - :py:obj:`~.cudaGraphInstantiateMultipleDevicesNotSupported`, if the
          graph is instantiated for device launch but a node¡¯s device differs
          from that of another node. This error can also be returned if a graph
          is not instantiated for device launch and it contains kernels which
          call device-side :py:obj:`~.cudaGraphLaunch()` from multiple devices.
          `errNode_out` will be set to this node.

        If instantiation is successful, `result_out` will be set to
        :py:obj:`~.cudaGraphInstantiateSuccess`, and `hErrNode_out` will be set
        to NULL.

        Parameters
        ----------
        graph : :py:obj:`~.CUgraph` or :py:obj:`~.cudaGraph_t`
            Graph to instantiate
        instantiateParams : :py:obj:`~.cudaGraphInstantiateParams`
            Instantiation parameters

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`
        pGraphExec : :py:obj:`~.cudaGraphExec_t`
            Returns instantiated graph

        See Also
        --------
        :py:obj:`~.cudaGraphCreate`, :py:obj:`~.cudaGraphInstantiate`, :py:obj:`~.cudaGraphInstantiateWithFlags`, :py:obj:`~.cudaGraphExecDestroy`
    """


def cudaGraphKernelNodeCopyAttributes(hSrc, hDst):
    """
    cudaGraphKernelNodeCopyAttributes(hSrc, hDst)
     Copies attributes from source node to destination node.

        Copies attributes from source node `src` to destination node `dst`.
        Both node must have the same context.

        Parameters
        ----------
        dst : :py:obj:`~.CUgraphNode` or :py:obj:`~.cudaGraphNode_t`
            Destination node
        src : :py:obj:`~.CUgraphNode` or :py:obj:`~.cudaGraphNode_t`
            Source node For list of attributes see
            :py:obj:`~.cudaKernelNodeAttrID`

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidContext`

        See Also
        --------
        :py:obj:`~.cudaAccessPolicyWindow`
    """


def cudaGraphKernelNodeGetAttribute(hNode, attr: 'cudaKernelNodeAttrID'):
    """
    cudaGraphKernelNodeGetAttribute(hNode, attr: cudaKernelNodeAttrID)
     Queries node attribute.

        Queries attribute `attr` from node `hNode` and stores it in
        corresponding member of `value_out`.

        Parameters
        ----------
        hNode : :py:obj:`~.CUgraphNode` or :py:obj:`~.cudaGraphNode_t`

        attr : :py:obj:`~.cudaKernelNodeAttrID`


        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidResourceHandle`
        value_out : :py:obj:`~.cudaKernelNodeAttrValue`


        See Also
        --------
        :py:obj:`~.cudaAccessPolicyWindow`
    """


def cudaGraphKernelNodeGetParams(node):
    """
    cudaGraphKernelNodeGetParams(node)
     Returns a kernel node's parameters.

        Returns the parameters of kernel node `node` in `pNodeParams`. The
        `kernelParams` or `extra` array returned in `pNodeParams`, as well as
        the argument values it points to, are owned by the node. This memory
        remains valid until the node is destroyed or its parameters are
        modified, and should not be modified directly. Use
        :py:obj:`~.cudaGraphKernelNodeSetParams` to update the parameters of
        this node.

        The params will contain either `kernelParams` or `extra`, according to
        which of these was most recently set on the node.

        Parameters
        ----------
        node : :py:obj:`~.CUgraphNode` or :py:obj:`~.cudaGraphNode_t`
            Node to get the parameters for

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidDeviceFunction`
        pNodeParams : :py:obj:`~.cudaKernelNodeParams`
            Pointer to return the parameters

        See Also
        --------
        :py:obj:`~.cudaLaunchKernel`, :py:obj:`~.cudaGraphAddKernelNode`, :py:obj:`~.cudaGraphKernelNodeSetParams`
    """

cudaGraphKernelNodePortDefault: int
cudaGraphKernelNodePortLaunchCompletion: int
cudaGraphKernelNodePortProgrammatic: int

def cudaGraphKernelNodeSetAttribute(hNode, attr: 'cudaKernelNodeAttrID', value: 'Optional[cudaKernelNodeAttrValue]'):
    """
    cudaGraphKernelNodeSetAttribute(hNode, attr: cudaKernelNodeAttrID, cudaKernelNodeAttrValue value: Optional[cudaKernelNodeAttrValue])
     Sets node attribute.

        Sets attribute `attr` on node `hNode` from corresponding attribute of
        `value`.

        Parameters
        ----------
        hNode : :py:obj:`~.CUgraphNode` or :py:obj:`~.cudaGraphNode_t`

        attr : :py:obj:`~.cudaKernelNodeAttrID`

        value : :py:obj:`~.cudaKernelNodeAttrValue`


        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidResourceHandle`

        See Also
        --------
        :py:obj:`~.cudaAccessPolicyWindow`
    """


def cudaGraphKernelNodeSetParams(node, pNodeParams: 'Optional[cudaKernelNodeParams]'):
    """
    cudaGraphKernelNodeSetParams(node, cudaKernelNodeParams pNodeParams: Optional[cudaKernelNodeParams])
     Sets a kernel node's parameters.

        Sets the parameters of kernel node `node` to `pNodeParams`.

        Parameters
        ----------
        node : :py:obj:`~.CUgraphNode` or :py:obj:`~.cudaGraphNode_t`
            Node to set the parameters for
        pNodeParams : :py:obj:`~.cudaKernelNodeParams`
            Parameters to copy

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidResourceHandle`, :py:obj:`~.cudaErrorMemoryAllocation`

        See Also
        --------
        :py:obj:`~.cudaGraphNodeSetParams`, :py:obj:`~.cudaLaunchKernel`, :py:obj:`~.cudaGraphAddKernelNode`, :py:obj:`~.cudaGraphKernelNodeGetParams`
    """


def cudaGraphLaunch(graphExec, stream):
    """
    cudaGraphLaunch(graphExec, stream)
     Launches an executable graph in a stream.

        Executes `graphExec` in `stream`. Only one instance of `graphExec` may
        be executing at a time. Each launch is ordered behind both any previous
        work in `stream` and any previous launches of `graphExec`. To execute a
        graph concurrently, it must be instantiated multiple times into
        multiple executable graphs.

        If any allocations created by `graphExec` remain unfreed (from a
        previous launch) and `graphExec` was not instantiated with
        :py:obj:`~.cudaGraphInstantiateFlagAutoFreeOnLaunch`, the launch will
        fail with :py:obj:`~.cudaErrorInvalidValue`.

        Parameters
        ----------
        graphExec : :py:obj:`~.CUgraphExec` or :py:obj:`~.cudaGraphExec_t`
            Executable graph to launch
        stream : :py:obj:`~.CUstream` or :py:obj:`~.cudaStream_t`
            Stream in which to launch the graph

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`

        See Also
        --------
        :py:obj:`~.cudaGraphInstantiate`, :py:obj:`~.cudaGraphUpload`, :py:obj:`~.cudaGraphExecDestroy`
    """


def cudaGraphMemAllocNodeGetParams(node):
    """
    cudaGraphMemAllocNodeGetParams(node)
     Returns a memory alloc node's parameters.

        Returns the parameters of a memory alloc node `hNode` in `params_out`.
        The `poolProps` and `accessDescs` returned in `params_out`, are owned
        by the node. This memory remains valid until the node is destroyed. The
        returned parameters must not be modified.

        Parameters
        ----------
        node : :py:obj:`~.CUgraphNode` or :py:obj:`~.cudaGraphNode_t`
            Node to get the parameters for

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`
        params_out : :py:obj:`~.cudaMemAllocNodeParams`
            Pointer to return the parameters

        See Also
        --------
        :py:obj:`~.cudaGraphAddMemAllocNode`, :py:obj:`~.cudaGraphMemFreeNodeGetParams`
    """


def cudaGraphMemFreeNodeGetParams(node):
    """
    cudaGraphMemFreeNodeGetParams(node)
     Returns a memory free node's parameters.

        Returns the address of a memory free node `hNode` in `dptr_out`.

        Parameters
        ----------
        node : :py:obj:`~.CUgraphNode` or :py:obj:`~.cudaGraphNode_t`
            Node to get the parameters for

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`
        dptr_out : Any
            Pointer to return the device address

        See Also
        --------
        :py:obj:`~.cudaGraphAddMemFreeNode`, :py:obj:`~.cudaGraphMemFreeNodeGetParams`
    """


def cudaGraphMemcpyNodeGetParams(node):
    """
    cudaGraphMemcpyNodeGetParams(node)
     Returns a memcpy node's parameters.

        Returns the parameters of memcpy node `node` in `pNodeParams`.

        Parameters
        ----------
        node : :py:obj:`~.CUgraphNode` or :py:obj:`~.cudaGraphNode_t`
            Node to get the parameters for

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`
        pNodeParams : :py:obj:`~.cudaMemcpy3DParms`
            Pointer to return the parameters

        See Also
        --------
        :py:obj:`~.cudaMemcpy3D`, :py:obj:`~.cudaGraphAddMemcpyNode`, :py:obj:`~.cudaGraphMemcpyNodeSetParams`
    """


def cudaGraphMemcpyNodeSetParams(node, pNodeParams: 'Optional[cudaMemcpy3DParms]'):
    """
    cudaGraphMemcpyNodeSetParams(node, cudaMemcpy3DParms pNodeParams: Optional[cudaMemcpy3DParms])
     Sets a memcpy node's parameters.

        Sets the parameters of memcpy node `node` to `pNodeParams`.

        Parameters
        ----------
        node : :py:obj:`~.CUgraphNode` or :py:obj:`~.cudaGraphNode_t`
            Node to set the parameters for
        pNodeParams : :py:obj:`~.cudaMemcpy3DParms`
            Parameters to copy

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`,

        See Also
        --------
        :py:obj:`~.cudaGraphNodeSetParams`, :py:obj:`~.cudaMemcpy3D`, :py:obj:`~.cudaGraphMemcpyNodeSetParamsToSymbol`, :py:obj:`~.cudaGraphMemcpyNodeSetParamsFromSymbol`, :py:obj:`~.cudaGraphMemcpyNodeSetParams1D`, :py:obj:`~.cudaGraphAddMemcpyNode`, :py:obj:`~.cudaGraphMemcpyNodeGetParams`
    """


def cudaGraphMemcpyNodeSetParams1D(node, dst, src, count, kind: 'cudaMemcpyKind'):
    """
    cudaGraphMemcpyNodeSetParams1D(node, dst, src, size_t count, kind: cudaMemcpyKind)
     Sets a memcpy node's parameters to perform a 1-dimensional copy.

        Sets the parameters of memcpy node `node` to the copy described by the
        provided parameters.

        When the graph is launched, the node will copy `count` bytes from the
        memory area pointed to by `src` to the memory area pointed to by `dst`,
        where `kind` specifies the direction of the copy, and must be one of
        :py:obj:`~.cudaMemcpyHostToHost`, :py:obj:`~.cudaMemcpyHostToDevice`,
        :py:obj:`~.cudaMemcpyDeviceToHost`,
        :py:obj:`~.cudaMemcpyDeviceToDevice`, or :py:obj:`~.cudaMemcpyDefault`.
        Passing :py:obj:`~.cudaMemcpyDefault` is recommended, in which case the
        type of transfer is inferred from the pointer values. However,
        :py:obj:`~.cudaMemcpyDefault` is only allowed on systems that support
        unified virtual addressing. Launching a memcpy node with dst and src
        pointers that do not match the direction of the copy results in an
        undefined behavior.

        Parameters
        ----------
        node : :py:obj:`~.CUgraphNode` or :py:obj:`~.cudaGraphNode_t`
            Node to set the parameters for
        dst : Any
            Destination memory address
        src : Any
            Source memory address
        count : size_t
            Size in bytes to copy
        kind : :py:obj:`~.cudaMemcpyKind`
            Type of transfer

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`

        See Also
        --------
        :py:obj:`~.cudaMemcpy`, :py:obj:`~.cudaGraphMemcpyNodeSetParams`, :py:obj:`~.cudaGraphAddMemcpyNode`, :py:obj:`~.cudaGraphMemcpyNodeGetParams`
    """


def cudaGraphMemsetNodeGetParams(node):
    """
    cudaGraphMemsetNodeGetParams(node)
     Returns a memset node's parameters.

        Returns the parameters of memset node `node` in `pNodeParams`.

        Parameters
        ----------
        node : :py:obj:`~.CUgraphNode` or :py:obj:`~.cudaGraphNode_t`
            Node to get the parameters for

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`
        pNodeParams : :py:obj:`~.cudaMemsetParams`
            Pointer to return the parameters

        See Also
        --------
        :py:obj:`~.cudaMemset2D`, :py:obj:`~.cudaGraphAddMemsetNode`, :py:obj:`~.cudaGraphMemsetNodeSetParams`
    """


def cudaGraphMemsetNodeSetParams(node, pNodeParams: 'Optional[cudaMemsetParams]'):
    """
    cudaGraphMemsetNodeSetParams(node, cudaMemsetParams pNodeParams: Optional[cudaMemsetParams])
     Sets a memset node's parameters.

        Sets the parameters of memset node `node` to `pNodeParams`.

        Parameters
        ----------
        node : :py:obj:`~.CUgraphNode` or :py:obj:`~.cudaGraphNode_t`
            Node to set the parameters for
        pNodeParams : :py:obj:`~.cudaMemsetParams`
            Parameters to copy

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`

        See Also
        --------
        :py:obj:`~.cudaGraphNodeSetParams`, :py:obj:`~.cudaMemset2D`, :py:obj:`~.cudaGraphAddMemsetNode`, :py:obj:`~.cudaGraphMemsetNodeGetParams`
    """


def cudaGraphNodeFindInClone(originalNode, clonedGraph):
    """
    cudaGraphNodeFindInClone(originalNode, clonedGraph)
     Finds a cloned version of a node.

        This function returns the node in `clonedGraph` corresponding to
        `originalNode` in the original graph.

        `clonedGraph` must have been cloned from `originalGraph` via
        :py:obj:`~.cudaGraphClone`. `originalNode` must have been in
        `originalGraph` at the time of the call to :py:obj:`~.cudaGraphClone`,
        and the corresponding cloned node in `clonedGraph` must not have been
        removed. The cloned node is then returned via `pClonedNode`.

        Parameters
        ----------
        originalNode : :py:obj:`~.CUgraphNode` or :py:obj:`~.cudaGraphNode_t`
            Handle to the original node
        clonedGraph : :py:obj:`~.CUgraph` or :py:obj:`~.cudaGraph_t`
            Cloned graph to query

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`
        pNode : :py:obj:`~.cudaGraphNode_t`
            Returns handle to the cloned node

        See Also
        --------
        :py:obj:`~.cudaGraphClone`
    """


def cudaGraphNodeGetDependencies(node, pNumDependencies=0):
    """
    cudaGraphNodeGetDependencies(node, size_t pNumDependencies=0)
     Returns a node's dependencies.

        Returns a list of `node's` dependencies. `pDependencies` may be NULL,
        in which case this function will return the number of dependencies in
        `pNumDependencies`. Otherwise, `pNumDependencies` entries will be
        filled in. If `pNumDependencies` is higher than the actual number of
        dependencies, the remaining entries in `pDependencies` will be set to
        NULL, and the number of nodes actually obtained will be returned in
        `pNumDependencies`.

        Parameters
        ----------
        node : :py:obj:`~.CUgraphNode` or :py:obj:`~.cudaGraphNode_t`
            Node to query
        pNumDependencies : int
            See description

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`
        pDependencies : List[:py:obj:`~.cudaGraphNode_t`]
            Pointer to return the dependencies
        pNumDependencies : int
            See description

        See Also
        --------
        :py:obj:`~.cudaGraphNodeGetDependentNodes`, :py:obj:`~.cudaGraphGetNodes`, :py:obj:`~.cudaGraphGetRootNodes`, :py:obj:`~.cudaGraphGetEdges`, :py:obj:`~.cudaGraphAddDependencies`, :py:obj:`~.cudaGraphRemoveDependencies`
    """


def cudaGraphNodeGetDependencies_v2(node, pNumDependencies=0):
    """
    cudaGraphNodeGetDependencies_v2(node, size_t pNumDependencies=0)
     Returns a node's dependencies (12.3+)

        Returns a list of `node's` dependencies. `pDependencies` may be NULL,
        in which case this function will return the number of dependencies in
        `pNumDependencies`. Otherwise, `pNumDependencies` entries will be
        filled in. If `pNumDependencies` is higher than the actual number of
        dependencies, the remaining entries in `pDependencies` will be set to
        NULL, and the number of nodes actually obtained will be returned in
        `pNumDependencies`.

        Note that if an edge has non-zero (non-default) edge data and
        `edgeData` is NULL, this API will return
        :py:obj:`~.cudaErrorLossyQuery`. If `edgeData` is non-NULL, then
        `pDependencies` must be as well.

        Parameters
        ----------
        node : :py:obj:`~.CUgraphNode` or :py:obj:`~.cudaGraphNode_t`
            Node to query
        pNumDependencies : int
            See description

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorLossyQuery`, :py:obj:`~.cudaErrorInvalidValue`
        pDependencies : List[:py:obj:`~.cudaGraphNode_t`]
            Pointer to return the dependencies
        edgeData : List[:py:obj:`~.cudaGraphEdgeData`]
            Optional array to return edge data for each dependency
        pNumDependencies : int
            See description

        See Also
        --------
        :py:obj:`~.cudaGraphNodeGetDependentNodes`, :py:obj:`~.cudaGraphGetNodes`, :py:obj:`~.cudaGraphGetRootNodes`, :py:obj:`~.cudaGraphGetEdges`, :py:obj:`~.cudaGraphAddDependencies`, :py:obj:`~.cudaGraphRemoveDependencies`
    """


def cudaGraphNodeGetDependentNodes(node, pNumDependentNodes=0):
    """
    cudaGraphNodeGetDependentNodes(node, size_t pNumDependentNodes=0)
     Returns a node's dependent nodes.

        Returns a list of `node's` dependent nodes. `pDependentNodes` may be
        NULL, in which case this function will return the number of dependent
        nodes in `pNumDependentNodes`. Otherwise, `pNumDependentNodes` entries
        will be filled in. If `pNumDependentNodes` is higher than the actual
        number of dependent nodes, the remaining entries in `pDependentNodes`
        will be set to NULL, and the number of nodes actually obtained will be
        returned in `pNumDependentNodes`.

        Parameters
        ----------
        node : :py:obj:`~.CUgraphNode` or :py:obj:`~.cudaGraphNode_t`
            Node to query
        pNumDependentNodes : int
            See description

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`
        pDependentNodes : List[:py:obj:`~.cudaGraphNode_t`]
            Pointer to return the dependent nodes
        pNumDependentNodes : int
            See description

        See Also
        --------
        :py:obj:`~.cudaGraphNodeGetDependencies`, :py:obj:`~.cudaGraphGetNodes`, :py:obj:`~.cudaGraphGetRootNodes`, :py:obj:`~.cudaGraphGetEdges`, :py:obj:`~.cudaGraphAddDependencies`, :py:obj:`~.cudaGraphRemoveDependencies`
    """


def cudaGraphNodeGetDependentNodes_v2(node, pNumDependentNodes=0):
    """
    cudaGraphNodeGetDependentNodes_v2(node, size_t pNumDependentNodes=0)
     Returns a node's dependent nodes (12.3+)

        Returns a list of `node's` dependent nodes. `pDependentNodes` may be
        NULL, in which case this function will return the number of dependent
        nodes in `pNumDependentNodes`. Otherwise, `pNumDependentNodes` entries
        will be filled in. If `pNumDependentNodes` is higher than the actual
        number of dependent nodes, the remaining entries in `pDependentNodes`
        will be set to NULL, and the number of nodes actually obtained will be
        returned in `pNumDependentNodes`.

        Note that if an edge has non-zero (non-default) edge data and
        `edgeData` is NULL, this API will return
        :py:obj:`~.cudaErrorLossyQuery`. If `edgeData` is non-NULL, then
        `pDependentNodes` must be as well.

        Parameters
        ----------
        node : :py:obj:`~.CUgraphNode` or :py:obj:`~.cudaGraphNode_t`
            Node to query
        pNumDependentNodes : int
            See description

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorLossyQuery`, :py:obj:`~.cudaErrorInvalidValue`
        pDependentNodes : List[:py:obj:`~.cudaGraphNode_t`]
            Pointer to return the dependent nodes
        edgeData : List[:py:obj:`~.cudaGraphEdgeData`]
            Optional pointer to return edge data for dependent nodes
        pNumDependentNodes : int
            See description

        See Also
        --------
        :py:obj:`~.cudaGraphNodeGetDependencies`, :py:obj:`~.cudaGraphGetNodes`, :py:obj:`~.cudaGraphGetRootNodes`, :py:obj:`~.cudaGraphGetEdges`, :py:obj:`~.cudaGraphAddDependencies`, :py:obj:`~.cudaGraphRemoveDependencies`
    """


def cudaGraphNodeGetEnabled(hGraphExec, hNode):
    """
    cudaGraphNodeGetEnabled(hGraphExec, hNode)
     Query whether a node in the given graphExec is enabled.

        Sets isEnabled to 1 if `hNode` is enabled, or 0 if `hNode` is disabled.

        The node is identified by the corresponding node `hNode` in the non-
        executable graph, from which the executable graph was instantiated.

        `hNode` must not have been removed from the original graph.

        Parameters
        ----------
        hGraphExec : :py:obj:`~.CUgraphExec` or :py:obj:`~.cudaGraphExec_t`
            The executable graph in which to set the specified node
        hNode : :py:obj:`~.CUgraphNode` or :py:obj:`~.cudaGraphNode_t`
            Node from the graph from which graphExec was instantiated

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`,
        isEnabled : unsigned int
            Location to return the enabled status of the node

        See Also
        --------
        :py:obj:`~.cudaGraphNodeSetEnabled`, :py:obj:`~.cudaGraphExecUpdate`, :py:obj:`~.cudaGraphInstantiate` :py:obj:`~.cudaGraphLaunch`

        Notes
        -----
        Currently only kernel, memset and memcpy nodes are supported.
    """


def cudaGraphNodeGetType(node):
    """
    cudaGraphNodeGetType(node)
     Returns a node's type.

        Returns the node type of `node` in `pType`.

        Parameters
        ----------
        node : :py:obj:`~.CUgraphNode` or :py:obj:`~.cudaGraphNode_t`
            Node to query

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`
        pType : :py:obj:`~.cudaGraphNodeType`
            Pointer to return the node type

        See Also
        --------
        :py:obj:`~.cudaGraphGetNodes`, :py:obj:`~.cudaGraphGetRootNodes`, :py:obj:`~.cudaGraphChildGraphNodeGetGraph`, :py:obj:`~.cudaGraphKernelNodeGetParams`, :py:obj:`~.cudaGraphKernelNodeSetParams`, :py:obj:`~.cudaGraphHostNodeGetParams`, :py:obj:`~.cudaGraphHostNodeSetParams`, :py:obj:`~.cudaGraphMemcpyNodeGetParams`, :py:obj:`~.cudaGraphMemcpyNodeSetParams`, :py:obj:`~.cudaGraphMemsetNodeGetParams`, :py:obj:`~.cudaGraphMemsetNodeSetParams`
    """


def cudaGraphNodeSetEnabled(hGraphExec, hNode, isEnabled):
    """
    cudaGraphNodeSetEnabled(hGraphExec, hNode, unsigned int isEnabled)
     Enables or disables the specified node in the given graphExec.

        Sets `hNode` to be either enabled or disabled. Disabled nodes are
        functionally equivalent to empty nodes until they are reenabled.
        Existing node parameters are not affected by disabling/enabling the
        node.

        The node is identified by the corresponding node `hNode` in the non-
        executable graph, from which the executable graph was instantiated.

        `hNode` must not have been removed from the original graph.

        The modifications only affect future launches of `hGraphExec`. Already
        enqueued or running launches of `hGraphExec` are not affected by this
        call. `hNode` is also not modified by this call.

        Parameters
        ----------
        hGraphExec : :py:obj:`~.CUgraphExec` or :py:obj:`~.cudaGraphExec_t`
            The executable graph in which to set the specified node
        hNode : :py:obj:`~.CUgraphNode` or :py:obj:`~.cudaGraphNode_t`
            Node from the graph from which graphExec was instantiated
        isEnabled : unsigned int
            Node is enabled if != 0, otherwise the node is disabled

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`,

        See Also
        --------
        :py:obj:`~.cudaGraphNodeGetEnabled`, :py:obj:`~.cudaGraphExecUpdate`, :py:obj:`~.cudaGraphInstantiate` :py:obj:`~.cudaGraphLaunch`

        Notes
        -----
        Currently only kernel, memset and memcpy nodes are supported.
    """


def cudaGraphNodeSetParams(node, nodeParams: 'Optional[cudaGraphNodeParams]'):
    """
    cudaGraphNodeSetParams(node, cudaGraphNodeParams nodeParams: Optional[cudaGraphNodeParams])
     Update's a graph node's parameters.

        Sets the parameters of graph node `node` to `nodeParams`. The node type
        specified by `nodeParams->type` must match the type of `node`.
        `nodeParams` must be fully initialized and all unused bytes (reserved,
        padding) zeroed.

        Modifying parameters is not supported for node types
        cudaGraphNodeTypeMemAlloc and cudaGraphNodeTypeMemFree.

        Parameters
        ----------
        node : :py:obj:`~.CUgraphNode` or :py:obj:`~.cudaGraphNode_t`
            Node to set the parameters for
        nodeParams : :py:obj:`~.cudaGraphNodeParams`
            Parameters to copy

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidDeviceFunction`, :py:obj:`~.cudaErrorNotSupported`

        See Also
        --------
        :py:obj:`~.cudaGraphAddNode`, :py:obj:`~.cudaGraphExecNodeSetParams`
    """


def cudaGraphReleaseUserObject(graph, object, count):
    """
    cudaGraphReleaseUserObject(graph, object, unsigned int count)
     Release a user object reference from a graph.

        Releases user object references owned by a graph.

        See CUDA User Objects in the CUDA C++ Programming Guide for more
        information on user objects.

        Parameters
        ----------
        graph : :py:obj:`~.CUgraph` or :py:obj:`~.cudaGraph_t`
            The graph that will release the reference
        object : :py:obj:`~.cudaUserObject_t`
            The user object to release a reference for
        count : unsigned int
            The number of references to release, typically 1. Must be nonzero
            and not larger than INT_MAX.

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`

        See Also
        --------
        :py:obj:`~.cudaUserObjectCreate` :py:obj:`~.cudaUserObjectRetain`, :py:obj:`~.cudaUserObjectRelease`, :py:obj:`~.cudaGraphRetainUserObject`, :py:obj:`~.cudaGraphCreate`
    """


def cudaGraphRemoveDependencies(graph, from_: 'Optional[Tuple[cudaGraphNode_t] | List[cudaGraphNode_t]]', to: 'Optional[Tuple[cudaGraphNode_t] | List[cudaGraphNode_t]]', numDependencies):
    """
    cudaGraphRemoveDependencies(graph, from_: Optional[Tuple[cudaGraphNode_t] | List[cudaGraphNode_t]], to: Optional[Tuple[cudaGraphNode_t] | List[cudaGraphNode_t]], size_t numDependencies)
     Removes dependency edges from a graph.

        The number of `pDependencies` to be removed is defined by
        `numDependencies`. Elements in `pFrom` and `pTo` at corresponding
        indices define a dependency. Each node in `pFrom` and `pTo` must belong
        to `graph`.

        If `numDependencies` is 0, elements in `pFrom` and `pTo` will be
        ignored. Specifying a non-existing dependency will return an error.

        Parameters
        ----------
        graph : :py:obj:`~.CUgraph` or :py:obj:`~.cudaGraph_t`
            Graph from which to remove dependencies
        from : List[:py:obj:`~.cudaGraphNode_t`]
            Array of nodes that provide the dependencies
        to : List[:py:obj:`~.cudaGraphNode_t`]
            Array of dependent nodes
        numDependencies : size_t
            Number of dependencies to be removed

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`

        See Also
        --------
        :py:obj:`~.cudaGraphAddDependencies`, :py:obj:`~.cudaGraphGetEdges`, :py:obj:`~.cudaGraphNodeGetDependencies`, :py:obj:`~.cudaGraphNodeGetDependentNodes`
    """


def cudaGraphRemoveDependencies_v2(graph, from_: 'Optional[Tuple[cudaGraphNode_t] | List[cudaGraphNode_t]]', to: 'Optional[Tuple[cudaGraphNode_t] | List[cudaGraphNode_t]]', edgeData: 'Optional[Tuple[cudaGraphEdgeData] | List[cudaGraphEdgeData]]', numDependencies):
    """
    cudaGraphRemoveDependencies_v2(graph, from_: Optional[Tuple[cudaGraphNode_t] | List[cudaGraphNode_t]], to: Optional[Tuple[cudaGraphNode_t] | List[cudaGraphNode_t]], edgeData: Optional[Tuple[cudaGraphEdgeData] | List[cudaGraphEdgeData]], size_t numDependencies)
     Removes dependency edges from a graph. (12.3+)

        The number of `pDependencies` to be removed is defined by
        `numDependencies`. Elements in `pFrom` and `pTo` at corresponding
        indices define a dependency. Each node in `pFrom` and `pTo` must belong
        to `graph`.

        If `numDependencies` is 0, elements in `pFrom` and `pTo` will be
        ignored. Specifying an edge that does not exist in the graph, with data
        matching `edgeData`, results in an error. `edgeData` is nullable, which
        is equivalent to passing default (zeroed) data for each edge.

        Parameters
        ----------
        graph : :py:obj:`~.CUgraph` or :py:obj:`~.cudaGraph_t`
            Graph from which to remove dependencies
        from : List[:py:obj:`~.cudaGraphNode_t`]
            Array of nodes that provide the dependencies
        to : List[:py:obj:`~.cudaGraphNode_t`]
            Array of dependent nodes
        edgeData : List[:py:obj:`~.cudaGraphEdgeData`]
            Optional array of edge data. If NULL, edge data is assumed to be
            default (zeroed).
        numDependencies : size_t
            Number of dependencies to be removed

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`

        See Also
        --------
        :py:obj:`~.cudaGraphAddDependencies`, :py:obj:`~.cudaGraphGetEdges`, :py:obj:`~.cudaGraphNodeGetDependencies`, :py:obj:`~.cudaGraphNodeGetDependentNodes`
    """


def cudaGraphRetainUserObject(graph, object, count, flags):
    """
    cudaGraphRetainUserObject(graph, object, unsigned int count, unsigned int flags)
     Retain a reference to a user object from a graph.

        Creates or moves user object references that will be owned by a CUDA
        graph.

        See CUDA User Objects in the CUDA C++ Programming Guide for more
        information on user objects.

        Parameters
        ----------
        graph : :py:obj:`~.CUgraph` or :py:obj:`~.cudaGraph_t`
            The graph to associate the reference with
        object : :py:obj:`~.cudaUserObject_t`
            The user object to retain a reference for
        count : unsigned int
            The number of references to add to the graph, typically 1. Must be
            nonzero and not larger than INT_MAX.
        flags : unsigned int
            The optional flag :py:obj:`~.cudaGraphUserObjectMove` transfers
            references from the calling thread, rather than create new
            references. Pass 0 to create new references.

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`

        See Also
        --------
        :py:obj:`~.cudaUserObjectCreate` :py:obj:`~.cudaUserObjectRetain`, :py:obj:`~.cudaUserObjectRelease`, :py:obj:`~.cudaGraphReleaseUserObject`, :py:obj:`~.cudaGraphCreate`
    """


def cudaGraphUpload(graphExec, stream):
    """
    cudaGraphUpload(graphExec, stream)
     Uploads an executable graph in a stream.

        Uploads `hGraphExec` to the device in `hStream` without executing it.
        Uploads of the same `hGraphExec` will be serialized. Each upload is
        ordered behind both any previous work in `hStream` and any previous
        launches of `hGraphExec`. Uses memory cached by `stream` to back the
        allocations owned by `graphExec`.

        Parameters
        ----------
        hGraphExec : :py:obj:`~.CUgraphExec` or :py:obj:`~.cudaGraphExec_t`
            Executable graph to upload
        hStream : :py:obj:`~.CUstream` or :py:obj:`~.cudaStream_t`
            Stream in which to upload the graph

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`,

        See Also
        --------
        :py:obj:`~.cudaGraphInstantiate`, :py:obj:`~.cudaGraphLaunch`, :py:obj:`~.cudaGraphExecDestroy`
    """


def cudaGraphicsEGLRegisterImage(image, flags):
    """
    cudaGraphicsEGLRegisterImage(image, unsigned int flags)
     Registers an EGL image.

        Registers the EGLImageKHR specified by `image` for access by CUDA. A
        handle to the registered object is returned as `pCudaResource`.
        Additional Mapping/Unmapping is not required for the registered
        resource and :py:obj:`~.cudaGraphicsResourceGetMappedEglFrame` can be
        directly called on the `pCudaResource`.

        The application will be responsible for synchronizing access to shared
        objects. The application must ensure that any pending operation which
        access the objects have completed before passing control to CUDA. This
        may be accomplished by issuing and waiting for glFinish command on all
        GLcontexts (for OpenGL and likewise for other APIs). The application
        will be also responsible for ensuring that any pending operation on the
        registered CUDA resource has completed prior to executing subsequent
        commands in other APIs accesing the same memory objects. This can be
        accomplished by calling cuCtxSynchronize or cuEventSynchronize
        (preferably).

        The surface's intended usage is specified using `flags`, as follows:

        - :py:obj:`~.cudaGraphicsRegisterFlagsNone`: Specifies no hints about
          how this resource will be used. It is therefore assumed that this
          resource will be read from and written to by CUDA. This is the
          default value.

        - :py:obj:`~.cudaGraphicsRegisterFlagsReadOnly`: Specifies that CUDA
          will not write to this resource.

        - :py:obj:`~.cudaGraphicsRegisterFlagsWriteDiscard`: Specifies that
          CUDA will not read from this resource and will write over the entire
          contents of the resource, so none of the data previously stored in
          the resource will be preserved.

        The EGLImageKHR is an object which can be used to create EGLImage
        target resource. It is defined as a void pointer. typedef void*
        EGLImageKHR

        Parameters
        ----------
        image : :py:obj:`~.EGLImageKHR`
            An EGLImageKHR image which can be used to create target resource.
        flags : unsigned int
            Map flags

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidResourceHandle`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorUnknown`
        pCudaResource : :py:obj:`~.cudaGraphicsResource`
            Pointer to the returned object handle

        See Also
        --------
        :py:obj:`~.cudaGraphicsUnregisterResource`, :py:obj:`~.cudaGraphicsResourceGetMappedEglFrame`, :py:obj:`~.cuGraphicsEGLRegisterImage`
    """


def cudaGraphicsGLRegisterBuffer(buffer, flags):
    """
    cudaGraphicsGLRegisterBuffer(buffer, unsigned int flags)
     Registers an OpenGL buffer object.

        Registers the buffer object specified by `buffer` for access by CUDA. A
        handle to the registered object is returned as `resource`. The register
        flags `flags` specify the intended usage, as follows:

        - :py:obj:`~.cudaGraphicsRegisterFlagsNone`: Specifies no hints about
          how this resource will be used. It is therefore assumed that this
          resource will be read from and written to by CUDA. This is the
          default value.

        - :py:obj:`~.cudaGraphicsRegisterFlagsReadOnly`: Specifies that CUDA
          will not write to this resource.

        - :py:obj:`~.cudaGraphicsRegisterFlagsWriteDiscard`: Specifies that
          CUDA will not read from this resource and will write over the entire
          contents of the resource, so none of the data previously stored in
          the resource will be preserved.

        Parameters
        ----------
        buffer : :py:obj:`~.GLuint`
            name of buffer object to be registered
        flags : unsigned int
            Register flags

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidDevice`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidResourceHandle`, :py:obj:`~.cudaErrorOperatingSystem`, :py:obj:`~.cudaErrorUnknown`
        resource : :py:obj:`~.cudaGraphicsResource`
            Pointer to the returned object handle

        See Also
        --------
        :py:obj:`~.cudaGraphicsUnregisterResource`, :py:obj:`~.cudaGraphicsMapResources`, :py:obj:`~.cudaGraphicsResourceGetMappedPointer`, :py:obj:`~.cuGraphicsGLRegisterBuffer`
    """


def cudaGraphicsGLRegisterImage(image, target, flags):
    """
    cudaGraphicsGLRegisterImage(image, target, unsigned int flags)
     Register an OpenGL texture or renderbuffer object.

        Registers the texture or renderbuffer object specified by `image` for
        access by CUDA. A handle to the registered object is returned as
        `resource`.

        `target` must match the type of the object, and must be one of
        :py:obj:`~.GL_TEXTURE_2D`, :py:obj:`~.GL_TEXTURE_RECTANGLE`,
        :py:obj:`~.GL_TEXTURE_CUBE_MAP`, :py:obj:`~.GL_TEXTURE_3D`,
        :py:obj:`~.GL_TEXTURE_2D_ARRAY`, or :py:obj:`~.GL_RENDERBUFFER`.

        The register flags `flags` specify the intended usage, as follows:

        - :py:obj:`~.cudaGraphicsRegisterFlagsNone`: Specifies no hints about
          how this resource will be used. It is therefore assumed that this
          resource will be read from and written to by CUDA. This is the
          default value.

        - :py:obj:`~.cudaGraphicsRegisterFlagsReadOnly`: Specifies that CUDA
          will not write to this resource.

        - :py:obj:`~.cudaGraphicsRegisterFlagsWriteDiscard`: Specifies that
          CUDA will not read from this resource and will write over the entire
          contents of the resource, so none of the data previously stored in
          the resource will be preserved.

        - :py:obj:`~.cudaGraphicsRegisterFlagsSurfaceLoadStore`: Specifies that
          CUDA will bind this resource to a surface reference.

        - :py:obj:`~.cudaGraphicsRegisterFlagsTextureGather`: Specifies that
          CUDA will perform texture gather operations on this resource.

        The following image formats are supported. For brevity's sake, the list
        is abbreviated. For ex., {GL_R, GL_RG} X {8, 16} would expand to the
        following 4 formats {GL_R8, GL_R16, GL_RG8, GL_RG16} :

        - GL_RED, GL_RG, GL_RGBA, GL_LUMINANCE, GL_ALPHA, GL_LUMINANCE_ALPHA,
          GL_INTENSITY

        - {GL_R, GL_RG, GL_RGBA} X {8, 16, 16F, 32F, 8UI, 16UI, 32UI, 8I, 16I,
          32I}

        - {GL_LUMINANCE, GL_ALPHA, GL_LUMINANCE_ALPHA, GL_INTENSITY} X {8, 16,
          16F_ARB, 32F_ARB, 8UI_EXT, 16UI_EXT, 32UI_EXT, 8I_EXT, 16I_EXT,
          32I_EXT}

        The following image classes are currently disallowed:

        - Textures with borders

        - Multisampled renderbuffers

        Parameters
        ----------
        image : :py:obj:`~.GLuint`
            name of texture or renderbuffer object to be registered
        target : :py:obj:`~.GLenum`
            Identifies the type of object specified by `image`
        flags : unsigned int
            Register flags

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidDevice`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidResourceHandle`, :py:obj:`~.cudaErrorOperatingSystem`, :py:obj:`~.cudaErrorUnknown`
        resource : :py:obj:`~.cudaGraphicsResource`
            Pointer to the returned object handle

        See Also
        --------
        :py:obj:`~.cudaGraphicsUnregisterResource`, :py:obj:`~.cudaGraphicsMapResources`, :py:obj:`~.cudaGraphicsSubResourceGetMappedArray`, :py:obj:`~.cuGraphicsGLRegisterImage`
    """


def cudaGraphicsMapResources(count, resources, stream):
    """
    cudaGraphicsMapResources(int count, resources, stream)
     Map graphics resources for access by CUDA.

        Maps the `count` graphics resources in `resources` for access by CUDA.

        The resources in `resources` may be accessed by CUDA until they are
        unmapped. The graphics API from which `resources` were registered
        should not access any resources while they are mapped by CUDA. If an
        application does so, the results are undefined.

        This function provides the synchronization guarantee that any graphics
        calls issued before :py:obj:`~.cudaGraphicsMapResources()` will
        complete before any subsequent CUDA work issued in `stream` begins.

        If `resources` contains any duplicate entries then
        :py:obj:`~.cudaErrorInvalidResourceHandle` is returned. If any of
        `resources` are presently mapped for access by CUDA then
        :py:obj:`~.cudaErrorUnknown` is returned.

        Parameters
        ----------
        count : int
            Number of resources to map
        resources : :py:obj:`~.cudaGraphicsResource_t`
            Resources to map for CUDA
        stream : :py:obj:`~.CUstream` or :py:obj:`~.cudaStream_t`
            Stream for synchronization

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidResourceHandle`, :py:obj:`~.cudaErrorUnknown`

        See Also
        --------
        :py:obj:`~.cudaGraphicsResourceGetMappedPointer`, :py:obj:`~.cudaGraphicsSubResourceGetMappedArray`, :py:obj:`~.cudaGraphicsUnmapResources`, :py:obj:`~.cuGraphicsMapResources`
    """


def cudaGraphicsResourceGetMappedEglFrame(resource, index, mipLevel):
    """
    cudaGraphicsResourceGetMappedEglFrame(resource, unsigned int index, unsigned int mipLevel)
     Get an eglFrame through which to access a registered EGL graphics resource.

        Returns in `*eglFrame` an eglFrame pointer through which the registered
        graphics resource `resource` may be accessed. This API can only be
        called for EGL graphics resources.

        The :py:obj:`~.cudaEglFrame` is defined as

        **View CUDA Toolkit Documentation for a C++ code example**

        Parameters
        ----------
        resource : :py:obj:`~.cudaGraphicsResource_t`
            Registered resource to access.
        index : unsigned int
            Index for cubemap surfaces.
        mipLevel : unsigned int
            Mipmap level for the subresource to access.

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorUnknown`
        eglFrame : :py:obj:`~.cudaEglFrame`
            Returned eglFrame.

        See Also
        --------
        :py:obj:`~.cudaGraphicsSubResourceGetMappedArray`, :py:obj:`~.cudaGraphicsResourceGetMappedPointer`, :py:obj:`~.cuGraphicsResourceGetMappedEglFrame`

        Notes
        -----
        Note that in case of multiplanar `*eglFrame`, pitch of only first plane (unsigned int :py:obj:`~.cudaEglPlaneDesc.pitch`) is to be considered by the application.
    """


def cudaGraphicsResourceGetMappedMipmappedArray(resource):
    """
    cudaGraphicsResourceGetMappedMipmappedArray(resource)
     Get a mipmapped array through which to access a mapped graphics resource.

        Returns in `*mipmappedArray` a mipmapped array through which the mapped
        graphics resource `resource` may be accessed. The value set in
        `mipmappedArray` may change every time that `resource` is mapped.

        If `resource` is not a texture then it cannot be accessed via an array
        and :py:obj:`~.cudaErrorUnknown` is returned. If `resource` is not
        mapped then :py:obj:`~.cudaErrorUnknown` is returned.

        Parameters
        ----------
        resource : :py:obj:`~.cudaGraphicsResource_t`
            Mapped resource to access

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidResourceHandle`, :py:obj:`~.cudaErrorUnknown`
        mipmappedArray : :py:obj:`~.cudaMipmappedArray_t`
            Returned mipmapped array through which `resource` may be accessed

        See Also
        --------
        :py:obj:`~.cudaGraphicsResourceGetMappedPointer`, :py:obj:`~.cuGraphicsResourceGetMappedMipmappedArray`
    """


def cudaGraphicsResourceGetMappedPointer(resource):
    """
    cudaGraphicsResourceGetMappedPointer(resource)
     Get an device pointer through which to access a mapped graphics resource.

        Returns in `*devPtr` a pointer through which the mapped graphics
        resource `resource` may be accessed. Returns in `*size` the size of the
        memory in bytes which may be accessed from that pointer. The value set
        in `devPtr` may change every time that `resource` is mapped.

        If `resource` is not a buffer then it cannot be accessed via a pointer
        and :py:obj:`~.cudaErrorUnknown` is returned. If `resource` is not
        mapped then :py:obj:`~.cudaErrorUnknown` is returned.

        Parameters
        ----------
        resource : :py:obj:`~.cudaGraphicsResource_t`
            None

        Returns
        -------
        cudaError_t

        devPtr : Any
            None
        size : int
            None
    """


def cudaGraphicsResourceSetMapFlags(resource, flags):
    """
    cudaGraphicsResourceSetMapFlags(resource, unsigned int flags)
     Set usage flags for mapping a graphics resource.

        Set `flags` for mapping the graphics resource `resource`.

        Changes to `flags` will take effect the next time `resource` is mapped.
        The `flags` argument may be any of the following:

        - :py:obj:`~.cudaGraphicsMapFlagsNone`: Specifies no hints about how
          `resource` will be used. It is therefore assumed that CUDA may read
          from or write to `resource`.

        - :py:obj:`~.cudaGraphicsMapFlagsReadOnly`: Specifies that CUDA will
          not write to `resource`.

        - :py:obj:`~.cudaGraphicsMapFlagsWriteDiscard`: Specifies CUDA will not
          read from `resource` and will write over the entire contents of
          `resource`, so none of the data previously stored in `resource` will
          be preserved.

        If `resource` is presently mapped for access by CUDA then
        :py:obj:`~.cudaErrorUnknown` is returned. If `flags` is not one of the
        above values then :py:obj:`~.cudaErrorInvalidValue` is returned.

        Parameters
        ----------
        resource : :py:obj:`~.cudaGraphicsResource_t`
            Registered resource to set flags for
        flags : unsigned int
            Parameters for resource mapping

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidResourceHandle`, :py:obj:`~.cudaErrorUnknown`,

        See Also
        --------
        :py:obj:`~.cudaGraphicsMapResources`, :py:obj:`~.cuGraphicsResourceSetMapFlags`
    """


def cudaGraphicsSubResourceGetMappedArray(resource, arrayIndex, mipLevel):
    """
    cudaGraphicsSubResourceGetMappedArray(resource, unsigned int arrayIndex, unsigned int mipLevel)
     Get an array through which to access a subresource of a mapped graphics resource.

        Returns in `*array` an array through which the subresource of the
        mapped graphics resource `resource` which corresponds to array index
        `arrayIndex` and mipmap level `mipLevel` may be accessed. The value set
        in `array` may change every time that `resource` is mapped.

        If `resource` is not a texture then it cannot be accessed via an array
        and :py:obj:`~.cudaErrorUnknown` is returned. If `arrayIndex` is not a
        valid array index for `resource` then :py:obj:`~.cudaErrorInvalidValue`
        is returned. If `mipLevel` is not a valid mipmap level for `resource`
        then :py:obj:`~.cudaErrorInvalidValue` is returned. If `resource` is
        not mapped then :py:obj:`~.cudaErrorUnknown` is returned.

        Parameters
        ----------
        resource : :py:obj:`~.cudaGraphicsResource_t`
            Mapped resource to access
        arrayIndex : unsigned int
            Array index for array textures or cubemap face index as defined by
            :py:obj:`~.cudaGraphicsCubeFace` for cubemap textures for the
            subresource to access
        mipLevel : unsigned int
            Mipmap level for the subresource to access

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidResourceHandle`, :py:obj:`~.cudaErrorUnknown`
        array : :py:obj:`~.cudaArray_t`
            Returned array through which a subresource of `resource` may be
            accessed

        See Also
        --------
        :py:obj:`~.cudaGraphicsResourceGetMappedPointer`, :py:obj:`~.cuGraphicsSubResourceGetMappedArray`
    """


def cudaGraphicsUnmapResources(count, resources, stream):
    """
    cudaGraphicsUnmapResources(int count, resources, stream)
     Unmap graphics resources.

        Unmaps the `count` graphics resources in `resources`.

        Once unmapped, the resources in `resources` may not be accessed by CUDA
        until they are mapped again.

        This function provides the synchronization guarantee that any CUDA work
        issued in `stream` before :py:obj:`~.cudaGraphicsUnmapResources()` will
        complete before any subsequently issued graphics work begins.

        If `resources` contains any duplicate entries then
        :py:obj:`~.cudaErrorInvalidResourceHandle` is returned. If any of
        `resources` are not presently mapped for access by CUDA then
        :py:obj:`~.cudaErrorUnknown` is returned.

        Parameters
        ----------
        count : int
            Number of resources to unmap
        resources : :py:obj:`~.cudaGraphicsResource_t`
            Resources to unmap
        stream : :py:obj:`~.CUstream` or :py:obj:`~.cudaStream_t`
            Stream for synchronization

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidResourceHandle`, :py:obj:`~.cudaErrorUnknown`

        See Also
        --------
        :py:obj:`~.cudaGraphicsMapResources`, :py:obj:`~.cuGraphicsUnmapResources`
    """


def cudaGraphicsUnregisterResource(resource):
    """
    cudaGraphicsUnregisterResource(resource)
     Unregisters a graphics resource for access by CUDA.

        Unregisters the graphics resource `resource` so it is not accessible by
        CUDA unless registered again.

        If `resource` is invalid then
        :py:obj:`~.cudaErrorInvalidResourceHandle` is returned.

        Parameters
        ----------
        resource : :py:obj:`~.cudaGraphicsResource_t`
            Resource to unregister

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidResourceHandle`, :py:obj:`~.cudaErrorUnknown`

        See Also
        --------
        :py:obj:`~.cudaGraphicsD3D9RegisterResource`, :py:obj:`~.cudaGraphicsD3D10RegisterResource`, :py:obj:`~.cudaGraphicsD3D11RegisterResource`, :py:obj:`~.cudaGraphicsGLRegisterBuffer`, :py:obj:`~.cudaGraphicsGLRegisterImage`, :py:obj:`~.cuGraphicsUnregisterResource`
    """


def cudaGraphicsVDPAURegisterOutputSurface(vdpSurface, flags):
    """
    cudaGraphicsVDPAURegisterOutputSurface(vdpSurface, unsigned int flags)
     Register a VdpOutputSurface object.

        Registers the VdpOutputSurface specified by `vdpSurface` for access by
        CUDA. A handle to the registered object is returned as `resource`. The
        surface's intended usage is specified using `flags`, as follows:

        - :py:obj:`~.cudaGraphicsMapFlagsNone`: Specifies no hints about how
          this resource will be used. It is therefore assumed that this
          resource will be read from and written to by CUDA. This is the
          default value.

        - :py:obj:`~.cudaGraphicsMapFlagsReadOnly`: Specifies that CUDA will
          not write to this resource.

        - :py:obj:`~.cudaGraphicsMapFlagsWriteDiscard`: Specifies that CUDA
          will not read from this resource and will write over the entire
          contents of the resource, so none of the data previously stored in
          the resource will be preserved.

        Parameters
        ----------
        vdpSurface : :py:obj:`~.VdpOutputSurface`
            VDPAU object to be registered
        flags : unsigned int
            Map flags

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidDevice`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidResourceHandle`, :py:obj:`~.cudaErrorUnknown`
        resource : :py:obj:`~.cudaGraphicsResource`
            Pointer to the returned object handle

        See Also
        --------
        :py:obj:`~.cudaVDPAUSetVDPAUDevice`, :py:obj:`~.cudaGraphicsUnregisterResource`, :py:obj:`~.cudaGraphicsSubResourceGetMappedArray`, :py:obj:`~.cuGraphicsVDPAURegisterOutputSurface`
    """


def cudaGraphicsVDPAURegisterVideoSurface(vdpSurface, flags):
    """
    cudaGraphicsVDPAURegisterVideoSurface(vdpSurface, unsigned int flags)
     Register a VdpVideoSurface object.

        Registers the VdpVideoSurface specified by `vdpSurface` for access by
        CUDA. A handle to the registered object is returned as `resource`. The
        surface's intended usage is specified using `flags`, as follows:

        - :py:obj:`~.cudaGraphicsMapFlagsNone`: Specifies no hints about how
          this resource will be used. It is therefore assumed that this
          resource will be read from and written to by CUDA. This is the
          default value.

        - :py:obj:`~.cudaGraphicsMapFlagsReadOnly`: Specifies that CUDA will
          not write to this resource.

        - :py:obj:`~.cudaGraphicsMapFlagsWriteDiscard`: Specifies that CUDA
          will not read from this resource and will write over the entire
          contents of the resource, so none of the data previously stored in
          the resource will be preserved.

        Parameters
        ----------
        vdpSurface : :py:obj:`~.VdpVideoSurface`
            VDPAU object to be registered
        flags : unsigned int
            Map flags

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidDevice`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidResourceHandle`, :py:obj:`~.cudaErrorUnknown`
        resource : :py:obj:`~.cudaGraphicsResource`
            Pointer to the returned object handle

        See Also
        --------
        :py:obj:`~.cudaVDPAUSetVDPAUDevice`, :py:obj:`~.cudaGraphicsUnregisterResource`, :py:obj:`~.cudaGraphicsSubResourceGetMappedArray`, :py:obj:`~.cuGraphicsVDPAURegisterVideoSurface`
    """


def cudaHostAlloc(size, flags):
    """
    cudaHostAlloc(size_t size, unsigned int flags)
     Allocates page-locked memory on the host.

        Allocates `size` bytes of host memory that is page-locked and
        accessible to the device. The driver tracks the virtual memory ranges
        allocated with this function and automatically accelerates calls to
        functions such as :py:obj:`~.cudaMemcpy()`. Since the memory can be
        accessed directly by the device, it can be read or written with much
        higher bandwidth than pageable memory obtained with functions such as
        :py:obj:`~.malloc()`. Allocating excessive amounts of pinned memory may
        degrade system performance, since it reduces the amount of memory
        available to the system for paging. As a result, this function is best
        used sparingly to allocate staging areas for data exchange between host
        and device.

        The `flags` parameter enables different options to be specified that
        affect the allocation, as follows.

        - :py:obj:`~.cudaHostAllocDefault`: This flag's value is defined to be
          0 and causes :py:obj:`~.cudaHostAlloc()` to emulate
          :py:obj:`~.cudaMallocHost()`.

        - :py:obj:`~.cudaHostAllocPortable`: The memory returned by this call
          will be considered as pinned memory by all CUDA contexts, not just
          the one that performed the allocation.

        - :py:obj:`~.cudaHostAllocMapped`: Maps the allocation into the CUDA
          address space. The device pointer to the memory may be obtained by
          calling :py:obj:`~.cudaHostGetDevicePointer()`.

        - :py:obj:`~.cudaHostAllocWriteCombined`: Allocates the memory as
          write-combined (WC). WC memory can be transferred across the PCI
          Express bus more quickly on some system configurations, but cannot be
          read efficiently by most CPUs. WC memory is a good option for buffers
          that will be written by the CPU and read by the device via mapped
          pinned memory or host->device transfers.

        All of these flags are orthogonal to one another: a developer may
        allocate memory that is portable, mapped and/or write-combined with no
        restrictions.

        In order for the :py:obj:`~.cudaHostAllocMapped` flag to have any
        effect, the CUDA context must support the :py:obj:`~.cudaDeviceMapHost`
        flag, which can be checked via :py:obj:`~.cudaGetDeviceFlags()`. The
        :py:obj:`~.cudaDeviceMapHost` flag is implicitly set for contexts
        created via the runtime API.

        The :py:obj:`~.cudaHostAllocMapped` flag may be specified on CUDA
        contexts for devices that do not support mapped pinned memory. The
        failure is deferred to :py:obj:`~.cudaHostGetDevicePointer()` because
        the memory may be mapped into other CUDA contexts via the
        :py:obj:`~.cudaHostAllocPortable` flag.

        Memory allocated by this function must be freed with
        :py:obj:`~.cudaFreeHost()`.

        Parameters
        ----------
        size : size_t
            Requested allocation size in bytes
        flags : unsigned int
            Requested properties of allocated memory

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorMemoryAllocation`
        pHost : Any
            Device pointer to allocated memory

        See Also
        --------
        :py:obj:`~.cudaSetDeviceFlags`, :py:obj:`~.cudaMallocHost (C API)`, :py:obj:`~.cudaFreeHost`, :py:obj:`~.cudaGetDeviceFlags`, :py:obj:`~.cuMemHostAlloc`
    """

cudaHostAllocDefault: int
cudaHostAllocMapped: int
cudaHostAllocPortable: int
cudaHostAllocWriteCombined: int

def cudaHostGetDevicePointer(pHost, flags):
    """
    cudaHostGetDevicePointer(pHost, unsigned int flags)
     Passes back device pointer of mapped host memory allocated by cudaHostAlloc or registered by cudaHostRegister.

        Passes back the device pointer corresponding to the mapped, pinned host
        buffer allocated by :py:obj:`~.cudaHostAlloc()` or registered by
        :py:obj:`~.cudaHostRegister()`.

        :py:obj:`~.cudaHostGetDevicePointer()` will fail if the
        :py:obj:`~.cudaDeviceMapHost` flag was not specified before deferred
        context creation occurred, or if called on a device that does not
        support mapped, pinned memory.

        For devices that have a non-zero value for the device attribute
        :py:obj:`~.cudaDevAttrCanUseHostPointerForRegisteredMem`, the memory
        can also be accessed from the device using the host pointer `pHost`.
        The device pointer returned by :py:obj:`~.cudaHostGetDevicePointer()`
        may or may not match the original host pointer `pHost` and depends on
        the devices visible to the application. If all devices visible to the
        application have a non-zero value for the device attribute, the device
        pointer returned by :py:obj:`~.cudaHostGetDevicePointer()` will match
        the original pointer `pHost`. If any device visible to the application
        has a zero value for the device attribute, the device pointer returned
        by :py:obj:`~.cudaHostGetDevicePointer()` will not match the original
        host pointer `pHost`, but it will be suitable for use on all devices
        provided Unified Virtual Addressing is enabled. In such systems, it is
        valid to access the memory using either pointer on devices that have a
        non-zero value for the device attribute. Note however that such devices
        should access the memory using only of the two pointers and not both.

        `flags` provides for future releases. For now, it must be set to 0.

        Parameters
        ----------
        pHost : Any
            Requested host pointer mapping
        flags : unsigned int
            Flags for extensions (must be 0 for now)

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorMemoryAllocation`
        pDevice : Any
            Returned device pointer for mapped memory

        See Also
        --------
        :py:obj:`~.cudaSetDeviceFlags`, :py:obj:`~.cudaHostAlloc`, :py:obj:`~.cuMemHostGetDevicePointer`
    """


def cudaHostGetFlags(pHost):
    """
    cudaHostGetFlags(pHost)
     Passes back flags used to allocate pinned host memory allocated by cudaHostAlloc.

        :py:obj:`~.cudaHostGetFlags()` will fail if the input pointer does not
        reside in an address range allocated by :py:obj:`~.cudaHostAlloc()`.

        Parameters
        ----------
        pHost : Any
            Host pointer

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`
        pFlags : unsigned int
            Returned flags word

        See Also
        --------
        :py:obj:`~.cudaHostAlloc`, :py:obj:`~.cuMemHostGetFlags`
    """


def cudaHostRegister(ptr, size, flags):
    """
    cudaHostRegister(ptr, size_t size, unsigned int flags)
     Registers an existing host memory range for use by CUDA.

        Page-locks the memory range specified by `ptr` and `size` and maps it
        for the device(s) as specified by `flags`. This memory range also is
        added to the same tracking mechanism as :py:obj:`~.cudaHostAlloc()` to
        automatically accelerate calls to functions such as
        :py:obj:`~.cudaMemcpy()`. Since the memory can be accessed directly by
        the device, it can be read or written with much higher bandwidth than
        pageable memory that has not been registered. Page-locking excessive
        amounts of memory may degrade system performance, since it reduces the
        amount of memory available to the system for paging. As a result, this
        function is best used sparingly to register staging areas for data
        exchange between host and device.

        On systems where :py:obj:`~.pageableMemoryAccessUsesHostPageTables` is
        true, :py:obj:`~.cudaHostRegister` will not page-lock the memory range
        specified by `ptr` but only populate unpopulated pages.

        :py:obj:`~.cudaHostRegister` is supported only on I/O coherent devices
        that have a non-zero value for the device attribute
        :py:obj:`~.cudaDevAttrHostRegisterSupported`.

        The `flags` parameter enables different options to be specified that
        affect the allocation, as follows.

        - :py:obj:`~.cudaHostRegisterDefault`: On a system with unified virtual
          addressing, the memory will be both mapped and portable. On a system
          with no unified virtual addressing, the memory will be neither mapped
          nor portable.

        - :py:obj:`~.cudaHostRegisterPortable`: The memory returned by this
          call will be considered as pinned memory by all CUDA contexts, not
          just the one that performed the allocation.

        - :py:obj:`~.cudaHostRegisterMapped`: Maps the allocation into the CUDA
          address space. The device pointer to the memory may be obtained by
          calling :py:obj:`~.cudaHostGetDevicePointer()`.

        - :py:obj:`~.cudaHostRegisterIoMemory`: The passed memory pointer is
          treated as pointing to some memory-mapped I/O space, e.g. belonging
          to a third-party PCIe device, and it will marked as non cache-
          coherent and contiguous.

        - :py:obj:`~.cudaHostRegisterReadOnly`: The passed memory pointer is
          treated as pointing to memory that is considered read-only by the
          device. On platforms without
          :py:obj:`~.cudaDevAttrPageableMemoryAccessUsesHostPageTables`, this
          flag is required in order to register memory mapped to the CPU as
          read-only. Support for the use of this flag can be queried from the
          device attribute
          :py:obj:`~.cudaDevAttrHostRegisterReadOnlySupported`. Using this flag
          with a current context associated with a device that does not have
          this attribute set will cause :py:obj:`~.cudaHostRegister` to error
          with cudaErrorNotSupported.

        All of these flags are orthogonal to one another: a developer may page-
        lock memory that is portable or mapped with no restrictions.

        The CUDA context must have been created with the
        :py:obj:`~.cudaMapHost` flag in order for the
        :py:obj:`~.cudaHostRegisterMapped` flag to have any effect.

        The :py:obj:`~.cudaHostRegisterMapped` flag may be specified on CUDA
        contexts for devices that do not support mapped pinned memory. The
        failure is deferred to :py:obj:`~.cudaHostGetDevicePointer()` because
        the memory may be mapped into other CUDA contexts via the
        :py:obj:`~.cudaHostRegisterPortable` flag.

        For devices that have a non-zero value for the device attribute
        :py:obj:`~.cudaDevAttrCanUseHostPointerForRegisteredMem`, the memory
        can also be accessed from the device using the host pointer `ptr`. The
        device pointer returned by :py:obj:`~.cudaHostGetDevicePointer()` may
        or may not match the original host pointer `ptr` and depends on the
        devices visible to the application. If all devices visible to the
        application have a non-zero value for the device attribute, the device
        pointer returned by :py:obj:`~.cudaHostGetDevicePointer()` will match
        the original pointer `ptr`. If any device visible to the application
        has a zero value for the device attribute, the device pointer returned
        by :py:obj:`~.cudaHostGetDevicePointer()` will not match the original
        host pointer `ptr`, but it will be suitable for use on all devices
        provided Unified Virtual Addressing is enabled. In such systems, it is
        valid to access the memory using either pointer on devices that have a
        non-zero value for the device attribute. Note however that such devices
        should access the memory using only of the two pointers and not both.

        The memory page-locked by this function must be unregistered with
        :py:obj:`~.cudaHostUnregister()`.

        Parameters
        ----------
        ptr : Any
            Host pointer to memory to page-lock
        size : size_t
            Size in bytes of the address range to page-lock in bytes
        flags : unsigned int
            Flags for allocation request

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorMemoryAllocation`, :py:obj:`~.cudaErrorHostMemoryAlreadyRegistered`, :py:obj:`~.cudaErrorNotSupported`

        See Also
        --------
        :py:obj:`~.cudaHostUnregister`, :py:obj:`~.cudaHostGetFlags`, :py:obj:`~.cudaHostGetDevicePointer`, :py:obj:`~.cuMemHostRegister`
    """

cudaHostRegisterDefault: int
cudaHostRegisterIoMemory: int
cudaHostRegisterMapped: int
cudaHostRegisterPortable: int
cudaHostRegisterReadOnly: int

def cudaHostUnregister(ptr):
    """
    cudaHostUnregister(ptr)
     Unregisters a memory range that was registered with cudaHostRegister.

        Unmaps the memory range whose base address is specified by `ptr`, and
        makes it pageable again.

        The base address must be the same one specified to
        :py:obj:`~.cudaHostRegister()`.

        Parameters
        ----------
        ptr : Any
            Host pointer to memory to unregister

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorHostMemoryNotRegistered`

        See Also
        --------
        :py:obj:`~.cudaHostUnregister`, :py:obj:`~.cuMemHostUnregister`
    """


def cudaImportExternalMemory(memHandleDesc: 'Optional[cudaExternalMemoryHandleDesc]'):
    """
    cudaImportExternalMemory(cudaExternalMemoryHandleDesc memHandleDesc: Optional[cudaExternalMemoryHandleDesc])
     Imports an external memory object.

        Imports an externally allocated memory object and returns a handle to
        that in `extMem_out`.

        The properties of the handle being imported must be described in
        `memHandleDesc`. The :py:obj:`~.cudaExternalMemoryHandleDesc` structure
        is defined as follows:

        **View CUDA Toolkit Documentation for a C++ code example**

        where :py:obj:`~.cudaExternalMemoryHandleDesc.type` specifies the type
        of handle being imported. :py:obj:`~.cudaExternalMemoryHandleType` is
        defined as:

        **View CUDA Toolkit Documentation for a C++ code example**

        If :py:obj:`~.cudaExternalMemoryHandleDesc.type` is
        :py:obj:`~.cudaExternalMemoryHandleTypeOpaqueFd`, then
        :py:obj:`~.cudaExternalMemoryHandleDesc`::handle::fd must be a valid
        file descriptor referencing a memory object. Ownership of the file
        descriptor is transferred to the CUDA driver when the handle is
        imported successfully. Performing any operations on the file descriptor
        after it is imported results in undefined behavior.

        If :py:obj:`~.cudaExternalMemoryHandleDesc.type` is
        :py:obj:`~.cudaExternalMemoryHandleTypeOpaqueWin32`, then exactly one
        of :py:obj:`~.cudaExternalMemoryHandleDesc`::handle::win32::handle and
        :py:obj:`~.cudaExternalMemoryHandleDesc`::handle::win32::name must not
        be NULL. If
        :py:obj:`~.cudaExternalMemoryHandleDesc`::handle::win32::handle is not
        NULL, then it must represent a valid shared NT handle that references a
        memory object. Ownership of this handle is not transferred to CUDA
        after the import operation, so the application must release the handle
        using the appropriate system call. If
        :py:obj:`~.cudaExternalMemoryHandleDesc`::handle::win32::name is not
        NULL, then it must point to a NULL-terminated array of UTF-16
        characters that refers to a memory object.

        If :py:obj:`~.cudaExternalMemoryHandleDesc.type` is
        :py:obj:`~.cudaExternalMemoryHandleTypeOpaqueWin32Kmt`, then
        :py:obj:`~.cudaExternalMemoryHandleDesc`::handle::win32::handle must be
        non-NULL and
        :py:obj:`~.cudaExternalMemoryHandleDesc`::handle::win32::name must be
        NULL. The handle specified must be a globally shared KMT handle. This
        handle does not hold a reference to the underlying object, and thus
        will be invalid when all references to the memory object are destroyed.

        If :py:obj:`~.cudaExternalMemoryHandleDesc.type` is
        :py:obj:`~.cudaExternalMemoryHandleTypeD3D12Heap`, then exactly one of
        :py:obj:`~.cudaExternalMemoryHandleDesc`::handle::win32::handle and
        :py:obj:`~.cudaExternalMemoryHandleDesc`::handle::win32::name must not
        be NULL. If
        :py:obj:`~.cudaExternalMemoryHandleDesc`::handle::win32::handle is not
        NULL, then it must represent a valid shared NT handle that is returned
        by ID3D12Device::CreateSharedHandle when referring to a ID3D12Heap
        object. This handle holds a reference to the underlying object. If
        :py:obj:`~.cudaExternalMemoryHandleDesc`::handle::win32::name is not
        NULL, then it must point to a NULL-terminated array of UTF-16
        characters that refers to a ID3D12Heap object.

        If :py:obj:`~.cudaExternalMemoryHandleDesc.type` is
        :py:obj:`~.cudaExternalMemoryHandleTypeD3D12Resource`, then exactly one
        of :py:obj:`~.cudaExternalMemoryHandleDesc`::handle::win32::handle and
        :py:obj:`~.cudaExternalMemoryHandleDesc`::handle::win32::name must not
        be NULL. If
        :py:obj:`~.cudaExternalMemoryHandleDesc`::handle::win32::handle is not
        NULL, then it must represent a valid shared NT handle that is returned
        by ID3D12Device::CreateSharedHandle when referring to a ID3D12Resource
        object. This handle holds a reference to the underlying object. If
        :py:obj:`~.cudaExternalMemoryHandleDesc`::handle::win32::name is not
        NULL, then it must point to a NULL-terminated array of UTF-16
        characters that refers to a ID3D12Resource object.

        If :py:obj:`~.cudaExternalMemoryHandleDesc.type` is
        :py:obj:`~.cudaExternalMemoryHandleTypeD3D11Resource`,then exactly one
        of :py:obj:`~.cudaExternalMemoryHandleDesc`::handle::win32::handle and
        :py:obj:`~.cudaExternalMemoryHandleDesc`::handle::win32::name must not
        be NULL. If
        :py:obj:`~.cudaExternalMemoryHandleDesc`::handle::win32::handle is
        not NULL, then it must represent a valid shared NT handle that is
        returned by IDXGIResource1::CreateSharedHandle when referring to a
        ID3D11Resource object. If
        :py:obj:`~.cudaExternalMemoryHandleDesc`::handle::win32::name is not
        NULL, then it must point to a NULL-terminated array of UTF-16
        characters that refers to a ID3D11Resource object.

        If :py:obj:`~.cudaExternalMemoryHandleDesc.type` is
        :py:obj:`~.cudaExternalMemoryHandleTypeD3D11ResourceKmt`, then
        :py:obj:`~.cudaExternalMemoryHandleDesc`::handle::win32::handle must be
        non-NULL and
        :py:obj:`~.cudaExternalMemoryHandleDesc`::handle::win32::name must be
        NULL. The handle specified must be a valid shared KMT handle that is
        returned by IDXGIResource::GetSharedHandle when referring to a
        ID3D11Resource object.

        If :py:obj:`~.cudaExternalMemoryHandleDesc.type` is
        :py:obj:`~.cudaExternalMemoryHandleTypeNvSciBuf`, then
        :py:obj:`~.cudaExternalMemoryHandleDesc`::handle::nvSciBufObject must
        be NON-NULL and reference a valid NvSciBuf object. If the NvSciBuf
        object imported into CUDA is also mapped by other drivers, then the
        application must use :py:obj:`~.cudaWaitExternalSemaphoresAsync` or
        :py:obj:`~.cudaSignalExternalSemaphoresAsync` as approprriate barriers
        to maintain coherence between CUDA and the other drivers. See
        :py:obj:`~.cudaExternalSemaphoreWaitSkipNvSciBufMemSync` and
        :py:obj:`~.cudaExternalSemaphoreSignalSkipNvSciBufMemSync` for memory
        synchronization.

        The size of the memory object must be specified in
        :py:obj:`~.cudaExternalMemoryHandleDesc.size`.

        Specifying the flag :py:obj:`~.cudaExternalMemoryDedicated` in
        :py:obj:`~.cudaExternalMemoryHandleDesc.flags` indicates that the
        resource is a dedicated resource. The definition of what a dedicated
        resource is outside the scope of this extension. This flag must be set
        if :py:obj:`~.cudaExternalMemoryHandleDesc.type` is one of the
        following: :py:obj:`~.cudaExternalMemoryHandleTypeD3D12Resource`
        :py:obj:`~.cudaExternalMemoryHandleTypeD3D11Resource`
        :py:obj:`~.cudaExternalMemoryHandleTypeD3D11ResourceKmt`

        Parameters
        ----------
        memHandleDesc : :py:obj:`~.cudaExternalMemoryHandleDesc`
            Memory import handle descriptor

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidResourceHandle`, :py:obj:`~.cudaErrorOperatingSystem`
        extMem_out : :py:obj:`~.cudaExternalMemory_t`
            Returned handle to an external memory object

        See Also
        --------
        :py:obj:`~.cudaDestroyExternalMemory`, :py:obj:`~.cudaExternalMemoryGetMappedBuffer`, :py:obj:`~.cudaExternalMemoryGetMappedMipmappedArray`

        Notes
        -----
        If the Vulkan memory imported into CUDA is mapped on the CPU then the application must use vkInvalidateMappedMemoryRanges/vkFlushMappedMemoryRanges as well as appropriate Vulkan pipeline barriers to maintain coherence between CPU and GPU. For more information on these APIs, please refer to "Synchronization
        and Cache Control" chapter from Vulkan specification.
    """


def cudaImportExternalSemaphore(semHandleDesc: 'Optional[cudaExternalSemaphoreHandleDesc]'):
    """
    cudaImportExternalSemaphore(cudaExternalSemaphoreHandleDesc semHandleDesc: Optional[cudaExternalSemaphoreHandleDesc])
     Imports an external semaphore.

        Imports an externally allocated synchronization object and returns a
        handle to that in `extSem_out`.

        The properties of the handle being imported must be described in
        `semHandleDesc`. The :py:obj:`~.cudaExternalSemaphoreHandleDesc` is
        defined as follows:

        **View CUDA Toolkit Documentation for a C++ code example**

        where :py:obj:`~.cudaExternalSemaphoreHandleDesc.type` specifies the
        type of handle being imported.
        :py:obj:`~.cudaExternalSemaphoreHandleType` is defined as:

        **View CUDA Toolkit Documentation for a C++ code example**

        If :py:obj:`~.cudaExternalSemaphoreHandleDesc.type` is
        :py:obj:`~.cudaExternalSemaphoreHandleTypeOpaqueFd`, then
        :py:obj:`~.cudaExternalSemaphoreHandleDesc`::handle::fd must be a valid
        file descriptor referencing a synchronization object. Ownership of the
        file descriptor is transferred to the CUDA driver when the handle is
        imported successfully. Performing any operations on the file descriptor
        after it is imported results in undefined behavior.

        If :py:obj:`~.cudaExternalSemaphoreHandleDesc.type` is
        :py:obj:`~.cudaExternalSemaphoreHandleTypeOpaqueWin32`, then exactly
        one of
        :py:obj:`~.cudaExternalSemaphoreHandleDesc`::handle::win32::handle and
        :py:obj:`~.cudaExternalSemaphoreHandleDesc`::handle::win32::name must
        not be NULL. If
        :py:obj:`~.cudaExternalSemaphoreHandleDesc`::handle::win32::handle is
        not NULL, then it must represent a valid shared NT handle that
        references a synchronization object. Ownership of this handle is not
        transferred to CUDA after the import operation, so the application must
        release the handle using the appropriate system call. If
        :py:obj:`~.cudaExternalSemaphoreHandleDesc`::handle::win32::name is not
        NULL, then it must name a valid synchronization object.

        If :py:obj:`~.cudaExternalSemaphoreHandleDesc.type` is
        :py:obj:`~.cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt`, then
        :py:obj:`~.cudaExternalSemaphoreHandleDesc`::handle::win32::handle must
        be non-NULL and
        :py:obj:`~.cudaExternalSemaphoreHandleDesc`::handle::win32::name must
        be NULL. The handle specified must be a globally shared KMT handle.
        This handle does not hold a reference to the underlying object, and
        thus will be invalid when all references to the synchronization object
        are destroyed.

        If :py:obj:`~.cudaExternalSemaphoreHandleDesc.type` is
        :py:obj:`~.cudaExternalSemaphoreHandleTypeD3D12Fence`, then exactly one
        of :py:obj:`~.cudaExternalSemaphoreHandleDesc`::handle::win32::handle
        and :py:obj:`~.cudaExternalSemaphoreHandleDesc`::handle::win32::name
        must not be NULL. If
        :py:obj:`~.cudaExternalSemaphoreHandleDesc`::handle::win32::handle is
        not NULL, then it must represent a valid shared NT handle that is
        returned by ID3D12Device::CreateSharedHandle when referring to a
        ID3D12Fence object. This handle holds a reference to the underlying
        object. If
        :py:obj:`~.cudaExternalSemaphoreHandleDesc`::handle::win32::name is not
        NULL, then it must name a valid synchronization object that refers to a
        valid ID3D12Fence object.

        If :py:obj:`~.cudaExternalSemaphoreHandleDesc.type` is
        :py:obj:`~.cudaExternalSemaphoreHandleTypeD3D11Fence`, then exactly one
        of :py:obj:`~.cudaExternalSemaphoreHandleDesc`::handle::win32::handle
        and :py:obj:`~.cudaExternalSemaphoreHandleDesc`::handle::win32::name
        must not be NULL. If
        :py:obj:`~.cudaExternalSemaphoreHandleDesc`::handle::win32::handle is
        not NULL, then it must represent a valid shared NT handle that is
        returned by ID3D11Fence::CreateSharedHandle. If
        :py:obj:`~.cudaExternalSemaphoreHandleDesc`::handle::win32::name is not
        NULL, then it must name a valid synchronization object that refers to a
        valid ID3D11Fence object.

        If :py:obj:`~.cudaExternalSemaphoreHandleDesc.type` is
        :py:obj:`~.cudaExternalSemaphoreHandleTypeNvSciSync`, then
        :py:obj:`~.cudaExternalSemaphoreHandleDesc`::handle::nvSciSyncObj
        represents a valid NvSciSyncObj.

        :py:obj:`~.cudaExternalSemaphoreHandleTypeKeyedMutex`, then exactly one
        of :py:obj:`~.cudaExternalSemaphoreHandleDesc`::handle::win32::handle
        and :py:obj:`~.cudaExternalSemaphoreHandleDesc`::handle::win32::name
        must not be NULL. If
        :py:obj:`~.cudaExternalSemaphoreHandleDesc`::handle::win32::handle is
        not NULL, then it represent a valid shared NT handle that is returned
        by IDXGIResource1::CreateSharedHandle when referring to a
        IDXGIKeyedMutex object.

        If :py:obj:`~.cudaExternalSemaphoreHandleDesc.type` is
        :py:obj:`~.cudaExternalSemaphoreHandleTypeKeyedMutexKmt`, then
        :py:obj:`~.cudaExternalSemaphoreHandleDesc`::handle::win32::handle must
        be non-NULL and
        :py:obj:`~.cudaExternalSemaphoreHandleDesc`::handle::win32::name must
        be NULL. The handle specified must represent a valid KMT handle that is
        returned by IDXGIResource::GetSharedHandle when referring to a
        IDXGIKeyedMutex object.

        If :py:obj:`~.cudaExternalSemaphoreHandleDesc.type` is
        :py:obj:`~.cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd`, then
        :py:obj:`~.cudaExternalSemaphoreHandleDesc`::handle::fd must be a valid
        file descriptor referencing a synchronization object. Ownership of the
        file descriptor is transferred to the CUDA driver when the handle is
        imported successfully. Performing any operations on the file descriptor
        after it is imported results in undefined behavior.

        If :py:obj:`~.cudaExternalSemaphoreHandleDesc.type` is
        :py:obj:`~.cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32`, then
        exactly one of
        :py:obj:`~.cudaExternalSemaphoreHandleDesc`::handle::win32::handle and
        :py:obj:`~.cudaExternalSemaphoreHandleDesc`::handle::win32::name must
        not be NULL. If
        :py:obj:`~.cudaExternalSemaphoreHandleDesc`::handle::win32::handle is
        not NULL, then it must represent a valid shared NT handle that
        references a synchronization object. Ownership of this handle is not
        transferred to CUDA after the import operation, so the application must
        release the handle using the appropriate system call. If
        :py:obj:`~.cudaExternalSemaphoreHandleDesc`::handle::win32::name is not
        NULL, then it must name a valid synchronization object.

        Parameters
        ----------
        semHandleDesc : :py:obj:`~.cudaExternalSemaphoreHandleDesc`
            Semaphore import handle descriptor

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidResourceHandle`, :py:obj:`~.cudaErrorOperatingSystem`
        extSem_out : :py:obj:`~.cudaExternalSemaphore_t`
            Returned handle to an external semaphore

        See Also
        --------
        :py:obj:`~.cudaDestroyExternalSemaphore`, :py:obj:`~.cudaSignalExternalSemaphoresAsync`, :py:obj:`~.cudaWaitExternalSemaphoresAsync`
    """


def cudaInitDevice(device, deviceFlags, flags):
    """
    cudaInitDevice(int device, unsigned int deviceFlags, unsigned int flags)
     Initialize device to be used for GPU executions.

        This function will initialize the CUDA Runtime structures and primary
        context on `device` when called, but the context will not be made
        current to `device`.

        When :py:obj:`~.cudaInitDeviceFlagsAreValid` is set in `flags`,
        deviceFlags are applied to the requested device. The values of
        deviceFlags match those of the flags parameters in
        :py:obj:`~.cudaSetDeviceFlags`. The effect may be verified by
        :py:obj:`~.cudaGetDeviceFlags`.

        This function will return an error if the device is in
        :py:obj:`~.cudaComputeModeExclusiveProcess` and is occupied by another
        process or if the device is in :py:obj:`~.cudaComputeModeProhibited`.

        Parameters
        ----------
        device : int
            Device on which the runtime will initialize itself.
        deviceFlags : unsigned int
            Parameters for device operation.
        flags : unsigned int
            Flags for controlling the device initialization.

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidDevice`,

        See Also
        --------
        :py:obj:`~.cudaGetDeviceCount`, :py:obj:`~.cudaGetDevice`, :py:obj:`~.cudaGetDeviceProperties`, :py:obj:`~.cudaChooseDevice`, :py:obj:`~.cudaSetDevice` :py:obj:`~.cuCtxSetCurrent`
    """

cudaInitDeviceFlagsAreValid: int
cudaInvalidDeviceId: int

def cudaIpcCloseMemHandle(devPtr):
    """
    cudaIpcCloseMemHandle(devPtr)
     Attempts to close memory mapped with cudaIpcOpenMemHandle.

        Decrements the reference count of the memory returnd by
        :py:obj:`~.cudaIpcOpenMemHandle` by 1. When the reference count reaches
        0, this API unmaps the memory. The original allocation in the exporting
        process as well as imported mappings in other processes will be
        unaffected.

        Any resources used to enable peer access will be freed if this is the
        last mapping using them.

        IPC functionality is restricted to devices with support for unified
        addressing on Linux and Windows operating systems. IPC functionality on
        Windows is supported for compatibility purposes but not recommended as
        it comes with performance cost. Users can test their device for IPC
        functionality by calling :py:obj:`~.cudaDeviceGetAttribute` with
        :py:obj:`~.cudaDevAttrIpcEventSupport`

        Parameters
        ----------
        devPtr : Any
            Device pointer returned by :py:obj:`~.cudaIpcOpenMemHandle`

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorMapBufferObjectFailed`, :py:obj:`~.cudaErrorNotSupported`, :py:obj:`~.cudaErrorInvalidValue`

        See Also
        --------
        :py:obj:`~.cudaMalloc`, :py:obj:`~.cudaFree`, :py:obj:`~.cudaIpcGetEventHandle`, :py:obj:`~.cudaIpcOpenEventHandle`, :py:obj:`~.cudaIpcGetMemHandle`, :py:obj:`~.cudaIpcOpenMemHandle`, :py:obj:`~.cuIpcCloseMemHandle`
    """


def cudaIpcGetEventHandle(event):
    """
    cudaIpcGetEventHandle(event)
     Gets an interprocess handle for a previously allocated event.

        Takes as input a previously allocated event. This event must have been
        created with the :py:obj:`~.cudaEventInterprocess` and
        :py:obj:`~.cudaEventDisableTiming` flags set. This opaque handle may be
        copied into other processes and opened with
        :py:obj:`~.cudaIpcOpenEventHandle` to allow efficient hardware
        synchronization between GPU work in different processes.

        After the event has been been opened in the importing process,
        :py:obj:`~.cudaEventRecord`, :py:obj:`~.cudaEventSynchronize`,
        :py:obj:`~.cudaStreamWaitEvent` and :py:obj:`~.cudaEventQuery` may be
        used in either process. Performing operations on the imported event
        after the exported event has been freed with
        :py:obj:`~.cudaEventDestroy` will result in undefined behavior.

        IPC functionality is restricted to devices with support for unified
        addressing on Linux and Windows operating systems. IPC functionality on
        Windows is supported for compatibility purposes but not recommended as
        it comes with performance cost. Users can test their device for IPC
        functionality by calling :py:obj:`~.cudaDeviceGetAttribute` with
        :py:obj:`~.cudaDevAttrIpcEventSupport`

        Parameters
        ----------
        event : :py:obj:`~.CUevent` or :py:obj:`~.cudaEvent_t`
            Event allocated with :py:obj:`~.cudaEventInterprocess` and
            :py:obj:`~.cudaEventDisableTiming` flags.

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidResourceHandle`, :py:obj:`~.cudaErrorMemoryAllocation`, :py:obj:`~.cudaErrorMapBufferObjectFailed`, :py:obj:`~.cudaErrorNotSupported`, :py:obj:`~.cudaErrorInvalidValue`
        handle : :py:obj:`~.cudaIpcEventHandle_t`
            Pointer to a user allocated cudaIpcEventHandle in which to return
            the opaque event handle

        See Also
        --------
        :py:obj:`~.cudaEventCreate`, :py:obj:`~.cudaEventDestroy`, :py:obj:`~.cudaEventSynchronize`, :py:obj:`~.cudaEventQuery`, :py:obj:`~.cudaStreamWaitEvent`, :py:obj:`~.cudaIpcOpenEventHandle`, :py:obj:`~.cudaIpcGetMemHandle`, :py:obj:`~.cudaIpcOpenMemHandle`, :py:obj:`~.cudaIpcCloseMemHandle`, :py:obj:`~.cuIpcGetEventHandle`
    """


def cudaIpcGetMemHandle(devPtr):
    """
    cudaIpcGetMemHandle(devPtr)
     Gets an interprocess memory handle for an existing device memory allocation.

        Takes a pointer to the base of an existing device memory allocation
        created with :py:obj:`~.cudaMalloc` and exports it for use in another
        process. This is a lightweight operation and may be called multiple
        times on an allocation without adverse effects.

        If a region of memory is freed with :py:obj:`~.cudaFree` and a
        subsequent call to :py:obj:`~.cudaMalloc` returns memory with the same
        device address, :py:obj:`~.cudaIpcGetMemHandle` will return a unique
        handle for the new memory.

        IPC functionality is restricted to devices with support for unified
        addressing on Linux and Windows operating systems. IPC functionality on
        Windows is supported for compatibility purposes but not recommended as
        it comes with performance cost. Users can test their device for IPC
        functionality by calling :py:obj:`~.cudaDeviceGetAttribute` with
        :py:obj:`~.cudaDevAttrIpcEventSupport`

        Parameters
        ----------
        devPtr : Any
            Base pointer to previously allocated device memory

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorMemoryAllocation`, :py:obj:`~.cudaErrorMapBufferObjectFailed`, :py:obj:`~.cudaErrorNotSupported`, :py:obj:`~.cudaErrorInvalidValue`
        handle : :py:obj:`~.cudaIpcMemHandle_t`
            Pointer to user allocated :py:obj:`~.cudaIpcMemHandle` to return
            the handle in.

        See Also
        --------
        :py:obj:`~.cudaMalloc`, :py:obj:`~.cudaFree`, :py:obj:`~.cudaIpcGetEventHandle`, :py:obj:`~.cudaIpcOpenEventHandle`, :py:obj:`~.cudaIpcOpenMemHandle`, :py:obj:`~.cudaIpcCloseMemHandle`, :py:obj:`~.cuIpcGetMemHandle`
    """

cudaIpcMemLazyEnablePeerAccess: int

def cudaIpcOpenEventHandle(handle: 'cudaIpcEventHandle_t'):
    """
    cudaIpcOpenEventHandle(cudaIpcEventHandle_t handle: cudaIpcEventHandle_t)
     Opens an interprocess event handle for use in the current process.

        Opens an interprocess event handle exported from another process with
        :py:obj:`~.cudaIpcGetEventHandle`. This function returns a
        :py:obj:`~.cudaEvent_t` that behaves like a locally created event with
        the :py:obj:`~.cudaEventDisableTiming` flag specified. This event must
        be freed with :py:obj:`~.cudaEventDestroy`.

        Performing operations on the imported event after the exported event
        has been freed with :py:obj:`~.cudaEventDestroy` will result in
        undefined behavior.

        IPC functionality is restricted to devices with support for unified
        addressing on Linux and Windows operating systems. IPC functionality on
        Windows is supported for compatibility purposes but not recommended as
        it comes with performance cost. Users can test their device for IPC
        functionality by calling :py:obj:`~.cudaDeviceGetAttribute` with
        :py:obj:`~.cudaDevAttrIpcEventSupport`

        Parameters
        ----------
        handle : :py:obj:`~.cudaIpcEventHandle_t`
            Interprocess handle to open

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorMapBufferObjectFailed`, :py:obj:`~.cudaErrorNotSupported`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorDeviceUninitialized`
        event : :py:obj:`~.cudaEvent_t`
            Returns the imported event

        See Also
        --------
        :py:obj:`~.cudaEventCreate`, :py:obj:`~.cudaEventDestroy`, :py:obj:`~.cudaEventSynchronize`, :py:obj:`~.cudaEventQuery`, :py:obj:`~.cudaStreamWaitEvent`, :py:obj:`~.cudaIpcGetEventHandle`, :py:obj:`~.cudaIpcGetMemHandle`, :py:obj:`~.cudaIpcOpenMemHandle`, :py:obj:`~.cudaIpcCloseMemHandle`, :py:obj:`~.cuIpcOpenEventHandle`
    """


def cudaIpcOpenMemHandle(handle: 'cudaIpcMemHandle_t', flags):
    """
    cudaIpcOpenMemHandle(cudaIpcMemHandle_t handle: cudaIpcMemHandle_t, unsigned int flags)
     Opens an interprocess memory handle exported from another process and returns a device pointer usable in the local process.

        Maps memory exported from another process with
        :py:obj:`~.cudaIpcGetMemHandle` into the current device address space.
        For contexts on different devices :py:obj:`~.cudaIpcOpenMemHandle` can
        attempt to enable peer access between the devices as if the user called
        :py:obj:`~.cudaDeviceEnablePeerAccess`. This behavior is controlled by
        the :py:obj:`~.cudaIpcMemLazyEnablePeerAccess` flag.
        :py:obj:`~.cudaDeviceCanAccessPeer` can determine if a mapping is
        possible.

        :py:obj:`~.cudaIpcOpenMemHandle` can open handles to devices that may
        not be visible in the process calling the API.

        Contexts that may open :py:obj:`~.cudaIpcMemHandles` are restricted in
        the following way. :py:obj:`~.cudaIpcMemHandles` from each device in a
        given process may only be opened by one context per device per other
        process.

        If the memory handle has already been opened by the current context,
        the reference count on the handle is incremented by 1 and the existing
        device pointer is returned.

        Memory returned from :py:obj:`~.cudaIpcOpenMemHandle` must be freed
        with :py:obj:`~.cudaIpcCloseMemHandle`.

        Calling :py:obj:`~.cudaFree` on an exported memory region before
        calling :py:obj:`~.cudaIpcCloseMemHandle` in the importing context will
        result in undefined behavior.

        IPC functionality is restricted to devices with support for unified
        addressing on Linux and Windows operating systems. IPC functionality on
        Windows is supported for compatibility purposes but not recommended as
        it comes with performance cost. Users can test their device for IPC
        functionality by calling :py:obj:`~.cudaDeviceGetAttribute` with
        :py:obj:`~.cudaDevAttrIpcEventSupport`

        Parameters
        ----------
        handle : :py:obj:`~.cudaIpcMemHandle_t`
            :py:obj:`~.cudaIpcMemHandle` to open
        flags : unsigned int
            Flags for this operation. Must be specified as
            :py:obj:`~.cudaIpcMemLazyEnablePeerAccess`

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorMapBufferObjectFailed`, :py:obj:`~.cudaErrorInvalidResourceHandle`, :py:obj:`~.cudaErrorDeviceUninitialized`, :py:obj:`~.cudaErrorTooManyPeers`, :py:obj:`~.cudaErrorNotSupported`, :py:obj:`~.cudaErrorInvalidValue`
        devPtr : Any
            Returned device pointer

        See Also
        --------
        :py:obj:`~.cudaMalloc`, :py:obj:`~.cudaFree`, :py:obj:`~.cudaIpcGetEventHandle`, :py:obj:`~.cudaIpcOpenEventHandle`, :py:obj:`~.cudaIpcGetMemHandle`, :py:obj:`~.cudaIpcCloseMemHandle`, :py:obj:`~.cudaDeviceEnablePeerAccess`, :py:obj:`~.cudaDeviceCanAccessPeer`, :py:obj:`~.cuIpcOpenMemHandle`

        Notes
        -----
        No guarantees are made about the address returned in `*devPtr`. 
         In particular, multiple processes may not receive the same address for the same `handle`.
    """

cudaKernelNodeAttributeAccessPolicyWindow: int
cudaKernelNodeAttributeClusterDimension: int
cudaKernelNodeAttributeClusterSchedulingPolicyPreference: int
cudaKernelNodeAttributeCooperative: int
cudaKernelNodeAttributeDeviceUpdatableKernelNode: int
cudaKernelNodeAttributeMemSyncDomain: int
cudaKernelNodeAttributeMemSyncDomainMap: int
cudaKernelNodeAttributePreferredSharedMemoryCarveout: int
cudaKernelNodeAttributePriority: int

def cudaKernelSetAttributeForDevice(kernel, attr: 'cudaFuncAttribute', value, device):
    """
    cudaKernelSetAttributeForDevice(kernel, attr: cudaFuncAttribute, int value, int device)
     Sets information about a kernel.

        This call sets the value of a specified attribute `attr` on the kernel
        `kernel` for the requested device `device` to an integer value
        specified by `value`. This function returns :py:obj:`~.cudaSuccess` if
        the new value of the attribute could be successfully set. If the set
        fails, this call will return an error. Not all attributes can have
        values set. Attempting to set a value on a read-only attribute will
        result in an error (:py:obj:`~.cudaErrorInvalidValue`)

        Note that attributes set using :py:obj:`~.cudaFuncSetAttribute()` will
        override the attribute set by this API irrespective of whether the call
        to :py:obj:`~.cudaFuncSetAttribute()` is made before or after this API
        call. Because of this and the stricter locking requirements mentioned
        below it is suggested that this call be used during the initialization
        path and not on each thread accessing `kernel` such as on kernel
        launches or on the critical path.

        Valid values for `attr` are:

        - :py:obj:`~.cudaFuncAttributeMaxDynamicSharedMemorySize` - The
          requested maximum size in bytes of dynamically-allocated shared
          memory. The sum of this value and the function attribute
          :py:obj:`~.sharedSizeBytes` cannot exceed the device attribute
          :py:obj:`~.cudaDevAttrMaxSharedMemoryPerBlockOptin`. The maximal size
          of requestable dynamic shared memory may differ by GPU architecture.

        - :py:obj:`~.cudaFuncAttributePreferredSharedMemoryCarveout` - On
          devices where the L1 cache and shared memory use the same hardware
          resources, this sets the shared memory carveout preference, in
          percent of the total shared memory. See
          :py:obj:`~.cudaDevAttrMaxSharedMemoryPerMultiprocessor`. This is only
          a hint, and the driver can choose a different ratio if required to
          execute the function.

        - :py:obj:`~.cudaFuncAttributeRequiredClusterWidth`: The required
          cluster width in blocks. The width, height, and depth values must
          either all be 0 or all be positive. The validity of the cluster
          dimensions is checked at launch time. If the value is set during
          compile time, it cannot be set at runtime. Setting it at runtime will
          return cudaErrorNotPermitted.

        - :py:obj:`~.cudaFuncAttributeRequiredClusterHeight`: The required
          cluster height in blocks. The width, height, and depth values must
          either all be 0 or all be positive. The validity of the cluster
          dimensions is checked at launch time. If the value is set during
          compile time, it cannot be set at runtime. Setting it at runtime will
          return cudaErrorNotPermitted.

        - :py:obj:`~.cudaFuncAttributeRequiredClusterDepth`: The required
          cluster depth in blocks. The width, height, and depth values must
          either all be 0 or all be positive. The validity of the cluster
          dimensions is checked at launch time. If the value is set during
          compile time, it cannot be set at runtime. Setting it at runtime will
          return cudaErrorNotPermitted.

        - :py:obj:`~.cudaFuncAttributeNonPortableClusterSizeAllowed`: Indicates
          whether the function can be launched with non-portable cluster size.
          1 is allowed, 0 is disallowed.

        - :py:obj:`~.cudaFuncAttributeClusterSchedulingPolicyPreference`: The
          block scheduling policy of a function. The value type is
          cudaClusterSchedulingPolicy.

        Parameters
        ----------
        kernel : :py:obj:`~.cudaKernel_t`
            Kernel to set attribute of
        attr : :py:obj:`~.cudaFuncAttribute`
            Attribute requested
        value : int
            Value to set
        device : int
            Device to set attribute of

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidDeviceFunction`, :py:obj:`~.cudaErrorInvalidValue`

        See Also
        --------
        :py:obj:`~.cudaLibraryLoadData`, :py:obj:`~.cudaLibraryLoadFromFile`, :py:obj:`~.cudaLibraryUnload`, :py:obj:`~.cudaLibraryGetKernel`, :py:obj:`~.cudaLaunchKernel`, :py:obj:`~.cudaFuncSetAttribute`, :py:obj:`~.cuKernelSetAttribute`

        Notes
        -----
        The API has stricter locking requirements in comparison to its legacy counterpart :py:obj:`~.cudaFuncSetAttribute()` due to device-wide semantics. If multiple threads are trying to set the same attribute on the same device simultaneously, the attribute setting will depend on the interleavings chosen by the OS scheduler and memory consistency.
    """


def cudaLaunchHostFunc(stream, fn, userData):
    """
    cudaLaunchHostFunc(stream, fn, userData)
     Enqueues a host function call in a stream.

        Enqueues a host function to run in a stream. The function will be
        called after currently enqueued work and will block work added after
        it.

        The host function must not make any CUDA API calls. Attempting to use a
        CUDA API may result in :py:obj:`~.cudaErrorNotPermitted`, but this is
        not required. The host function must not perform any synchronization
        that may depend on outstanding CUDA work not mandated to run earlier.
        Host functions without a mandated order (such as in independent
        streams) execute in undefined order and may be serialized.

        For the purposes of Unified Memory, execution makes a number of
        guarantees:

        - The stream is considered idle for the duration of the function's
          execution. Thus, for example, the function may always use memory
          attached to the stream it was enqueued in.

        - The start of execution of the function has the same effect as
          synchronizing an event recorded in the same stream immediately prior
          to the function. It thus synchronizes streams which have been
          "joined" prior to the function.

        - Adding device work to any stream does not have the effect of making
          the stream active until all preceding host functions and stream
          callbacks have executed. Thus, for example, a function might use
          global attached memory even if work has been added to another stream,
          if the work has been ordered behind the function call with an event.

        - Completion of the function does not cause a stream to become active
          except as described above. The stream will remain idle if no device
          work follows the function, and will remain idle across consecutive
          host functions or stream callbacks without device work in between.
          Thus, for example, stream synchronization can be done by signaling
          from a host function at the end of the stream.

        Note that, in constrast to :py:obj:`~.cuStreamAddCallback`, the
        function will not be called in the event of an error in the CUDA
        context.

        Parameters
        ----------
        hStream : :py:obj:`~.CUstream` or :py:obj:`~.cudaStream_t`
            Stream to enqueue function call in
        fn : :py:obj:`~.cudaHostFn_t`
            The function to call once preceding stream operations are complete
        userData : Any
            User-specified data to be passed to the function

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidResourceHandle`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorNotSupported`

        See Also
        --------
        :py:obj:`~.cudaStreamCreate`, :py:obj:`~.cudaStreamQuery`, :py:obj:`~.cudaStreamSynchronize`, :py:obj:`~.cudaStreamWaitEvent`, :py:obj:`~.cudaStreamDestroy`, :py:obj:`~.cudaMallocManaged`, :py:obj:`~.cudaStreamAttachMemAsync`, :py:obj:`~.cudaStreamAddCallback`, :py:obj:`~.cuLaunchHostFunc`
    """


def cudaLibraryEnumerateKernels(numKernels, lib):
    """
    cudaLibraryEnumerateKernels(unsigned int numKernels, lib)
     Retrieve the kernel handles within a library.

        Returns in `kernels` a maximum number of `numKernels` kernel handles
        within `lib`. The returned kernel handle becomes invalid when the
        library is unloaded.

        Parameters
        ----------
        numKernels : unsigned int
            Maximum number of kernel handles may be returned to the buffer
        lib : :py:obj:`~.cudaLibrary_t`
            Library to query from

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorCudartUnloading`, :py:obj:`~.cudaErrorInitializationError`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidResourceHandle`
        kernels : List[:py:obj:`~.cudaKernel_t`]
            Buffer where the kernel handles are returned to

        See Also
        --------
        :py:obj:`~.cudaLibraryGetKernelCount`, :py:obj:`~.cuLibraryEnumerateKernels`
    """


def cudaLibraryGetGlobal(library, name):
    """
    cudaLibraryGetGlobal(library, char *name)
     Returns a global device pointer.

        Returns in `*dptr` and `*bytes` the base pointer and size of the global
        with name `name` for the requested library `library` and the current
        device. If no global for the requested name `name` exists, the call
        returns :py:obj:`~.cudaErrorSymbolNotFound`. One of the parameters
        `dptr` or `numbytes` (not both) can be NULL in which case it is
        ignored. The returned `dptr` cannot be passed to the Symbol APIs such
        as :py:obj:`~.cudaMemcpyToSymbol`, :py:obj:`~.cudaMemcpyFromSymbol`,
        :py:obj:`~.cudaGetSymbolAddress`, or :py:obj:`~.cudaGetSymbolSize`.

        Parameters
        ----------
        library : :py:obj:`~.cudaLibrary_t`
            Library to retrieve global from
        name : bytes
            Name of global to retrieve

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorCudartUnloading`, :py:obj:`~.cudaErrorInitializationError`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidResourceHandle`, :py:obj:`~.cudaErrorSymbolNotFound` :py:obj:`~.cudaErrorDeviceUninitialized`, :py:obj:`~.cudaErrorContextIsDestroyed`
        dptr : Any
            Returned global device pointer for the requested library
        numbytes : int
            Returned global size in bytes

        See Also
        --------
        :py:obj:`~.cudaLibraryLoadData`, :py:obj:`~.cudaLibraryLoadFromFile`, :py:obj:`~.cudaLibraryUnload`, :py:obj:`~.cudaLibraryGetManaged`, :py:obj:`~.cuLibraryGetGlobal`
    """


def cudaLibraryGetKernel(library, name):
    """
    cudaLibraryGetKernel(library, char *name)
     Returns a kernel handle.

        Returns in `pKernel` the handle of the kernel with name `name` located
        in library `library`. If kernel handle is not found, the call returns
        :py:obj:`~.cudaErrorSymbolNotFound`.

        Parameters
        ----------
        library : :py:obj:`~.cudaLibrary_t`
            Library to retrieve kernel from
        name : bytes
            Name of kernel to retrieve

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorCudartUnloading`, :py:obj:`~.cudaErrorInitializationError`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidResourceHandle`, :py:obj:`~.cudaErrorSymbolNotFound`
        pKernel : :py:obj:`~.cudaKernel_t`
            Returned kernel handle

        See Also
        --------
        :py:obj:`~.cudaLibraryLoadData`, :py:obj:`~.cudaLibraryLoadFromFile`, :py:obj:`~.cudaLibraryUnload`, :py:obj:`~.cuLibraryGetKernel`
    """


def cudaLibraryGetKernelCount(lib):
    """
    cudaLibraryGetKernelCount(lib)
     Returns the number of kernels within a library.

        Returns in `count` the number of kernels in `lib`.

        Parameters
        ----------
        lib : :py:obj:`~.cudaLibrary_t`
            Library to query

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorCudartUnloading`, :py:obj:`~.cudaErrorInitializationError`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidResourceHandle`
        count : unsigned int
            Number of kernels found within the library

        See Also
        --------
        :py:obj:`~.cudaLibraryEnumerateKernels`, :py:obj:`~.cudaLibraryLoadFromFile`, :py:obj:`~.cudaLibraryLoadData`, :py:obj:`~.cuLibraryGetKernelCount`
    """


def cudaLibraryGetManaged(library, name):
    """
    cudaLibraryGetManaged(library, char *name)
     Returns a pointer to managed memory.

        Returns in `*dptr` and `*bytes` the base pointer and size of the
        managed memory with name `name` for the requested library `library`. If
        no managed memory with the requested name `name` exists, the call
        returns :py:obj:`~.cudaErrorSymbolNotFound`. One of the parameters
        `dptr` or `numbytes` (not both) can be NULL in which case it is
        ignored. Note that managed memory for library `library` is shared
        across devices and is registered when the library is loaded. The
        returned `dptr` cannot be passed to the Symbol APIs such as
        :py:obj:`~.cudaMemcpyToSymbol`, :py:obj:`~.cudaMemcpyFromSymbol`,
        :py:obj:`~.cudaGetSymbolAddress`, or :py:obj:`~.cudaGetSymbolSize`.

        Parameters
        ----------
        library : :py:obj:`~.cudaLibrary_t`
            Library to retrieve managed memory from
        name : bytes
            Name of managed memory to retrieve

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorCudartUnloading`, :py:obj:`~.cudaErrorInitializationError`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidResourceHandle`, :py:obj:`~.cudaErrorSymbolNotFound`
        dptr : Any
            Returned pointer to the managed memory
        numbytes : int
            Returned memory size in bytes

        See Also
        --------
        :py:obj:`~.cudaLibraryLoadData`, :py:obj:`~.cudaLibraryLoadFromFile`, :py:obj:`~.cudaLibraryUnload`, :py:obj:`~.cudaLibraryGetGlobal`, :py:obj:`~.cuLibraryGetManaged`
    """


def cudaLibraryGetUnifiedFunction(library, symbol):
    """
    cudaLibraryGetUnifiedFunction(library, char *symbol)
     Returns a pointer to a unified function.

        Returns in `*fptr` the function pointer to a unified function denoted
        by `symbol`. If no unified function with name `symbol` exists, the call
        returns :py:obj:`~.cudaErrorSymbolNotFound`. If there is no device with
        attribute :py:obj:`~.cudaDeviceProp.unifiedFunctionPointers` present in
        the system, the call may return :py:obj:`~.cudaErrorSymbolNotFound`.

        Parameters
        ----------
        library : :py:obj:`~.cudaLibrary_t`
            Library to retrieve function pointer memory from
        symbol : bytes
            Name of function pointer to retrieve

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorCudartUnloading`, :py:obj:`~.cudaErrorInitializationError`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidResourceHandle`, :py:obj:`~.cudaErrorSymbolNotFound`
        fptr : Any
            Returned pointer to a unified function

        See Also
        --------
        :py:obj:`~.cudaLibraryLoadData`, :py:obj:`~.cudaLibraryLoadFromFile`, :py:obj:`~.cudaLibraryUnload`, :py:obj:`~.cuLibraryGetUnifiedFunction`
    """


def cudaLibraryLoadData(code, jitOptions: 'Optional[Tuple[cudaJitOption] | List[cudaJitOption]]', jitOptionsValues: 'Optional[Tuple[Any] | List[Any]]', numJitOptions, libraryOptions: 'Optional[Tuple[cudaLibraryOption] | List[cudaLibraryOption]]', libraryOptionValues: 'Optional[Tuple[Any] | List[Any]]', numLibraryOptions):
    """
    cudaLibraryLoadData(code, jitOptions: Optional[Tuple[cudaJitOption] | List[cudaJitOption]], jitOptionsValues: Optional[Tuple[Any] | List[Any]], unsigned int numJitOptions, libraryOptions: Optional[Tuple[cudaLibraryOption] | List[cudaLibraryOption]], libraryOptionValues: Optional[Tuple[Any] | List[Any]], unsigned int numLibraryOptions)
     Load a library with specified code and options.

        Takes a pointer `code` and loads the corresponding library `library`
        based on the application defined library loading mode:

        - If module loading is set to EAGER, via the environment variables
          described in "Module loading", `library` is loaded eagerly into all
          contexts at the time of the call and future contexts at the time of
          creation until the library is unloaded with
          :py:obj:`~.cudaLibraryUnload()`.

        - If the environment variables are set to LAZY, `library` is not
          immediately loaded onto all existent contexts and will only be loaded
          when a function is needed for that context, such as a kernel launch.

        These environment variables are described in the CUDA programming guide
        under the "CUDA environment variables" section.

        The `code` may be a `cubin` or `fatbin` as output by nvcc, or a NULL-
        terminated `PTX`, either as output by nvcc or hand-written. A fatbin
        should also contain relocatable code when doing separate compilation.
        Please also see the documentation for nvrtc
        (https://docs.nvidia.com/cuda/nvrtc/index.html), nvjitlink
        (https://docs.nvidia.com/cuda/nvjitlink/index.html), and nvfatbin
        (https://docs.nvidia.com/cuda/nvfatbin/index.html) for more information
        on generating loadable code at runtime.

        Options are passed as an array via `jitOptions` and any corresponding
        parameters are passed in `jitOptionsValues`. The number of total JIT
        options is supplied via `numJitOptions`. Any outputs will be returned
        via `jitOptionsValues`.

        Library load options are passed as an array via `libraryOptions` and
        any corresponding parameters are passed in `libraryOptionValues`. The
        number of total library load options is supplied via
        `numLibraryOptions`.

        Parameters
        ----------
        code : Any
            Code to load
        jitOptions : List[:py:obj:`~.cudaJitOption`]
            Options for JIT
        jitOptionsValues : List[Any]
            Option values for JIT
        numJitOptions : unsigned int
            Number of options
        libraryOptions : List[:py:obj:`~.cudaLibraryOption`]
            Options for loading
        libraryOptionValues : List[Any]
            Option values for loading
        numLibraryOptions : unsigned int
            Number of options for loading

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorMemoryAllocation`, :py:obj:`~.cudaErrorInitializationError`, :py:obj:`~.cudaErrorCudartUnloading`, :py:obj:`~.cudaErrorInvalidPtx`, :py:obj:`~.cudaErrorUnsupportedPtxVersion`, :py:obj:`~.cudaErrorNoKernelImageForDevice`, :py:obj:`~.cudaErrorSharedObjectSymbolNotFound`, :py:obj:`~.cudaErrorSharedObjectInitFailed`, :py:obj:`~.cudaErrorJitCompilerNotFound`
        library : :py:obj:`~.cudaLibrary_t`
            Returned library

        See Also
        --------
        :py:obj:`~.cudaLibraryLoadFromFile`, :py:obj:`~.cudaLibraryUnload`, :py:obj:`~.cuLibraryLoadData`
    """


def cudaLibraryLoadFromFile(fileName, jitOptions: 'Optional[Tuple[cudaJitOption] | List[cudaJitOption]]', jitOptionsValues: 'Optional[Tuple[Any] | List[Any]]', numJitOptions, libraryOptions: 'Optional[Tuple[cudaLibraryOption] | List[cudaLibraryOption]]', libraryOptionValues: 'Optional[Tuple[Any] | List[Any]]', numLibraryOptions):
    """
    cudaLibraryLoadFromFile(char *fileName, jitOptions: Optional[Tuple[cudaJitOption] | List[cudaJitOption]], jitOptionsValues: Optional[Tuple[Any] | List[Any]], unsigned int numJitOptions, libraryOptions: Optional[Tuple[cudaLibraryOption] | List[cudaLibraryOption]], libraryOptionValues: Optional[Tuple[Any] | List[Any]], unsigned int numLibraryOptions)
     Load a library with specified file and options.

        Takes a pointer `code` and loads the corresponding library `library`
        based on the application defined library loading mode:

        - If module loading is set to EAGER, via the environment variables
          described in "Module loading", `library` is loaded eagerly into all
          contexts at the time of the call and future contexts at the time of
          creation until the library is unloaded with
          :py:obj:`~.cudaLibraryUnload()`.

        - If the environment variables are set to LAZY, `library` is not
          immediately loaded onto all existent contexts and will only be loaded
          when a function is needed for that context, such as a kernel launch.

        These environment variables are described in the CUDA programming guide
        under the "CUDA environment variables" section.

        The file should be a `cubin` file as output by nvcc, or a `PTX` file
        either as output by nvcc or handwritten, or a `fatbin` file as output
        by nvcc. A fatbin should also contain relocatable code when doing
        separate compilation. Please also see the documentation for nvrtc
        (https://docs.nvidia.com/cuda/nvrtc/index.html), nvjitlink
        (https://docs.nvidia.com/cuda/nvjitlink/index.html), and nvfatbin
        (https://docs.nvidia.com/cuda/nvfatbin/index.html) for more information
        on generating loadable code at runtime.

        Options are passed as an array via `jitOptions` and any corresponding
        parameters are passed in `jitOptionsValues`. The number of total
        options is supplied via `numJitOptions`. Any outputs will be returned
        via `jitOptionsValues`.

        Library load options are passed as an array via `libraryOptions` and
        any corresponding parameters are passed in `libraryOptionValues`. The
        number of total library load options is supplied via
        `numLibraryOptions`.

        Parameters
        ----------
        fileName : bytes
            File to load from
        jitOptions : List[:py:obj:`~.cudaJitOption`]
            Options for JIT
        jitOptionsValues : List[Any]
            Option values for JIT
        numJitOptions : unsigned int
            Number of options
        libraryOptions : List[:py:obj:`~.cudaLibraryOption`]
            Options for loading
        libraryOptionValues : List[Any]
            Option values for loading
        numLibraryOptions : unsigned int
            Number of options for loading

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorMemoryAllocation`, :py:obj:`~.cudaErrorInitializationError`, :py:obj:`~.cudaErrorCudartUnloading`, :py:obj:`~.cudaErrorInvalidPtx`, :py:obj:`~.cudaErrorUnsupportedPtxVersion`, :py:obj:`~.cudaErrorNoKernelImageForDevice`, :py:obj:`~.cudaErrorSharedObjectSymbolNotFound`, :py:obj:`~.cudaErrorSharedObjectInitFailed`, :py:obj:`~.cudaErrorJitCompilerNotFound`
        library : :py:obj:`~.cudaLibrary_t`
            Returned library

        See Also
        --------
        :py:obj:`~.cudaLibraryLoadData`, :py:obj:`~.cudaLibraryUnload`, :py:obj:`~.cuLibraryLoadFromFile`
    """


def cudaLibraryUnload(library):
    """
    cudaLibraryUnload(library)
     Unloads a library.

        Unloads the library specified with `library`

        Parameters
        ----------
        library : :py:obj:`~.cudaLibrary_t`
            Library to unload

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorCudartUnloading`, :py:obj:`~.cudaErrorInitializationError`, :py:obj:`~.cudaErrorInvalidValue`

        See Also
        --------
        :py:obj:`~.cudaLibraryLoadData`, :py:obj:`~.cudaLibraryLoadFromFile`, :py:obj:`~.cuLibraryUnload`
    """


def cudaMalloc(size):
    """
    cudaMalloc(size_t size)
     Allocate memory on the device.

        Allocates `size` bytes of linear memory on the device and returns in
        `*devPtr` a pointer to the allocated memory. The allocated memory is
        suitably aligned for any kind of variable. The memory is not cleared.
        :py:obj:`~.cudaMalloc()` returns :py:obj:`~.cudaErrorMemoryAllocation`
        in case of failure.

        The device version of :py:obj:`~.cudaFree` cannot be used with a
        `*devPtr` allocated using the host API, and vice versa.

        Parameters
        ----------
        size : size_t
            Requested allocation size in bytes

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorMemoryAllocation`
        devPtr : Any
            Pointer to allocated device memory

        See Also
        --------
        :py:obj:`~.cudaMallocPitch`, :py:obj:`~.cudaFree`, :py:obj:`~.cudaMallocArray`, :py:obj:`~.cudaFreeArray`, :py:obj:`~.cudaMalloc3D`, :py:obj:`~.cudaMalloc3DArray`, :py:obj:`~.cudaMallocHost (C API)`, :py:obj:`~.cudaFreeHost`, :py:obj:`~.cudaHostAlloc`, :py:obj:`~.cuMemAlloc`
    """


def cudaMalloc3D(extent: 'cudaExtent'):
    """
    cudaMalloc3D(cudaExtent extent: cudaExtent)
     Allocates logical 1D, 2D, or 3D memory objects on the device.

        Allocates at least `width` * `height` * `depth` bytes of linear memory
        on the device and returns a :py:obj:`~.cudaPitchedPtr` in which `ptr`
        is a pointer to the allocated memory. The function may pad the
        allocation to ensure hardware alignment requirements are met. The pitch
        returned in the `pitch` field of `pitchedDevPtr` is the width in bytes
        of the allocation.

        The returned :py:obj:`~.cudaPitchedPtr` contains additional fields
        `xsize` and `ysize`, the logical width and height of the allocation,
        which are equivalent to the `width` and `height` `extent` parameters
        provided by the programmer during allocation.

        For allocations of 2D and 3D objects, it is highly recommended that
        programmers perform allocations using :py:obj:`~.cudaMalloc3D()` or
        :py:obj:`~.cudaMallocPitch()`. Due to alignment restrictions in the
        hardware, this is especially true if the application will be performing
        memory copies involving 2D or 3D objects (whether linear memory or CUDA
        arrays).

        Parameters
        ----------
        extent : :py:obj:`~.cudaExtent`
            Requested allocation size (`width` field in bytes)

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorMemoryAllocation`
        pitchedDevPtr : :py:obj:`~.cudaPitchedPtr`
            Pointer to allocated pitched device memory

        See Also
        --------
        :py:obj:`~.cudaMallocPitch`, :py:obj:`~.cudaFree`, :py:obj:`~.cudaMemcpy3D`, :py:obj:`~.cudaMemset3D`, :py:obj:`~.cudaMalloc3DArray`, :py:obj:`~.cudaMallocArray`, :py:obj:`~.cudaFreeArray`, :py:obj:`~.cudaMallocHost (C API)`, :py:obj:`~.cudaFreeHost`, :py:obj:`~.cudaHostAlloc`, :py:obj:`~.make_cudaPitchedPtr`, :py:obj:`~.make_cudaExtent`, :py:obj:`~.cuMemAllocPitch`
    """


def cudaMalloc3DArray(desc: 'Optional[cudaChannelFormatDesc]', extent: 'cudaExtent', flags):
    """
    cudaMalloc3DArray(cudaChannelFormatDesc desc: Optional[cudaChannelFormatDesc], cudaExtent extent: cudaExtent, unsigned int flags)
     Allocate an array on the device.

        Allocates a CUDA array according to the
        :py:obj:`~.cudaChannelFormatDesc` structure `desc` and returns a handle
        to the new CUDA array in `*array`.

        The :py:obj:`~.cudaChannelFormatDesc` is defined as:

        **View CUDA Toolkit Documentation for a C++ code example**

        where :py:obj:`~.cudaChannelFormatKind` is one of
        :py:obj:`~.cudaChannelFormatKindSigned`,
        :py:obj:`~.cudaChannelFormatKindUnsigned`, or
        :py:obj:`~.cudaChannelFormatKindFloat`.

        :py:obj:`~.cudaMalloc3DArray()` can allocate the following:

        - A 1D array is allocated if the height and depth extents are both
          zero.

        - A 2D array is allocated if only the depth extent is zero.

        - A 3D array is allocated if all three extents are non-zero.

        - A 1D layered CUDA array is allocated if only the height extent is
          zero and the cudaArrayLayered flag is set. Each layer is a 1D array.
          The number of layers is determined by the depth extent.

        - A 2D layered CUDA array is allocated if all three extents are non-
          zero and the cudaArrayLayered flag is set. Each layer is a 2D array.
          The number of layers is determined by the depth extent.

        - A cubemap CUDA array is allocated if all three extents are non-zero
          and the cudaArrayCubemap flag is set. Width must be equal to height,
          and depth must be six. A cubemap is a special type of 2D layered CUDA
          array, where the six layers represent the six faces of a cube. The
          order of the six layers in memory is the same as that listed in
          :py:obj:`~.cudaGraphicsCubeFace`.

        - A cubemap layered CUDA array is allocated if all three extents are
          non-zero, and both, cudaArrayCubemap and cudaArrayLayered flags are
          set. Width must be equal to height, and depth must be a multiple of
          six. A cubemap layered CUDA array is a special type of 2D layered
          CUDA array that consists of a collection of cubemaps. The first six
          layers represent the first cubemap, the next six layers form the
          second cubemap, and so on.

        The `flags` parameter enables different options to be specified that
        affect the allocation, as follows.

        - :py:obj:`~.cudaArrayDefault`: This flag's value is defined to be 0
          and provides default array allocation

        - :py:obj:`~.cudaArrayLayered`: Allocates a layered CUDA array, with
          the depth extent indicating the number of layers

        - :py:obj:`~.cudaArrayCubemap`: Allocates a cubemap CUDA array. Width
          must be equal to height, and depth must be six. If the
          cudaArrayLayered flag is also set, depth must be a multiple of six.

        - :py:obj:`~.cudaArraySurfaceLoadStore`: Allocates a CUDA array that
          could be read from or written to using a surface reference.

        - :py:obj:`~.cudaArrayTextureGather`: This flag indicates that texture
          gather operations will be performed on the CUDA array. Texture gather
          can only be performed on 2D CUDA arrays.

        - :py:obj:`~.cudaArraySparse`: Allocates a CUDA array without physical
          backing memory. The subregions within this sparse array can later be
          mapped onto a physical memory allocation by calling
          :py:obj:`~.cuMemMapArrayAsync`. This flag can only be used for
          creating 2D, 3D or 2D layered sparse CUDA arrays. The physical
          backing memory must be allocated via :py:obj:`~.cuMemCreate`.

        - :py:obj:`~.cudaArrayDeferredMapping`: Allocates a CUDA array without
          physical backing memory. The entire array can later be mapped onto a
          physical memory allocation by calling :py:obj:`~.cuMemMapArrayAsync`.
          The physical backing memory must be allocated via
          :py:obj:`~.cuMemCreate`.

        The width, height and depth extents must meet certain size requirements
        as listed in the following table. All values are specified in elements.

        Note that 2D CUDA arrays have different size requirements if the
        :py:obj:`~.cudaArrayTextureGather` flag is set. In that case, the valid
        range for (width, height, depth) is ((1,maxTexture2DGather[0]),
        (1,maxTexture2DGather[1]), 0).

        **View CUDA Toolkit Documentation for a table example**

        Parameters
        ----------
        desc : :py:obj:`~.cudaChannelFormatDesc`
            Requested channel format
        extent : :py:obj:`~.cudaExtent`
            Requested allocation size (`width` field in elements)
        flags : unsigned int
            Flags for extensions

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorMemoryAllocation`
        array : :py:obj:`~.cudaArray_t`
            Pointer to allocated array in device memory

        See Also
        --------
        :py:obj:`~.cudaMalloc3D`, :py:obj:`~.cudaMalloc`, :py:obj:`~.cudaMallocPitch`, :py:obj:`~.cudaFree`, :py:obj:`~.cudaFreeArray`, :py:obj:`~.cudaMallocHost (C API)`, :py:obj:`~.cudaFreeHost`, :py:obj:`~.cudaHostAlloc`, :py:obj:`~.make_cudaExtent`, :py:obj:`~.cuArray3DCreate`
    """


def cudaMallocArray(desc: 'Optional[cudaChannelFormatDesc]', width, height, flags):
    """
    cudaMallocArray(cudaChannelFormatDesc desc: Optional[cudaChannelFormatDesc], size_t width, size_t height, unsigned int flags)
     Allocate an array on the device.

        Allocates a CUDA array according to the
        :py:obj:`~.cudaChannelFormatDesc` structure `desc` and returns a handle
        to the new CUDA array in `*array`.

        The :py:obj:`~.cudaChannelFormatDesc` is defined as:

        **View CUDA Toolkit Documentation for a C++ code example**

        where :py:obj:`~.cudaChannelFormatKind` is one of
        :py:obj:`~.cudaChannelFormatKindSigned`,
        :py:obj:`~.cudaChannelFormatKindUnsigned`, or
        :py:obj:`~.cudaChannelFormatKindFloat`.

        The `flags` parameter enables different options to be specified that
        affect the allocation, as follows.

        - :py:obj:`~.cudaArrayDefault`: This flag's value is defined to be 0
          and provides default array allocation

        - :py:obj:`~.cudaArraySurfaceLoadStore`: Allocates an array that can be
          read from or written to using a surface reference

        - :py:obj:`~.cudaArrayTextureGather`: This flag indicates that texture
          gather operations will be performed on the array.

        - :py:obj:`~.cudaArraySparse`: Allocates a CUDA array without physical
          backing memory. The subregions within this sparse array can later be
          mapped onto a physical memory allocation by calling
          :py:obj:`~.cuMemMapArrayAsync`. The physical backing memory must be
          allocated via :py:obj:`~.cuMemCreate`.

        - :py:obj:`~.cudaArrayDeferredMapping`: Allocates a CUDA array without
          physical backing memory. The entire array can later be mapped onto a
          physical memory allocation by calling :py:obj:`~.cuMemMapArrayAsync`.
          The physical backing memory must be allocated via
          :py:obj:`~.cuMemCreate`.

        `width` and `height` must meet certain size requirements. See
        :py:obj:`~.cudaMalloc3DArray()` for more details.

        Parameters
        ----------
        desc : :py:obj:`~.cudaChannelFormatDesc`
            Requested channel format
        width : size_t
            Requested array allocation width
        height : size_t
            Requested array allocation height
        flags : unsigned int
            Requested properties of allocated array

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorMemoryAllocation`
        array : :py:obj:`~.cudaArray_t`
            Pointer to allocated array in device memory

        See Also
        --------
        :py:obj:`~.cudaMalloc`, :py:obj:`~.cudaMallocPitch`, :py:obj:`~.cudaFree`, :py:obj:`~.cudaFreeArray`, :py:obj:`~.cudaMallocHost (C API)`, :py:obj:`~.cudaFreeHost`, :py:obj:`~.cudaMalloc3D`, :py:obj:`~.cudaMalloc3DArray`, :py:obj:`~.cudaHostAlloc`, :py:obj:`~.cuArrayCreate`
    """


def cudaMallocAsync(size, hStream):
    """
    cudaMallocAsync(size_t size, hStream)
     Allocates memory with stream ordered semantics.

        Inserts an allocation operation into `hStream`. A pointer to the
        allocated memory is returned immediately in *dptr. The allocation must
        not be accessed until the the allocation operation completes. The
        allocation comes from the memory pool associated with the stream's
        device.

        Parameters
        ----------
        size : size_t
            Number of bytes to allocate
        hStream : :py:obj:`~.CUstream` or :py:obj:`~.cudaStream_t`
            The stream establishing the stream ordering contract and the memory
            pool to allocate from

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorNotSupported`, :py:obj:`~.cudaErrorOutOfMemory`,
        devPtr : Any
            Returned device pointer

        See Also
        --------
        :py:obj:`~.cuMemAllocAsync`, cudaMallocAsync (C++ API), :py:obj:`~.cudaMallocFromPoolAsync`, :py:obj:`~.cudaFreeAsync`, :py:obj:`~.cudaDeviceSetMemPool`, :py:obj:`~.cudaDeviceGetDefaultMemPool`, :py:obj:`~.cudaDeviceGetMemPool`, :py:obj:`~.cudaMemPoolSetAccess`, :py:obj:`~.cudaMemPoolSetAttribute`, :py:obj:`~.cudaMemPoolGetAttribute`

        Notes
        -----
        The default memory pool of a device contains device memory from that device.

        Basic stream ordering allows future work submitted into the same stream to use the allocation. Stream query, stream synchronize, and CUDA events can be used to guarantee that the allocation operation completes before work submitted in a separate stream runs.

        During stream capture, this function results in the creation of an allocation node. In this case, the allocation is owned by the graph instead of the memory pool. The memory pool's properties are used to set the node's creation parameters.
    """


def cudaMallocFromPoolAsync(size, memPool, stream):
    """
    cudaMallocFromPoolAsync(size_t size, memPool, stream)
     Allocates memory from a specified pool with stream ordered semantics.

        Inserts an allocation operation into `hStream`. A pointer to the
        allocated memory is returned immediately in *dptr. The allocation must
        not be accessed until the the allocation operation completes. The
        allocation comes from the specified memory pool.

        Parameters
        ----------
        bytesize : size_t
            Number of bytes to allocate
        memPool : :py:obj:`~.CUmemoryPool` or :py:obj:`~.cudaMemPool_t`
            The pool to allocate from
        stream : :py:obj:`~.CUstream` or :py:obj:`~.cudaStream_t`
            The stream establishing the stream ordering semantic

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorNotSupported`, :py:obj:`~.cudaErrorOutOfMemory`
        ptr : Any
            Returned device pointer

        See Also
        --------
        :py:obj:`~.cuMemAllocFromPoolAsync`, cudaMallocAsync (C++ API), :py:obj:`~.cudaMallocAsync`, :py:obj:`~.cudaFreeAsync`, :py:obj:`~.cudaDeviceGetDefaultMemPool`, :py:obj:`~.cudaMemPoolCreate`, :py:obj:`~.cudaMemPoolSetAccess`, :py:obj:`~.cudaMemPoolSetAttribute`

        Notes
        -----
        During stream capture, this function results in the creation of an allocation node. In this case, the allocation is owned by the graph instead of the memory pool. The memory pool's properties are used to set the node's creation parameters.
    """


def cudaMallocHost(size):
    """
    cudaMallocHost(size_t size)
     Allocates page-locked memory on the host.

        Allocates `size` bytes of host memory that is page-locked and
        accessible to the device. The driver tracks the virtual memory ranges
        allocated with this function and automatically accelerates calls to
        functions such as :py:obj:`~.cudaMemcpy`*(). Since the memory can be
        accessed directly by the device, it can be read or written with much
        higher bandwidth than pageable memory obtained with functions such as
        :py:obj:`~.malloc()`.

        On systems where :py:obj:`~.pageableMemoryAccessUsesHostPageTables` is
        true, :py:obj:`~.cudaMallocHost` may not page-lock the allocated
        memory.

        Page-locking excessive amounts of memory with
        :py:obj:`~.cudaMallocHost()` may degrade system performance, since it
        reduces the amount of memory available to the system for paging. As a
        result, this function is best used sparingly to allocate staging areas
        for data exchange between host and device.

        Parameters
        ----------
        size : size_t
            Requested allocation size in bytes

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorMemoryAllocation`
        ptr : Any
            Pointer to allocated host memory

        See Also
        --------
        :py:obj:`~.cudaMalloc`, :py:obj:`~.cudaMallocPitch`, :py:obj:`~.cudaMallocArray`, :py:obj:`~.cudaMalloc3D`, :py:obj:`~.cudaMalloc3DArray`, :py:obj:`~.cudaHostAlloc`, :py:obj:`~.cudaFree`, :py:obj:`~.cudaFreeArray`, cudaMallocHost (C++ API), :py:obj:`~.cudaFreeHost`, :py:obj:`~.cudaHostAlloc`, :py:obj:`~.cuMemAllocHost`
    """


def cudaMallocManaged(size, flags):
    """
    cudaMallocManaged(size_t size, unsigned int flags)
     Allocates memory that will be automatically managed by the Unified Memory system.

        Allocates `size` bytes of managed memory on the device and returns in
        `*devPtr` a pointer to the allocated memory. If the device doesn't
        support allocating managed memory, :py:obj:`~.cudaErrorNotSupported` is
        returned. Support for managed memory can be queried using the device
        attribute :py:obj:`~.cudaDevAttrManagedMemory`. The allocated memory is
        suitably aligned for any kind of variable. The memory is not cleared.
        If `size` is 0, :py:obj:`~.cudaMallocManaged` returns
        :py:obj:`~.cudaErrorInvalidValue`. The pointer is valid on the CPU and
        on all GPUs in the system that support managed memory. All accesses to
        this pointer must obey the Unified Memory programming model.

        `flags` specifies the default stream association for this allocation.
        `flags` must be one of :py:obj:`~.cudaMemAttachGlobal` or
        :py:obj:`~.cudaMemAttachHost`. The default value for `flags` is
        :py:obj:`~.cudaMemAttachGlobal`. If :py:obj:`~.cudaMemAttachGlobal` is
        specified, then this memory is accessible from any stream on any
        device. If :py:obj:`~.cudaMemAttachHost` is specified, then the
        allocation should not be accessed from devices that have a zero value
        for the device attribute
        :py:obj:`~.cudaDevAttrConcurrentManagedAccess`; an explicit call to
        :py:obj:`~.cudaStreamAttachMemAsync` will be required to enable access
        on such devices.

        If the association is later changed via
        :py:obj:`~.cudaStreamAttachMemAsync` to a single stream, the default
        association, as specifed during :py:obj:`~.cudaMallocManaged`, is
        restored when that stream is destroyed. For managed variables, the
        default association is always :py:obj:`~.cudaMemAttachGlobal`. Note
        that destroying a stream is an asynchronous operation, and as a result,
        the change to default association won't happen until all work in the
        stream has completed.

        Memory allocated with :py:obj:`~.cudaMallocManaged` should be released
        with :py:obj:`~.cudaFree`.

        Device memory oversubscription is possible for GPUs that have a non-
        zero value for the device attribute
        :py:obj:`~.cudaDevAttrConcurrentManagedAccess`. Managed memory on such
        GPUs may be evicted from device memory to host memory at any time by
        the Unified Memory driver in order to make room for other allocations.

        In a system where all GPUs have a non-zero value for the device
        attribute :py:obj:`~.cudaDevAttrConcurrentManagedAccess`, managed
        memory may not be populated when this API returns and instead may be
        populated on access. In such systems, managed memory can migrate to any
        processor's memory at any time. The Unified Memory driver will employ
        heuristics to maintain data locality and prevent excessive page faults
        to the extent possible. The application can also guide the driver about
        memory usage patterns via :py:obj:`~.cudaMemAdvise`. The application
        can also explicitly migrate memory to a desired processor's memory via
        :py:obj:`~.cudaMemPrefetchAsync`.

        In a multi-GPU system where all of the GPUs have a zero value for the
        device attribute :py:obj:`~.cudaDevAttrConcurrentManagedAccess` and all
        the GPUs have peer-to-peer support with each other, the physical
        storage for managed memory is created on the GPU which is active at the
        time :py:obj:`~.cudaMallocManaged` is called. All other GPUs will
        reference the data at reduced bandwidth via peer mappings over the PCIe
        bus. The Unified Memory driver does not migrate memory among such GPUs.

        In a multi-GPU system where not all GPUs have peer-to-peer support with
        each other and where the value of the device attribute
        :py:obj:`~.cudaDevAttrConcurrentManagedAccess` is zero for at least one
        of those GPUs, the location chosen for physical storage of managed
        memory is system-dependent.

        - On Linux, the location chosen will be device memory as long as the
          current set of active contexts are on devices that either have peer-
          to-peer support with each other or have a non-zero value for the
          device attribute :py:obj:`~.cudaDevAttrConcurrentManagedAccess`. If
          there is an active context on a GPU that does not have a non-zero
          value for that device attribute and it does not have peer-to-peer
          support with the other devices that have active contexts on them,
          then the location for physical storage will be 'zero-copy' or host
          memory. Note that this means that managed memory that is located in
          device memory is migrated to host memory if a new context is created
          on a GPU that doesn't have a non-zero value for the device attribute
          and does not support peer-to-peer with at least one of the other
          devices that has an active context. This in turn implies that context
          creation may fail if there is insufficient host memory to migrate all
          managed allocations.

        - On Windows, the physical storage is always created in 'zero-copy' or
          host memory. All GPUs will reference the data at reduced bandwidth
          over the PCIe bus. In these circumstances, use of the environment
          variable CUDA_VISIBLE_DEVICES is recommended to restrict CUDA to only
          use those GPUs that have peer-to-peer support. Alternatively, users
          can also set CUDA_MANAGED_FORCE_DEVICE_ALLOC to a non-zero value to
          force the driver to always use device memory for physical storage.
          When this environment variable is set to a non-zero value, all
          devices used in that process that support managed memory have to be
          peer-to-peer compatible with each other. The error
          :py:obj:`~.cudaErrorInvalidDevice` will be returned if a device that
          supports managed memory is used and it is not peer-to-peer compatible
          with any of the other managed memory supporting devices that were
          previously used in that process, even if :py:obj:`~.cudaDeviceReset`
          has been called on those devices. These environment variables are
          described in the CUDA programming guide under the "CUDA environment
          variables" section.

        Parameters
        ----------
        size : size_t
            Requested allocation size in bytes
        flags : unsigned int
            Must be either :py:obj:`~.cudaMemAttachGlobal` or
            :py:obj:`~.cudaMemAttachHost` (defaults to
            :py:obj:`~.cudaMemAttachGlobal`)

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorMemoryAllocation`, :py:obj:`~.cudaErrorNotSupported`, :py:obj:`~.cudaErrorInvalidValue`
        devPtr : Any
            Pointer to allocated device memory

        See Also
        --------
        :py:obj:`~.cudaMallocPitch`, :py:obj:`~.cudaFree`, :py:obj:`~.cudaMallocArray`, :py:obj:`~.cudaFreeArray`, :py:obj:`~.cudaMalloc3D`, :py:obj:`~.cudaMalloc3DArray`, :py:obj:`~.cudaMallocHost (C API)`, :py:obj:`~.cudaFreeHost`, :py:obj:`~.cudaHostAlloc`, :py:obj:`~.cudaDeviceGetAttribute`, :py:obj:`~.cudaStreamAttachMemAsync`, :py:obj:`~.cuMemAllocManaged`
    """


def cudaMallocMipmappedArray(desc: 'Optional[cudaChannelFormatDesc]', extent: 'cudaExtent', numLevels, flags):
    """
    cudaMallocMipmappedArray(cudaChannelFormatDesc desc: Optional[cudaChannelFormatDesc], cudaExtent extent: cudaExtent, unsigned int numLevels, unsigned int flags)
     Allocate a mipmapped array on the device.

        Allocates a CUDA mipmapped array according to the
        :py:obj:`~.cudaChannelFormatDesc` structure `desc` and returns a handle
        to the new CUDA mipmapped array in `*mipmappedArray`. `numLevels`
        specifies the number of mipmap levels to be allocated. This value is
        clamped to the range [1, 1 + floor(log2(max(width, height, depth)))].

        The :py:obj:`~.cudaChannelFormatDesc` is defined as:

        **View CUDA Toolkit Documentation for a C++ code example**

        where :py:obj:`~.cudaChannelFormatKind` is one of
        :py:obj:`~.cudaChannelFormatKindSigned`,
        :py:obj:`~.cudaChannelFormatKindUnsigned`, or
        :py:obj:`~.cudaChannelFormatKindFloat`.

        :py:obj:`~.cudaMallocMipmappedArray()` can allocate the following:

        - A 1D mipmapped array is allocated if the height and depth extents are
          both zero.

        - A 2D mipmapped array is allocated if only the depth extent is zero.

        - A 3D mipmapped array is allocated if all three extents are non-zero.

        - A 1D layered CUDA mipmapped array is allocated if only the height
          extent is zero and the cudaArrayLayered flag is set. Each layer is a
          1D mipmapped array. The number of layers is determined by the depth
          extent.

        - A 2D layered CUDA mipmapped array is allocated if all three extents
          are non-zero and the cudaArrayLayered flag is set. Each layer is a 2D
          mipmapped array. The number of layers is determined by the depth
          extent.

        - A cubemap CUDA mipmapped array is allocated if all three extents are
          non-zero and the cudaArrayCubemap flag is set. Width must be equal to
          height, and depth must be six. The order of the six layers in memory
          is the same as that listed in :py:obj:`~.cudaGraphicsCubeFace`.

        - A cubemap layered CUDA mipmapped array is allocated if all three
          extents are non-zero, and both, cudaArrayCubemap and cudaArrayLayered
          flags are set. Width must be equal to height, and depth must be a
          multiple of six. A cubemap layered CUDA mipmapped array is a special
          type of 2D layered CUDA mipmapped array that consists of a collection
          of cubemap mipmapped arrays. The first six layers represent the first
          cubemap mipmapped array, the next six layers form the second cubemap
          mipmapped array, and so on.

        The `flags` parameter enables different options to be specified that
        affect the allocation, as follows.

        - :py:obj:`~.cudaArrayDefault`: This flag's value is defined to be 0
          and provides default mipmapped array allocation

        - :py:obj:`~.cudaArrayLayered`: Allocates a layered CUDA mipmapped
          array, with the depth extent indicating the number of layers

        - :py:obj:`~.cudaArrayCubemap`: Allocates a cubemap CUDA mipmapped
          array. Width must be equal to height, and depth must be six. If the
          cudaArrayLayered flag is also set, depth must be a multiple of six.

        - :py:obj:`~.cudaArraySurfaceLoadStore`: This flag indicates that
          individual mipmap levels of the CUDA mipmapped array will be read
          from or written to using a surface reference.

        - :py:obj:`~.cudaArrayTextureGather`: This flag indicates that texture
          gather operations will be performed on the CUDA array. Texture gather
          can only be performed on 2D CUDA mipmapped arrays, and the gather
          operations are performed only on the most detailed mipmap level.

        - :py:obj:`~.cudaArraySparse`: Allocates a CUDA mipmapped array without
          physical backing memory. The subregions within this sparse array can
          later be mapped onto a physical memory allocation by calling
          :py:obj:`~.cuMemMapArrayAsync`. This flag can only be used for
          creating 2D, 3D or 2D layered sparse CUDA mipmapped arrays. The
          physical backing memory must be allocated via
          :py:obj:`~.cuMemCreate`.

        - :py:obj:`~.cudaArrayDeferredMapping`: Allocates a CUDA mipmapped
          array without physical backing memory. The entire array can later be
          mapped onto a physical memory allocation by calling
          :py:obj:`~.cuMemMapArrayAsync`. The physical backing memory must be
          allocated via :py:obj:`~.cuMemCreate`.

        The width, height and depth extents must meet certain size requirements
        as listed in the following table. All values are specified in elements.

        **View CUDA Toolkit Documentation for a table example**

        Parameters
        ----------
        desc : :py:obj:`~.cudaChannelFormatDesc`
            Requested channel format
        extent : :py:obj:`~.cudaExtent`
            Requested allocation size (`width` field in elements)
        numLevels : unsigned int
            Number of mipmap levels to allocate
        flags : unsigned int
            Flags for extensions

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorMemoryAllocation`
        mipmappedArray : :py:obj:`~.cudaMipmappedArray_t`
            Pointer to allocated mipmapped array in device memory

        See Also
        --------
        :py:obj:`~.cudaMalloc3D`, :py:obj:`~.cudaMalloc`, :py:obj:`~.cudaMallocPitch`, :py:obj:`~.cudaFree`, :py:obj:`~.cudaFreeArray`, :py:obj:`~.cudaMallocHost (C API)`, :py:obj:`~.cudaFreeHost`, :py:obj:`~.cudaHostAlloc`, :py:obj:`~.make_cudaExtent`, :py:obj:`~.cuMipmappedArrayCreate`
    """


def cudaMallocPitch(width, height):
    """
    cudaMallocPitch(size_t width, size_t height)
     Allocates pitched memory on the device.

        Allocates at least `width` (in bytes) * `height` bytes of linear memory
        on the device and returns in `*devPtr` a pointer to the allocated
        memory. The function may pad the allocation to ensure that
        corresponding pointers in any given row will continue to meet the
        alignment requirements for coalescing as the address is updated from
        row to row. The pitch returned in `*pitch` by
        :py:obj:`~.cudaMallocPitch()` is the width in bytes of the allocation.
        The intended usage of `pitch` is as a separate parameter of the
        allocation, used to compute addresses within the 2D array. Given the
        row and column of an array element of type `T`, the address is computed
        as:

        **View CUDA Toolkit Documentation for a C++ code example**

        For allocations of 2D arrays, it is recommended that programmers
        consider performing pitch allocations using
        :py:obj:`~.cudaMallocPitch()`. Due to pitch alignment restrictions in
        the hardware, this is especially true if the application will be
        performing 2D memory copies between different regions of device memory
        (whether linear memory or CUDA arrays).

        Parameters
        ----------
        width : size_t
            Requested pitched allocation width (in bytes)
        height : size_t
            Requested pitched allocation height

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorMemoryAllocation`
        devPtr : Any
            Pointer to allocated pitched device memory
        pitch : int
            Pitch for allocation

        See Also
        --------
        :py:obj:`~.cudaMalloc`, :py:obj:`~.cudaFree`, :py:obj:`~.cudaMallocArray`, :py:obj:`~.cudaFreeArray`, :py:obj:`~.cudaMallocHost (C API)`, :py:obj:`~.cudaFreeHost`, :py:obj:`~.cudaMalloc3D`, :py:obj:`~.cudaMalloc3DArray`, :py:obj:`~.cudaHostAlloc`, :py:obj:`~.cuMemAllocPitch`
    """


def cudaMemAdvise(devPtr, count, advice: 'cudaMemoryAdvise', device):
    """
    cudaMemAdvise(devPtr, size_t count, advice: cudaMemoryAdvise, int device)
     Advise about the usage of a given memory range.

        Advise the Unified Memory subsystem about the usage pattern for the
        memory range starting at `devPtr` with a size of `count` bytes. The
        start address and end address of the memory range will be rounded down
        and rounded up respectively to be aligned to CPU page size before the
        advice is applied. The memory range must refer to managed memory
        allocated via :py:obj:`~.cudaMallocManaged` or declared via managed
        variables. The memory range could also refer to system-allocated
        pageable memory provided it represents a valid, host-accessible region
        of memory and all additional constraints imposed by `advice` as
        outlined below are also satisfied. Specifying an invalid system-
        allocated pageable memory range results in an error being returned.

        The `advice` parameter can take the following values:

        - :py:obj:`~.cudaMemAdviseSetReadMostly`: This implies that the data is
          mostly going to be read from and only occasionally written to. Any
          read accesses from any processor to this region will create a read-
          only copy of at least the accessed pages in that processor's memory.
          Additionally, if :py:obj:`~.cudaMemPrefetchAsync` is called on this
          region, it will create a read-only copy of the data on the
          destination processor. If any processor writes to this region, all
          copies of the corresponding page will be invalidated except for the
          one where the write occurred. The `device` argument is ignored for
          this advice. Note that for a page to be read-duplicated, the
          accessing processor must either be the CPU or a GPU that has a non-
          zero value for the device attribute
          :py:obj:`~.cudaDevAttrConcurrentManagedAccess`. Also, if a context is
          created on a device that does not have the device attribute
          :py:obj:`~.cudaDevAttrConcurrentManagedAccess` set, then read-
          duplication will not occur until all such contexts are destroyed. If
          the memory region refers to valid system-allocated pageable memory,
          then the accessing device must have a non-zero value for the device
          attribute :py:obj:`~.cudaDevAttrPageableMemoryAccess` for a read-only
          copy to be created on that device. Note however that if the accessing
          device also has a non-zero value for the device attribute
          :py:obj:`~.cudaDevAttrPageableMemoryAccessUsesHostPageTables`, then
          setting this advice will not create a read-only copy when that device
          accesses this memory region.

        - :py:obj:`~.cudaMemAdviceUnsetReadMostly`: Undoes the effect of
          :py:obj:`~.cudaMemAdviceReadMostly` and also prevents the Unified
          Memory driver from attempting heuristic read-duplication on the
          memory range. Any read-duplicated copies of the data will be
          collapsed into a single copy. The location for the collapsed copy
          will be the preferred location if the page has a preferred location
          and one of the read-duplicated copies was resident at that location.
          Otherwise, the location chosen is arbitrary.

        - :py:obj:`~.cudaMemAdviseSetPreferredLocation`: This advice sets the
          preferred location for the data to be the memory belonging to
          `device`. Passing in cudaCpuDeviceId for `device` sets the preferred
          location as host memory. If `device` is a GPU, then it must have a
          non-zero value for the device attribute
          :py:obj:`~.cudaDevAttrConcurrentManagedAccess`. Setting the preferred
          location does not cause data to migrate to that location immediately.
          Instead, it guides the migration policy when a fault occurs on that
          memory region. If the data is already in its preferred location and
          the faulting processor can establish a mapping without requiring the
          data to be migrated, then data migration will be avoided. On the
          other hand, if the data is not in its preferred location or if a
          direct mapping cannot be established, then it will be migrated to the
          processor accessing it. It is important to note that setting the
          preferred location does not prevent data prefetching done using
          :py:obj:`~.cudaMemPrefetchAsync`. Having a preferred location can
          override the page thrash detection and resolution logic in the
          Unified Memory driver. Normally, if a page is detected to be
          constantly thrashing between for example host and device memory, the
          page may eventually be pinned to host memory by the Unified Memory
          driver. But if the preferred location is set as device memory, then
          the page will continue to thrash indefinitely. If
          :py:obj:`~.cudaMemAdviseSetReadMostly` is also set on this memory
          region or any subset of it, then the policies associated with that
          advice will override the policies of this advice, unless read
          accesses from `device` will not result in a read-only copy being
          created on that device as outlined in description for the advice
          :py:obj:`~.cudaMemAdviseSetReadMostly`. If the memory region refers
          to valid system-allocated pageable memory, then `device` must have a
          non-zero value for the device attribute
          :py:obj:`~.cudaDevAttrPageableMemoryAccess`.

        - :py:obj:`~.cudaMemAdviseUnsetPreferredLocation`: Undoes the effect of
          :py:obj:`~.cudaMemAdviseSetPreferredLocation` and changes the
          preferred location to none.

        - :py:obj:`~.cudaMemAdviseSetAccessedBy`: This advice implies that the
          data will be accessed by `device`. Passing in
          :py:obj:`~.cudaCpuDeviceId` for `device` will set the advice for the
          CPU. If `device` is a GPU, then the device attribute
          :py:obj:`~.cudaDevAttrConcurrentManagedAccess` must be non-zero. This
          advice does not cause data migration and has no impact on the
          location of the data per se. Instead, it causes the data to always be
          mapped in the specified processor's page tables, as long as the
          location of the data permits a mapping to be established. If the data
          gets migrated for any reason, the mappings are updated accordingly.
          This advice is recommended in scenarios where data locality is not
          important, but avoiding faults is. Consider for example a system
          containing multiple GPUs with peer-to-peer access enabled, where the
          data located on one GPU is occasionally accessed by peer GPUs. In
          such scenarios, migrating data over to the other GPUs is not as
          important because the accesses are infrequent and the overhead of
          migration may be too high. But preventing faults can still help
          improve performance, and so having a mapping set up in advance is
          useful. Note that on CPU access of this data, the data may be
          migrated to host memory because the CPU typically cannot access
          device memory directly. Any GPU that had the
          :py:obj:`~.cudaMemAdviceSetAccessedBy` flag set for this data will
          now have its mapping updated to point to the page in host memory. If
          :py:obj:`~.cudaMemAdviseSetReadMostly` is also set on this memory
          region or any subset of it, then the policies associated with that
          advice will override the policies of this advice. Additionally, if
          the preferred location of this memory region or any subset of it is
          also `device`, then the policies associated with
          :py:obj:`~.cudaMemAdviseSetPreferredLocation` will override the
          policies of this advice. If the memory region refers to valid system-
          allocated pageable memory, then `device` must have a non-zero value
          for the device attribute :py:obj:`~.cudaDevAttrPageableMemoryAccess`.
          Additionally, if `device` has a non-zero value for the device
          attribute
          :py:obj:`~.cudaDevAttrPageableMemoryAccessUsesHostPageTables`, then
          this call has no effect.

        - :py:obj:`~.cudaMemAdviseUnsetAccessedBy`: Undoes the effect of
          :py:obj:`~.cudaMemAdviseSetAccessedBy`. Any mappings to the data from
          `device` may be removed at any time causing accesses to result in
          non-fatal page faults. If the memory region refers to valid system-
          allocated pageable memory, then `device` must have a non-zero value
          for the device attribute :py:obj:`~.cudaDevAttrPageableMemoryAccess`.
          Additionally, if `device` has a non-zero value for the device
          attribute
          :py:obj:`~.cudaDevAttrPageableMemoryAccessUsesHostPageTables`, then
          this call has no effect.

        Parameters
        ----------
        devPtr : Any
            Pointer to memory to set the advice for
        count : size_t
            Size in bytes of the memory range
        advice : :py:obj:`~.cudaMemoryAdvise`
            Advice to be applied for the specified memory range
        device : int
            Device to apply the advice for

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidDevice`

        See Also
        --------
        :py:obj:`~.cudaMemcpy`, :py:obj:`~.cudaMemcpyPeer`, :py:obj:`~.cudaMemcpyAsync`, :py:obj:`~.cudaMemcpy3DPeerAsync`, :py:obj:`~.cudaMemPrefetchAsync`, :py:obj:`~.cuMemAdvise`
    """


def cudaMemAdvise_v2(devPtr, count, advice: 'cudaMemoryAdvise', location: 'cudaMemLocation'):
    """
    cudaMemAdvise_v2(devPtr, size_t count, advice: cudaMemoryAdvise, cudaMemLocation location: cudaMemLocation)
     Advise about the usage of a given memory range.

        Advise the Unified Memory subsystem about the usage pattern for the
        memory range starting at `devPtr` with a size of `count` bytes. The
        start address and end address of the memory range will be rounded down
        and rounded up respectively to be aligned to CPU page size before the
        advice is applied. The memory range must refer to managed memory
        allocated via :py:obj:`~.cudaMallocManaged` or declared via managed
        variables. The memory range could also refer to system-allocated
        pageable memory provided it represents a valid, host-accessible region
        of memory and all additional constraints imposed by `advice` as
        outlined below are also satisfied. Specifying an invalid system-
        allocated pageable memory range results in an error being returned.

        The `advice` parameter can take the following values:

        - :py:obj:`~.cudaMemAdviseSetReadMostly`: This implies that the data is
          mostly going to be read from and only occasionally written to. Any
          read accesses from any processor to this region will create a read-
          only copy of at least the accessed pages in that processor's memory.
          Additionally, if :py:obj:`~.cudaMemPrefetchAsync` or
          :py:obj:`~.cudaMemPrefetchAsync_v2` is called on this region, it will
          create a read-only copy of the data on the destination processor. If
          the target location for :py:obj:`~.cudaMemPrefetchAsync_v2` is a host
          NUMA node and a read-only copy already exists on another host NUMA
          node, that copy will be migrated to the targeted host NUMA node. If
          any processor writes to this region, all copies of the corresponding
          page will be invalidated except for the one where the write occurred.
          If the writing processor is the CPU and the preferred location of the
          page is a host NUMA node, then the page will also be migrated to that
          host NUMA node. The `location` argument is ignored for this advice.
          Note that for a page to be read-duplicated, the accessing processor
          must either be the CPU or a GPU that has a non-zero value for the
          device attribute :py:obj:`~.cudaDevAttrConcurrentManagedAccess`.
          Also, if a context is created on a device that does not have the
          device attribute :py:obj:`~.cudaDevAttrConcurrentManagedAccess` set,
          then read-duplication will not occur until all such contexts are
          destroyed. If the memory region refers to valid system-allocated
          pageable memory, then the accessing device must have a non-zero value
          for the device attribute :py:obj:`~.cudaDevAttrPageableMemoryAccess`
          for a read-only copy to be created on that device. Note however that
          if the accessing device also has a non-zero value for the device
          attribute
          :py:obj:`~.cudaDevAttrPageableMemoryAccessUsesHostPageTables`, then
          setting this advice will not create a read-only copy when that device
          accesses this memory region.

        - :py:obj:`~.cudaMemAdviceUnsetReadMostly`: Undoes the effect of
          :py:obj:`~.cudaMemAdviseSetReadMostly` and also prevents the Unified
          Memory driver from attempting heuristic read-duplication on the
          memory range. Any read-duplicated copies of the data will be
          collapsed into a single copy. The location for the collapsed copy
          will be the preferred location if the page has a preferred location
          and one of the read-duplicated copies was resident at that location.
          Otherwise, the location chosen is arbitrary. Note: The `location`
          argument is ignored for this advice.

        - :py:obj:`~.cudaMemAdviseSetPreferredLocation`: This advice sets the
          preferred location for the data to be the memory belonging to
          `location`. When :py:obj:`~.cudaMemLocation.type` is
          :py:obj:`~.cudaMemLocationTypeHost`, :py:obj:`~.cudaMemLocation.id`
          is ignored and the preferred location is set to be host memory. To
          set the preferred location to a specific host NUMA node, applications
          must set :py:obj:`~.cudaMemLocation.type` to
          :py:obj:`~.cudaMemLocationTypeHostNuma` and
          :py:obj:`~.cudaMemLocation.id` must specify the NUMA ID of the host
          NUMA node. If :py:obj:`~.cudaMemLocation.type` is set to
          :py:obj:`~.cudaMemLocationTypeHostNumaCurrent`,
          :py:obj:`~.cudaMemLocation.id` will be ignored and the host NUMA node
          closest to the calling thread's CPU will be used as the preferred
          location. If :py:obj:`~.cudaMemLocation.type` is a
          :py:obj:`~.cudaMemLocationTypeDevice`, then
          :py:obj:`~.cudaMemLocation.id` must be a valid device ordinal and the
          device must have a non-zero value for the device attribute
          :py:obj:`~.cudaDevAttrConcurrentManagedAccess`. Setting the preferred
          location does not cause data to migrate to that location immediately.
          Instead, it guides the migration policy when a fault occurs on that
          memory region. If the data is already in its preferred location and
          the faulting processor can establish a mapping without requiring the
          data to be migrated, then data migration will be avoided. On the
          other hand, if the data is not in its preferred location or if a
          direct mapping cannot be established, then it will be migrated to the
          processor accessing it. It is important to note that setting the
          preferred location does not prevent data prefetching done using
          :py:obj:`~.cudaMemPrefetchAsync`. Having a preferred location can
          override the page thrash detection and resolution logic in the
          Unified Memory driver. Normally, if a page is detected to be
          constantly thrashing between for example host and device memory, the
          page may eventually be pinned to host memory by the Unified Memory
          driver. But if the preferred location is set as device memory, then
          the page will continue to thrash indefinitely. If
          :py:obj:`~.cudaMemAdviseSetReadMostly` is also set on this memory
          region or any subset of it, then the policies associated with that
          advice will override the policies of this advice, unless read
          accesses from `location` will not result in a read-only copy being
          created on that procesor as outlined in description for the advice
          :py:obj:`~.cudaMemAdviseSetReadMostly`. If the memory region refers
          to valid system-allocated pageable memory, and
          :py:obj:`~.cudaMemLocation.type` is
          :py:obj:`~.cudaMemLocationTypeDevice` then
          :py:obj:`~.cudaMemLocation.id` must be a valid device that has a non-
          zero alue for the device attribute
          :py:obj:`~.cudaDevAttrPageableMemoryAccess`.

        - :py:obj:`~.cudaMemAdviseUnsetPreferredLocation`: Undoes the effect of
          :py:obj:`~.cudaMemAdviseSetPreferredLocation` and changes the
          preferred location to none. The `location` argument is ignored for
          this advice.

        - :py:obj:`~.cudaMemAdviseSetAccessedBy`: This advice implies that the
          data will be accessed by processor `location`. The
          :py:obj:`~.cudaMemLocation.type` must be either
          :py:obj:`~.cudaMemLocationTypeDevice` with
          :py:obj:`~.cudaMemLocation.id` representing a valid device ordinal or
          :py:obj:`~.cudaMemLocationTypeHost` and
          :py:obj:`~.cudaMemLocation.id` will be ignored. All other location
          types are invalid. If :py:obj:`~.cudaMemLocation.id` is a GPU, then
          the device attribute :py:obj:`~.cudaDevAttrConcurrentManagedAccess`
          must be non-zero. This advice does not cause data migration and has
          no impact on the location of the data per se. Instead, it causes the
          data to always be mapped in the specified processor's page tables, as
          long as the location of the data permits a mapping to be established.
          If the data gets migrated for any reason, the mappings are updated
          accordingly. This advice is recommended in scenarios where data
          locality is not important, but avoiding faults is. Consider for
          example a system containing multiple GPUs with peer-to-peer access
          enabled, where the data located on one GPU is occasionally accessed
          by peer GPUs. In such scenarios, migrating data over to the other
          GPUs is not as important because the accesses are infrequent and the
          overhead of migration may be too high. But preventing faults can
          still help improve performance, and so having a mapping set up in
          advance is useful. Note that on CPU access of this data, the data may
          be migrated to host memory because the CPU typically cannot access
          device memory directly. Any GPU that had the
          :py:obj:`~.cudaMemAdviseSetAccessedBy` flag set for this data will
          now have its mapping updated to point to the page in host memory. If
          :py:obj:`~.cudaMemAdviseSetReadMostly` is also set on this memory
          region or any subset of it, then the policies associated with that
          advice will override the policies of this advice. Additionally, if
          the preferred location of this memory region or any subset of it is
          also `location`, then the policies associated with
          :py:obj:`~.CU_MEM_ADVISE_SET_PREFERRED_LOCATION` will override the
          policies of this advice. If the memory region refers to valid system-
          allocated pageable memory, and :py:obj:`~.cudaMemLocation.type` is
          :py:obj:`~.cudaMemLocationTypeDevice` then device in
          :py:obj:`~.cudaMemLocation.id` must have a non-zero value for the
          device attribute :py:obj:`~.cudaDevAttrPageableMemoryAccess`.
          Additionally, if :py:obj:`~.cudaMemLocation.id` has a non-zero value
          for the device attribute
          :py:obj:`~.cudaDevAttrPageableMemoryAccessUsesHostPageTables`, then
          this call has no effect.

        - :py:obj:`~.CU_MEM_ADVISE_UNSET_ACCESSED_BY`: Undoes the effect of
          :py:obj:`~.cudaMemAdviseSetAccessedBy`. Any mappings to the data from
          `location` may be removed at any time causing accesses to result in
          non-fatal page faults. If the memory region refers to valid system-
          allocated pageable memory, and :py:obj:`~.cudaMemLocation.type` is
          :py:obj:`~.cudaMemLocationTypeDevice` then device in
          :py:obj:`~.cudaMemLocation.id` must have a non-zero value for the
          device attribute :py:obj:`~.cudaDevAttrPageableMemoryAccess`.
          Additionally, if :py:obj:`~.cudaMemLocation.id` has a non-zero value
          for the device attribute
          :py:obj:`~.cudaDevAttrPageableMemoryAccessUsesHostPageTables`, then
          this call has no effect.

        Parameters
        ----------
        devPtr : Any
            Pointer to memory to set the advice for
        count : size_t
            Size in bytes of the memory range
        advice : :py:obj:`~.cudaMemoryAdvise`
            Advice to be applied for the specified memory range
        location : :py:obj:`~.cudaMemLocation`
            location to apply the advice for

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidDevice`

        See Also
        --------
        :py:obj:`~.cudaMemcpy`, :py:obj:`~.cudaMemcpyPeer`, :py:obj:`~.cudaMemcpyAsync`, :py:obj:`~.cudaMemcpy3DPeerAsync`, :py:obj:`~.cudaMemPrefetchAsync`, :py:obj:`~.cuMemAdvise`, :py:obj:`~.cuMemAdvise_v2`
    """

cudaMemAttachGlobal: int
cudaMemAttachHost: int
cudaMemAttachSingle: int

def cudaMemGetInfo():
    """
    cudaMemGetInfo()
     Gets free and total device memory.

        Returns in `*total` the total amount of memory available to the the
        current context. Returns in `*free` the amount of memory on the device
        that is free according to the OS. CUDA is not guaranteed to be able to
        allocate all of the memory that the OS reports as free. In a multi-
        tenet situation, free estimate returned is prone to race condition
        where a new allocation/free done by a different process or a different
        thread in the same process between the time when free memory was
        estimated and reported, will result in deviation in free value reported
        and actual free memory.

        The integrated GPU on Tegra shares memory with CPU and other component
        of the SoC. The free and total values returned by the API excludes the
        SWAP memory space maintained by the OS on some platforms. The OS may
        move some of the memory pages into swap area as the GPU or CPU allocate
        or access memory. See Tegra app note on how to calculate total and free
        memory on Tegra.

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorLaunchFailure`
        free : int
            Returned free memory in bytes
        total : int
            Returned total memory in bytes

        See Also
        --------
        :py:obj:`~.cuMemGetInfo`
    """


def cudaMemPoolCreate(poolProps: 'Optional[cudaMemPoolProps]'):
    """
    cudaMemPoolCreate(cudaMemPoolProps poolProps: Optional[cudaMemPoolProps])
     Creates a memory pool.

        Creates a CUDA memory pool and returns the handle in `pool`. The
        `poolProps` determines the properties of the pool such as the backing
        device and IPC capabilities.

        To create a memory pool targeting a specific host NUMA node,
        applications must set
        :py:obj:`~.cudaMemPoolProps`::cudaMemLocation::type to
        :py:obj:`~.cudaMemLocationTypeHostNuma` and
        :py:obj:`~.cudaMemPoolProps`::cudaMemLocation::id must specify the NUMA
        ID of the host memory node. Specifying
        :py:obj:`~.cudaMemLocationTypeHostNumaCurrent` or
        :py:obj:`~.cudaMemLocationTypeHost` as the
        :py:obj:`~.cudaMemPoolProps`::cudaMemLocation::type will result in
        :py:obj:`~.cudaErrorInvalidValue`. By default, the pool's memory will
        be accessible from the device it is allocated on. In the case of pools
        created with :py:obj:`~.cudaMemLocationTypeHostNuma`, their default
        accessibility will be from the host CPU. Applications can control the
        maximum size of the pool by specifying a non-zero value for
        :py:obj:`~.cudaMemPoolProps.maxSize`. If set to 0, the maximum size of
        the pool will default to a system dependent value.

        Applications that intend to use :py:obj:`~.CU_MEM_HANDLE_TYPE_FABRIC`
        based memory sharing must ensure: (1) `nvidia-caps-imex-channels`
        character device is created by the driver and is listed under
        /proc/devices (2) have at least one IMEX channel file accessible by the
        user launching the application.

        When exporter and importer CUDA processes have been granted access to
        the same IMEX channel, they can securely share memory.

        The IMEX channel security model works on a per user basis. Which means
        all processes under a user can share memory if the user has access to a
        valid IMEX channel. When multi-user isolation is desired, a separate
        IMEX channel is required for each user.

        These channel files exist in /dev/nvidia-caps-imex-channels/channel*
        and can be created using standard OS native calls like mknod on Linux.
        For example: To create channel0 with the major number from
        /proc/devices users can execute the following command: `mknod
        /dev/nvidia-caps-imex-channels/channel0 c <major number> 0`

        Parameters
        ----------
        poolProps : :py:obj:`~.cudaMemPoolProps`
            None

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorNotSupported`
        memPool : :py:obj:`~.cudaMemPool_t`
            None

        See Also
        --------
        :py:obj:`~.cuMemPoolCreate`, :py:obj:`~.cudaDeviceSetMemPool`, :py:obj:`~.cudaMallocFromPoolAsync`, :py:obj:`~.cudaMemPoolExportToShareableHandle`, :py:obj:`~.cudaDeviceGetDefaultMemPool`, :py:obj:`~.cudaDeviceGetMemPool`

        Notes
        -----
        Specifying cudaMemHandleTypeNone creates a memory pool that will not support IPC.
    """

cudaMemPoolCreateUsageHwDecompress: int

def cudaMemPoolDestroy(memPool):
    """
    cudaMemPoolDestroy(memPool)
     Destroys the specified memory pool.

        If any pointers obtained from this pool haven't been freed or the pool
        has free operations that haven't completed when
        :py:obj:`~.cudaMemPoolDestroy` is invoked, the function will return
        immediately and the resources associated with the pool will be released
        automatically once there are no more outstanding allocations.

        Destroying the current mempool of a device sets the default mempool of
        that device as the current mempool for that device.

        Parameters
        ----------
        memPool : :py:obj:`~.CUmemoryPool` or :py:obj:`~.cudaMemPool_t`
            None

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`

        See Also
        --------
        cuMemPoolDestroy, :py:obj:`~.cudaFreeAsync`, :py:obj:`~.cudaDeviceSetMemPool`, :py:obj:`~.cudaDeviceGetDefaultMemPool`, :py:obj:`~.cudaDeviceGetMemPool`, :py:obj:`~.cudaMemPoolCreate`

        Notes
        -----
        A device's default memory pool cannot be destroyed.
    """


def cudaMemPoolExportPointer(ptr):
    """
    cudaMemPoolExportPointer(ptr)
     Export data to share a memory pool allocation between processes.

        Constructs `shareData_out` for sharing a specific allocation from an
        already shared memory pool. The recipient process can import the
        allocation with the :py:obj:`~.cudaMemPoolImportPointer` api. The data
        is not a handle and may be shared through any IPC mechanism.

        Parameters
        ----------
        ptr : Any
            pointer to memory being exported

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorOutOfMemory`
        shareData_out : :py:obj:`~.cudaMemPoolPtrExportData`
            Returned export data

        See Also
        --------
        :py:obj:`~.cuMemPoolExportPointer`, :py:obj:`~.cudaMemPoolExportToShareableHandle`, :py:obj:`~.cudaMemPoolImportFromShareableHandle`, :py:obj:`~.cudaMemPoolImportPointer`
    """


def cudaMemPoolExportToShareableHandle(memPool, handleType: 'cudaMemAllocationHandleType', flags):
    """
    cudaMemPoolExportToShareableHandle(memPool, handleType: cudaMemAllocationHandleType, unsigned int flags)
     Exports a memory pool to the requested handle type.

        Given an IPC capable mempool, create an OS handle to share the pool
        with another process. A recipient process can convert the shareable
        handle into a mempool with
        :py:obj:`~.cudaMemPoolImportFromShareableHandle`. Individual pointers
        can then be shared with the :py:obj:`~.cudaMemPoolExportPointer` and
        :py:obj:`~.cudaMemPoolImportPointer` APIs. The implementation of what
        the shareable handle is and how it can be transferred is defined by the
        requested handle type.

        Parameters
        ----------
        pool : :py:obj:`~.CUmemoryPool` or :py:obj:`~.cudaMemPool_t`
            pool to export
        handleType : :py:obj:`~.cudaMemAllocationHandleType`
            the type of handle to create
        flags : unsigned int
            must be 0

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorOutOfMemory`
        handle_out : Any
            pointer to the location in which to store the requested handle

        See Also
        --------
        :py:obj:`~.cuMemPoolExportToShareableHandle`, :py:obj:`~.cudaMemPoolImportFromShareableHandle`, :py:obj:`~.cudaMemPoolExportPointer`, :py:obj:`~.cudaMemPoolImportPointer`

        Notes
        -----
        : To create an IPC capable mempool, create a mempool with a CUmemAllocationHandleType other than cudaMemHandleTypeNone.
    """


def cudaMemPoolGetAccess(memPool, location: 'Optional[cudaMemLocation]'):
    """
    cudaMemPoolGetAccess(memPool, cudaMemLocation location: Optional[cudaMemLocation])
     Returns the accessibility of a pool from a device.

        Returns the accessibility of the pool's memory from the specified
        location.

        Parameters
        ----------
        memPool : :py:obj:`~.CUmemoryPool` or :py:obj:`~.cudaMemPool_t`
            the pool being queried
        location : :py:obj:`~.cudaMemLocation`
            the location accessing the pool

        Returns
        -------
        cudaError_t

        flags : :py:obj:`~.cudaMemAccessFlags`
            the accessibility of the pool from the specified location

        See Also
        --------
        :py:obj:`~.cuMemPoolGetAccess`, :py:obj:`~.cudaMemPoolSetAccess`
    """


def cudaMemPoolGetAttribute(memPool, attr: 'cudaMemPoolAttr'):
    """
    cudaMemPoolGetAttribute(memPool, attr: cudaMemPoolAttr)
     Gets attributes of a memory pool.

        Supported attributes are:

        - :py:obj:`~.cudaMemPoolAttrReleaseThreshold`: (value type =
          cuuint64_t) Amount of reserved memory in bytes to hold onto before
          trying to release memory back to the OS. When more than the release
          threshold bytes of memory are held by the memory pool, the allocator
          will try to release memory back to the OS on the next call to stream,
          event or context synchronize. (default 0)

        - :py:obj:`~.cudaMemPoolReuseFollowEventDependencies`: (value type =
          int) Allow :py:obj:`~.cudaMallocAsync` to use memory asynchronously
          freed in another stream as long as a stream ordering dependency of
          the allocating stream on the free action exists. Cuda events and null
          stream interactions can create the required stream ordered
          dependencies. (default enabled)

        - :py:obj:`~.cudaMemPoolReuseAllowOpportunistic`: (value type = int)
          Allow reuse of already completed frees when there is no dependency
          between the free and allocation. (default enabled)

        - :py:obj:`~.cudaMemPoolReuseAllowInternalDependencies`: (value type =
          int) Allow :py:obj:`~.cudaMallocAsync` to insert new stream
          dependencies in order to establish the stream ordering required to
          reuse a piece of memory released by :py:obj:`~.cudaFreeAsync`
          (default enabled).

        - :py:obj:`~.cudaMemPoolAttrReservedMemCurrent`: (value type =
          cuuint64_t) Amount of backing memory currently allocated for the
          mempool.

        - :py:obj:`~.cudaMemPoolAttrReservedMemHigh`: (value type = cuuint64_t)
          High watermark of backing memory allocated for the mempool since the
          last time it was reset.

        - :py:obj:`~.cudaMemPoolAttrUsedMemCurrent`: (value type = cuuint64_t)
          Amount of memory from the pool that is currently in use by the
          application.

        - :py:obj:`~.cudaMemPoolAttrUsedMemHigh`: (value type = cuuint64_t)
          High watermark of the amount of memory from the pool that was in use
          by the application since the last time it was reset.

        Parameters
        ----------
        pool : :py:obj:`~.CUmemoryPool` or :py:obj:`~.cudaMemPool_t`
            The memory pool to get attributes of
        attr : :py:obj:`~.cudaMemPoolAttr`
            The attribute to get

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`
        value : Any
            Retrieved value

        See Also
        --------
        :py:obj:`~.cuMemPoolGetAttribute`, :py:obj:`~.cudaMallocAsync`, :py:obj:`~.cudaFreeAsync`, :py:obj:`~.cudaDeviceGetDefaultMemPool`, :py:obj:`~.cudaDeviceGetMemPool`, :py:obj:`~.cudaMemPoolCreate`
    """


def cudaMemPoolImportFromShareableHandle(shareableHandle, handleType: 'cudaMemAllocationHandleType', flags):
    """
    cudaMemPoolImportFromShareableHandle(shareableHandle, handleType: cudaMemAllocationHandleType, unsigned int flags)
     imports a memory pool from a shared handle.

        Specific allocations can be imported from the imported pool with
        :py:obj:`~.cudaMemPoolImportPointer`.

        Parameters
        ----------
        handle : Any
            OS handle of the pool to open
        handleType : :py:obj:`~.cudaMemAllocationHandleType`
            The type of handle being imported
        flags : unsigned int
            must be 0

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorOutOfMemory`
        pool_out : :py:obj:`~.cudaMemPool_t`
            Returned memory pool

        See Also
        --------
        :py:obj:`~.cuMemPoolImportFromShareableHandle`, :py:obj:`~.cudaMemPoolExportToShareableHandle`, :py:obj:`~.cudaMemPoolExportPointer`, :py:obj:`~.cudaMemPoolImportPointer`

        Notes
        -----
        Imported memory pools do not support creating new allocations. As such imported memory pools may not be used in :py:obj:`~.cudaDeviceSetMemPool` or :py:obj:`~.cudaMallocFromPoolAsync` calls.
    """


def cudaMemPoolImportPointer(memPool, exportData: 'Optional[cudaMemPoolPtrExportData]'):
    """
    cudaMemPoolImportPointer(memPool, cudaMemPoolPtrExportData exportData: Optional[cudaMemPoolPtrExportData])
     Import a memory pool allocation from another process.

        Returns in `ptr_out` a pointer to the imported memory. The imported
        memory must not be accessed before the allocation operation completes
        in the exporting process. The imported memory must be freed from all
        importing processes before being freed in the exporting process. The
        pointer may be freed with cudaFree or cudaFreeAsync. If
        :py:obj:`~.cudaFreeAsync` is used, the free must be completed on the
        importing process before the free operation on the exporting process.

        Parameters
        ----------
        pool : :py:obj:`~.CUmemoryPool` or :py:obj:`~.cudaMemPool_t`
            pool from which to import
        shareData : :py:obj:`~.cudaMemPoolPtrExportData`
            data specifying the memory to import

        Returns
        -------
        cudaError_t
            :py:obj:`~.CUDA_SUCCESS`, :py:obj:`~.CUDA_ERROR_INVALID_VALUE`, :py:obj:`~.CUDA_ERROR_NOT_INITIALIZED`, :py:obj:`~.CUDA_ERROR_OUT_OF_MEMORY`
        ptr_out : Any
            pointer to imported memory

        See Also
        --------
        :py:obj:`~.cuMemPoolImportPointer`, :py:obj:`~.cudaMemPoolExportToShareableHandle`, :py:obj:`~.cudaMemPoolImportFromShareableHandle`, :py:obj:`~.cudaMemPoolExportPointer`

        Notes
        -----
        The :py:obj:`~.cudaFreeAsync` api may be used in the exporting process before the :py:obj:`~.cudaFreeAsync` operation completes in its stream as long as the :py:obj:`~.cudaFreeAsync` in the exporting process specifies a stream with a stream dependency on the importing process's :py:obj:`~.cudaFreeAsync`.
    """


def cudaMemPoolSetAccess(memPool, descList: 'Optional[Tuple[cudaMemAccessDesc] | List[cudaMemAccessDesc]]', count):
    """
    cudaMemPoolSetAccess(memPool, descList: Optional[Tuple[cudaMemAccessDesc] | List[cudaMemAccessDesc]], size_t count)
     Controls visibility of pools between devices.

        Parameters
        ----------
        pool : :py:obj:`~.CUmemoryPool` or :py:obj:`~.cudaMemPool_t`
            The pool being modified
        map : List[:py:obj:`~.cudaMemAccessDesc`]
            Array of access descriptors. Each descriptor instructs the access
            to enable for a single gpu
        count : size_t
            Number of descriptors in the map array.

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`

        See Also
        --------
        :py:obj:`~.cuMemPoolSetAccess`, :py:obj:`~.cudaMemPoolGetAccess`, :py:obj:`~.cudaMallocAsync`, :py:obj:`~.cudaFreeAsync`
    """


def cudaMemPoolSetAttribute(memPool, attr: 'cudaMemPoolAttr', value):
    """
    cudaMemPoolSetAttribute(memPool, attr: cudaMemPoolAttr, value)
     Sets attributes of a memory pool.

        Supported attributes are:

        - :py:obj:`~.cudaMemPoolAttrReleaseThreshold`: (value type =
          cuuint64_t) Amount of reserved memory in bytes to hold onto before
          trying to release memory back to the OS. When more than the release
          threshold bytes of memory are held by the memory pool, the allocator
          will try to release memory back to the OS on the next call to stream,
          event or context synchronize. (default 0)

        - :py:obj:`~.cudaMemPoolReuseFollowEventDependencies`: (value type =
          int) Allow :py:obj:`~.cudaMallocAsync` to use memory asynchronously
          freed in another stream as long as a stream ordering dependency of
          the allocating stream on the free action exists. Cuda events and null
          stream interactions can create the required stream ordered
          dependencies. (default enabled)

        - :py:obj:`~.cudaMemPoolReuseAllowOpportunistic`: (value type = int)
          Allow reuse of already completed frees when there is no dependency
          between the free and allocation. (default enabled)

        - :py:obj:`~.cudaMemPoolReuseAllowInternalDependencies`: (value type =
          int) Allow :py:obj:`~.cudaMallocAsync` to insert new stream
          dependencies in order to establish the stream ordering required to
          reuse a piece of memory released by :py:obj:`~.cudaFreeAsync`
          (default enabled).

        - :py:obj:`~.cudaMemPoolAttrReservedMemHigh`: (value type = cuuint64_t)
          Reset the high watermark that tracks the amount of backing memory
          that was allocated for the memory pool. It is illegal to set this
          attribute to a non-zero value.

        - :py:obj:`~.cudaMemPoolAttrUsedMemHigh`: (value type = cuuint64_t)
          Reset the high watermark that tracks the amount of used memory that
          was allocated for the memory pool. It is illegal to set this
          attribute to a non-zero value.

        Parameters
        ----------
        pool : :py:obj:`~.CUmemoryPool` or :py:obj:`~.cudaMemPool_t`
            The memory pool to modify
        attr : :py:obj:`~.cudaMemPoolAttr`
            The attribute to modify
        value : Any
            Pointer to the value to assign

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`

        See Also
        --------
        :py:obj:`~.cuMemPoolSetAttribute`, :py:obj:`~.cudaMallocAsync`, :py:obj:`~.cudaFreeAsync`, :py:obj:`~.cudaDeviceGetDefaultMemPool`, :py:obj:`~.cudaDeviceGetMemPool`, :py:obj:`~.cudaMemPoolCreate`
    """


def cudaMemPoolTrimTo(memPool, minBytesToKeep):
    """
    cudaMemPoolTrimTo(memPool, size_t minBytesToKeep)
     Tries to release memory back to the OS.

        Releases memory back to the OS until the pool contains fewer than
        minBytesToKeep reserved bytes, or there is no more memory that the
        allocator can safely release. The allocator cannot release OS
        allocations that back outstanding asynchronous allocations. The OS
        allocations may happen at different granularity from the user
        allocations.

        Parameters
        ----------
        pool : :py:obj:`~.CUmemoryPool` or :py:obj:`~.cudaMemPool_t`
            The memory pool to trim
        minBytesToKeep : size_t
            If the pool has less than minBytesToKeep reserved, the TrimTo
            operation is a no-op. Otherwise the pool will be guaranteed to have
            at least minBytesToKeep bytes reserved after the operation.

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`

        See Also
        --------
        :py:obj:`~.cuMemPoolTrimTo`, :py:obj:`~.cudaMallocAsync`, :py:obj:`~.cudaFreeAsync`, :py:obj:`~.cudaDeviceGetDefaultMemPool`, :py:obj:`~.cudaDeviceGetMemPool`, :py:obj:`~.cudaMemPoolCreate`

        Notes
        -----
        : Allocations that have not been freed count as outstanding.

        : Allocations that have been asynchronously freed but whose completion has not been observed on the host (eg. by a synchronize) can count as outstanding.
    """


def cudaMemPrefetchAsync(devPtr, count, dstDevice, stream):
    """
    cudaMemPrefetchAsync(devPtr, size_t count, int dstDevice, stream)
     Prefetches memory to the specified destination device.

        Prefetches memory to the specified destination device. `devPtr` is the
        base device pointer of the memory to be prefetched and `dstDevice` is
        the destination device. `count` specifies the number of bytes to copy.
        `stream` is the stream in which the operation is enqueued. The memory
        range must refer to managed memory allocated via
        :py:obj:`~.cudaMallocManaged` or declared via managed variables, or it
        may also refer to system-allocated memory on systems with non-zero
        cudaDevAttrPageableMemoryAccess.

        Passing in cudaCpuDeviceId for `dstDevice` will prefetch the data to
        host memory. If `dstDevice` is a GPU, then the device attribute
        :py:obj:`~.cudaDevAttrConcurrentManagedAccess` must be non-zero.
        Additionally, `stream` must be associated with a device that has a non-
        zero value for the device attribute
        :py:obj:`~.cudaDevAttrConcurrentManagedAccess`.

        The start address and end address of the memory range will be rounded
        down and rounded up respectively to be aligned to CPU page size before
        the prefetch operation is enqueued in the stream.

        If no physical memory has been allocated for this region, then this
        memory region will be populated and mapped on the destination device.
        If there's insufficient memory to prefetch the desired region, the
        Unified Memory driver may evict pages from other
        :py:obj:`~.cudaMallocManaged` allocations to host memory in order to
        make room. Device memory allocated using :py:obj:`~.cudaMalloc` or
        :py:obj:`~.cudaMallocArray` will not be evicted.

        By default, any mappings to the previous location of the migrated pages
        are removed and mappings for the new location are only setup on
        `dstDevice`. The exact behavior however also depends on the settings
        applied to this memory range via :py:obj:`~.cudaMemAdvise` as described
        below:

        If :py:obj:`~.cudaMemAdviseSetReadMostly` was set on any subset of this
        memory range, then that subset will create a read-only copy of the
        pages on `dstDevice`.

        If :py:obj:`~.cudaMemAdviseSetPreferredLocation` was called on any
        subset of this memory range, then the pages will be migrated to
        `dstDevice` even if `dstDevice` is not the preferred location of any
        pages in the memory range.

        If :py:obj:`~.cudaMemAdviseSetAccessedBy` was called on any subset of
        this memory range, then mappings to those pages from all the
        appropriate processors are updated to refer to the new location if
        establishing such a mapping is possible. Otherwise, those mappings are
        cleared.

        Note that this API is not required for functionality and only serves to
        improve performance by allowing the application to migrate data to a
        suitable location before it is accessed. Memory accesses to this range
        are always coherent and are allowed even when the data is actively
        being migrated.

        Note that this function is asynchronous with respect to the host and
        all work on other devices.

        Parameters
        ----------
        devPtr : Any
            Pointer to be prefetched
        count : size_t
            Size in bytes
        dstDevice : int
            Destination device to prefetch to
        stream : :py:obj:`~.CUstream` or :py:obj:`~.cudaStream_t`
            Stream to enqueue prefetch operation

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidDevice`

        See Also
        --------
        :py:obj:`~.cudaMemcpy`, :py:obj:`~.cudaMemcpyPeer`, :py:obj:`~.cudaMemcpyAsync`, :py:obj:`~.cudaMemcpy3DPeerAsync`, :py:obj:`~.cudaMemAdvise`, :py:obj:`~.cudaMemAdvise_v2` :py:obj:`~.cuMemPrefetchAsync`
    """


def cudaMemPrefetchAsync_v2(devPtr, count, location: 'cudaMemLocation', flags, stream):
    """
    cudaMemPrefetchAsync_v2(devPtr, size_t count, cudaMemLocation location: cudaMemLocation, unsigned int flags, stream)
     Prefetches memory to the specified destination location.

        Prefetches memory to the specified destination location. `devPtr` is
        the base device pointer of the memory to be prefetched and `location`
        specifies the destination location. `count` specifies the number of
        bytes to copy. `stream` is the stream in which the operation is
        enqueued. The memory range must refer to managed memory allocated via
        :py:obj:`~.cudaMallocManaged` or declared via managed variables, or it
        may also refer to system-allocated memory on systems with non-zero
        cudaDevAttrPageableMemoryAccess.

        Specifying :py:obj:`~.cudaMemLocationTypeDevice` for
        :py:obj:`~.cudaMemLocation.type` will prefetch memory to GPU specified
        by device ordinal :py:obj:`~.cudaMemLocation.id` which must have non-
        zero value for the device attribute
        :py:obj:`~.concurrentManagedAccess`. Additionally, `stream` must be
        associated with a device that has a non-zero value for the device
        attribute :py:obj:`~.concurrentManagedAccess`. Specifying
        :py:obj:`~.cudaMemLocationTypeHost` as :py:obj:`~.cudaMemLocation.type`
        will prefetch data to host memory. Applications can request prefetching
        memory to a specific host NUMA node by specifying
        :py:obj:`~.cudaMemLocationTypeHostNuma` for
        :py:obj:`~.cudaMemLocation.type` and a valid host NUMA node id in
        :py:obj:`~.cudaMemLocation.id` Users can also request prefetching
        memory to the host NUMA node closest to the current thread's CPU by
        specifying :py:obj:`~.cudaMemLocationTypeHostNumaCurrent` for
        :py:obj:`~.cudaMemLocation.type`. Note when
        :py:obj:`~.cudaMemLocation.type` is etiher
        :py:obj:`~.cudaMemLocationTypeHost` OR
        :py:obj:`~.cudaMemLocationTypeHostNumaCurrent`,
        :py:obj:`~.cudaMemLocation.id` will be ignored.

        The start address and end address of the memory range will be rounded
        down and rounded up respectively to be aligned to CPU page size before
        the prefetch operation is enqueued in the stream.

        If no physical memory has been allocated for this region, then this
        memory region will be populated and mapped on the destination device.
        If there's insufficient memory to prefetch the desired region, the
        Unified Memory driver may evict pages from other
        :py:obj:`~.cudaMallocManaged` allocations to host memory in order to
        make room. Device memory allocated using :py:obj:`~.cudaMalloc` or
        :py:obj:`~.cudaMallocArray` will not be evicted.

        By default, any mappings to the previous location of the migrated pages
        are removed and mappings for the new location are only setup on the
        destination location. The exact behavior however also depends on the
        settings applied to this memory range via :py:obj:`~.cuMemAdvise` as
        described below:

        If :py:obj:`~.cudaMemAdviseSetReadMostly` was set on any subset of this
        memory range, then that subset will create a read-only copy of the
        pages on destination location. If however the destination location is a
        host NUMA node, then any pages of that subset that are already in
        another host NUMA node will be transferred to the destination.

        If :py:obj:`~.cudaMemAdviseSetPreferredLocation` was called on any
        subset of this memory range, then the pages will be migrated to
        `location` even if `location` is not the preferred location of any
        pages in the memory range.

        If :py:obj:`~.cudaMemAdviseSetAccessedBy` was called on any subset of
        this memory range, then mappings to those pages from all the
        appropriate processors are updated to refer to the new location if
        establishing such a mapping is possible. Otherwise, those mappings are
        cleared.

        Note that this API is not required for functionality and only serves to
        improve performance by allowing the application to migrate data to a
        suitable location before it is accessed. Memory accesses to this range
        are always coherent and are allowed even when the data is actively
        being migrated.

        Note that this function is asynchronous with respect to the host and
        all work on other devices.

        Parameters
        ----------
        devPtr : Any
            Pointer to be prefetched
        count : size_t
            Size in bytes
        location : :py:obj:`~.cudaMemLocation`
            location to prefetch to
        flags : unsigned int
            flags for future use, must be zero now.
        stream : :py:obj:`~.CUstream` or :py:obj:`~.cudaStream_t`
            Stream to enqueue prefetch operation

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidDevice`

        See Also
        --------
        :py:obj:`~.cudaMemcpy`, :py:obj:`~.cudaMemcpyPeer`, :py:obj:`~.cudaMemcpyAsync`, :py:obj:`~.cudaMemcpy3DPeerAsync`, :py:obj:`~.cudaMemAdvise`, :py:obj:`~.cudaMemAdvise_v2` :py:obj:`~.cuMemPrefetchAsync`
    """


def cudaMemRangeGetAttribute(dataSize, attribute: 'cudaMemRangeAttribute', devPtr, count):
    """
    cudaMemRangeGetAttribute(size_t dataSize, attribute: cudaMemRangeAttribute, devPtr, size_t count)
     Query an attribute of a given memory range.

        Query an attribute about the memory range starting at `devPtr` with a
        size of `count` bytes. The memory range must refer to managed memory
        allocated via :py:obj:`~.cudaMallocManaged` or declared via managed
        variables.

        The `attribute` parameter can take the following values:

        - :py:obj:`~.cudaMemRangeAttributeReadMostly`: If this attribute is
          specified, `data` will be interpreted as a 32-bit integer, and
          `dataSize` must be 4. The result returned will be 1 if all pages in
          the given memory range have read-duplication enabled, or 0 otherwise.

        - :py:obj:`~.cudaMemRangeAttributePreferredLocation`: If this attribute
          is specified, `data` will be interpreted as a 32-bit integer, and
          `dataSize` must be 4. The result returned will be a GPU device id if
          all pages in the memory range have that GPU as their preferred
          location, or it will be cudaCpuDeviceId if all pages in the memory
          range have the CPU as their preferred location, or it will be
          cudaInvalidDeviceId if either all the pages don't have the same
          preferred location or some of the pages don't have a preferred
          location at all. Note that the actual location of the pages in the
          memory range at the time of the query may be different from the
          preferred location.

        - :py:obj:`~.cudaMemRangeAttributeAccessedBy`: If this attribute is
          specified, `data` will be interpreted as an array of 32-bit integers,
          and `dataSize` must be a non-zero multiple of 4. The result returned
          will be a list of device ids that had
          :py:obj:`~.cudaMemAdviceSetAccessedBy` set for that entire memory
          range. If any device does not have that advice set for the entire
          memory range, that device will not be included. If `data` is larger
          than the number of devices that have that advice set for that memory
          range, cudaInvalidDeviceId will be returned in all the extra space
          provided. For ex., if `dataSize` is 12 (i.e. `data` has 3 elements)
          and only device 0 has the advice set, then the result returned will
          be { 0, cudaInvalidDeviceId, cudaInvalidDeviceId }. If `data` is
          smaller than the number of devices that have that advice set, then
          only as many devices will be returned as can fit in the array. There
          is no guarantee on which specific devices will be returned, however.

        - :py:obj:`~.cudaMemRangeAttributeLastPrefetchLocation`: If this
          attribute is specified, `data` will be interpreted as a 32-bit
          integer, and `dataSize` must be 4. The result returned will be the
          last location to which all pages in the memory range were prefetched
          explicitly via :py:obj:`~.cudaMemPrefetchAsync`. This will either be
          a GPU id or cudaCpuDeviceId depending on whether the last location
          for prefetch was a GPU or the CPU respectively. If any page in the
          memory range was never explicitly prefetched or if all pages were not
          prefetched to the same location, cudaInvalidDeviceId will be
          returned. Note that this simply returns the last location that the
          applicaton requested to prefetch the memory range to. It gives no
          indication as to whether the prefetch operation to that location has
          completed or even begun.

          - :py:obj:`~.cudaMemRangeAttributePreferredLocationType`: If this
            attribute is specified, `data` will be interpreted as a
            :py:obj:`~.cudaMemLocationType`, and `dataSize` must be
            sizeof(cudaMemLocationType). The :py:obj:`~.cudaMemLocationType`
            returned will be :py:obj:`~.cudaMemLocationTypeDevice` if all pages
            in the memory range have the same GPU as their preferred location,
            or :py:obj:`~.cudaMemLocationType` will be
            :py:obj:`~.cudaMemLocationTypeHost` if all pages in the memory
            range have the CPU as their preferred location, or or it will be
            :py:obj:`~.cudaMemLocationTypeHostNuma` if all the pages in the
            memory range have the same host NUMA node ID as their preferred
            location or it will be :py:obj:`~.cudaMemLocationTypeInvalid` if
            either all the pages don't have the same preferred location or some
            of the pages don't have a preferred location at all. Note that the
            actual location type of the pages in the memory range at the time
            of the query may be different from the preferred location type.

        - :py:obj:`~.cudaMemRangeAttributePreferredLocationId`: If this
        attribute is specified, `data` will be interpreted as a 32-bit integer,
        and `dataSize` must be 4. If the
        :py:obj:`~.cudaMemRangeAttributePreferredLocationType` query for the
        same address range returns :py:obj:`~.cudaMemLocationTypeDevice`, it
        will be a valid device ordinal or if it returns
        :py:obj:`~.cudaMemLocationTypeHostNuma`, it will be a valid host NUMA
        node ID or if it returns any other location type, the id should be
        ignored.

          - :py:obj:`~.cudaMemRangeAttributeLastPrefetchLocationType`: If this
            attribute is specified, `data` will be interpreted as a
            :py:obj:`~.cudaMemLocationType`, and `dataSize` must be
            sizeof(cudaMemLocationType). The result returned will be the last
            location type to which all pages in the memory range were
            prefetched explicitly via :py:obj:`~.cuMemPrefetchAsync`. The
            :py:obj:`~.cudaMemLocationType` returned will be
            :py:obj:`~.cudaMemLocationTypeDevice` if the last prefetch location
            was the GPU or :py:obj:`~.cudaMemLocationTypeHost` if it was the
            CPU or :py:obj:`~.cudaMemLocationTypeHostNuma` if the last prefetch
            location was a specific host NUMA node. If any page in the memory
            range was never explicitly prefetched or if all pages were not
            prefetched to the same location, :py:obj:`~.CUmemLocationType` will
            be :py:obj:`~.cudaMemLocationTypeInvalid`. Note that this simply
            returns the last location type that the application requested to
            prefetch the memory range to. It gives no indication as to whether
            the prefetch operation to that location has completed or even
            begun.

        - :py:obj:`~.cudaMemRangeAttributeLastPrefetchLocationId`: If this
        attribute is specified, `data` will be interpreted as a 32-bit integer,
        and `dataSize` must be 4. If the
        :py:obj:`~.cudaMemRangeAttributeLastPrefetchLocationType` query for the
        same address range returns :py:obj:`~.cudaMemLocationTypeDevice`, it
        will be a valid device ordinal or if it returns
        :py:obj:`~.cudaMemLocationTypeHostNuma`, it will be a valid host NUMA
        node ID or if it returns any other location type, the id should be
        ignored.

        Parameters
        ----------
        dataSize : size_t
            Array containing the size of data
        attribute : :py:obj:`~.cudaMemRangeAttribute`
            The attribute to query
        devPtr : Any
            Start of the range to query
        count : size_t
            Size of the range to query

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`
        data : Any
            A pointers to a memory location where the result of each attribute
            query will be written to.

        See Also
        --------
        :py:obj:`~.cudaMemRangeGetAttributes`, :py:obj:`~.cudaMemPrefetchAsync`, :py:obj:`~.cudaMemAdvise`, :py:obj:`~.cuMemRangeGetAttribute`
    """


def cudaMemRangeGetAttributes(dataSizes: 'Tuple[int] | List[int]', attributes: 'Optional[Tuple[cudaMemRangeAttribute] | List[cudaMemRangeAttribute]]', numAttributes, devPtr, count):
    """
    cudaMemRangeGetAttributes(dataSizes: Tuple[int] | List[int], attributes: Optional[Tuple[cudaMemRangeAttribute] | List[cudaMemRangeAttribute]], size_t numAttributes, devPtr, size_t count)
     Query attributes of a given memory range.

        Query attributes of the memory range starting at `devPtr` with a size
        of `count` bytes. The memory range must refer to managed memory
        allocated via :py:obj:`~.cudaMallocManaged` or declared via managed
        variables. The `attributes` array will be interpreted to have
        `numAttributes` entries. The `dataSizes` array will also be interpreted
        to have `numAttributes` entries. The results of the query will be
        stored in `data`.

        The list of supported attributes are given below. Please refer to
        :py:obj:`~.cudaMemRangeGetAttribute` for attribute descriptions and
        restrictions.

        - :py:obj:`~.cudaMemRangeAttributeReadMostly`

        - :py:obj:`~.cudaMemRangeAttributePreferredLocation`

        - :py:obj:`~.cudaMemRangeAttributeAccessedBy`

        - :py:obj:`~.cudaMemRangeAttributeLastPrefetchLocation`

        - :: cudaMemRangeAttributePreferredLocationType

        - :: cudaMemRangeAttributePreferredLocationId

        - :: cudaMemRangeAttributeLastPrefetchLocationType

        - :: cudaMemRangeAttributeLastPrefetchLocationId

        Parameters
        ----------
        dataSizes : List[int]
            Array containing the sizes of each result
        attributes : List[:py:obj:`~.cudaMemRangeAttribute`]
            An array of attributes to query (numAttributes and the number of
            attributes in this array should match)
        numAttributes : size_t
            Number of attributes to query
        devPtr : Any
            Start of the range to query
        count : size_t
            Size of the range to query

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`
        data : List[Any]
            A two-dimensional array containing pointers to memory locations
            where the result of each attribute query will be written to.

        See Also
        --------
        :py:obj:`~.cudaMemRangeGetAttribute`, :py:obj:`~.cudaMemAdvise`, :py:obj:`~.cudaMemPrefetchAsync`, :py:obj:`~.cuMemRangeGetAttributes`
    """


def cudaMemcpy(dst, src, count, kind: 'cudaMemcpyKind'):
    """
    cudaMemcpy(dst, src, size_t count, kind: cudaMemcpyKind)
     Copies data between host and device.

        Copies `count` bytes from the memory area pointed to by `src` to the
        memory area pointed to by `dst`, where `kind` specifies the direction
        of the copy, and must be one of :py:obj:`~.cudaMemcpyHostToHost`,
        :py:obj:`~.cudaMemcpyHostToDevice`, :py:obj:`~.cudaMemcpyDeviceToHost`,
        :py:obj:`~.cudaMemcpyDeviceToDevice`, or :py:obj:`~.cudaMemcpyDefault`.
        Passing :py:obj:`~.cudaMemcpyDefault` is recommended, in which case the
        type of transfer is inferred from the pointer values. However,
        :py:obj:`~.cudaMemcpyDefault` is only allowed on systems that support
        unified virtual addressing. Calling :py:obj:`~.cudaMemcpy()` with dst
        and src pointers that do not match the direction of the copy results in
        an undefined behavior.

    
    ote_sync

        Parameters
        ----------
        dst : Any
            Destination memory address
        src : Any
            Source memory address
        count : size_t
            Size in bytes to copy
        kind : :py:obj:`~.cudaMemcpyKind`
            Type of transfer

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidMemcpyDirection`

        See Also
        --------
        :py:obj:`~.cudaMemcpy2D`, :py:obj:`~.cudaMemcpy2DToArray`, :py:obj:`~.cudaMemcpy2DFromArray`, :py:obj:`~.cudaMemcpy2DArrayToArray`, :py:obj:`~.cudaMemcpyToSymbol`, :py:obj:`~.cudaMemcpyFromSymbol`, :py:obj:`~.cudaMemcpyAsync`, :py:obj:`~.cudaMemcpy2DAsync`, :py:obj:`~.cudaMemcpy2DToArrayAsync`, :py:obj:`~.cudaMemcpy2DFromArrayAsync`, :py:obj:`~.cudaMemcpyToSymbolAsync`, :py:obj:`~.cudaMemcpyFromSymbolAsync`, :py:obj:`~.cuMemcpyDtoH`, :py:obj:`~.cuMemcpyHtoD`, :py:obj:`~.cuMemcpyDtoD`, :py:obj:`~.cuMemcpy`
    """


def cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind: 'cudaMemcpyKind'):
    """
    cudaMemcpy2D(dst, size_t dpitch, src, size_t spitch, size_t width, size_t height, kind: cudaMemcpyKind)
     Copies data between host and device.

        Copies a matrix (`height` rows of `width` bytes each) from the memory
        area pointed to by `src` to the memory area pointed to by `dst`, where
        `kind` specifies the direction of the copy, and must be one of
        :py:obj:`~.cudaMemcpyHostToHost`, :py:obj:`~.cudaMemcpyHostToDevice`,
        :py:obj:`~.cudaMemcpyDeviceToHost`,
        :py:obj:`~.cudaMemcpyDeviceToDevice`, or :py:obj:`~.cudaMemcpyDefault`.
        Passing :py:obj:`~.cudaMemcpyDefault` is recommended, in which case the
        type of transfer is inferred from the pointer values. However,
        :py:obj:`~.cudaMemcpyDefault` is only allowed on systems that support
        unified virtual addressing. `dpitch` and `spitch` are the widths in
        memory in bytes of the 2D arrays pointed to by `dst` and `src`,
        including any padding added to the end of each row. The memory areas
        may not overlap. `width` must not exceed either `dpitch` or `spitch`.
        Calling :py:obj:`~.cudaMemcpy2D()` with `dst` and `src` pointers that
        do not match the direction of the copy results in an undefined
        behavior. :py:obj:`~.cudaMemcpy2D()` returns an error if `dpitch` or
        `spitch` exceeds the maximum allowed.

        Parameters
        ----------
        dst : Any
            Destination memory address
        dpitch : size_t
            Pitch of destination memory
        src : Any
            Source memory address
        spitch : size_t
            Pitch of source memory
        width : size_t
            Width of matrix transfer (columns in bytes)
        height : size_t
            Height of matrix transfer (rows)
        kind : :py:obj:`~.cudaMemcpyKind`
            Type of transfer

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidPitchValue`, :py:obj:`~.cudaErrorInvalidMemcpyDirection`

        See Also
        --------
        :py:obj:`~.cudaMemcpy`, :py:obj:`~.cudaMemcpy2DToArray`, :py:obj:`~.cudaMemcpy2DFromArray`, :py:obj:`~.cudaMemcpy2DArrayToArray`, :py:obj:`~.cudaMemcpyToSymbol`, :py:obj:`~.cudaMemcpyFromSymbol`, :py:obj:`~.cudaMemcpyAsync`, :py:obj:`~.cudaMemcpy2DAsync`, :py:obj:`~.cudaMemcpy2DToArrayAsync`, :py:obj:`~.cudaMemcpy2DFromArrayAsync`, :py:obj:`~.cudaMemcpyToSymbolAsync`, :py:obj:`~.cudaMemcpyFromSymbolAsync`, :py:obj:`~.cuMemcpy2D`, :py:obj:`~.cuMemcpy2DUnaligned`
    """


def cudaMemcpy2DArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, width, height, kind: 'cudaMemcpyKind'):
    """
    cudaMemcpy2DArrayToArray(dst, size_t wOffsetDst, size_t hOffsetDst, src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, kind: cudaMemcpyKind)
     Copies data between host and device.

        Copies a matrix (`height` rows of `width` bytes each) from the CUDA
        array `src` starting at `hOffsetSrc` rows and `wOffsetSrc` bytes from
        the upper left corner to the CUDA array `dst` starting at `hOffsetDst`
        rows and `wOffsetDst` bytes from the upper left corner, where `kind`
        specifies the direction of the copy, and must be one of
        :py:obj:`~.cudaMemcpyHostToHost`, :py:obj:`~.cudaMemcpyHostToDevice`,
        :py:obj:`~.cudaMemcpyDeviceToHost`,
        :py:obj:`~.cudaMemcpyDeviceToDevice`, or :py:obj:`~.cudaMemcpyDefault`.
        Passing :py:obj:`~.cudaMemcpyDefault` is recommended, in which case the
        type of transfer is inferred from the pointer values. However,
        :py:obj:`~.cudaMemcpyDefault` is only allowed on systems that support
        unified virtual addressing. `wOffsetDst` + `width` must not exceed the
        width of the CUDA array `dst`. `wOffsetSrc` + `width` must not exceed
        the width of the CUDA array `src`.

        Parameters
        ----------
        dst : :py:obj:`~.cudaArray_t`
            Destination memory address
        wOffsetDst : size_t
            Destination starting X offset (columns in bytes)
        hOffsetDst : size_t
            Destination starting Y offset (rows)
        src : :py:obj:`~.cudaArray_const_t`
            Source memory address
        wOffsetSrc : size_t
            Source starting X offset (columns in bytes)
        hOffsetSrc : size_t
            Source starting Y offset (rows)
        width : size_t
            Width of matrix transfer (columns in bytes)
        height : size_t
            Height of matrix transfer (rows)
        kind : :py:obj:`~.cudaMemcpyKind`
            Type of transfer

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidMemcpyDirection`

        See Also
        --------
        :py:obj:`~.cudaMemcpy`, :py:obj:`~.cudaMemcpy2D`, :py:obj:`~.cudaMemcpy2DToArray`, :py:obj:`~.cudaMemcpy2DFromArray`, :py:obj:`~.cudaMemcpyToSymbol`, :py:obj:`~.cudaMemcpyFromSymbol`, :py:obj:`~.cudaMemcpyAsync`, :py:obj:`~.cudaMemcpy2DAsync`, :py:obj:`~.cudaMemcpy2DToArrayAsync`, :py:obj:`~.cudaMemcpy2DFromArrayAsync`, :py:obj:`~.cudaMemcpyToSymbolAsync`, :py:obj:`~.cudaMemcpyFromSymbolAsync`, :py:obj:`~.cuMemcpy2D`, :py:obj:`~.cuMemcpy2DUnaligned`
    """


def cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind: 'cudaMemcpyKind', stream):
    """
    cudaMemcpy2DAsync(dst, size_t dpitch, src, size_t spitch, size_t width, size_t height, kind: cudaMemcpyKind, stream)
     Copies data between host and device.

        Copies a matrix (`height` rows of `width` bytes each) from the memory
        area pointed to by `src` to the memory area pointed to by `dst`, where
        `kind` specifies the direction of the copy, and must be one of
        :py:obj:`~.cudaMemcpyHostToHost`, :py:obj:`~.cudaMemcpyHostToDevice`,
        :py:obj:`~.cudaMemcpyDeviceToHost`,
        :py:obj:`~.cudaMemcpyDeviceToDevice`, or :py:obj:`~.cudaMemcpyDefault`.
        Passing :py:obj:`~.cudaMemcpyDefault` is recommended, in which case the
        type of transfer is inferred from the pointer values. However,
        :py:obj:`~.cudaMemcpyDefault` is only allowed on systems that support
        unified virtual addressing. `dpitch` and `spitch` are the widths in
        memory in bytes of the 2D arrays pointed to by `dst` and `src`,
        including any padding added to the end of each row. The memory areas
        may not overlap. `width` must not exceed either `dpitch` or `spitch`.

        Calling :py:obj:`~.cudaMemcpy2DAsync()` with `dst` and `src` pointers
        that do not match the direction of the copy results in an undefined
        behavior. :py:obj:`~.cudaMemcpy2DAsync()` returns an error if `dpitch`
        or `spitch` is greater than the maximum allowed.

        :py:obj:`~.cudaMemcpy2DAsync()` is asynchronous with respect to the
        host, so the call may return before the copy is complete. The copy can
        optionally be associated to a stream by passing a non-zero `stream`
        argument. If `kind` is :py:obj:`~.cudaMemcpyHostToDevice` or
        :py:obj:`~.cudaMemcpyDeviceToHost` and `stream` is non-zero, the copy
        may overlap with operations in other streams.

        The device version of this function only handles device to device
        copies and cannot be given local or shared pointers.

        Parameters
        ----------
        dst : Any
            Destination memory address
        dpitch : size_t
            Pitch of destination memory
        src : Any
            Source memory address
        spitch : size_t
            Pitch of source memory
        width : size_t
            Width of matrix transfer (columns in bytes)
        height : size_t
            Height of matrix transfer (rows)
        kind : :py:obj:`~.cudaMemcpyKind`
            Type of transfer
        stream : :py:obj:`~.CUstream` or :py:obj:`~.cudaStream_t`
            Stream identifier

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidPitchValue`, :py:obj:`~.cudaErrorInvalidMemcpyDirection`

        See Also
        --------
        :py:obj:`~.cudaMemcpy`, :py:obj:`~.cudaMemcpy2D`, :py:obj:`~.cudaMemcpy2DToArray`, :py:obj:`~.cudaMemcpy2DFromArray`, :py:obj:`~.cudaMemcpy2DArrayToArray`, :py:obj:`~.cudaMemcpyToSymbol`, :py:obj:`~.cudaMemcpyFromSymbol`, :py:obj:`~.cudaMemcpyAsync`, :py:obj:`~.cudaMemcpy2DToArrayAsync`, :py:obj:`~.cudaMemcpy2DFromArrayAsync`, :py:obj:`~.cudaMemcpyToSymbolAsync`, :py:obj:`~.cudaMemcpyFromSymbolAsync`, :py:obj:`~.cuMemcpy2DAsync`
    """


def cudaMemcpy2DFromArray(dst, dpitch, src, wOffset, hOffset, width, height, kind: 'cudaMemcpyKind'):
    """
    cudaMemcpy2DFromArray(dst, size_t dpitch, src, size_t wOffset, size_t hOffset, size_t width, size_t height, kind: cudaMemcpyKind)
     Copies data between host and device.

        Copies a matrix (`height` rows of `width` bytes each) from the CUDA
        array `src` starting at `hOffset` rows and `wOffset` bytes from the
        upper left corner to the memory area pointed to by `dst`, where `kind`
        specifies the direction of the copy, and must be one of
        :py:obj:`~.cudaMemcpyHostToHost`, :py:obj:`~.cudaMemcpyHostToDevice`,
        :py:obj:`~.cudaMemcpyDeviceToHost`,
        :py:obj:`~.cudaMemcpyDeviceToDevice`, or :py:obj:`~.cudaMemcpyDefault`.
        Passing :py:obj:`~.cudaMemcpyDefault` is recommended, in which case the
        type of transfer is inferred from the pointer values. However,
        :py:obj:`~.cudaMemcpyDefault` is only allowed on systems that support
        unified virtual addressing. `dpitch` is the width in memory in bytes of
        the 2D array pointed to by `dst`, including any padding added to the
        end of each row. `wOffset` + `width` must not exceed the width of the
        CUDA array `src`. `width` must not exceed `dpitch`.
        :py:obj:`~.cudaMemcpy2DFromArray()` returns an error if `dpitch`
        exceeds the maximum allowed.

        Parameters
        ----------
        dst : Any
            Destination memory address
        dpitch : size_t
            Pitch of destination memory
        src : :py:obj:`~.cudaArray_const_t`
            Source memory address
        wOffset : size_t
            Source starting X offset (columns in bytes)
        hOffset : size_t
            Source starting Y offset (rows)
        width : size_t
            Width of matrix transfer (columns in bytes)
        height : size_t
            Height of matrix transfer (rows)
        kind : :py:obj:`~.cudaMemcpyKind`
            Type of transfer

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidPitchValue`, :py:obj:`~.cudaErrorInvalidMemcpyDirection`

        See Also
        --------
        :py:obj:`~.cudaMemcpy`, :py:obj:`~.cudaMemcpy2D`, :py:obj:`~.cudaMemcpy2DToArray`, :py:obj:`~.cudaMemcpy2DArrayToArray`, :py:obj:`~.cudaMemcpyToSymbol`, :py:obj:`~.cudaMemcpyFromSymbol`, :py:obj:`~.cudaMemcpyAsync`, :py:obj:`~.cudaMemcpy2DAsync`, :py:obj:`~.cudaMemcpy2DToArrayAsync`, :py:obj:`~.cudaMemcpy2DFromArrayAsync`, :py:obj:`~.cudaMemcpyToSymbolAsync`, :py:obj:`~.cudaMemcpyFromSymbolAsync`, :py:obj:`~.cuMemcpy2D`, :py:obj:`~.cuMemcpy2DUnaligned`
    """


def cudaMemcpy2DFromArrayAsync(dst, dpitch, src, wOffset, hOffset, width, height, kind: 'cudaMemcpyKind', stream):
    """
    cudaMemcpy2DFromArrayAsync(dst, size_t dpitch, src, size_t wOffset, size_t hOffset, size_t width, size_t height, kind: cudaMemcpyKind, stream)
     Copies data between host and device.

        Copies a matrix (`height` rows of `width` bytes each) from the CUDA
        array `src` starting at `hOffset` rows and `wOffset` bytes from the
        upper left corner to the memory area pointed to by `dst`, where `kind`
        specifies the direction of the copy, and must be one of
        :py:obj:`~.cudaMemcpyHostToHost`, :py:obj:`~.cudaMemcpyHostToDevice`,
        :py:obj:`~.cudaMemcpyDeviceToHost`,
        :py:obj:`~.cudaMemcpyDeviceToDevice`, or :py:obj:`~.cudaMemcpyDefault`.
        Passing :py:obj:`~.cudaMemcpyDefault` is recommended, in which case the
        type of transfer is inferred from the pointer values. However,
        :py:obj:`~.cudaMemcpyDefault` is only allowed on systems that support
        unified virtual addressing. `dpitch` is the width in memory in bytes of
        the 2D array pointed to by `dst`, including any padding added to the
        end of each row. `wOffset` + `width` must not exceed the width of the
        CUDA array `src`. `width` must not exceed `dpitch`.
        :py:obj:`~.cudaMemcpy2DFromArrayAsync()` returns an error if `dpitch`
        exceeds the maximum allowed.

        :py:obj:`~.cudaMemcpy2DFromArrayAsync()` is asynchronous with respect
        to the host, so the call may return before the copy is complete. The
        copy can optionally be associated to a stream by passing a non-zero
        `stream` argument. If `kind` is :py:obj:`~.cudaMemcpyHostToDevice` or
        :py:obj:`~.cudaMemcpyDeviceToHost` and `stream` is non-zero, the copy
        may overlap with operations in other streams.

        :py:obj:`~.cudaMemcpyToSymbolAsync`,
        :py:obj:`~.cudaMemcpyFromSymbolAsync`, :py:obj:`~.cuMemcpy2DAsync`

        Parameters
        ----------
        dst : Any
            Destination memory address
        dpitch : size_t
            Pitch of destination memory
        src : :py:obj:`~.cudaArray_const_t`
            Source memory address
        wOffset : size_t
            Source starting X offset (columns in bytes)
        hOffset : size_t
            Source starting Y offset (rows)
        width : size_t
            Width of matrix transfer (columns in bytes)
        height : size_t
            Height of matrix transfer (rows)
        kind : :py:obj:`~.cudaMemcpyKind`
            Type of transfer
        stream : :py:obj:`~.CUstream` or :py:obj:`~.cudaStream_t`
            Stream identifier

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidPitchValue`, :py:obj:`~.cudaErrorInvalidMemcpyDirection`

        See Also
        --------
        :py:obj:`~.cudaMemcpy`, :py:obj:`~.cudaMemcpy2D`, :py:obj:`~.cudaMemcpy2DToArray`, :py:obj:`~.cudaMemcpy2DFromArray`, :py:obj:`~.cudaMemcpy2DArrayToArray`, :py:obj:`~.cudaMemcpyToSymbol`, :py:obj:`~.cudaMemcpyFromSymbol`, :py:obj:`~.cudaMemcpyAsync`, :py:obj:`~.cudaMemcpy2DAsync`, :py:obj:`~.cudaMemcpy2DToArrayAsync`,
    """


def cudaMemcpy2DToArray(dst, wOffset, hOffset, src, spitch, width, height, kind: 'cudaMemcpyKind'):
    """
    cudaMemcpy2DToArray(dst, size_t wOffset, size_t hOffset, src, size_t spitch, size_t width, size_t height, kind: cudaMemcpyKind)
     Copies data between host and device.

        Copies a matrix (`height` rows of `width` bytes each) from the memory
        area pointed to by `src` to the CUDA array `dst` starting at `hOffset`
        rows and `wOffset` bytes from the upper left corner, where `kind`
        specifies the direction of the copy, and must be one of
        :py:obj:`~.cudaMemcpyHostToHost`, :py:obj:`~.cudaMemcpyHostToDevice`,
        :py:obj:`~.cudaMemcpyDeviceToHost`,
        :py:obj:`~.cudaMemcpyDeviceToDevice`, or :py:obj:`~.cudaMemcpyDefault`.
        Passing :py:obj:`~.cudaMemcpyDefault` is recommended, in which case the
        type of transfer is inferred from the pointer values. However,
        :py:obj:`~.cudaMemcpyDefault` is only allowed on systems that support
        unified virtual addressing. `spitch` is the width in memory in bytes of
        the 2D array pointed to by `src`, including any padding added to the
        end of each row. `wOffset` + `width` must not exceed the width of the
        CUDA array `dst`. `width` must not exceed `spitch`.
        :py:obj:`~.cudaMemcpy2DToArray()` returns an error if `spitch` exceeds
        the maximum allowed.

        Parameters
        ----------
        dst : :py:obj:`~.cudaArray_t`
            Destination memory address
        wOffset : size_t
            Destination starting X offset (columns in bytes)
        hOffset : size_t
            Destination starting Y offset (rows)
        src : Any
            Source memory address
        spitch : size_t
            Pitch of source memory
        width : size_t
            Width of matrix transfer (columns in bytes)
        height : size_t
            Height of matrix transfer (rows)
        kind : :py:obj:`~.cudaMemcpyKind`
            Type of transfer

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidPitchValue`, :py:obj:`~.cudaErrorInvalidMemcpyDirection`

        See Also
        --------
        :py:obj:`~.cudaMemcpy`, :py:obj:`~.cudaMemcpy2D`, :py:obj:`~.cudaMemcpy2DFromArray`, :py:obj:`~.cudaMemcpy2DArrayToArray`, :py:obj:`~.cudaMemcpyToSymbol`, :py:obj:`~.cudaMemcpyFromSymbol`, :py:obj:`~.cudaMemcpyAsync`, :py:obj:`~.cudaMemcpy2DAsync`, :py:obj:`~.cudaMemcpy2DToArrayAsync`, :py:obj:`~.cudaMemcpy2DFromArrayAsync`, :py:obj:`~.cudaMemcpyToSymbolAsync`, :py:obj:`~.cudaMemcpyFromSymbolAsync`, :py:obj:`~.cuMemcpy2D`, :py:obj:`~.cuMemcpy2DUnaligned`
    """


def cudaMemcpy2DToArrayAsync(dst, wOffset, hOffset, src, spitch, width, height, kind: 'cudaMemcpyKind', stream):
    """
    cudaMemcpy2DToArrayAsync(dst, size_t wOffset, size_t hOffset, src, size_t spitch, size_t width, size_t height, kind: cudaMemcpyKind, stream)
     Copies data between host and device.

        Copies a matrix (`height` rows of `width` bytes each) from the memory
        area pointed to by `src` to the CUDA array `dst` starting at `hOffset`
        rows and `wOffset` bytes from the upper left corner, where `kind`
        specifies the direction of the copy, and must be one of
        :py:obj:`~.cudaMemcpyHostToHost`, :py:obj:`~.cudaMemcpyHostToDevice`,
        :py:obj:`~.cudaMemcpyDeviceToHost`,
        :py:obj:`~.cudaMemcpyDeviceToDevice`, or :py:obj:`~.cudaMemcpyDefault`.
        Passing :py:obj:`~.cudaMemcpyDefault` is recommended, in which case the
        type of transfer is inferred from the pointer values. However,
        :py:obj:`~.cudaMemcpyDefault` is only allowed on systems that support
        unified virtual addressing. `spitch` is the width in memory in bytes of
        the 2D array pointed to by `src`, including any padding added to the
        end of each row. `wOffset` + `width` must not exceed the width of the
        CUDA array `dst`. `width` must not exceed `spitch`.
        :py:obj:`~.cudaMemcpy2DToArrayAsync()` returns an error if `spitch`
        exceeds the maximum allowed.

        :py:obj:`~.cudaMemcpy2DToArrayAsync()` is asynchronous with respect to
        the host, so the call may return before the copy is complete. The copy
        can optionally be associated to a stream by passing a non-zero `stream`
        argument. If `kind` is :py:obj:`~.cudaMemcpyHostToDevice` or
        :py:obj:`~.cudaMemcpyDeviceToHost` and `stream` is non-zero, the copy
        may overlap with operations in other streams.

        :py:obj:`~.cudaMemcpy2DFromArrayAsync`,
        :py:obj:`~.cudaMemcpyToSymbolAsync`,
        :py:obj:`~.cudaMemcpyFromSymbolAsync`, :py:obj:`~.cuMemcpy2DAsync`

        Parameters
        ----------
        dst : :py:obj:`~.cudaArray_t`
            Destination memory address
        wOffset : size_t
            Destination starting X offset (columns in bytes)
        hOffset : size_t
            Destination starting Y offset (rows)
        src : Any
            Source memory address
        spitch : size_t
            Pitch of source memory
        width : size_t
            Width of matrix transfer (columns in bytes)
        height : size_t
            Height of matrix transfer (rows)
        kind : :py:obj:`~.cudaMemcpyKind`
            Type of transfer
        stream : :py:obj:`~.CUstream` or :py:obj:`~.cudaStream_t`
            Stream identifier

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidPitchValue`, :py:obj:`~.cudaErrorInvalidMemcpyDirection`

        See Also
        --------
        :py:obj:`~.cudaMemcpy`, :py:obj:`~.cudaMemcpy2D`, :py:obj:`~.cudaMemcpy2DToArray`, :py:obj:`~.cudaMemcpy2DFromArray`, :py:obj:`~.cudaMemcpy2DArrayToArray`, :py:obj:`~.cudaMemcpyToSymbol`, :py:obj:`~.cudaMemcpyFromSymbol`, :py:obj:`~.cudaMemcpyAsync`, :py:obj:`~.cudaMemcpy2DAsync`,
    """


def cudaMemcpy3D(p: 'Optional[cudaMemcpy3DParms]'):
    """
    cudaMemcpy3D(cudaMemcpy3DParms p: Optional[cudaMemcpy3DParms])
     Copies data between 3D objects.

        **View CUDA Toolkit Documentation for a C++ code example**

        :py:obj:`~.cudaMemcpy3D()` copies data betwen two 3D objects. The
        source and destination objects may be in either host memory, device
        memory, or a CUDA array. The source, destination, extent, and kind of
        copy performed is specified by the :py:obj:`~.cudaMemcpy3DParms` struct
        which should be initialized to zero before use:

        **View CUDA Toolkit Documentation for a C++ code example**

        The struct passed to :py:obj:`~.cudaMemcpy3D()` must specify one of
        `srcArray` or `srcPtr` and one of `dstArray` or `dstPtr`. Passing more
        than one non-zero source or destination will cause
        :py:obj:`~.cudaMemcpy3D()` to return an error.

        The `srcPos` and `dstPos` fields are optional offsets into the source
        and destination objects and are defined in units of each object's
        elements. The element for a host or device pointer is assumed to be
        unsigned char.

        The `extent` field defines the dimensions of the transferred area in
        elements. If a CUDA array is participating in the copy, the extent is
        defined in terms of that array's elements. If no CUDA array is
        participating in the copy then the extents are defined in elements of
        unsigned char.

        The `kind` field defines the direction of the copy. It must be one of
        :py:obj:`~.cudaMemcpyHostToHost`, :py:obj:`~.cudaMemcpyHostToDevice`,
        :py:obj:`~.cudaMemcpyDeviceToHost`,
        :py:obj:`~.cudaMemcpyDeviceToDevice`, or :py:obj:`~.cudaMemcpyDefault`.
        Passing :py:obj:`~.cudaMemcpyDefault` is recommended, in which case the
        type of transfer is inferred from the pointer values. However,
        :py:obj:`~.cudaMemcpyDefault` is only allowed on systems that support
        unified virtual addressing. For :py:obj:`~.cudaMemcpyHostToHost` or
        :py:obj:`~.cudaMemcpyHostToDevice` or
        :py:obj:`~.cudaMemcpyDeviceToHost` passed as kind and cudaArray type
        passed as source or destination, if the kind implies cudaArray type to
        be present on the host, :py:obj:`~.cudaMemcpy3D()` will disregard that
        implication and silently correct the kind based on the fact that
        cudaArray type can only be present on the device.

        If the source and destination are both arrays,
        :py:obj:`~.cudaMemcpy3D()` will return an error if they do not have the
        same element size.

        The source and destination object may not overlap. If overlapping
        source and destination objects are specified, undefined behavior will
        result.

        The source object must entirely contain the region defined by `srcPos`
        and `extent`. The destination object must entirely contain the region
        defined by `dstPos` and `extent`.

        :py:obj:`~.cudaMemcpy3D()` returns an error if the pitch of `srcPtr` or
        `dstPtr` exceeds the maximum allowed. The pitch of a
        :py:obj:`~.cudaPitchedPtr` allocated with :py:obj:`~.cudaMalloc3D()`
        will always be valid.

        Parameters
        ----------
        p : :py:obj:`~.cudaMemcpy3DParms`
            3D memory copy parameters

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidPitchValue`, :py:obj:`~.cudaErrorInvalidMemcpyDirection`

        See Also
        --------
        :py:obj:`~.cudaMalloc3D`, :py:obj:`~.cudaMalloc3DArray`, :py:obj:`~.cudaMemset3D`, :py:obj:`~.cudaMemcpy3DAsync`, :py:obj:`~.cudaMemcpy`, :py:obj:`~.cudaMemcpy2D`, :py:obj:`~.cudaMemcpy2DToArray`, :py:obj:`~.cudaMemcpy2DFromArray`, :py:obj:`~.cudaMemcpy2DArrayToArray`, :py:obj:`~.cudaMemcpyToSymbol`, :py:obj:`~.cudaMemcpyFromSymbol`, :py:obj:`~.cudaMemcpyAsync`, :py:obj:`~.cudaMemcpy2DAsync`, :py:obj:`~.cudaMemcpy2DToArrayAsync`, :py:obj:`~.cudaMemcpy2DFromArrayAsync`, :py:obj:`~.cudaMemcpyToSymbolAsync`, :py:obj:`~.cudaMemcpyFromSymbolAsync`, :py:obj:`~.make_cudaExtent`, :py:obj:`~.make_cudaPos`, :py:obj:`~.cuMemcpy3D`
    """


def cudaMemcpy3DAsync(p: 'Optional[cudaMemcpy3DParms]', stream):
    """
    cudaMemcpy3DAsync(cudaMemcpy3DParms p: Optional[cudaMemcpy3DParms], stream)
     Copies data between 3D objects.

        **View CUDA Toolkit Documentation for a C++ code example**

        :py:obj:`~.cudaMemcpy3DAsync()` copies data betwen two 3D objects. The
        source and destination objects may be in either host memory, device
        memory, or a CUDA array. The source, destination, extent, and kind of
        copy performed is specified by the :py:obj:`~.cudaMemcpy3DParms` struct
        which should be initialized to zero before use:

        **View CUDA Toolkit Documentation for a C++ code example**

        The struct passed to :py:obj:`~.cudaMemcpy3DAsync()` must specify one
        of `srcArray` or `srcPtr` and one of `dstArray` or `dstPtr`. Passing
        more than one non-zero source or destination will cause
        :py:obj:`~.cudaMemcpy3DAsync()` to return an error.

        The `srcPos` and `dstPos` fields are optional offsets into the source
        and destination objects and are defined in units of each object's
        elements. The element for a host or device pointer is assumed to be
        unsigned char. For CUDA arrays, positions must be in the range [0,
        2048) for any dimension.

        The `extent` field defines the dimensions of the transferred area in
        elements. If a CUDA array is participating in the copy, the extent is
        defined in terms of that array's elements. If no CUDA array is
        participating in the copy then the extents are defined in elements of
        unsigned char.

        The `kind` field defines the direction of the copy. It must be one of
        :py:obj:`~.cudaMemcpyHostToHost`, :py:obj:`~.cudaMemcpyHostToDevice`,
        :py:obj:`~.cudaMemcpyDeviceToHost`,
        :py:obj:`~.cudaMemcpyDeviceToDevice`, or :py:obj:`~.cudaMemcpyDefault`.
        Passing :py:obj:`~.cudaMemcpyDefault` is recommended, in which case the
        type of transfer is inferred from the pointer values. However,
        :py:obj:`~.cudaMemcpyDefault` is only allowed on systems that support
        unified virtual addressing. For :py:obj:`~.cudaMemcpyHostToHost` or
        :py:obj:`~.cudaMemcpyHostToDevice` or
        :py:obj:`~.cudaMemcpyDeviceToHost` passed as kind and cudaArray type
        passed as source or destination, if the kind implies cudaArray type to
        be present on the host, :py:obj:`~.cudaMemcpy3DAsync()` will disregard
        that implication and silently correct the kind based on the fact that
        cudaArray type can only be present on the device.

        If the source and destination are both arrays,
        :py:obj:`~.cudaMemcpy3DAsync()` will return an error if they do not
        have the same element size.

        The source and destination object may not overlap. If overlapping
        source and destination objects are specified, undefined behavior will
        result.

        The source object must lie entirely within the region defined by
        `srcPos` and `extent`. The destination object must lie entirely within
        the region defined by `dstPos` and `extent`.

        :py:obj:`~.cudaMemcpy3DAsync()` returns an error if the pitch of
        `srcPtr` or `dstPtr` exceeds the maximum allowed. The pitch of a
        :py:obj:`~.cudaPitchedPtr` allocated with :py:obj:`~.cudaMalloc3D()`
        will always be valid.

        :py:obj:`~.cudaMemcpy3DAsync()` is asynchronous with respect to the
        host, so the call may return before the copy is complete. The copy can
        optionally be associated to a stream by passing a non-zero `stream`
        argument. If `kind` is :py:obj:`~.cudaMemcpyHostToDevice` or
        :py:obj:`~.cudaMemcpyDeviceToHost` and `stream` is non-zero, the copy
        may overlap with operations in other streams.

        The device version of this function only handles device to device
        copies and cannot be given local or shared pointers.

        Parameters
        ----------
        p : :py:obj:`~.cudaMemcpy3DParms`
            3D memory copy parameters
        stream : :py:obj:`~.CUstream` or :py:obj:`~.cudaStream_t`
            Stream identifier

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidPitchValue`, :py:obj:`~.cudaErrorInvalidMemcpyDirection`

        See Also
        --------
        :py:obj:`~.cudaMalloc3D`, :py:obj:`~.cudaMalloc3DArray`, :py:obj:`~.cudaMemset3D`, :py:obj:`~.cudaMemcpy3D`, :py:obj:`~.cudaMemcpy`, :py:obj:`~.cudaMemcpy2D`, :py:obj:`~.cudaMemcpy2DToArray`, ::::py:obj:`~.cudaMemcpy2DFromArray`, :py:obj:`~.cudaMemcpy2DArrayToArray`, :py:obj:`~.cudaMemcpyToSymbol`, :py:obj:`~.cudaMemcpyFromSymbol`, :py:obj:`~.cudaMemcpyAsync`, :py:obj:`~.cudaMemcpy2DAsync`, :py:obj:`~.cudaMemcpy2DToArrayAsync`, :py:obj:`~.cudaMemcpy2DFromArrayAsync`, :py:obj:`~.cudaMemcpyToSymbolAsync`, :py:obj:`~.cudaMemcpyFromSymbolAsync`, :py:obj:`~.make_cudaExtent`, :py:obj:`~.make_cudaPos`, :py:obj:`~.cuMemcpy3DAsync`
    """


def cudaMemcpy3DBatchAsync(numOps, opList: 'Optional[Tuple[cudaMemcpy3DBatchOp] | List[cudaMemcpy3DBatchOp]]', flags, stream):
    """
    cudaMemcpy3DBatchAsync(size_t numOps, opList: Optional[Tuple[cudaMemcpy3DBatchOp] | List[cudaMemcpy3DBatchOp]], unsigned long long flags, stream)
     Performs a batch of 3D memory copies asynchronously.

        Performs a batch of memory copies. The batch as a whole executes in
        stream order but copies within a batch are not guaranteed to execute in
        any specific order. Note that this means specifying any dependent
        copies within a batch will result in undefined behavior.

        Performs memory copies as specified in the `opList` array. The length
        of this array is specified in `numOps`. Each entry in this array
        describes a copy operation. This includes among other things, the
        source and destination operands for the copy as specified in
        :py:obj:`~.cudaMemcpy3DBatchOp.src` and
        :py:obj:`~.cudaMemcpy3DBatchOp.dst` respectively. The source and
        destination operands of a copy can either be a pointer or a CUDA array.
        The width, height and depth of a copy is specified in
        :py:obj:`~.cudaMemcpy3DBatchOp.extent`. The width, height and depth of
        a copy are specified in elements and must not be zero. For pointer-to-
        pointer copies, the element size is considered to be 1. For pointer to
        CUDA array or vice versa copies, the element size is determined by the
        CUDA array. For CUDA array to CUDA array copies, the element size of
        the two CUDA arrays must match.

        For a given operand, if :py:obj:`~.cudaMemcpy3DOperand`::type is
        specified as :py:obj:`~.cudaMemcpyOperandTypePointer`, then
        :py:obj:`~.cudaMemcpy3DOperand`::op::ptr will be used. The
        :py:obj:`~.cudaMemcpy3DOperand`::op::ptr::ptr field must contain the
        pointer where the copy should begin. The
        :py:obj:`~.cudaMemcpy3DOperand`::op::ptr::rowLength field specifies the
        length of each row in elements and must either be zero or be greater
        than or equal to the width of the copy specified in
        :py:obj:`~.cudaMemcpy3DBatchOp`::extent::width. The
        :py:obj:`~.cudaMemcpy3DOperand`::op::ptr::layerHeight field specifies
        the height of each layer and must either be zero or be greater than or
        equal to the height of the copy specified in
        :py:obj:`~.cudaMemcpy3DBatchOp`::extent::height. When either of these
        values is zero, that aspect of the operand is considered to be tightly
        packed according to the copy extent. For managed memory pointers on
        devices where :py:obj:`~.cudaDevAttrConcurrentManagedAccess` is true or
        system-allocated pageable memory on devices where
        :py:obj:`~.cudaDevAttrPageableMemoryAccess` is true, the
        :py:obj:`~.cudaMemcpy3DOperand`::op::ptr::locHint field can be used to
        hint the location of the operand.

        If an operand's type is specified as
        :py:obj:`~.cudaMemcpyOperandTypeArray`, then
        :py:obj:`~.cudaMemcpy3DOperand`::op::array will be used. The
        :py:obj:`~.cudaMemcpy3DOperand`::op::array::array field specifies the
        CUDA array and :py:obj:`~.cudaMemcpy3DOperand`::op::array::offset
        specifies the 3D offset into that array where the copy begins.

        The :py:obj:`~.cudaMemcpyAttributes.srcAccessOrder` indicates the
        source access ordering to be observed for copies associated with the
        attribute. If the source access order is set to
        :py:obj:`~.cudaMemcpySrcAccessOrderStream`, then the source will be
        accessed in stream order. If the source access order is set to
        :py:obj:`~.cudaMemcpySrcAccessOrderDuringApiCall` then it indicates
        that access to the source pointer can be out of stream order and all
        accesses must be complete before the API call returns. This flag is
        suited for ephemeral sources (ex., stack variables) when it's known
        that no prior operations in the stream can be accessing the memory and
        also that the lifetime of the memory is limited to the scope that the
        source variable was declared in. Specifying this flag allows the driver
        to optimize the copy and removes the need for the user to synchronize
        the stream after the API call. If the source access order is set to
        :py:obj:`~.cudaMemcpySrcAccessOrderAny` then it indicates that access
        to the source pointer can be out of stream order and the accesses can
        happen even after the API call returns. This flag is suited for host
        pointers allocated outside CUDA (ex., via malloc) when it's known that
        no prior operations in the stream can be accessing the memory.
        Specifying this flag allows the driver to optimize the copy on certain
        platforms. Each memcopy operation in `opList` must have a valid
        srcAccessOrder setting, otherwise this API will return
        :py:obj:`~.cudaErrorInvalidValue`.

        The :py:obj:`~.cudaMemcpyAttributes.flags` field can be used to specify
        certain flags for copies. Setting the
        :py:obj:`~.cudaMemcpyFlagPreferOverlapWithCompute` flag indicates that
        the associated copies should preferably overlap with any compute work.
        Note that this flag is a hint and can be ignored depending on the
        platform and other parameters of the copy.

        If any error is encountered while parsing the batch, the index within
        the batch where the error was encountered will be returned in
        `failIdx`.

        Parameters
        ----------
        numOps : size_t
            Total number of memcpy operations.
        opList : List[:py:obj:`~.cudaMemcpy3DBatchOp`]
            Array of size `numOps` containing the actual memcpy operations.
        flags : unsigned long long
            Flags for future use, must be zero now.
        hStream : :py:obj:`~.CUstream` or :py:obj:`~.cudaStream_t`
            The stream to enqueue the operations in. Must not be default NULL
            stream.

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess` :py:obj:`~.cudaErrorInvalidValue`
        failIdx : int
            Pointer to a location to return the index of the copy where a
            failure was encountered. The value will be SIZE_MAX if the error
            doesn't pertain to any specific copy.
    """


def cudaMemcpy3DPeer(p: 'Optional[cudaMemcpy3DPeerParms]'):
    """
    cudaMemcpy3DPeer(cudaMemcpy3DPeerParms p: Optional[cudaMemcpy3DPeerParms])
     Copies memory between devices.

        Perform a 3D memory copy according to the parameters specified in `p`.
        See the definition of the :py:obj:`~.cudaMemcpy3DPeerParms` structure
        for documentation of its parameters.

        Note that this function is synchronous with respect to the host only if
        the source or destination of the transfer is host memory. Note also
        that this copy is serialized with respect to all pending and future
        asynchronous work in to the current device, the copy's source device,
        and the copy's destination device (use
        :py:obj:`~.cudaMemcpy3DPeerAsync` to avoid this synchronization).

        Parameters
        ----------
        p : :py:obj:`~.cudaMemcpy3DPeerParms`
            Parameters for the memory copy

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidDevice`, :py:obj:`~.cudaErrorInvalidPitchValue`

        See Also
        --------
        :py:obj:`~.cudaMemcpy`, :py:obj:`~.cudaMemcpyPeer`, :py:obj:`~.cudaMemcpyAsync`, :py:obj:`~.cudaMemcpyPeerAsync`, :py:obj:`~.cudaMemcpy3DPeerAsync`, :py:obj:`~.cuMemcpy3DPeer`
    """


def cudaMemcpy3DPeerAsync(p: 'Optional[cudaMemcpy3DPeerParms]', stream):
    """
    cudaMemcpy3DPeerAsync(cudaMemcpy3DPeerParms p: Optional[cudaMemcpy3DPeerParms], stream)
     Copies memory between devices asynchronously.

        Perform a 3D memory copy according to the parameters specified in `p`.
        See the definition of the :py:obj:`~.cudaMemcpy3DPeerParms` structure
        for documentation of its parameters.

        Parameters
        ----------
        p : :py:obj:`~.cudaMemcpy3DPeerParms`
            Parameters for the memory copy
        stream : :py:obj:`~.CUstream` or :py:obj:`~.cudaStream_t`
            Stream identifier

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidDevice`, :py:obj:`~.cudaErrorInvalidPitchValue`

        See Also
        --------
        :py:obj:`~.cudaMemcpy`, :py:obj:`~.cudaMemcpyPeer`, :py:obj:`~.cudaMemcpyAsync`, :py:obj:`~.cudaMemcpyPeerAsync`, :py:obj:`~.cudaMemcpy3DPeerAsync`, :py:obj:`~.cuMemcpy3DPeerAsync`
    """


def cudaMemcpyArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, count, kind: 'cudaMemcpyKind'):
    """
    cudaMemcpyArrayToArray(dst, size_t wOffsetDst, size_t hOffsetDst, src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, kind: cudaMemcpyKind)
     Copies data between host and device.

        [Deprecated]

        Copies `count` bytes from the CUDA array `src` starting at `hOffsetSrc`
        rows and `wOffsetSrc` bytes from the upper left corner to the CUDA
        array `dst` starting at `hOffsetDst` rows and `wOffsetDst` bytes from
        the upper left corner, where `kind` specifies the direction of the
        copy, and must be one of :py:obj:`~.cudaMemcpyHostToHost`,
        :py:obj:`~.cudaMemcpyHostToDevice`, :py:obj:`~.cudaMemcpyDeviceToHost`,
        :py:obj:`~.cudaMemcpyDeviceToDevice`, or :py:obj:`~.cudaMemcpyDefault`.
        Passing :py:obj:`~.cudaMemcpyDefault` is recommended, in which case the
        type of transfer is inferred from the pointer values. However,
        :py:obj:`~.cudaMemcpyDefault` is only allowed on systems that support
        unified virtual addressing.

        Parameters
        ----------
        dst : :py:obj:`~.cudaArray_t`
            Destination memory address
        wOffsetDst : size_t
            Destination starting X offset (columns in bytes)
        hOffsetDst : size_t
            Destination starting Y offset (rows)
        src : :py:obj:`~.cudaArray_const_t`
            Source memory address
        wOffsetSrc : size_t
            Source starting X offset (columns in bytes)
        hOffsetSrc : size_t
            Source starting Y offset (rows)
        count : size_t
            Size in bytes to copy
        kind : :py:obj:`~.cudaMemcpyKind`
            Type of transfer

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidMemcpyDirection`

        See Also
        --------
        :py:obj:`~.cudaMemcpy`, :py:obj:`~.cudaMemcpy2D`, :py:obj:`~.cudaMemcpyToArray`, :py:obj:`~.cudaMemcpy2DToArray`, :py:obj:`~.cudaMemcpyFromArray`, :py:obj:`~.cudaMemcpy2DFromArray`, :py:obj:`~.cudaMemcpy2DArrayToArray`, :py:obj:`~.cudaMemcpyToSymbol`, :py:obj:`~.cudaMemcpyFromSymbol`, :py:obj:`~.cudaMemcpyAsync`, :py:obj:`~.cudaMemcpy2DAsync`, :py:obj:`~.cudaMemcpyToArrayAsync`, :py:obj:`~.cudaMemcpy2DToArrayAsync`, :py:obj:`~.cudaMemcpyFromArrayAsync`, :py:obj:`~.cudaMemcpy2DFromArrayAsync`, :py:obj:`~.cudaMemcpyToSymbolAsync`, :py:obj:`~.cudaMemcpyFromSymbolAsync`, :py:obj:`~.cuMemcpyAtoA`
    """


def cudaMemcpyAsync(dst, src, count, kind: 'cudaMemcpyKind', stream):
    """
    cudaMemcpyAsync(dst, src, size_t count, kind: cudaMemcpyKind, stream)
     Copies data between host and device.

        Copies `count` bytes from the memory area pointed to by `src` to the
        memory area pointed to by `dst`, where `kind` specifies the direction
        of the copy, and must be one of :py:obj:`~.cudaMemcpyHostToHost`,
        :py:obj:`~.cudaMemcpyHostToDevice`, :py:obj:`~.cudaMemcpyDeviceToHost`,
        :py:obj:`~.cudaMemcpyDeviceToDevice`, or :py:obj:`~.cudaMemcpyDefault`.
        Passing :py:obj:`~.cudaMemcpyDefault` is recommended, in which case the
        type of transfer is inferred from the pointer values. However,
        :py:obj:`~.cudaMemcpyDefault` is only allowed on systems that support
        unified virtual addressing.

        The memory areas may not overlap. Calling :py:obj:`~.cudaMemcpyAsync()`
        with `dst` and `src` pointers that do not match the direction of the
        copy results in an undefined behavior.

        :py:obj:`~.cudaMemcpyAsync()` is asynchronous with respect to the host,
        so the call may return before the copy is complete. The copy can
        optionally be associated to a stream by passing a non-zero `stream`
        argument. If `kind` is :py:obj:`~.cudaMemcpyHostToDevice` or
        :py:obj:`~.cudaMemcpyDeviceToHost` and the `stream` is non-zero, the
        copy may overlap with operations in other streams.

        The device version of this function only handles device to device
        copies and cannot be given local or shared pointers.

        Parameters
        ----------
        dst : Any
            Destination memory address
        src : Any
            Source memory address
        count : size_t
            Size in bytes to copy
        kind : :py:obj:`~.cudaMemcpyKind`
            Type of transfer
        stream : :py:obj:`~.CUstream` or :py:obj:`~.cudaStream_t`
            Stream identifier

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidMemcpyDirection`

        See Also
        --------
        :py:obj:`~.cudaMemcpy`, :py:obj:`~.cudaMemcpy2D`, :py:obj:`~.cudaMemcpy2DToArray`, :py:obj:`~.cudaMemcpy2DFromArray`, :py:obj:`~.cudaMemcpy2DArrayToArray`, :py:obj:`~.cudaMemcpyToSymbol`, :py:obj:`~.cudaMemcpyFromSymbol`, :py:obj:`~.cudaMemcpy2DAsync`, :py:obj:`~.cudaMemcpy2DToArrayAsync`, :py:obj:`~.cudaMemcpy2DFromArrayAsync`, :py:obj:`~.cudaMemcpyToSymbolAsync`, :py:obj:`~.cudaMemcpyFromSymbolAsync`, :py:obj:`~.cuMemcpyAsync`, :py:obj:`~.cuMemcpyDtoHAsync`, :py:obj:`~.cuMemcpyHtoDAsync`, :py:obj:`~.cuMemcpyDtoDAsync`
    """


def cudaMemcpyBatchAsync(dsts: 'Optional[Tuple[Any] | List[Any]]', srcs: 'Optional[Tuple[Any] | List[Any]]', sizes: 'Tuple[int] | List[int]', count, attrs: 'Optional[Tuple[cudaMemcpyAttributes] | List[cudaMemcpyAttributes]]', attrsIdxs: 'Tuple[int] | List[int]', numAttrs, stream):
    """
    cudaMemcpyBatchAsync(dsts: Optional[Tuple[Any] | List[Any]], srcs: Optional[Tuple[Any] | List[Any]], sizes: Tuple[int] | List[int], size_t count, attrs: Optional[Tuple[cudaMemcpyAttributes] | List[cudaMemcpyAttributes]], attrsIdxs: Tuple[int] | List[int], size_t numAttrs, stream)
     Performs a batch of memory copies asynchronously.

        Performs a batch of memory copies. The batch as a whole executes in
        stream order but copies within a batch are not guaranteed to execute in
        any specific order. This API only supports pointer-to-pointer copies.
        For copies involving CUDA arrays, please see
        :py:obj:`~.cudaMemcpy3DBatchAsync`.

        Performs memory copies from source buffers specified in `srcs` to
        destination buffers specified in `dsts`. The size of each copy is
        specified in `sizes`. All three arrays must be of the same length as
        specified by `count`. Since there are no ordering guarantees for copies
        within a batch, specifying any dependent copies within a batch will
        result in undefined behavior.

        Every copy in the batch has to be associated with a set of attributes
        specified in the `attrs` array. Each entry in this array can apply to
        more than one copy. This can be done by specifying in the `attrsIdxs`
        array, the index of the first copy that the corresponding entry in the
        `attrs` array applies to. Both `attrs` and `attrsIdxs` must be of the
        same length as specified by `numAttrs`. For example, if a batch has 10
        copies listed in dst/src/sizes, the first 6 of which have one set of
        attributes and the remaining 4 another, then `numAttrs` will be 2,
        `attrsIdxs` will be {0, 6} and `attrs` will contains the two sets of
        attributes. Note that the first entry in `attrsIdxs` must always be 0.
        Also, each entry must be greater than the previous entry and the last
        entry should be less than `count`. Furthermore, `numAttrs` must be
        lesser than or equal to `count`.

        The :py:obj:`~.cudaMemcpyAttributes.srcAccessOrder` indicates the
        source access ordering to be observed for copies associated with the
        attribute. If the source access order is set to
        :py:obj:`~.cudaMemcpySrcAccessOrderStream`, then the source will be
        accessed in stream order. If the source access order is set to
        :py:obj:`~.cudaMemcpySrcAccessOrderDuringApiCall` then it indicates
        that access to the source pointer can be out of stream order and all
        accesses must be complete before the API call returns. This flag is
        suited for ephemeral sources (ex., stack variables) when it's known
        that no prior operations in the stream can be accessing the memory and
        also that the lifetime of the memory is limited to the scope that the
        source variable was declared in. Specifying this flag allows the driver
        to optimize the copy and removes the need for the user to synchronize
        the stream after the API call. If the source access order is set to
        :py:obj:`~.cudaMemcpySrcAccessOrderAny` then it indicates that access
        to the source pointer can be out of stream order and the accesses can
        happen even after the API call returns. This flag is suited for host
        pointers allocated outside CUDA (ex., via malloc) when it's known that
        no prior operations in the stream can be accessing the memory.
        Specifying this flag allows the driver to optimize the copy on certain
        platforms. Each memcpy operation in the batch must have a valid
        :py:obj:`~.cudaMemcpyAttributes` corresponding to it including the
        appropriate srcAccessOrder setting, otherwise the API will return
        :py:obj:`~.cudaErrorInvalidValue`.

        The :py:obj:`~.cudaMemcpyAttributes.srcLocHint` and
        :py:obj:`~.cudaMemcpyAttributes.dstLocHint` allows applications to
        specify hint locations for operands of a copy when the operand doesn't
        have a fixed location. That is, these hints are only applicable for
        managed memory pointers on devices where
        :py:obj:`~.cudaDevAttrConcurrentManagedAccess` is true or system-
        allocated pageable memory on devices where
        :py:obj:`~.cudaDevAttrPageableMemoryAccess` is true. For other cases,
        these hints are ignored.

        The :py:obj:`~.cudaMemcpyAttributes.flags` field can be used to specify
        certain flags for copies. Setting the
        :py:obj:`~.cudaMemcpyFlagPreferOverlapWithCompute` flag indicates that
        the associated copies should preferably overlap with any compute work.
        Note that this flag is a hint and can be ignored depending on the
        platform and other parameters of the copy.

        If any error is encountered while parsing the batch, the index within
        the batch where the error was encountered will be returned in
        `failIdx`.

        Parameters
        ----------
        dsts : List[Any]
            Array of destination pointers.
        srcs : List[Any]
            Array of memcpy source pointers.
        sizes : List[int]
            Array of sizes for memcpy operations.
        count : size_t
            Size of `dsts`, `srcs` and `sizes` arrays
        attrs : List[:py:obj:`~.cudaMemcpyAttributes`]
            Array of memcpy attributes.
        attrsIdxs : List[int]
            Array of indices to specify which copies each entry in the `attrs`
            array applies to. The attributes specified in attrs[k] will be
            applied to copies starting from attrsIdxs[k] through attrsIdxs[k+1]
            - 1. Also attrs[numAttrs-1] will apply to copies starting from
            attrsIdxs[numAttrs-1] through count - 1.
        numAttrs : size_t
            Size of `attrs` and `attrsIdxs` arrays.
        hStream : :py:obj:`~.CUstream` or :py:obj:`~.cudaStream_t`
            The stream to enqueue the operations in. Must not be legacy NULL
            stream.

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess` :py:obj:`~.cudaErrorInvalidValue`
        failIdx : int
            Pointer to a location to return the index of the copy where a
            failure was encountered. The value will be SIZE_MAX if the error
            doesn't pertain to any specific copy.
    """


def cudaMemcpyFromArray(dst, src, wOffset, hOffset, count, kind: 'cudaMemcpyKind'):
    """
    cudaMemcpyFromArray(dst, src, size_t wOffset, size_t hOffset, size_t count, kind: cudaMemcpyKind)
     Copies data between host and device.

        [Deprecated]

        Copies `count` bytes from the CUDA array `src` starting at `hOffset`
        rows and `wOffset` bytes from the upper left corner to the memory area
        pointed to by `dst`, where `kind` specifies the direction of the copy,
        and must be one of :py:obj:`~.cudaMemcpyHostToHost`,
        :py:obj:`~.cudaMemcpyHostToDevice`, :py:obj:`~.cudaMemcpyDeviceToHost`,
        :py:obj:`~.cudaMemcpyDeviceToDevice`, or :py:obj:`~.cudaMemcpyDefault`.
        Passing :py:obj:`~.cudaMemcpyDefault` is recommended, in which case the
        type of transfer is inferred from the pointer values. However,
        :py:obj:`~.cudaMemcpyDefault` is only allowed on systems that support
        unified virtual addressing.

        Parameters
        ----------
        dst : Any
            Destination memory address
        src : :py:obj:`~.cudaArray_const_t`
            Source memory address
        wOffset : size_t
            Source starting X offset (columns in bytes)
        hOffset : size_t
            Source starting Y offset (rows)
        count : size_t
            Size in bytes to copy
        kind : :py:obj:`~.cudaMemcpyKind`
            Type of transfer

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidMemcpyDirection`

        See Also
        --------
        :py:obj:`~.cudaMemcpy`, :py:obj:`~.cudaMemcpy2D`, :py:obj:`~.cudaMemcpyToArray`, :py:obj:`~.cudaMemcpy2DToArray`, :py:obj:`~.cudaMemcpy2DFromArray`, :py:obj:`~.cudaMemcpyArrayToArray`, :py:obj:`~.cudaMemcpy2DArrayToArray`, :py:obj:`~.cudaMemcpyToSymbol`, :py:obj:`~.cudaMemcpyFromSymbol`, :py:obj:`~.cudaMemcpyAsync`, :py:obj:`~.cudaMemcpy2DAsync`, :py:obj:`~.cudaMemcpyToArrayAsync`, :py:obj:`~.cudaMemcpy2DToArrayAsync`, :py:obj:`~.cudaMemcpyFromArrayAsync`, :py:obj:`~.cudaMemcpy2DFromArrayAsync`, :py:obj:`~.cudaMemcpyToSymbolAsync`, :py:obj:`~.cudaMemcpyFromSymbolAsync`, :py:obj:`~.cuMemcpyAtoH`, :py:obj:`~.cuMemcpyAtoD`
    """


def cudaMemcpyFromArrayAsync(dst, src, wOffset, hOffset, count, kind: 'cudaMemcpyKind', stream):
    """
    cudaMemcpyFromArrayAsync(dst, src, size_t wOffset, size_t hOffset, size_t count, kind: cudaMemcpyKind, stream)
     Copies data between host and device.

        [Deprecated]

        Copies `count` bytes from the CUDA array `src` starting at `hOffset`
        rows and `wOffset` bytes from the upper left corner to the memory area
        pointed to by `dst`, where `kind` specifies the direction of the copy,
        and must be one of :py:obj:`~.cudaMemcpyHostToHost`,
        :py:obj:`~.cudaMemcpyHostToDevice`, :py:obj:`~.cudaMemcpyDeviceToHost`,
        :py:obj:`~.cudaMemcpyDeviceToDevice`, or :py:obj:`~.cudaMemcpyDefault`.
        Passing :py:obj:`~.cudaMemcpyDefault` is recommended, in which case the
        type of transfer is inferred from the pointer values. However,
        :py:obj:`~.cudaMemcpyDefault` is only allowed on systems that support
        unified virtual addressing.

        :py:obj:`~.cudaMemcpyFromArrayAsync()` is asynchronous with respect to
        the host, so the call may return before the copy is complete. The copy
        can optionally be associated to a stream by passing a non-zero `stream`
        argument. If `kind` is :py:obj:`~.cudaMemcpyHostToDevice` or
        :py:obj:`~.cudaMemcpyDeviceToHost` and `stream` is non-zero, the copy
        may overlap with operations in other streams.

        Parameters
        ----------
        dst : Any
            Destination memory address
        src : :py:obj:`~.cudaArray_const_t`
            Source memory address
        wOffset : size_t
            Source starting X offset (columns in bytes)
        hOffset : size_t
            Source starting Y offset (rows)
        count : size_t
            Size in bytes to copy
        kind : :py:obj:`~.cudaMemcpyKind`
            Type of transfer
        stream : :py:obj:`~.CUstream` or :py:obj:`~.cudaStream_t`
            Stream identifier

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidMemcpyDirection`

        See Also
        --------
        :py:obj:`~.cudaMemcpy`, :py:obj:`~.cudaMemcpy2D`, :py:obj:`~.cudaMemcpyToArray`, :py:obj:`~.cudaMemcpy2DToArray`, :py:obj:`~.cudaMemcpyFromArray`, :py:obj:`~.cudaMemcpy2DFromArray`, :py:obj:`~.cudaMemcpyArrayToArray`, :py:obj:`~.cudaMemcpy2DArrayToArray`, :py:obj:`~.cudaMemcpyToSymbol`, :py:obj:`~.cudaMemcpyFromSymbol`, :py:obj:`~.cudaMemcpyAsync`, :py:obj:`~.cudaMemcpy2DAsync`, :py:obj:`~.cudaMemcpyToArrayAsync`, :py:obj:`~.cudaMemcpy2DToArrayAsync`, :py:obj:`~.cudaMemcpy2DFromArrayAsync`, :py:obj:`~.cudaMemcpyToSymbolAsync`, :py:obj:`~.cudaMemcpyFromSymbolAsync`, :py:obj:`~.cuMemcpyAtoHAsync`, :py:obj:`~.cuMemcpy2DAsync`
    """


def cudaMemcpyPeer(dst, dstDevice, src, srcDevice, count):
    """
    cudaMemcpyPeer(dst, int dstDevice, src, int srcDevice, size_t count)
     Copies memory between two devices.

        Copies memory from one device to memory on another device. `dst` is the
        base device pointer of the destination memory and `dstDevice` is the
        destination device. `src` is the base device pointer of the source
        memory and `srcDevice` is the source device. `count` specifies the
        number of bytes to copy.

        Note that this function is asynchronous with respect to the host, but
        serialized with respect all pending and future asynchronous work in to
        the current device, `srcDevice`, and `dstDevice` (use
        :py:obj:`~.cudaMemcpyPeerAsync` to avoid this synchronization).

        Parameters
        ----------
        dst : Any
            Destination device pointer
        dstDevice : int
            Destination device
        src : Any
            Source device pointer
        srcDevice : int
            Source device
        count : size_t
            Size of memory copy in bytes

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidDevice`

        See Also
        --------
        :py:obj:`~.cudaMemcpy`, :py:obj:`~.cudaMemcpyAsync`, :py:obj:`~.cudaMemcpyPeerAsync`, :py:obj:`~.cudaMemcpy3DPeerAsync`, :py:obj:`~.cuMemcpyPeer`
    """


def cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, count, stream):
    """
    cudaMemcpyPeerAsync(dst, int dstDevice, src, int srcDevice, size_t count, stream)
     Copies memory between two devices asynchronously.

        Copies memory from one device to memory on another device. `dst` is the
        base device pointer of the destination memory and `dstDevice` is the
        destination device. `src` is the base device pointer of the source
        memory and `srcDevice` is the source device. `count` specifies the
        number of bytes to copy.

        Note that this function is asynchronous with respect to the host and
        all work on other devices.

        Parameters
        ----------
        dst : Any
            Destination device pointer
        dstDevice : int
            Destination device
        src : Any
            Source device pointer
        srcDevice : int
            Source device
        count : size_t
            Size of memory copy in bytes
        stream : :py:obj:`~.CUstream` or :py:obj:`~.cudaStream_t`
            Stream identifier

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidDevice`

        See Also
        --------
        :py:obj:`~.cudaMemcpy`, :py:obj:`~.cudaMemcpyPeer`, :py:obj:`~.cudaMemcpyAsync`, :py:obj:`~.cudaMemcpy3DPeerAsync`, :py:obj:`~.cuMemcpyPeerAsync`
    """


def cudaMemcpyToArray(dst, wOffset, hOffset, src, count, kind: 'cudaMemcpyKind'):
    """
    cudaMemcpyToArray(dst, size_t wOffset, size_t hOffset, src, size_t count, kind: cudaMemcpyKind)
     Copies data between host and device.

        [Deprecated]

        Copies `count` bytes from the memory area pointed to by `src` to the
        CUDA array `dst` starting at `hOffset` rows and `wOffset` bytes from
        the upper left corner, where `kind` specifies the direction of the
        copy, and must be one of :py:obj:`~.cudaMemcpyHostToHost`,
        :py:obj:`~.cudaMemcpyHostToDevice`, :py:obj:`~.cudaMemcpyDeviceToHost`,
        :py:obj:`~.cudaMemcpyDeviceToDevice`, or :py:obj:`~.cudaMemcpyDefault`.
        Passing :py:obj:`~.cudaMemcpyDefault` is recommended, in which case the
        type of transfer is inferred from the pointer values. However,
        :py:obj:`~.cudaMemcpyDefault` is only allowed on systems that support
        unified virtual addressing.

        Parameters
        ----------
        dst : :py:obj:`~.cudaArray_t`
            Destination memory address
        wOffset : size_t
            Destination starting X offset (columns in bytes)
        hOffset : size_t
            Destination starting Y offset (rows)
        src : Any
            Source memory address
        count : size_t
            Size in bytes to copy
        kind : :py:obj:`~.cudaMemcpyKind`
            Type of transfer

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidMemcpyDirection`

        See Also
        --------
        :py:obj:`~.cudaMemcpy`, :py:obj:`~.cudaMemcpy2D`, :py:obj:`~.cudaMemcpy2DToArray`, :py:obj:`~.cudaMemcpyFromArray`, :py:obj:`~.cudaMemcpy2DFromArray`, :py:obj:`~.cudaMemcpyArrayToArray`, :py:obj:`~.cudaMemcpy2DArrayToArray`, :py:obj:`~.cudaMemcpyToSymbol`, :py:obj:`~.cudaMemcpyFromSymbol`, :py:obj:`~.cudaMemcpyAsync`, :py:obj:`~.cudaMemcpy2DAsync`, :py:obj:`~.cudaMemcpyToArrayAsync`, :py:obj:`~.cudaMemcpy2DToArrayAsync`, :py:obj:`~.cudaMemcpyFromArrayAsync`, :py:obj:`~.cudaMemcpy2DFromArrayAsync`, :py:obj:`~.cudaMemcpyToSymbolAsync`, :py:obj:`~.cudaMemcpyFromSymbolAsync`, :py:obj:`~.cuMemcpyHtoA`, :py:obj:`~.cuMemcpyDtoA`
    """


def cudaMemcpyToArrayAsync(dst, wOffset, hOffset, src, count, kind: 'cudaMemcpyKind', stream):
    """
    cudaMemcpyToArrayAsync(dst, size_t wOffset, size_t hOffset, src, size_t count, kind: cudaMemcpyKind, stream)
     Copies data between host and device.

        [Deprecated]

        Copies `count` bytes from the memory area pointed to by `src` to the
        CUDA array `dst` starting at `hOffset` rows and `wOffset` bytes from
        the upper left corner, where `kind` specifies the direction of the
        copy, and must be one of :py:obj:`~.cudaMemcpyHostToHost`,
        :py:obj:`~.cudaMemcpyHostToDevice`, :py:obj:`~.cudaMemcpyDeviceToHost`,
        :py:obj:`~.cudaMemcpyDeviceToDevice`, or :py:obj:`~.cudaMemcpyDefault`.
        Passing :py:obj:`~.cudaMemcpyDefault` is recommended, in which case the
        type of transfer is inferred from the pointer values. However,
        :py:obj:`~.cudaMemcpyDefault` is only allowed on systems that support
        unified virtual addressing.

        :py:obj:`~.cudaMemcpyToArrayAsync()` is asynchronous with respect to
        the host, so the call may return before the copy is complete. The copy
        can optionally be associated to a stream by passing a non-zero `stream`
        argument. If `kind` is :py:obj:`~.cudaMemcpyHostToDevice` or
        :py:obj:`~.cudaMemcpyDeviceToHost` and `stream` is non-zero, the copy
        may overlap with operations in other streams.

        Parameters
        ----------
        dst : :py:obj:`~.cudaArray_t`
            Destination memory address
        wOffset : size_t
            Destination starting X offset (columns in bytes)
        hOffset : size_t
            Destination starting Y offset (rows)
        src : Any
            Source memory address
        count : size_t
            Size in bytes to copy
        kind : :py:obj:`~.cudaMemcpyKind`
            Type of transfer
        stream : :py:obj:`~.CUstream` or :py:obj:`~.cudaStream_t`
            Stream identifier

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidMemcpyDirection`

        See Also
        --------
        :py:obj:`~.cudaMemcpy`, :py:obj:`~.cudaMemcpy2D`, :py:obj:`~.cudaMemcpyToArray`, :py:obj:`~.cudaMemcpy2DToArray`, :py:obj:`~.cudaMemcpyFromArray`, :py:obj:`~.cudaMemcpy2DFromArray`, :py:obj:`~.cudaMemcpyArrayToArray`, :py:obj:`~.cudaMemcpy2DArrayToArray`, :py:obj:`~.cudaMemcpyToSymbol`, :py:obj:`~.cudaMemcpyFromSymbol`, :py:obj:`~.cudaMemcpyAsync`, :py:obj:`~.cudaMemcpy2DAsync`, :py:obj:`~.cudaMemcpy2DToArrayAsync`, :py:obj:`~.cudaMemcpyFromArrayAsync`, :py:obj:`~.cudaMemcpy2DFromArrayAsync`, :py:obj:`~.cudaMemcpyToSymbolAsync`, :py:obj:`~.cudaMemcpyFromSymbolAsync`, :py:obj:`~.cuMemcpyHtoAAsync`, :py:obj:`~.cuMemcpy2DAsync`
    """


def cudaMemset(devPtr, value, count):
    """
    cudaMemset(devPtr, int value, size_t count)
     Initializes or sets device memory to a value.

        Fills the first `count` bytes of the memory area pointed to by `devPtr`
        with the constant byte value `value`.

        Note that this function is asynchronous with respect to the host unless
        `devPtr` refers to pinned host memory.

        Parameters
        ----------
        devPtr : Any
            Pointer to device memory
        value : int
            Value to set for each byte of specified memory
        count : size_t
            Size in bytes to set

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`,

        See Also
        --------
        :py:obj:`~.cuMemsetD8`, :py:obj:`~.cuMemsetD16`, :py:obj:`~.cuMemsetD32`
    """


def cudaMemset2D(devPtr, pitch, value, width, height):
    """
    cudaMemset2D(devPtr, size_t pitch, int value, size_t width, size_t height)
     Initializes or sets device memory to a value.

        Sets to the specified value `value` a matrix (`height` rows of `width`
        bytes each) pointed to by `dstPtr`. `pitch` is the width in bytes of
        the 2D array pointed to by `dstPtr`, including any padding added to the
        end of each row. This function performs fastest when the pitch is one
        that has been passed back by :py:obj:`~.cudaMallocPitch()`.

        Note that this function is asynchronous with respect to the host unless
        `devPtr` refers to pinned host memory.

        Parameters
        ----------
        devPtr : Any
            Pointer to 2D device memory
        pitch : size_t
            Pitch in bytes of 2D device memory(Unused if `height` is 1)
        value : int
            Value to set for each byte of specified memory
        width : size_t
            Width of matrix set (columns in bytes)
        height : size_t
            Height of matrix set (rows)

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`,

        See Also
        --------
        :py:obj:`~.cudaMemset`, :py:obj:`~.cudaMemset3D`, :py:obj:`~.cudaMemsetAsync`, :py:obj:`~.cudaMemset2DAsync`, :py:obj:`~.cudaMemset3DAsync`, :py:obj:`~.cuMemsetD2D8`, :py:obj:`~.cuMemsetD2D16`, :py:obj:`~.cuMemsetD2D32`
    """


def cudaMemset2DAsync(devPtr, pitch, value, width, height, stream):
    """
    cudaMemset2DAsync(devPtr, size_t pitch, int value, size_t width, size_t height, stream)
     Initializes or sets device memory to a value.

        Sets to the specified value `value` a matrix (`height` rows of `width`
        bytes each) pointed to by `dstPtr`. `pitch` is the width in bytes of
        the 2D array pointed to by `dstPtr`, including any padding added to the
        end of each row. This function performs fastest when the pitch is one
        that has been passed back by :py:obj:`~.cudaMallocPitch()`.

        :py:obj:`~.cudaMemset2DAsync()` is asynchronous with respect to the
        host, so the call may return before the memset is complete. The
        operation can optionally be associated to a stream by passing a non-
        zero `stream` argument. If `stream` is non-zero, the operation may
        overlap with operations in other streams.

        The device version of this function only handles device to device
        copies and cannot be given local or shared pointers.

        Parameters
        ----------
        devPtr : Any
            Pointer to 2D device memory
        pitch : size_t
            Pitch in bytes of 2D device memory(Unused if `height` is 1)
        value : int
            Value to set for each byte of specified memory
        width : size_t
            Width of matrix set (columns in bytes)
        height : size_t
            Height of matrix set (rows)
        stream : :py:obj:`~.CUstream` or :py:obj:`~.cudaStream_t`
            Stream identifier

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`,

        See Also
        --------
        :py:obj:`~.cudaMemset`, :py:obj:`~.cudaMemset2D`, :py:obj:`~.cudaMemset3D`, :py:obj:`~.cudaMemsetAsync`, :py:obj:`~.cudaMemset3DAsync`, :py:obj:`~.cuMemsetD2D8Async`, :py:obj:`~.cuMemsetD2D16Async`, :py:obj:`~.cuMemsetD2D32Async`
    """


def cudaMemset3D(pitchedDevPtr: 'cudaPitchedPtr', value, extent: 'cudaExtent'):
    """
    cudaMemset3D(cudaPitchedPtr pitchedDevPtr: cudaPitchedPtr, int value, cudaExtent extent: cudaExtent)
     Initializes or sets device memory to a value.

        Initializes each element of a 3D array to the specified value `value`.
        The object to initialize is defined by `pitchedDevPtr`. The `pitch`
        field of `pitchedDevPtr` is the width in memory in bytes of the 3D
        array pointed to by `pitchedDevPtr`, including any padding added to the
        end of each row. The `xsize` field specifies the logical width of each
        row in bytes, while the `ysize` field specifies the height of each 2D
        slice in rows. The `pitch` field of `pitchedDevPtr` is ignored when
        `height` and `depth` are both equal to 1.

        The extents of the initialized region are specified as a `width` in
        bytes, a `height` in rows, and a `depth` in slices.

        Extents with `width` greater than or equal to the `xsize` of
        `pitchedDevPtr` may perform significantly faster than extents narrower
        than the `xsize`. Secondarily, extents with `height` equal to the
        `ysize` of `pitchedDevPtr` will perform faster than when the `height`
        is shorter than the `ysize`.

        This function performs fastest when the `pitchedDevPtr` has been
        allocated by :py:obj:`~.cudaMalloc3D()`.

        Note that this function is asynchronous with respect to the host unless
        `pitchedDevPtr` refers to pinned host memory.

        Parameters
        ----------
        pitchedDevPtr : :py:obj:`~.cudaPitchedPtr`
            Pointer to pitched device memory
        value : int
            Value to set for each byte of specified memory
        extent : :py:obj:`~.cudaExtent`
            Size parameters for where to set device memory (`width` field in
            bytes)

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`,

        See Also
        --------
        :py:obj:`~.cudaMemset`, :py:obj:`~.cudaMemset2D`, :py:obj:`~.cudaMemsetAsync`, :py:obj:`~.cudaMemset2DAsync`, :py:obj:`~.cudaMemset3DAsync`, :py:obj:`~.cudaMalloc3D`, :py:obj:`~.make_cudaPitchedPtr`, :py:obj:`~.make_cudaExtent`
    """


def cudaMemset3DAsync(pitchedDevPtr: 'cudaPitchedPtr', value, extent: 'cudaExtent', stream):
    """
    cudaMemset3DAsync(cudaPitchedPtr pitchedDevPtr: cudaPitchedPtr, int value, cudaExtent extent: cudaExtent, stream)
     Initializes or sets device memory to a value.

        Initializes each element of a 3D array to the specified value `value`.
        The object to initialize is defined by `pitchedDevPtr`. The `pitch`
        field of `pitchedDevPtr` is the width in memory in bytes of the 3D
        array pointed to by `pitchedDevPtr`, including any padding added to the
        end of each row. The `xsize` field specifies the logical width of each
        row in bytes, while the `ysize` field specifies the height of each 2D
        slice in rows. The `pitch` field of `pitchedDevPtr` is ignored when
        `height` and `depth` are both equal to 1.

        The extents of the initialized region are specified as a `width` in
        bytes, a `height` in rows, and a `depth` in slices.

        Extents with `width` greater than or equal to the `xsize` of
        `pitchedDevPtr` may perform significantly faster than extents narrower
        than the `xsize`. Secondarily, extents with `height` equal to the
        `ysize` of `pitchedDevPtr` will perform faster than when the `height`
        is shorter than the `ysize`.

        This function performs fastest when the `pitchedDevPtr` has been
        allocated by :py:obj:`~.cudaMalloc3D()`.

        :py:obj:`~.cudaMemset3DAsync()` is asynchronous with respect to the
        host, so the call may return before the memset is complete. The
        operation can optionally be associated to a stream by passing a non-
        zero `stream` argument. If `stream` is non-zero, the operation may
        overlap with operations in other streams.

        The device version of this function only handles device to device
        copies and cannot be given local or shared pointers.

        Parameters
        ----------
        pitchedDevPtr : :py:obj:`~.cudaPitchedPtr`
            Pointer to pitched device memory
        value : int
            Value to set for each byte of specified memory
        extent : :py:obj:`~.cudaExtent`
            Size parameters for where to set device memory (`width` field in
            bytes)
        stream : :py:obj:`~.CUstream` or :py:obj:`~.cudaStream_t`
            Stream identifier

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`,

        See Also
        --------
        :py:obj:`~.cudaMemset`, :py:obj:`~.cudaMemset2D`, :py:obj:`~.cudaMemset3D`, :py:obj:`~.cudaMemsetAsync`, :py:obj:`~.cudaMemset2DAsync`, :py:obj:`~.cudaMalloc3D`, :py:obj:`~.make_cudaPitchedPtr`, :py:obj:`~.make_cudaExtent`
    """


def cudaMemsetAsync(devPtr, value, count, stream):
    """
    cudaMemsetAsync(devPtr, int value, size_t count, stream)
     Initializes or sets device memory to a value.

        Fills the first `count` bytes of the memory area pointed to by `devPtr`
        with the constant byte value `value`.

        :py:obj:`~.cudaMemsetAsync()` is asynchronous with respect to the host,
        so the call may return before the memset is complete. The operation can
        optionally be associated to a stream by passing a non-zero `stream`
        argument. If `stream` is non-zero, the operation may overlap with
        operations in other streams.

        The device version of this function only handles device to device
        copies and cannot be given local or shared pointers.

        Parameters
        ----------
        devPtr : Any
            Pointer to device memory
        value : int
            Value to set for each byte of specified memory
        count : size_t
            Size in bytes to set
        stream : :py:obj:`~.CUstream` or :py:obj:`~.cudaStream_t`
            Stream identifier

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`,

        See Also
        --------
        :py:obj:`~.cudaMemset`, :py:obj:`~.cudaMemset2D`, :py:obj:`~.cudaMemset3D`, :py:obj:`~.cudaMemset2DAsync`, :py:obj:`~.cudaMemset3DAsync`, :py:obj:`~.cuMemsetD8Async`, :py:obj:`~.cuMemsetD16Async`, :py:obj:`~.cuMemsetD32Async`
    """


def cudaMipmappedArrayGetMemoryRequirements(mipmap, device):
    """
    cudaMipmappedArrayGetMemoryRequirements(mipmap, int device)
     Returns the memory requirements of a CUDA mipmapped array.

        Returns the memory requirements of a CUDA mipmapped array in
        `memoryRequirements` If the CUDA mipmapped array is not allocated with
        flag :py:obj:`~.cudaArrayDeferredMapping`
        :py:obj:`~.cudaErrorInvalidValue` will be returned.

        The returned value in :py:obj:`~.cudaArrayMemoryRequirements.size`
        represents the total size of the CUDA mipmapped array. The returned
        value in :py:obj:`~.cudaArrayMemoryRequirements.alignment` represents
        the alignment necessary for mapping the CUDA mipmapped array.

        Parameters
        ----------
        mipmap : :py:obj:`~.cudaMipmappedArray_t`
            CUDA mipmapped array to get the memory requirements of
        device : int
            Device to get the memory requirements for

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess` :py:obj:`~.cudaErrorInvalidValue`
        memoryRequirements : :py:obj:`~.cudaArrayMemoryRequirements`
            Pointer to :py:obj:`~.cudaArrayMemoryRequirements`

        See Also
        --------
        :py:obj:`~.cudaArrayGetMemoryRequirements`
    """


def cudaMipmappedArrayGetSparseProperties(mipmap):
    """
    cudaMipmappedArrayGetSparseProperties(mipmap)
     Returns the layout properties of a sparse CUDA mipmapped array.

        Returns the sparse array layout properties in `sparseProperties`. If
        the CUDA mipmapped array is not allocated with flag
        :py:obj:`~.cudaArraySparse` :py:obj:`~.cudaErrorInvalidValue` will be
        returned.

        For non-layered CUDA mipmapped arrays,
        :py:obj:`~.cudaArraySparseProperties.miptailSize` returns the size of
        the mip tail region. The mip tail region includes all mip levels whose
        width, height or depth is less than that of the tile. For layered CUDA
        mipmapped arrays, if :py:obj:`~.cudaArraySparseProperties.flags`
        contains :py:obj:`~.cudaArraySparsePropertiesSingleMipTail`, then
        :py:obj:`~.cudaArraySparseProperties.miptailSize` specifies the size of
        the mip tail of all layers combined. Otherwise,
        :py:obj:`~.cudaArraySparseProperties.miptailSize` specifies mip tail
        size per layer. The returned value of
        :py:obj:`~.cudaArraySparseProperties.miptailFirstLevel` is valid only
        if :py:obj:`~.cudaArraySparseProperties.miptailSize` is non-zero.

        Parameters
        ----------
        mipmap : :py:obj:`~.cudaMipmappedArray_t`
            The CUDA mipmapped array to get the sparse properties of

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess` :py:obj:`~.cudaErrorInvalidValue`
        sparseProperties : :py:obj:`~.cudaArraySparseProperties`
            Pointer to return :py:obj:`~.cudaArraySparseProperties`

        See Also
        --------
        :py:obj:`~.cudaArrayGetSparseProperties`, :py:obj:`~.cuMemMapArrayAsync`
    """

cudaNvSciSyncAttrSignal: int
cudaNvSciSyncAttrWait: int

def cudaOccupancyAvailableDynamicSMemPerBlock(func, numBlocks, blockSize):
    """
    cudaOccupancyAvailableDynamicSMemPerBlock(func, int numBlocks, int blockSize)
     Returns dynamic shared memory available per block when launching `numBlocks` blocks on SM.

        Returns in `*dynamicSmemSize` the maximum size of dynamic shared memory
        to allow `numBlocks` blocks per SM.

        Parameters
        ----------
        func : Any
            Kernel function for which occupancy is calculated
        numBlocks : int
            Number of blocks to fit on SM
        blockSize : int
            Size of the block

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidDevice`, :py:obj:`~.cudaErrorInvalidDeviceFunction`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorUnknown`,
        dynamicSmemSize : int
            Returned maximum dynamic shared memory

        See Also
        --------
        :py:obj:`~.cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags`, cudaOccupancyMaxPotentialBlockSize (C++ API), cudaOccupancyMaxPotentialBlockSizeWithFlags (C++ API), cudaOccupancyMaxPotentialBlockSizeVariableSMem (C++ API), cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags (C++ API), :py:obj:`~.cudaOccupancyAvailableDynamicSMemPerBlock`
    """

cudaOccupancyDefault: int
cudaOccupancyDisableCachingOverride: int

def cudaOccupancyMaxActiveBlocksPerMultiprocessor(func, blockSize, dynamicSMemSize):
    """
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(func, int blockSize, size_t dynamicSMemSize)
     Returns occupancy for a device function.

        Returns in `*numBlocks` the maximum number of active blocks per
        streaming multiprocessor for the device function.

        Parameters
        ----------
        func : Any
            Kernel function for which occupancy is calculated
        blockSize : int
            Block size the kernel is intended to be launched with
        dynamicSMemSize : size_t
            Per-block dynamic shared memory usage intended, in bytes

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidDevice`, :py:obj:`~.cudaErrorInvalidDeviceFunction`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorUnknown`,
        numBlocks : int
            Returned occupancy

        See Also
        --------
        :py:obj:`~.cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags`, cudaOccupancyMaxPotentialBlockSize (C++ API), cudaOccupancyMaxPotentialBlockSizeWithFlags (C++ API), cudaOccupancyMaxPotentialBlockSizeVariableSMem (C++ API), cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags (C++ API), cudaOccupancyAvailableDynamicSMemPerBlock (C++ API), :py:obj:`~.cuOccupancyMaxActiveBlocksPerMultiprocessor`
    """


def cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(func, blockSize, dynamicSMemSize, flags):
    """
    cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(func, int blockSize, size_t dynamicSMemSize, unsigned int flags)
     Returns occupancy for a device function with the specified flags.

        Returns in `*numBlocks` the maximum number of active blocks per
        streaming multiprocessor for the device function.

        The `flags` parameter controls how special cases are handled. Valid
        flags include:

        - :py:obj:`~.cudaOccupancyDefault`: keeps the default behavior as
          :py:obj:`~.cudaOccupancyMaxActiveBlocksPerMultiprocessor`

        - :py:obj:`~.cudaOccupancyDisableCachingOverride`: This flag suppresses
          the default behavior on platform where global caching affects
          occupancy. On such platforms, if caching is enabled, but per-block SM
          resource usage would result in zero occupancy, the occupancy
          calculator will calculate the occupancy as if caching is disabled.
          Setting this flag makes the occupancy calculator to return 0 in such
          cases. More information can be found about this feature in the
          "Unified L1/Texture Cache" section of the Maxwell tuning guide.

        Parameters
        ----------
        func : Any
            Kernel function for which occupancy is calculated
        blockSize : int
            Block size the kernel is intended to be launched with
        dynamicSMemSize : size_t
            Per-block dynamic shared memory usage intended, in bytes
        flags : unsigned int
            Requested behavior for the occupancy calculator

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidDevice`, :py:obj:`~.cudaErrorInvalidDeviceFunction`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorUnknown`,
        numBlocks : int
            Returned occupancy

        See Also
        --------
        :py:obj:`~.cudaOccupancyMaxActiveBlocksPerMultiprocessor`, cudaOccupancyMaxPotentialBlockSize (C++ API), cudaOccupancyMaxPotentialBlockSizeWithFlags (C++ API), cudaOccupancyMaxPotentialBlockSizeVariableSMem (C++ API), cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags (C++ API), cudaOccupancyAvailableDynamicSMemPerBlock (C++ API), :py:obj:`~.cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags`
    """


def cudaPeekAtLastError():
    """
    cudaPeekAtLastError()
     Returns the last error from a runtime call.

        Returns the last error that has been produced by any of the runtime
        calls in the same instance of the CUDA Runtime library in the host
        thread. This call does not reset the error to :py:obj:`~.cudaSuccess`
        like :py:obj:`~.cudaGetLastError()`.

        Note: Multiple instances of the CUDA Runtime library can be present in
        an application when using a library that statically links the CUDA
        Runtime.

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorMissingConfiguration`, :py:obj:`~.cudaErrorMemoryAllocation`, :py:obj:`~.cudaErrorInitializationError`, :py:obj:`~.cudaErrorLaunchFailure`, :py:obj:`~.cudaErrorLaunchTimeout`, :py:obj:`~.cudaErrorLaunchOutOfResources`, :py:obj:`~.cudaErrorInvalidDeviceFunction`, :py:obj:`~.cudaErrorInvalidConfiguration`, :py:obj:`~.cudaErrorInvalidDevice`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidPitchValue`, :py:obj:`~.cudaErrorInvalidSymbol`, :py:obj:`~.cudaErrorUnmapBufferObjectFailed`, :py:obj:`~.cudaErrorInvalidDevicePointer`, :py:obj:`~.cudaErrorInvalidTexture`, :py:obj:`~.cudaErrorInvalidTextureBinding`, :py:obj:`~.cudaErrorInvalidChannelDescriptor`, :py:obj:`~.cudaErrorInvalidMemcpyDirection`, :py:obj:`~.cudaErrorInvalidFilterSetting`, :py:obj:`~.cudaErrorInvalidNormSetting`, :py:obj:`~.cudaErrorUnknown`, :py:obj:`~.cudaErrorInvalidResourceHandle`, :py:obj:`~.cudaErrorInsufficientDriver`, :py:obj:`~.cudaErrorNoDevice`, :py:obj:`~.cudaErrorSetOnActiveProcess`, :py:obj:`~.cudaErrorStartupFailure`, :py:obj:`~.cudaErrorInvalidPtx`, :py:obj:`~.cudaErrorUnsupportedPtxVersion`, :py:obj:`~.cudaErrorNoKernelImageForDevice`, :py:obj:`~.cudaErrorJitCompilerNotFound`, :py:obj:`~.cudaErrorJitCompilationDisabled`

        See Also
        --------
        :py:obj:`~.cudaGetLastError`, :py:obj:`~.cudaGetErrorName`, :py:obj:`~.cudaGetErrorString`, :py:obj:`~.cudaError`
    """

cudaPeerAccessDefault: int

def cudaPointerGetAttributes(ptr):
    """
    cudaPointerGetAttributes(ptr)
     Returns attributes about a specified pointer.

        Returns in `*attributes` the attributes of the pointer `ptr`. If
        pointer was not allocated in, mapped by or registered with context
        supporting unified addressing :py:obj:`~.cudaErrorInvalidValue` is
        returned.

        The :py:obj:`~.cudaPointerAttributes` structure is defined as:

        **View CUDA Toolkit Documentation for a C++ code example**

        In this structure, the individual fields mean

        - :py:obj:`~.cudaPointerAttributes.type` identifies type of memory. It
          can be :py:obj:`~.cudaMemoryTypeUnregistered` for unregistered host
          memory, :py:obj:`~.cudaMemoryTypeHost` for registered host memory,
          :py:obj:`~.cudaMemoryTypeDevice` for device memory or
          :py:obj:`~.cudaMemoryTypeManaged` for managed memory.

        - :py:obj:`~.device` is the device against which `ptr` was allocated.
          If `ptr` has memory type :py:obj:`~.cudaMemoryTypeDevice` then this
          identifies the device on which the memory referred to by `ptr`
          physically resides. If `ptr` has memory type
          :py:obj:`~.cudaMemoryTypeHost` then this identifies the device which
          was current when the allocation was made (and if that device is
          deinitialized then this allocation will vanish with that device's
          state).

        - :py:obj:`~.devicePointer` is the device pointer alias through which
          the memory referred to by `ptr` may be accessed on the current
          device. If the memory referred to by `ptr` cannot be accessed
          directly by the current device then this is NULL.

        - :py:obj:`~.hostPointer` is the host pointer alias through which the
          memory referred to by `ptr` may be accessed on the host. If the
          memory referred to by `ptr` cannot be accessed directly by the host
          then this is NULL.

        Parameters
        ----------
        ptr : Any
            Pointer to get attributes for

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidDevice`, :py:obj:`~.cudaErrorInvalidValue`
        attributes : :py:obj:`~.cudaPointerAttributes`
            Attributes for the specified pointer

        See Also
        --------
        :py:obj:`~.cudaGetDeviceCount`, :py:obj:`~.cudaGetDevice`, :py:obj:`~.cudaSetDevice`, :py:obj:`~.cudaChooseDevice`, :py:obj:`~.cudaInitDevice`, :py:obj:`~.cuPointerGetAttributes`

        Notes
        -----
        In CUDA 11.0 forward passing host pointer will return :py:obj:`~.cudaMemoryTypeUnregistered` in :py:obj:`~.cudaPointerAttributes.type` and call will return :py:obj:`~.cudaSuccess`.
    """


def cudaProfilerStart():
    """
    cudaProfilerStart()
     Enable profiling.

        Enables profile collection by the active profiling tool for the current
        context. If profiling is already enabled, then
        :py:obj:`~.cudaProfilerStart()` has no effect.

        cudaProfilerStart and cudaProfilerStop APIs are used to
        programmatically control the profiling granularity by allowing
        profiling to be done only on selective pieces of code.

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`

        See Also
        --------
        :py:obj:`~.cudaProfilerStop`, :py:obj:`~.cuProfilerStart`
    """


def cudaProfilerStop():
    """
    cudaProfilerStop()
     Disable profiling.

        Disables profile collection by the active profiling tool for the
        current context. If profiling is already disabled, then
        :py:obj:`~.cudaProfilerStop()` has no effect.

        cudaProfilerStart and cudaProfilerStop APIs are used to
        programmatically control the profiling granularity by allowing
        profiling to be done only on selective pieces of code.

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`

        See Also
        --------
        :py:obj:`~.cudaProfilerStart`, :py:obj:`~.cuProfilerStop`
    """


def cudaRuntimeGetVersion():
    """
    cudaRuntimeGetVersion()
     Returns the CUDA Runtime version.

        Returns in `*runtimeVersion` the version number of the current CUDA
        Runtime instance. The version is returned as (1000 * major + 10 *
        minor). For example, CUDA 9.2 would be represented by 9020.

        As of CUDA 12.0, this function no longer initializes CUDA. The purpose
        of this API is solely to return a compile-time constant stating the
        CUDA Toolkit version in the above format.

        This function automatically returns :py:obj:`~.cudaErrorInvalidValue`
        if the `runtimeVersion` argument is NULL.

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`
        runtimeVersion : int
            Returns the CUDA Runtime version.

        See Also
        --------
        :py:obj:`~.cudaDriverGetVersion`, :py:obj:`~.cuDriverGetVersion`
    """


def cudaSetDevice(device):
    """
    cudaSetDevice(int device)
     Set device to be used for GPU executions.

        Sets `device` as the current device for the calling host thread. Valid
        device id's are 0 to (:py:obj:`~.cudaGetDeviceCount()` - 1).

        Any device memory subsequently allocated from this host thread using
        :py:obj:`~.cudaMalloc()`, :py:obj:`~.cudaMallocPitch()` or
        :py:obj:`~.cudaMallocArray()` will be physically resident on `device`.
        Any host memory allocated from this host thread using
        :py:obj:`~.cudaMallocHost()` or :py:obj:`~.cudaHostAlloc()` or
        :py:obj:`~.cudaHostRegister()` will have its lifetime associated with
        `device`. Any streams or events created from this host thread will be
        associated with `device`. Any kernels launched from this host thread
        using the <<<>>> operator or :py:obj:`~.cudaLaunchKernel()` will be
        executed on `device`.

        This call may be made from any host thread, to any device, and at any
        time. This function will do no synchronization with the previous or new
        device, and should only take significant time when it initializes the
        runtime's context state. This call will bind the primary context of the
        specified device to the calling thread and all the subsequent memory
        allocations, stream and event creations, and kernel launches will be
        associated with the primary context. This function will also
        immediately initialize the runtime state on the primary context, and
        the context will be current on `device` immediately. This function will
        return an error if the device is in
        :py:obj:`~.cudaComputeModeExclusiveProcess` and is occupied by another
        process or if the device is in :py:obj:`~.cudaComputeModeProhibited`.

        It is not required to call :py:obj:`~.cudaInitDevice` before using this
        function.

        Parameters
        ----------
        device : int
            Device on which the active host thread should execute the device
            code.

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidDevice`, :py:obj:`~.cudaErrorDeviceUnavailable`,

        See Also
        --------
        :py:obj:`~.cudaGetDeviceCount`, :py:obj:`~.cudaGetDevice`, :py:obj:`~.cudaGetDeviceProperties`, :py:obj:`~.cudaChooseDevice`, :py:obj:`~.cudaInitDevice`, :py:obj:`~.cuCtxSetCurrent`
    """


def cudaSetDeviceFlags(flags):
    """
    cudaSetDeviceFlags(unsigned int flags)
     Sets flags to be used for device executions.

        Records `flags` as the flags for the current device. If the current
        device has been set and that device has already been initialized, the
        previous flags are overwritten. If the current device has not been
        initialized, it is initialized with the provided flags. If no device
        has been made current to the calling thread, a default device is
        selected and initialized with the provided flags.

        The three LSBs of the `flags` parameter can be used to control how the
        CPU thread interacts with the OS scheduler when waiting for results
        from the device.

        - :py:obj:`~.cudaDeviceScheduleAuto`: The default value if the `flags`
          parameter is zero, uses a heuristic based on the number of active
          CUDA contexts in the process `C` and the number of logical processors
          in the system `P`. If `C` > `P`, then CUDA will yield to other OS
          threads when waiting for the device, otherwise CUDA will not yield
          while waiting for results and actively spin on the processor.
          Additionally, on Tegra devices, :py:obj:`~.cudaDeviceScheduleAuto`
          uses a heuristic based on the power profile of the platform and may
          choose :py:obj:`~.cudaDeviceScheduleBlockingSync` for low-powered
          devices.

        - :py:obj:`~.cudaDeviceScheduleSpin`: Instruct CUDA to actively spin
          when waiting for results from the device. This can decrease latency
          when waiting for the device, but may lower the performance of CPU
          threads if they are performing work in parallel with the CUDA thread.

        - :py:obj:`~.cudaDeviceScheduleYield`: Instruct CUDA to yield its
          thread when waiting for results from the device. This can increase
          latency when waiting for the device, but can increase the performance
          of CPU threads performing work in parallel with the device.

        - :py:obj:`~.cudaDeviceScheduleBlockingSync`: Instruct CUDA to block
          the CPU thread on a synchronization primitive when waiting for the
          device to finish work.

        - :py:obj:`~.cudaDeviceBlockingSync`: Instruct CUDA to block the CPU
          thread on a synchronization primitive when waiting for the device to
          finish work.   :py:obj:`~.Deprecated:` This flag was deprecated as of
          CUDA 4.0 and replaced with
          :py:obj:`~.cudaDeviceScheduleBlockingSync`.

        - :py:obj:`~.cudaDeviceMapHost`: This flag enables allocating pinned
          host memory that is accessible to the device. It is implicit for the
          runtime but may be absent if a context is created using the driver
          API. If this flag is not set, :py:obj:`~.cudaHostGetDevicePointer()`
          will always return a failure code.

        - :py:obj:`~.cudaDeviceLmemResizeToMax`: Instruct CUDA to not reduce
          local memory after resizing local memory for a kernel. This can
          prevent thrashing by local memory allocations when launching many
          kernels with high local memory usage at the cost of potentially
          increased memory usage.   :py:obj:`~.Deprecated:` This flag is
          deprecated and the behavior enabled by this flag is now the default
          and cannot be disabled.

        - :py:obj:`~.cudaDeviceSyncMemops`: Ensures that synchronous memory
          operations initiated on this context will always synchronize. See
          further documentation in the section titled "API Synchronization
          behavior" to learn more about cases when synchronous memory
          operations can exhibit asynchronous behavior.

        Parameters
        ----------
        flags : unsigned int
            Parameters for device operation

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`

        See Also
        --------
        :py:obj:`~.cudaGetDeviceFlags`, :py:obj:`~.cudaGetDeviceCount`, :py:obj:`~.cudaGetDevice`, :py:obj:`~.cudaGetDeviceProperties`, :py:obj:`~.cudaSetDevice`, :py:obj:`~.cudaSetValidDevices`, :py:obj:`~.cudaInitDevice`, :py:obj:`~.cudaChooseDevice`, :py:obj:`~.cuDevicePrimaryCtxSetFlags`
    """


def cudaSignalExternalSemaphoresAsync(extSemArray: 'Optional[Tuple[cudaExternalSemaphore_t] | List[cudaExternalSemaphore_t]]', paramsArray: 'Optional[Tuple[cudaExternalSemaphoreSignalParams] | List[cudaExternalSemaphoreSignalParams]]', numExtSems, stream):
    """
    cudaSignalExternalSemaphoresAsync(extSemArray: Optional[Tuple[cudaExternalSemaphore_t] | List[cudaExternalSemaphore_t]], paramsArray: Optional[Tuple[cudaExternalSemaphoreSignalParams] | List[cudaExternalSemaphoreSignalParams]], unsigned int numExtSems, stream)
     Signals a set of external semaphore objects.

        Enqueues a signal operation on a set of externally allocated semaphore
        object in the specified stream. The operations will be executed when
        all prior operations in the stream complete.

        The exact semantics of signaling a semaphore depends on the type of the
        object.

        If the semaphore object is any one of the following types:
        :py:obj:`~.cudaExternalSemaphoreHandleTypeOpaqueFd`,
        :py:obj:`~.cudaExternalSemaphoreHandleTypeOpaqueWin32`,
        :py:obj:`~.cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt` then
        signaling the semaphore will set it to the signaled state.

        If the semaphore object is any one of the following types:
        :py:obj:`~.cudaExternalSemaphoreHandleTypeD3D12Fence`,
        :py:obj:`~.cudaExternalSemaphoreHandleTypeD3D11Fence`,
        :py:obj:`~.cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd`,
        :py:obj:`~.cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32` then
        the semaphore will be set to the value specified in
        :py:obj:`~.cudaExternalSemaphoreSignalParams`::params::fence::value.

        If the semaphore object is of the type
        :py:obj:`~.cudaExternalSemaphoreHandleTypeNvSciSync` this API sets
        :py:obj:`~.cudaExternalSemaphoreSignalParams`::params::nvSciSync::fence
        to a value that can be used by subsequent waiters of the same NvSciSync
        object to order operations with those currently submitted in `stream`.
        Such an update will overwrite previous contents of
        :py:obj:`~.cudaExternalSemaphoreSignalParams`::params::nvSciSync::fence.
        By default, signaling such an external semaphore object causes
        appropriate memory synchronization operations to be performed over all
        the external memory objects that are imported as
        :py:obj:`~.cudaExternalMemoryHandleTypeNvSciBuf`. This ensures that any
        subsequent accesses made by other importers of the same set of NvSciBuf
        memory object(s) are coherent. These operations can be skipped by
        specifying the flag
        :py:obj:`~.cudaExternalSemaphoreSignalSkipNvSciBufMemSync`, which can
        be used as a performance optimization when data coherency is not
        required. But specifying this flag in scenarios where data coherency is
        required results in undefined behavior. Also, for semaphore object of
        the type :py:obj:`~.cudaExternalSemaphoreHandleTypeNvSciSync`, if the
        NvSciSyncAttrList used to create the NvSciSyncObj had not set the flags
        in :py:obj:`~.cudaDeviceGetNvSciSyncAttributes` to
        cudaNvSciSyncAttrSignal, this API will return cudaErrorNotSupported.

        :py:obj:`~.cudaExternalSemaphoreSignalParams`::params::nvSciSync::fence
        associated with semaphore object of the type
        :py:obj:`~.cudaExternalSemaphoreHandleTypeNvSciSync` can be
        deterministic. For this the NvSciSyncAttrList used to create the
        semaphore object must have value of
        NvSciSyncAttrKey_RequireDeterministicFences key set to true.
        Deterministic fences allow users to enqueue a wait over the semaphore
        object even before corresponding signal is enqueued. For such a
        semaphore object, CUDA guarantees that each signal operation will
        increment the fence value by '1'. Users are expected to track count of
        signals enqueued on the semaphore object and insert waits accordingly.
        When such a semaphore object is signaled from multiple streams, due to
        concurrent stream execution, it is possible that the order in which the
        semaphore gets signaled is indeterministic. This could lead to waiters
        of the semaphore getting unblocked incorrectly. Users are expected to
        handle such situations, either by not using the same semaphore object
        with deterministic fence support enabled in different streams or by
        adding explicit dependency amongst such streams so that the semaphore
        is signaled in order.

        If the semaphore object is any one of the following types:
        :py:obj:`~.cudaExternalSemaphoreHandleTypeKeyedMutex`,
        :py:obj:`~.cudaExternalSemaphoreHandleTypeKeyedMutexKmt`, then the
        keyed mutex will be released with the key specified in
        :py:obj:`~.cudaExternalSemaphoreSignalParams`::params::keyedmutex::key.

        Parameters
        ----------
        extSemArray : List[:py:obj:`~.cudaExternalSemaphore_t`]
            Set of external semaphores to be signaled
        paramsArray : List[:py:obj:`~.cudaExternalSemaphoreSignalParams`]
            Array of semaphore parameters
        numExtSems : unsigned int
            Number of semaphores to signal
        stream : :py:obj:`~.CUstream` or :py:obj:`~.cudaStream_t`
            Stream to enqueue the signal operations in

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidResourceHandle`

        See Also
        --------
        :py:obj:`~.cudaImportExternalSemaphore`, :py:obj:`~.cudaDestroyExternalSemaphore`, :py:obj:`~.cudaWaitExternalSemaphoresAsync`
    """


def cudaStreamAddCallback(stream, callback, userData, flags):
    """
    cudaStreamAddCallback(stream, callback, userData, unsigned int flags)
     Add a callback to a compute stream.

        Adds a callback to be called on the host after all currently enqueued
        items in the stream have completed. For each cudaStreamAddCallback
        call, a callback will be executed exactly once. The callback will block
        later work in the stream until it is finished.

        The callback may be passed :py:obj:`~.cudaSuccess` or an error code. In
        the event of a device error, all subsequently executed callbacks will
        receive an appropriate :py:obj:`~.cudaError_t`.

        Callbacks must not make any CUDA API calls. Attempting to use CUDA APIs
        may result in :py:obj:`~.cudaErrorNotPermitted`. Callbacks must not
        perform any synchronization that may depend on outstanding device work
        or other callbacks that are not mandated to run earlier. Callbacks
        without a mandated order (in independent streams) execute in undefined
        order and may be serialized.

        For the purposes of Unified Memory, callback execution makes a number
        of guarantees:

        - The callback stream is considered idle for the duration of the
          callback. Thus, for example, a callback may always use memory
          attached to the callback stream.

        - The start of execution of a callback has the same effect as
          synchronizing an event recorded in the same stream immediately prior
          to the callback. It thus synchronizes streams which have been
          "joined" prior to the callback.

        - Adding device work to any stream does not have the effect of making
          the stream active until all preceding callbacks have executed. Thus,
          for example, a callback might use global attached memory even if work
          has been added to another stream, if it has been properly ordered
          with an event.

        - Completion of a callback does not cause a stream to become active
          except as described above. The callback stream will remain idle if no
          device work follows the callback, and will remain idle across
          consecutive callbacks without device work in between. Thus, for
          example, stream synchronization can be done by signaling from a
          callback at the end of the stream.

        Parameters
        ----------
        stream : :py:obj:`~.CUstream` or :py:obj:`~.cudaStream_t`
            Stream to add callback to
        callback : :py:obj:`~.cudaStreamCallback_t`
            The function to call once preceding stream operations are complete
        userData : Any
            User specified data to be passed to the callback function
        flags : unsigned int
            Reserved for future use, must be 0

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidResourceHandle`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorNotSupported`

        See Also
        --------
        :py:obj:`~.cudaStreamCreate`, :py:obj:`~.cudaStreamCreateWithFlags`, :py:obj:`~.cudaStreamQuery`, :py:obj:`~.cudaStreamSynchronize`, :py:obj:`~.cudaStreamWaitEvent`, :py:obj:`~.cudaStreamDestroy`, :py:obj:`~.cudaMallocManaged`, :py:obj:`~.cudaStreamAttachMemAsync`, :py:obj:`~.cudaLaunchHostFunc`, :py:obj:`~.cuStreamAddCallback`

        Notes
        -----
        This function is slated for eventual deprecation and removal. If you do not require the callback to execute in case of a device error, consider using :py:obj:`~.cudaLaunchHostFunc`. Additionally, this function is not supported with :py:obj:`~.cudaStreamBeginCapture` and :py:obj:`~.cudaStreamEndCapture`, unlike :py:obj:`~.cudaLaunchHostFunc`.
    """


def cudaStreamAttachMemAsync(stream, devPtr, length, flags):
    """
    cudaStreamAttachMemAsync(stream, devPtr, size_t length, unsigned int flags)
     Attach memory to a stream asynchronously.

        Enqueues an operation in `stream` to specify stream association of
        `length` bytes of memory starting from `devPtr`. This function is a
        stream-ordered operation, meaning that it is dependent on, and will
        only take effect when, previous work in stream has completed. Any
        previous association is automatically replaced.

        `devPtr` must point to an one of the following types of memories:

        - managed memory declared using the managed keyword or allocated with
          :py:obj:`~.cudaMallocManaged`.

        - a valid host-accessible region of system-allocated pageable memory.
          This type of memory may only be specified if the device associated
          with the stream reports a non-zero value for the device attribute
          :py:obj:`~.cudaDevAttrPageableMemoryAccess`.

        For managed allocations, `length` must be either zero or the entire
        allocation's size. Both indicate that the entire allocation's stream
        association is being changed. Currently, it is not possible to change
        stream association for a portion of a managed allocation.

        For pageable allocations, `length` must be non-zero.

        The stream association is specified using `flags` which must be one of
        :py:obj:`~.cudaMemAttachGlobal`, :py:obj:`~.cudaMemAttachHost` or
        :py:obj:`~.cudaMemAttachSingle`. The default value for `flags` is
        :py:obj:`~.cudaMemAttachSingle` If the :py:obj:`~.cudaMemAttachGlobal`
        flag is specified, the memory can be accessed by any stream on any
        device. If the :py:obj:`~.cudaMemAttachHost` flag is specified, the
        program makes a guarantee that it won't access the memory on the device
        from any stream on a device that has a zero value for the device
        attribute :py:obj:`~.cudaDevAttrConcurrentManagedAccess`. If the
        :py:obj:`~.cudaMemAttachSingle` flag is specified and `stream` is
        associated with a device that has a zero value for the device attribute
        :py:obj:`~.cudaDevAttrConcurrentManagedAccess`, the program makes a
        guarantee that it will only access the memory on the device from
        `stream`. It is illegal to attach singly to the NULL stream, because
        the NULL stream is a virtual global stream and not a specific stream.
        An error will be returned in this case.

        When memory is associated with a single stream, the Unified Memory
        system will allow CPU access to this memory region so long as all
        operations in `stream` have completed, regardless of whether other
        streams are active. In effect, this constrains exclusive ownership of
        the managed memory region by an active GPU to per-stream activity
        instead of whole-GPU activity.

        Accessing memory on the device from streams that are not associated
        with it will produce undefined results. No error checking is performed
        by the Unified Memory system to ensure that kernels launched into other
        streams do not access this region.

        It is a program's responsibility to order calls to
        :py:obj:`~.cudaStreamAttachMemAsync` via events, synchronization or
        other means to ensure legal access to memory at all times. Data
        visibility and coherency will be changed appropriately for all kernels
        which follow a stream-association change.

        If `stream` is destroyed while data is associated with it, the
        association is removed and the association reverts to the default
        visibility of the allocation as specified at
        :py:obj:`~.cudaMallocManaged`. For managed variables, the default
        association is always :py:obj:`~.cudaMemAttachGlobal`. Note that
        destroying a stream is an asynchronous operation, and as a result, the
        change to default association won't happen until all work in the stream
        has completed.

        Parameters
        ----------
        stream : :py:obj:`~.CUstream` or :py:obj:`~.cudaStream_t`
            Stream in which to enqueue the attach operation
        devPtr : Any
            Pointer to memory (must be a pointer to managed memory or to a
            valid host-accessible region of system-allocated memory)
        length : size_t
            Length of memory (defaults to zero)
        flags : unsigned int
            Must be one of :py:obj:`~.cudaMemAttachGlobal`,
            :py:obj:`~.cudaMemAttachHost` or :py:obj:`~.cudaMemAttachSingle`
            (defaults to :py:obj:`~.cudaMemAttachSingle`)

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorNotReady`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidResourceHandle`

        See Also
        --------
        :py:obj:`~.cudaStreamCreate`, :py:obj:`~.cudaStreamCreateWithFlags`, :py:obj:`~.cudaStreamWaitEvent`, :py:obj:`~.cudaStreamSynchronize`, :py:obj:`~.cudaStreamAddCallback`, :py:obj:`~.cudaStreamDestroy`, :py:obj:`~.cudaMallocManaged`, :py:obj:`~.cuStreamAttachMemAsync`
    """

cudaStreamAttributeAccessPolicyWindow: int
cudaStreamAttributeMemSyncDomain: int
cudaStreamAttributeMemSyncDomainMap: int
cudaStreamAttributePriority: int
cudaStreamAttributeSynchronizationPolicy: int

def cudaStreamBeginCapture(stream, mode: 'cudaStreamCaptureMode'):
    """
    cudaStreamBeginCapture(stream, mode: cudaStreamCaptureMode)
     Begins graph capture on a stream.

        Begin graph capture on `stream`. When a stream is in capture mode, all
        operations pushed into the stream will not be executed, but will
        instead be captured into a graph, which will be returned via
        :py:obj:`~.cudaStreamEndCapture`. Capture may not be initiated if
        `stream` is :py:obj:`~.cudaStreamLegacy`. Capture must be ended on the
        same stream in which it was initiated, and it may only be initiated if
        the stream is not already in capture mode. The capture mode may be
        queried via :py:obj:`~.cudaStreamIsCapturing`. A unique id representing
        the capture sequence may be queried via
        :py:obj:`~.cudaStreamGetCaptureInfo`.

        If `mode` is not :py:obj:`~.cudaStreamCaptureModeRelaxed`,
        :py:obj:`~.cudaStreamEndCapture` must be called on this stream from the
        same thread.

        Parameters
        ----------
        stream : :py:obj:`~.CUstream` or :py:obj:`~.cudaStream_t`
            Stream in which to initiate capture
        mode : :py:obj:`~.cudaStreamCaptureMode`
            Controls the interaction of this capture sequence with other API
            calls that are potentially unsafe. For more details see
            :py:obj:`~.cudaThreadExchangeStreamCaptureMode`.

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`

        See Also
        --------
        :py:obj:`~.cudaStreamCreate`, :py:obj:`~.cudaStreamIsCapturing`, :py:obj:`~.cudaStreamEndCapture`, :py:obj:`~.cudaThreadExchangeStreamCaptureMode`

        Notes
        -----
        Kernels captured using this API must not use texture and surface references. Reading or writing through any texture or surface reference is undefined behavior. This restriction does not apply to texture and surface objects.
    """


def cudaStreamBeginCaptureToGraph(stream, graph, dependencies: 'Optional[Tuple[cudaGraphNode_t] | List[cudaGraphNode_t]]', dependencyData: 'Optional[Tuple[cudaGraphEdgeData] | List[cudaGraphEdgeData]]', numDependencies, mode: 'cudaStreamCaptureMode'):
    """
    cudaStreamBeginCaptureToGraph(stream, graph, dependencies: Optional[Tuple[cudaGraphNode_t] | List[cudaGraphNode_t]], dependencyData: Optional[Tuple[cudaGraphEdgeData] | List[cudaGraphEdgeData]], size_t numDependencies, mode: cudaStreamCaptureMode)
     Begins graph capture on a stream to an existing graph.

        Begin graph capture on `stream`. When a stream is in capture mode, all
        operations pushed into the stream will not be executed, but will
        instead be captured into `graph`, which will be returned via
        :py:obj:`~.cudaStreamEndCapture`.

        Capture may not be initiated if `stream` is
        :py:obj:`~.cudaStreamLegacy`. Capture must be ended on the same stream
        in which it was initiated, and it may only be initiated if the stream
        is not already in capture mode. The capture mode may be queried via
        :py:obj:`~.cudaStreamIsCapturing`. A unique id representing the capture
        sequence may be queried via :py:obj:`~.cudaStreamGetCaptureInfo`.

        If `mode` is not :py:obj:`~.cudaStreamCaptureModeRelaxed`,
        :py:obj:`~.cudaStreamEndCapture` must be called on this stream from the
        same thread.

        Parameters
        ----------
        stream : :py:obj:`~.CUstream` or :py:obj:`~.cudaStream_t`
            Stream in which to initiate capture.
        graph : :py:obj:`~.CUgraph` or :py:obj:`~.cudaGraph_t`
            Graph to capture into.
        dependencies : List[:py:obj:`~.cudaGraphNode_t`]
            Dependencies of the first node captured in the stream. Can be NULL
            if numDependencies is 0.
        dependencyData : List[:py:obj:`~.cudaGraphEdgeData`]
            Optional array of data associated with each dependency.
        numDependencies : size_t
            Number of dependencies.
        mode : :py:obj:`~.cudaStreamCaptureMode`
            Controls the interaction of this capture sequence with other API
            calls that are potentially unsafe. For more details see
            :py:obj:`~.cudaThreadExchangeStreamCaptureMode`.

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`

        See Also
        --------
        :py:obj:`~.cudaStreamCreate`, :py:obj:`~.cudaStreamIsCapturing`, :py:obj:`~.cudaStreamEndCapture`, :py:obj:`~.cudaThreadExchangeStreamCaptureMode`

        Notes
        -----
        Kernels captured using this API must not use texture and surface references. Reading or writing through any texture or surface reference is undefined behavior. This restriction does not apply to texture and surface objects.
    """


def cudaStreamCopyAttributes(dst, src):
    """
    cudaStreamCopyAttributes(dst, src)
     Copies attributes from source stream to destination stream.

        Copies attributes from source stream `src` to destination stream `dst`.
        Both streams must have the same context.

        Parameters
        ----------
        dst : :py:obj:`~.CUstream` or :py:obj:`~.cudaStream_t`
            Destination stream
        src : :py:obj:`~.CUstream` or :py:obj:`~.cudaStream_t`
            Source stream For attributes see :py:obj:`~.cudaStreamAttrID`

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorNotSupported`

        See Also
        --------
        :py:obj:`~.cudaAccessPolicyWindow`
    """


def cudaStreamCreate():
    """
    cudaStreamCreate()
     Create an asynchronous stream.

        Creates a new asynchronous stream on the context that is current to the
        calling host thread. If no context is current to the calling host
        thread, then the primary context for a device is selected, made current
        to the calling thread, and initialized before creating a stream on it.

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`
        pStream : :py:obj:`~.cudaStream_t`
            Pointer to new stream identifier

        See Also
        --------
        :py:obj:`~.cudaStreamCreateWithPriority`, :py:obj:`~.cudaStreamCreateWithFlags`, :py:obj:`~.cudaStreamGetPriority`, :py:obj:`~.cudaStreamGetFlags`, :py:obj:`~.cudaStreamGetDevice`, :py:obj:`~.cudaStreamQuery`, :py:obj:`~.cudaStreamSynchronize`, :py:obj:`~.cudaStreamWaitEvent`, :py:obj:`~.cudaStreamAddCallback`, :py:obj:`~.cudaSetDevice`, :py:obj:`~.cudaStreamDestroy`, :py:obj:`~.cuStreamCreate`
    """


def cudaStreamCreateWithFlags(flags):
    """
    cudaStreamCreateWithFlags(unsigned int flags)
     Create an asynchronous stream.

        Creates a new asynchronous stream on the context that is current to the
        calling host thread. If no context is current to the calling host
        thread, then the primary context for a device is selected, made current
        to the calling thread, and initialized before creating a stream on it.
        The `flags` argument determines the behaviors of the stream. Valid
        values for `flags` are

        - :py:obj:`~.cudaStreamDefault`: Default stream creation flag.

        - :py:obj:`~.cudaStreamNonBlocking`: Specifies that work running in the
          created stream may run concurrently with work in stream 0 (the NULL
          stream), and that the created stream should perform no implicit
          synchronization with stream 0.

        Parameters
        ----------
        flags : unsigned int
            Parameters for stream creation

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`
        pStream : :py:obj:`~.cudaStream_t`
            Pointer to new stream identifier

        See Also
        --------
        :py:obj:`~.cudaStreamCreate`, :py:obj:`~.cudaStreamCreateWithPriority`, :py:obj:`~.cudaStreamGetFlags`, :py:obj:`~.cudaStreamGetDevice`, :py:obj:`~.cudaStreamQuery`, :py:obj:`~.cudaStreamSynchronize`, :py:obj:`~.cudaStreamWaitEvent`, :py:obj:`~.cudaStreamAddCallback`, :py:obj:`~.cudaSetDevice`, :py:obj:`~.cudaStreamDestroy`, :py:obj:`~.cuStreamCreate`
    """


def cudaStreamCreateWithPriority(flags, priority):
    """
    cudaStreamCreateWithPriority(unsigned int flags, int priority)
     Create an asynchronous stream with the specified priority.

        Creates a stream with the specified priority and returns a handle in
        `pStream`. The stream is created on the context that is current to the
        calling host thread. If no context is current to the calling host
        thread, then the primary context for a device is selected, made current
        to the calling thread, and initialized before creating a stream on it.
        This affects the scheduling priority of work in the stream. Priorities
        provide a hint to preferentially run work with higher priority when
        possible, but do not preempt already-running work or provide any other
        functional guarantee on execution order.

        `priority` follows a convention where lower numbers represent higher
        priorities. '0' represents default priority. The range of meaningful
        numerical priorities can be queried using
        :py:obj:`~.cudaDeviceGetStreamPriorityRange`. If the specified priority
        is outside the numerical range returned by
        :py:obj:`~.cudaDeviceGetStreamPriorityRange`, it will automatically be
        clamped to the lowest or the highest number in the range.

        Parameters
        ----------
        flags : unsigned int
            Flags for stream creation. See
            :py:obj:`~.cudaStreamCreateWithFlags` for a list of valid flags
            that can be passed
        priority : int
            Priority of the stream. Lower numbers represent higher priorities.
            See :py:obj:`~.cudaDeviceGetStreamPriorityRange` for more
            information about the meaningful stream priorities that can be
            passed.

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`
        pStream : :py:obj:`~.cudaStream_t`
            Pointer to new stream identifier

        See Also
        --------
        :py:obj:`~.cudaStreamCreate`, :py:obj:`~.cudaStreamCreateWithFlags`, :py:obj:`~.cudaDeviceGetStreamPriorityRange`, :py:obj:`~.cudaStreamGetPriority`, :py:obj:`~.cudaStreamQuery`, :py:obj:`~.cudaStreamWaitEvent`, :py:obj:`~.cudaStreamAddCallback`, :py:obj:`~.cudaStreamSynchronize`, :py:obj:`~.cudaSetDevice`, :py:obj:`~.cudaStreamDestroy`, :py:obj:`~.cuStreamCreateWithPriority`

        Notes
        -----
        Stream priorities are supported only on GPUs with compute capability 3.5 or higher.

        In the current implementation, only compute kernels launched in priority streams are affected by the stream's priority. Stream priorities have no effect on host-to-device and device-to-host memory operations.
    """

cudaStreamDefault: int

def cudaStreamDestroy(stream):
    """
    cudaStreamDestroy(stream)
     Destroys and cleans up an asynchronous stream.

        Destroys and cleans up the asynchronous stream specified by `stream`.

        In case the device is still doing work in the stream `stream` when
        :py:obj:`~.cudaStreamDestroy()` is called, the function will return
        immediately and the resources associated with `stream` will be released
        automatically once the device has completed all work in `stream`.

        Parameters
        ----------
        stream : :py:obj:`~.CUstream` or :py:obj:`~.cudaStream_t`
            Stream identifier

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidResourceHandle`

        See Also
        --------
        :py:obj:`~.cudaStreamCreate`, :py:obj:`~.cudaStreamCreateWithFlags`, :py:obj:`~.cudaStreamQuery`, :py:obj:`~.cudaStreamWaitEvent`, :py:obj:`~.cudaStreamSynchronize`, :py:obj:`~.cudaStreamAddCallback`, :py:obj:`~.cuStreamDestroy`
    """


def cudaStreamEndCapture(stream):
    """
    cudaStreamEndCapture(stream)
     Ends capture on a stream, returning the captured graph.

        End capture on `stream`, returning the captured graph via `pGraph`.
        Capture must have been initiated on `stream` via a call to
        :py:obj:`~.cudaStreamBeginCapture`. If capture was invalidated, due to
        a violation of the rules of stream capture, then a NULL graph will be
        returned.

        If the `mode` argument to :py:obj:`~.cudaStreamBeginCapture` was not
        :py:obj:`~.cudaStreamCaptureModeRelaxed`, this call must be from the
        same thread as :py:obj:`~.cudaStreamBeginCapture`.

        Parameters
        ----------
        stream : :py:obj:`~.CUstream` or :py:obj:`~.cudaStream_t`
            Stream to query

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorStreamCaptureWrongThread`
        pGraph : :py:obj:`~.cudaGraph_t`
            The captured graph

        See Also
        --------
        :py:obj:`~.cudaStreamCreate`, :py:obj:`~.cudaStreamBeginCapture`, :py:obj:`~.cudaStreamIsCapturing`, :py:obj:`~.cudaGraphDestroy`
    """


def cudaStreamGetAttribute(hStream, attr: 'cudaStreamAttrID'):
    """
    cudaStreamGetAttribute(hStream, attr: cudaStreamAttrID)
     Queries stream attribute.

        Queries attribute `attr` from `hStream` and stores it in corresponding
        member of `value_out`.

        Parameters
        ----------
        hStream : :py:obj:`~.CUstream` or :py:obj:`~.cudaStream_t`

        attr : :py:obj:`~.cudaStreamAttrID`


        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidResourceHandle`
        value_out : :py:obj:`~.cudaStreamAttrValue`


        See Also
        --------
        :py:obj:`~.cudaAccessPolicyWindow`
    """


def cudaStreamGetCaptureInfo(stream):
    """
    cudaStreamGetCaptureInfo(stream)
     Query a stream's capture state.

        Query stream state related to stream capture.

        If called on :py:obj:`~.cudaStreamLegacy` (the "null stream") while a
        stream not created with :py:obj:`~.cudaStreamNonBlocking` is capturing,
        returns :py:obj:`~.cudaErrorStreamCaptureImplicit`.

        Valid data (other than capture status) is returned only if both of the
        following are true:

        - the call returns cudaSuccess

        - the returned capture status is
          :py:obj:`~.cudaStreamCaptureStatusActive`

        Parameters
        ----------
        stream : :py:obj:`~.CUstream` or :py:obj:`~.cudaStream_t`
            The stream to query

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorStreamCaptureImplicit`
        captureStatus_out : :py:obj:`~.cudaStreamCaptureStatus`
            Location to return the capture status of the stream; required
        id_out : unsigned long long
            Optional location to return an id for the capture sequence, which
            is unique over the lifetime of the process
        graph_out : :py:obj:`~.cudaGraph_t`
            Optional location to return the graph being captured into. All
            operations other than destroy and node removal are permitted on the
            graph while the capture sequence is in progress. This API does not
            transfer ownership of the graph, which is transferred or destroyed
            at :py:obj:`~.cudaStreamEndCapture`. Note that the graph handle may
            be invalidated before end of capture for certain errors. Nodes that
            are or become unreachable from the original stream at
            :py:obj:`~.cudaStreamEndCapture` due to direct actions on the graph
            do not trigger :py:obj:`~.cudaErrorStreamCaptureUnjoined`.
        dependencies_out : List[:py:obj:`~.cudaGraphNode_t`]
            Optional location to store a pointer to an array of nodes. The next
            node to be captured in the stream will depend on this set of nodes,
            absent operations such as event wait which modify this set. The
            array pointer is valid until the next API call which operates on
            the stream or until the capture is terminated. The node handles may
            be copied out and are valid until they or the graph is destroyed.
            The driver-owned array may also be passed directly to APIs that
            operate on the graph (not the stream) without copying.
        numDependencies_out : int
            Optional location to store the size of the array returned in
            dependencies_out.

        See Also
        --------
        :py:obj:`~.cudaStreamGetCaptureInfo_v3`, :py:obj:`~.cudaStreamBeginCapture`, :py:obj:`~.cudaStreamIsCapturing`, :py:obj:`~.cudaStreamUpdateCaptureDependencies`
    """


def cudaStreamGetCaptureInfo_v3(stream):
    """
    cudaStreamGetCaptureInfo_v3(stream)
     Query a stream's capture state (12.3+)

        Query stream state related to stream capture.

        If called on :py:obj:`~.cudaStreamLegacy` (the "null stream") while a
        stream not created with :py:obj:`~.cudaStreamNonBlocking` is capturing,
        returns :py:obj:`~.cudaErrorStreamCaptureImplicit`.

        Valid data (other than capture status) is returned only if both of the
        following are true:

        - the call returns cudaSuccess

        - the returned capture status is
          :py:obj:`~.cudaStreamCaptureStatusActive`

        If `edgeData_out` is non-NULL then `dependencies_out` must be as well.
        If `dependencies_out` is non-NULL and `edgeData_out` is NULL, but there
        is non-zero edge data for one or more of the current stream
        dependencies, the call will return :py:obj:`~.cudaErrorLossyQuery`.

        Parameters
        ----------
        stream : :py:obj:`~.CUstream` or :py:obj:`~.cudaStream_t`
            The stream to query

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorStreamCaptureImplicit`, :py:obj:`~.cudaErrorLossyQuery`
        captureStatus_out : :py:obj:`~.cudaStreamCaptureStatus`
            Location to return the capture status of the stream; required
        id_out : unsigned long long
            Optional location to return an id for the capture sequence, which
            is unique over the lifetime of the process
        graph_out : :py:obj:`~.cudaGraph_t`
            Optional location to return the graph being captured into. All
            operations other than destroy and node removal are permitted on the
            graph while the capture sequence is in progress. This API does not
            transfer ownership of the graph, which is transferred or destroyed
            at :py:obj:`~.cudaStreamEndCapture`. Note that the graph handle may
            be invalidated before end of capture for certain errors. Nodes that
            are or become unreachable from the original stream at
            :py:obj:`~.cudaStreamEndCapture` due to direct actions on the graph
            do not trigger :py:obj:`~.cudaErrorStreamCaptureUnjoined`.
        dependencies_out : List[:py:obj:`~.cudaGraphNode_t`]
            Optional location to store a pointer to an array of nodes. The next
            node to be captured in the stream will depend on this set of nodes,
            absent operations such as event wait which modify this set. The
            array pointer is valid until the next API call which operates on
            the stream or until the capture is terminated. The node handles may
            be copied out and are valid until they or the graph is destroyed.
            The driver-owned array may also be passed directly to APIs that
            operate on the graph (not the stream) without copying.
        edgeData_out : List[:py:obj:`~.cudaGraphEdgeData`]
            Optional location to store a pointer to an array of graph edge
            data. This array parallels `dependencies_out`; the next node to be
            added has an edge to `dependencies_out`[i] with annotation
            `edgeData_out`[i] for each `i`. The array pointer is valid until
            the next API call which operates on the stream or until the capture
            is terminated.
        numDependencies_out : int
            Optional location to store the size of the array returned in
            dependencies_out.

        See Also
        --------
        :py:obj:`~.cudaStreamBeginCapture`, :py:obj:`~.cudaStreamIsCapturing`, :py:obj:`~.cudaStreamUpdateCaptureDependencies`
    """


def cudaStreamGetDevice(hStream):
    """
    cudaStreamGetDevice(hStream)
     Query the device of a stream.

        Returns in `*device` the device of the stream.

        Parameters
        ----------
        hStream : :py:obj:`~.CUstream` or :py:obj:`~.cudaStream_t`
            Handle to the stream to be queried

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorDeviceUnavailable`,
        device : int
            Returns the device to which the stream belongs

        See Also
        --------
        :py:obj:`~.cudaSetDevice`, :py:obj:`~.cudaGetDevice`, :py:obj:`~.cudaStreamCreate`, :py:obj:`~.cudaStreamGetPriority`, :py:obj:`~.cudaStreamGetFlags`, :py:obj:`~.cuStreamGetId`
    """


def cudaStreamGetFlags(hStream):
    """
    cudaStreamGetFlags(hStream)
     Query the flags of a stream.

        Query the flags of a stream. The flags are returned in `flags`. See
        :py:obj:`~.cudaStreamCreateWithFlags` for a list of valid flags.

        Parameters
        ----------
        hStream : :py:obj:`~.CUstream` or :py:obj:`~.cudaStream_t`
            Handle to the stream to be queried

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidResourceHandle`
        flags : unsigned int
            Pointer to an unsigned integer in which the stream's flags are
            returned

        See Also
        --------
        :py:obj:`~.cudaStreamCreateWithPriority`, :py:obj:`~.cudaStreamCreateWithFlags`, :py:obj:`~.cudaStreamGetPriority`, :py:obj:`~.cudaStreamGetDevice`, :py:obj:`~.cuStreamGetFlags`
    """


def cudaStreamGetId(hStream):
    """
    cudaStreamGetId(hStream)
     Query the Id of a stream.

        Query the Id of a stream. The Id is returned in `streamId`. The Id is
        unique for the life of the program.

        The stream handle `hStream` can refer to any of the following:

        - a stream created via any of the CUDA runtime APIs such as
          :py:obj:`~.cudaStreamCreate`, :py:obj:`~.cudaStreamCreateWithFlags`
          and :py:obj:`~.cudaStreamCreateWithPriority`, or their driver API
          equivalents such as :py:obj:`~.cuStreamCreate` or
          :py:obj:`~.cuStreamCreateWithPriority`. Passing an invalid handle
          will result in undefined behavior.

        - any of the special streams such as the NULL stream,
          :py:obj:`~.cudaStreamLegacy` and :py:obj:`~.cudaStreamPerThread`
          respectively. The driver API equivalents of these are also accepted
          which are NULL, :py:obj:`~.CU_STREAM_LEGACY` and
          :py:obj:`~.CU_STREAM_PER_THREAD`.

        Parameters
        ----------
        hStream : :py:obj:`~.CUstream` or :py:obj:`~.cudaStream_t`
            Handle to the stream to be queried

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidResourceHandle`
        streamId : unsigned long long
            Pointer to an unsigned long long in which the stream Id is returned

        See Also
        --------
        :py:obj:`~.cudaStreamCreateWithPriority`, :py:obj:`~.cudaStreamCreateWithFlags`, :py:obj:`~.cudaStreamGetPriority`, :py:obj:`~.cudaStreamGetFlags`, :py:obj:`~.cuStreamGetId`
    """


def cudaStreamGetPriority(hStream):
    """
    cudaStreamGetPriority(hStream)
     Query the priority of a stream.

        Query the priority of a stream. The priority is returned in in
        `priority`. Note that if the stream was created with a priority outside
        the meaningful numerical range returned by
        :py:obj:`~.cudaDeviceGetStreamPriorityRange`, this function returns the
        clamped priority. See :py:obj:`~.cudaStreamCreateWithPriority` for
        details about priority clamping.

        Parameters
        ----------
        hStream : :py:obj:`~.CUstream` or :py:obj:`~.cudaStream_t`
            Handle to the stream to be queried

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidResourceHandle`
        priority : int
            Pointer to a signed integer in which the stream's priority is
            returned

        See Also
        --------
        :py:obj:`~.cudaStreamCreateWithPriority`, :py:obj:`~.cudaDeviceGetStreamPriorityRange`, :py:obj:`~.cudaStreamGetFlags`, :py:obj:`~.cudaStreamGetDevice`, :py:obj:`~.cuStreamGetPriority`
    """


def cudaStreamIsCapturing(stream):
    """
    cudaStreamIsCapturing(stream)
     Returns a stream's capture status.

        Return the capture status of `stream` via `pCaptureStatus`. After a
        successful call, `*pCaptureStatus` will contain one of the following:

        - :py:obj:`~.cudaStreamCaptureStatusNone`: The stream is not capturing.

        - :py:obj:`~.cudaStreamCaptureStatusActive`: The stream is capturing.

        - :py:obj:`~.cudaStreamCaptureStatusInvalidated`: The stream was
          capturing but an error has invalidated the capture sequence. The
          capture sequence must be terminated with
          :py:obj:`~.cudaStreamEndCapture` on the stream where it was initiated
          in order to continue using `stream`.

        Note that, if this is called on :py:obj:`~.cudaStreamLegacy` (the "null
        stream") while a blocking stream on the same device is capturing, it
        will return :py:obj:`~.cudaErrorStreamCaptureImplicit` and
        `*pCaptureStatus` is unspecified after the call. The blocking stream
        capture is not invalidated.

        When a blocking stream is capturing, the legacy stream is in an
        unusable state until the blocking stream capture is terminated. The
        legacy stream is not supported for stream capture, but attempted use
        would have an implicit dependency on the capturing stream(s).

        Parameters
        ----------
        stream : :py:obj:`~.CUstream` or :py:obj:`~.cudaStream_t`
            Stream to query

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorStreamCaptureImplicit`
        pCaptureStatus : :py:obj:`~.cudaStreamCaptureStatus`
            Returns the stream's capture status

        See Also
        --------
        :py:obj:`~.cudaStreamCreate`, :py:obj:`~.cudaStreamBeginCapture`, :py:obj:`~.cudaStreamEndCapture`
    """

cudaStreamLegacy: int
cudaStreamNonBlocking: int
cudaStreamPerThread: int

def cudaStreamQuery(stream):
    """
    cudaStreamQuery(stream)
     Queries an asynchronous stream for completion status.

        Returns :py:obj:`~.cudaSuccess` if all operations in `stream` have
        completed, or :py:obj:`~.cudaErrorNotReady` if not.

        For the purposes of Unified Memory, a return value of
        :py:obj:`~.cudaSuccess` is equivalent to having called
        :py:obj:`~.cudaStreamSynchronize()`.

        Parameters
        ----------
        stream : :py:obj:`~.CUstream` or :py:obj:`~.cudaStream_t`
            Stream identifier

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorNotReady`, :py:obj:`~.cudaErrorInvalidResourceHandle`

        See Also
        --------
        :py:obj:`~.cudaStreamCreate`, :py:obj:`~.cudaStreamCreateWithFlags`, :py:obj:`~.cudaStreamWaitEvent`, :py:obj:`~.cudaStreamSynchronize`, :py:obj:`~.cudaStreamAddCallback`, :py:obj:`~.cudaStreamDestroy`, :py:obj:`~.cuStreamQuery`
    """


def cudaStreamSetAttribute(hStream, attr: 'cudaStreamAttrID', value: 'Optional[cudaStreamAttrValue]'):
    """
    cudaStreamSetAttribute(hStream, attr: cudaStreamAttrID, cudaStreamAttrValue value: Optional[cudaStreamAttrValue])
     Sets stream attribute.

        Sets attribute `attr` on `hStream` from corresponding attribute of
        `value`. The updated attribute will be applied to subsequent work
        submitted to the stream. It will not affect previously submitted work.

        Parameters
        ----------
        hStream : :py:obj:`~.CUstream` or :py:obj:`~.cudaStream_t`

        attr : :py:obj:`~.cudaStreamAttrID`

        value : :py:obj:`~.cudaStreamAttrValue`


        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidResourceHandle`

        See Also
        --------
        :py:obj:`~.cudaAccessPolicyWindow`
    """


def cudaStreamSynchronize(stream):
    """
    cudaStreamSynchronize(stream)
     Waits for stream tasks to complete.

        Blocks until `stream` has completed all operations. If the
        :py:obj:`~.cudaDeviceScheduleBlockingSync` flag was set for this
        device, the host thread will block until the stream is finished with
        all of its tasks.

        Parameters
        ----------
        stream : :py:obj:`~.CUstream` or :py:obj:`~.cudaStream_t`
            Stream identifier

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidResourceHandle`

        See Also
        --------
        :py:obj:`~.cudaStreamCreate`, :py:obj:`~.cudaStreamCreateWithFlags`, :py:obj:`~.cudaStreamQuery`, :py:obj:`~.cudaStreamWaitEvent`, :py:obj:`~.cudaStreamAddCallback`, :py:obj:`~.cudaStreamDestroy`, :py:obj:`~.cuStreamSynchronize`
    """


def cudaStreamUpdateCaptureDependencies(stream, dependencies: 'Optional[Tuple[cudaGraphNode_t] | List[cudaGraphNode_t]]', numDependencies, flags):
    """
    cudaStreamUpdateCaptureDependencies(stream, dependencies: Optional[Tuple[cudaGraphNode_t] | List[cudaGraphNode_t]], size_t numDependencies, unsigned int flags)
     Update the set of dependencies in a capturing stream (11.3+)

        Modifies the dependency set of a capturing stream. The dependency set
        is the set of nodes that the next captured node in the stream will
        depend on.

        Valid flags are :py:obj:`~.cudaStreamAddCaptureDependencies` and
        :py:obj:`~.cudaStreamSetCaptureDependencies`. These control whether the
        set passed to the API is added to the existing set or replaces it. A
        flags value of 0 defaults to
        :py:obj:`~.cudaStreamAddCaptureDependencies`.

        Nodes that are removed from the dependency set via this API do not
        result in :py:obj:`~.cudaErrorStreamCaptureUnjoined` if they are
        unreachable from the stream at :py:obj:`~.cudaStreamEndCapture`.

        Returns :py:obj:`~.cudaErrorIllegalState` if the stream is not
        capturing.

        This API is new in CUDA 11.3. Developers requiring compatibility across
        minor versions of the CUDA driver to 11.0 should not use this API or
        provide a fallback.

        Parameters
        ----------
        stream : :py:obj:`~.CUstream` or :py:obj:`~.cudaStream_t`
            The stream to update
        dependencies : List[:py:obj:`~.cudaGraphNode_t`]
            The set of dependencies to add
        numDependencies : size_t
            The size of the dependencies array
        flags : unsigned int
            See above

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorIllegalState`

        See Also
        --------
        :py:obj:`~.cudaStreamBeginCapture`, :py:obj:`~.cudaStreamGetCaptureInfo`,
    """


def cudaStreamUpdateCaptureDependencies_v2(stream, dependencies: 'Optional[Tuple[cudaGraphNode_t] | List[cudaGraphNode_t]]', dependencyData: 'Optional[Tuple[cudaGraphEdgeData] | List[cudaGraphEdgeData]]', numDependencies, flags):
    """
    cudaStreamUpdateCaptureDependencies_v2(stream, dependencies: Optional[Tuple[cudaGraphNode_t] | List[cudaGraphNode_t]], dependencyData: Optional[Tuple[cudaGraphEdgeData] | List[cudaGraphEdgeData]], size_t numDependencies, unsigned int flags)
     Update the set of dependencies in a capturing stream (12.3+)

        Modifies the dependency set of a capturing stream. The dependency set
        is the set of nodes that the next captured node in the stream will
        depend on.

        Valid flags are :py:obj:`~.cudaStreamAddCaptureDependencies` and
        :py:obj:`~.cudaStreamSetCaptureDependencies`. These control whether the
        set passed to the API is added to the existing set or replaces it. A
        flags value of 0 defaults to
        :py:obj:`~.cudaStreamAddCaptureDependencies`.

        Nodes that are removed from the dependency set via this API do not
        result in :py:obj:`~.cudaErrorStreamCaptureUnjoined` if they are
        unreachable from the stream at :py:obj:`~.cudaStreamEndCapture`.

        Returns :py:obj:`~.cudaErrorIllegalState` if the stream is not
        capturing.

        Parameters
        ----------
        stream : :py:obj:`~.CUstream` or :py:obj:`~.cudaStream_t`
            The stream to update
        dependencies : List[:py:obj:`~.cudaGraphNode_t`]
            The set of dependencies to add
        dependencyData : List[:py:obj:`~.cudaGraphEdgeData`]
            Optional array of data associated with each dependency.
        numDependencies : size_t
            The size of the dependencies array
        flags : unsigned int
            See above

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorIllegalState`

        See Also
        --------
        :py:obj:`~.cudaStreamBeginCapture`, :py:obj:`~.cudaStreamGetCaptureInfo`,
    """


def cudaStreamWaitEvent(stream, event, flags):
    """
    cudaStreamWaitEvent(stream, event, unsigned int flags)
     Make a compute stream wait on an event.

        Makes all future work submitted to `stream` wait for all work captured
        in `event`. See :py:obj:`~.cudaEventRecord()` for details on what is
        captured by an event. The synchronization will be performed efficiently
        on the device when applicable. `event` may be from a different device
        than `stream`.

        flags include:

        - :py:obj:`~.cudaEventWaitDefault`: Default event creation flag.

        - :py:obj:`~.cudaEventWaitExternal`: Event is captured in the graph as
          an external event node when performing stream capture.

        Parameters
        ----------
        stream : :py:obj:`~.CUstream` or :py:obj:`~.cudaStream_t`
            Stream to wait
        event : :py:obj:`~.CUevent` or :py:obj:`~.cudaEvent_t`
            Event to wait on
        flags : unsigned int
            Parameters for the operation(See above)

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`, :py:obj:`~.cudaErrorInvalidResourceHandle`

        See Also
        --------
        :py:obj:`~.cudaStreamCreate`, :py:obj:`~.cudaStreamCreateWithFlags`, :py:obj:`~.cudaStreamQuery`, :py:obj:`~.cudaStreamSynchronize`, :py:obj:`~.cudaStreamAddCallback`, :py:obj:`~.cudaStreamDestroy`, :py:obj:`~.cuStreamWaitEvent`
    """

cudaSurfaceType1D: int
cudaSurfaceType1DLayered: int
cudaSurfaceType2D: int
cudaSurfaceType2DLayered: int
cudaSurfaceType3D: int
cudaSurfaceTypeCubemap: int
cudaSurfaceTypeCubemapLayered: int
cudaTextureType1D: int
cudaTextureType1DLayered: int
cudaTextureType2D: int
cudaTextureType2DLayered: int
cudaTextureType3D: int
cudaTextureTypeCubemap: int
cudaTextureTypeCubemapLayered: int

def cudaThreadExchangeStreamCaptureMode(mode: 'cudaStreamCaptureMode'):
    """
    cudaThreadExchangeStreamCaptureMode(mode: cudaStreamCaptureMode)
     Swaps the stream capture interaction mode for a thread.

        Sets the calling thread's stream capture interaction mode to the value
        contained in `*mode`, and overwrites `*mode` with the previous mode for
        the thread. To facilitate deterministic behavior across function or
        module boundaries, callers are encouraged to use this API in a push-pop
        fashion:

        **View CUDA Toolkit Documentation for a C++ code example**

        During stream capture (see :py:obj:`~.cudaStreamBeginCapture`), some
        actions, such as a call to :py:obj:`~.cudaMalloc`, may be unsafe. In
        the case of :py:obj:`~.cudaMalloc`, the operation is not enqueued
        asynchronously to a stream, and is not observed by stream capture.
        Therefore, if the sequence of operations captured via
        :py:obj:`~.cudaStreamBeginCapture` depended on the allocation being
        replayed whenever the graph is launched, the captured graph would be
        invalid.

        Therefore, stream capture places restrictions on API calls that can be
        made within or concurrently to a
        :py:obj:`~.cudaStreamBeginCapture`-:py:obj:`~.cudaStreamEndCapture`
        sequence. This behavior can be controlled via this API and flags to
        :py:obj:`~.cudaStreamBeginCapture`.

        A thread's mode is one of the following:

        - `cudaStreamCaptureModeGlobal:` This is the default mode. If the local
          thread has an ongoing capture sequence that was not initiated with
          `cudaStreamCaptureModeRelaxed` at `cuStreamBeginCapture`, or if any
          other thread has a concurrent capture sequence initiated with
          `cudaStreamCaptureModeGlobal`, this thread is prohibited from
          potentially unsafe API calls.

        - `cudaStreamCaptureModeThreadLocal:` If the local thread has an
          ongoing capture sequence not initiated with
          `cudaStreamCaptureModeRelaxed`, it is prohibited from potentially
          unsafe API calls. Concurrent capture sequences in other threads are
          ignored.

        - `cudaStreamCaptureModeRelaxed:` The local thread is not prohibited
          from potentially unsafe API calls. Note that the thread is still
          prohibited from API calls which necessarily conflict with stream
          capture, for example, attempting :py:obj:`~.cudaEventQuery` on an
          event that was last recorded inside a capture sequence.

        Parameters
        ----------
        mode : :py:obj:`~.cudaStreamCaptureMode`
            Pointer to mode value to swap with the current mode

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`
        mode : :py:obj:`~.cudaStreamCaptureMode`
            Pointer to mode value to swap with the current mode

        See Also
        --------
        :py:obj:`~.cudaStreamBeginCapture`
    """


def cudaUserObjectCreate(ptr, destroy, initialRefcount, flags):
    """
    cudaUserObjectCreate(ptr, destroy, unsigned int initialRefcount, unsigned int flags)
     Create a user object.

        Create a user object with the specified destructor callback and initial
        reference count. The initial references are owned by the caller.

        Destructor callbacks cannot make CUDA API calls and should avoid
        blocking behavior, as they are executed by a shared internal thread.
        Another thread may be signaled to perform such actions, if it does not
        block forward progress of tasks scheduled through CUDA.

        See CUDA User Objects in the CUDA C++ Programming Guide for more
        information on user objects.

        Parameters
        ----------
        ptr : Any
            The pointer to pass to the destroy function
        destroy : :py:obj:`~.cudaHostFn_t`
            Callback to free the user object when it is no longer in use
        initialRefcount : unsigned int
            The initial refcount to create the object with, typically 1. The
            initial references are owned by the calling thread.
        flags : unsigned int
            Currently it is required to pass
            :py:obj:`~.cudaUserObjectNoDestructorSync`, which is the only
            defined flag. This indicates that the destroy callback cannot be
            waited on by any CUDA API. Users requiring synchronization of the
            callback should signal its completion manually.

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`
        object_out : :py:obj:`~.cudaUserObject_t`
            Location to return the user object handle

        See Also
        --------
        :py:obj:`~.cudaUserObjectRetain`, :py:obj:`~.cudaUserObjectRelease`, :py:obj:`~.cudaGraphRetainUserObject`, :py:obj:`~.cudaGraphReleaseUserObject`, :py:obj:`~.cudaGraphCreate`
    """


def cudaUserObjectRelease(object, count):
    """
    cudaUserObjectRelease(object, unsigned int count)
     Release a reference to a user object.

        Releases user object references owned by the caller. The object's
        destructor is invoked if the reference count reaches zero.

        It is undefined behavior to release references not owned by the caller,
        or to use a user object handle after all references are released.

        See CUDA User Objects in the CUDA C++ Programming Guide for more
        information on user objects.

        Parameters
        ----------
        object : :py:obj:`~.cudaUserObject_t`
            The object to release
        count : unsigned int
            The number of references to release, typically 1. Must be nonzero
            and not larger than INT_MAX.

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`

        See Also
        --------
        :py:obj:`~.cudaUserObjectCreate`, :py:obj:`~.cudaUserObjectRetain`, :py:obj:`~.cudaGraphRetainUserObject`, :py:obj:`~.cudaGraphReleaseUserObject`, :py:obj:`~.cudaGraphCreate`
    """


def cudaUserObjectRetain(object, count):
    """
    cudaUserObjectRetain(object, unsigned int count)
     Retain a reference to a user object.

        Retains new references to a user object. The new references are owned
        by the caller.

        See CUDA User Objects in the CUDA C++ Programming Guide for more
        information on user objects.

        Parameters
        ----------
        object : :py:obj:`~.cudaUserObject_t`
            The object to retain
        count : unsigned int
            The number of references to retain, typically 1. Must be nonzero
            and not larger than INT_MAX.

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`

        See Also
        --------
        :py:obj:`~.cudaUserObjectCreate`, :py:obj:`~.cudaUserObjectRelease`, :py:obj:`~.cudaGraphRetainUserObject`, :py:obj:`~.cudaGraphReleaseUserObject`, :py:obj:`~.cudaGraphCreate`
    """


def cudaVDPAUGetDevice(vdpDevice, vdpGetProcAddress):
    """
    cudaVDPAUGetDevice(vdpDevice, vdpGetProcAddress)
     Gets the CUDA device associated with a VdpDevice.

        Returns the CUDA device associated with a VdpDevice, if applicable.

        Parameters
        ----------
        vdpDevice : :py:obj:`~.VdpDevice`
            A VdpDevice handle
        vdpGetProcAddress : :py:obj:`~.VdpGetProcAddress`
            VDPAU's VdpGetProcAddress function pointer

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`
        device : int
            Returns the device associated with vdpDevice, or -1 if the device
            associated with vdpDevice is not a compute device.

        See Also
        --------
        :py:obj:`~.cudaVDPAUSetVDPAUDevice`, :py:obj:`~.cuVDPAUGetDevice`
    """


def cudaVDPAUSetVDPAUDevice(device, vdpDevice, vdpGetProcAddress):
    """
    cudaVDPAUSetVDPAUDevice(int device, vdpDevice, vdpGetProcAddress)
     Sets a CUDA device to use VDPAU interoperability.

        Records `vdpDevice` as the VdpDevice for VDPAU interoperability with
        the CUDA device `device` and sets `device` as the current device for
        the calling host thread.

        This function will immediately initialize the primary context on
        `device` if needed.

        If `device` has already been initialized then this call will fail with
        the error :py:obj:`~.cudaErrorSetOnActiveProcess`. In this case it is
        necessary to reset `device` using :py:obj:`~.cudaDeviceReset()` before
        VDPAU interoperability on `device` may be enabled.

        Parameters
        ----------
        device : int
            Device to use for VDPAU interoperability
        vdpDevice : :py:obj:`~.VdpDevice`
            The VdpDevice to interoperate with
        vdpGetProcAddress : :py:obj:`~.VdpGetProcAddress`
            VDPAU's VdpGetProcAddress function pointer

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidDevice`, :py:obj:`~.cudaErrorSetOnActiveProcess`

        See Also
        --------
        :py:obj:`~.cudaGraphicsVDPAURegisterVideoSurface`, :py:obj:`~.cudaGraphicsVDPAURegisterOutputSurface`, :py:obj:`~.cudaDeviceReset`
    """


def cudaWaitExternalSemaphoresAsync(extSemArray: 'Optional[Tuple[cudaExternalSemaphore_t] | List[cudaExternalSemaphore_t]]', paramsArray: 'Optional[Tuple[cudaExternalSemaphoreWaitParams] | List[cudaExternalSemaphoreWaitParams]]', numExtSems, stream):
    """
    cudaWaitExternalSemaphoresAsync(extSemArray: Optional[Tuple[cudaExternalSemaphore_t] | List[cudaExternalSemaphore_t]], paramsArray: Optional[Tuple[cudaExternalSemaphoreWaitParams] | List[cudaExternalSemaphoreWaitParams]], unsigned int numExtSems, stream)
     Waits on a set of external semaphore objects.

        Enqueues a wait operation on a set of externally allocated semaphore
        object in the specified stream. The operations will be executed when
        all prior operations in the stream complete.

        The exact semantics of waiting on a semaphore depends on the type of
        the object.

        If the semaphore object is any one of the following types:
        :py:obj:`~.cudaExternalSemaphoreHandleTypeOpaqueFd`,
        :py:obj:`~.cudaExternalSemaphoreHandleTypeOpaqueWin32`,
        :py:obj:`~.cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt` then waiting
        on the semaphore will wait until the semaphore reaches the signaled
        state. The semaphore will then be reset to the unsignaled state.
        Therefore for every signal operation, there can only be one wait
        operation.

        If the semaphore object is any one of the following types:
        :py:obj:`~.cudaExternalSemaphoreHandleTypeD3D12Fence`,
        :py:obj:`~.cudaExternalSemaphoreHandleTypeD3D11Fence`,
        :py:obj:`~.cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd`,
        :py:obj:`~.cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32` then
        waiting on the semaphore will wait until the value of the semaphore is
        greater than or equal to
        :py:obj:`~.cudaExternalSemaphoreWaitParams`::params::fence::value.

        If the semaphore object is of the type
        :py:obj:`~.cudaExternalSemaphoreHandleTypeNvSciSync` then, waiting on
        the semaphore will wait until the
        :py:obj:`~.cudaExternalSemaphoreSignalParams`::params::nvSciSync::fence
        is signaled by the signaler of the NvSciSyncObj that was associated
        with this semaphore object. By default, waiting on such an external
        semaphore object causes appropriate memory synchronization operations
        to be performed over all external memory objects that are imported as
        :py:obj:`~.cudaExternalMemoryHandleTypeNvSciBuf`. This ensures that any
        subsequent accesses made by other importers of the same set of NvSciBuf
        memory object(s) are coherent. These operations can be skipped by
        specifying the flag
        :py:obj:`~.cudaExternalSemaphoreWaitSkipNvSciBufMemSync`, which can be
        used as a performance optimization when data coherency is not required.
        But specifying this flag in scenarios where data coherency is required
        results in undefined behavior. Also, for semaphore object of the type
        :py:obj:`~.cudaExternalSemaphoreHandleTypeNvSciSync`, if the
        NvSciSyncAttrList used to create the NvSciSyncObj had not set the flags
        in :py:obj:`~.cudaDeviceGetNvSciSyncAttributes` to
        cudaNvSciSyncAttrWait, this API will return cudaErrorNotSupported.

        If the semaphore object is any one of the following types:
        :py:obj:`~.cudaExternalSemaphoreHandleTypeKeyedMutex`,
        :py:obj:`~.cudaExternalSemaphoreHandleTypeKeyedMutexKmt`, then the
        keyed mutex will be acquired when it is released with the key specified
        in
        :py:obj:`~.cudaExternalSemaphoreSignalParams`::params::keyedmutex::key
        or until the timeout specified by
        :py:obj:`~.cudaExternalSemaphoreSignalParams`::params::keyedmutex::timeoutMs
        has lapsed. The timeout interval can either be a finite value specified
        in milliseconds or an infinite value. In case an infinite value is
        specified the timeout never elapses. The windows INFINITE macro must be
        used to specify infinite timeout

        Parameters
        ----------
        extSemArray : List[:py:obj:`~.cudaExternalSemaphore_t`]
            External semaphores to be waited on
        paramsArray : List[:py:obj:`~.cudaExternalSemaphoreWaitParams`]
            Array of semaphore parameters
        numExtSems : unsigned int
            Number of semaphores to wait on
        stream : :py:obj:`~.CUstream` or :py:obj:`~.cudaStream_t`
            Stream to enqueue the wait operations in

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidResourceHandle` :py:obj:`~.cudaErrorTimeout`

        See Also
        --------
        :py:obj:`~.cudaImportExternalSemaphore`, :py:obj:`~.cudaDestroyExternalSemaphore`, :py:obj:`~.cudaSignalExternalSemaphoresAsync`
    """


def getLocalRuntimeVersion():
    """
    getLocalRuntimeVersion()
     Returns the CUDA Runtime version of local shared library.

        Returns in `*runtimeVersion` the version number of the current CUDA
        Runtime instance. The version is returned as (1000 * major + 10 *
        minor). For example, CUDA 9.2 would be represented by 9020.

        As of CUDA 12.0, this function no longer initializes CUDA. The purpose
        of this API is solely to return a compile-time constant stating the
        CUDA Toolkit version in the above format.

        This function automatically returns :py:obj:`~.cudaErrorInvalidValue`
        if the `runtimeVersion` argument is NULL.

        Returns
        -------
        cudaError_t
            :py:obj:`~.cudaSuccess`, :py:obj:`~.cudaErrorInvalidValue`
        runtimeVersion : int
            Returns the CUDA Runtime version.

        See Also
        --------
        :py:obj:`~.cudaDriverGetVersion`, :py:obj:`~.cuDriverGetVersion`
    """


def make_cudaExtent(w, h, d):
    """
    make_cudaExtent(size_t w, size_t h, size_t d)
     Returns a :py:obj:`~.cudaExtent` based on input parameters.

        Returns a :py:obj:`~.cudaExtent` based on the specified input
        parameters `w`, `h`, and `d`.

        Parameters
        ----------
        w : size_t
            Width in elements when referring to array memory, in bytes when
            referring to linear memory
        h : size_t
            Height in elements
        d : size_t
            Depth in elements

        Returns
        -------
        cudaError_t.cudaSuccess
            cudaError_t.cudaSuccess
        :py:obj:`~.cudaExtent`
            :py:obj:`~.cudaExtent` specified by `w`, `h`, and `d`

        See Also
        --------
        make_cudaPitchedPtr, make_cudaPos
    """


def make_cudaPitchedPtr(d, p, xsz, ysz):
    """
    make_cudaPitchedPtr(d, size_t p, size_t xsz, size_t ysz)
     Returns a :py:obj:`~.cudaPitchedPtr` based on input parameters.

        Returns a :py:obj:`~.cudaPitchedPtr` based on the specified input
        parameters `d`, `p`, `xsz`, and `ysz`.

        Parameters
        ----------
        d : Any
            Pointer to allocated memory
        p : size_t
            Pitch of allocated memory in bytes
        xsz : size_t
            Logical width of allocation in elements
        ysz : size_t
            Logical height of allocation in elements

        Returns
        -------
        cudaError_t.cudaSuccess
            cudaError_t.cudaSuccess
        :py:obj:`~.cudaPitchedPtr`
            :py:obj:`~.cudaPitchedPtr` specified by `d`, `p`, `xsz`, and `ysz`

        See Also
        --------
        make_cudaExtent, make_cudaPos
    """


def make_cudaPos(x, y, z):
    """
    make_cudaPos(size_t x, size_t y, size_t z)
     Returns a :py:obj:`~.cudaPos` based on input parameters.

        Returns a :py:obj:`~.cudaPos` based on the specified input parameters
        `x`, `y`, and `z`.

        Parameters
        ----------
        x : size_t
            X position
        y : size_t
            Y position
        z : size_t
            Z position

        Returns
        -------
        cudaError_t.cudaSuccess
            cudaError_t.cudaSuccess
        :py:obj:`~.cudaPos`
            :py:obj:`~.cudaPos` specified by `x`, `y`, and `z`

        See Also
        --------
        make_cudaExtent, make_cudaPitchedPtr
    """


def sizeof(objType):
    """
    sizeof(objType)
     Returns the size of provided CUDA Python structure in bytes

        Parameters
        ----------
        objType : Any
            CUDA Python object

        Returns
        -------
        lowered_name : int
            The size of `objType` in bytes
    """


class CUuuid(CUuuid_st):
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""

class CUuuid_st:
    bytes: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """CUuuid_st.getPtr(self)"""
    def __reduce__(self):
        """CUuuid_st.__reduce_cython__(self)"""

class EGLImageKHR:
    def __init__(self, *args, **kwargs) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """EGLImageKHR.getPtr(self)"""
    def __index__(self) -> int:
        """Return self converted to an integer, if self is suitable for use as an index into a list."""
    def __int__(self) -> int:
        """int(self)"""
    def __reduce__(self):
        """EGLImageKHR.__reduce_cython__(self)"""

class EGLStreamKHR:
    def __init__(self, *args, **kwargs) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """EGLStreamKHR.getPtr(self)"""
    def __index__(self) -> int:
        """Return self converted to an integer, if self is suitable for use as an index into a list."""
    def __int__(self) -> int:
        """int(self)"""
    def __reduce__(self):
        """EGLStreamKHR.__reduce_cython__(self)"""

class EGLSyncKHR:
    def __init__(self, *args, **kwargs) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """EGLSyncKHR.getPtr(self)"""
    def __index__(self) -> int:
        """Return self converted to an integer, if self is suitable for use as an index into a list."""
    def __int__(self) -> int:
        """int(self)"""
    def __reduce__(self):
        """EGLSyncKHR.__reduce_cython__(self)"""

class EGLint:
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def getPtr(self) -> Any:
        """EGLint.getPtr(self)"""
    def __int__(self) -> int:
        """int(self)"""
    def __reduce__(self):
        """EGLint.__reduce_cython__(self)"""

class GLenum:
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def getPtr(self) -> Any:
        """GLenum.getPtr(self)"""
    def __int__(self) -> int:
        """int(self)"""
    def __reduce__(self):
        """GLenum.__reduce_cython__(self)"""

class GLuint:
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def getPtr(self) -> Any:
        """GLuint.getPtr(self)"""
    def __int__(self) -> int:
        """int(self)"""
    def __reduce__(self):
        """GLuint.__reduce_cython__(self)"""

class VdpDevice:
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def getPtr(self) -> Any:
        """VdpDevice.getPtr(self)"""
    def __int__(self) -> int:
        """int(self)"""
    def __reduce__(self):
        """VdpDevice.__reduce_cython__(self)"""

class VdpGetProcAddress:
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def getPtr(self) -> Any:
        """VdpGetProcAddress.getPtr(self)"""
    def __int__(self) -> int:
        """int(self)"""
    def __reduce__(self):
        """VdpGetProcAddress.__reduce_cython__(self)"""

class VdpOutputSurface:
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def getPtr(self) -> Any:
        """VdpOutputSurface.getPtr(self)"""
    def __int__(self) -> int:
        """int(self)"""
    def __reduce__(self):
        """VdpOutputSurface.__reduce_cython__(self)"""

class VdpVideoSurface:
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def getPtr(self) -> Any:
        """VdpVideoSurface.getPtr(self)"""
    def __int__(self) -> int:
        """int(self)"""
    def __reduce__(self):
        """VdpVideoSurface.__reduce_cython__(self)"""

class anon_struct0:
    depth: Incomplete
    height: Incomplete
    width: Incomplete
    def __init__(self, void_ptr_ptr) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """anon_struct0.getPtr(self)"""
    def __reduce__(self):
        """anon_struct0.__reduce_cython__(self)"""

class anon_struct1:
    array: Incomplete
    def __init__(self, void_ptr_ptr) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """anon_struct1.getPtr(self)"""
    def __reduce__(self):
        """anon_struct1.__reduce_cython__(self)"""

class anon_struct15:
    value: Incomplete
    def __init__(self, void_ptr_ptr) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """anon_struct15.getPtr(self)"""
    def __reduce__(self):
        """anon_struct15.__reduce_cython__(self)"""

class anon_struct16:
    key: Incomplete
    def __init__(self, void_ptr_ptr) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """anon_struct16.getPtr(self)"""
    def __reduce__(self):
        """anon_struct16.__reduce_cython__(self)"""

class anon_struct17:
    fence: Incomplete
    keyedMutex: Incomplete
    nvSciSync: Incomplete
    reserved: Incomplete
    def __init__(self, void_ptr_ptr) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """anon_struct17.getPtr(self)"""
    def __reduce__(self):
        """anon_struct17.__reduce_cython__(self)"""

class anon_struct18:
    value: Incomplete
    def __init__(self, void_ptr_ptr) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """anon_struct18.getPtr(self)"""
    def __reduce__(self):
        """anon_struct18.__reduce_cython__(self)"""

class anon_struct19:
    key: Incomplete
    timeoutMs: Incomplete
    def __init__(self, void_ptr_ptr) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """anon_struct19.getPtr(self)"""
    def __reduce__(self):
        """anon_struct19.__reduce_cython__(self)"""

class anon_struct2:
    mipmap: Incomplete
    def __init__(self, void_ptr_ptr) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """anon_struct2.getPtr(self)"""
    def __reduce__(self):
        """anon_struct2.__reduce_cython__(self)"""

class anon_struct20:
    fence: Incomplete
    keyedMutex: Incomplete
    nvSciSync: Incomplete
    reserved: Incomplete
    def __init__(self, void_ptr_ptr) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """anon_struct20.getPtr(self)"""
    def __reduce__(self):
        """anon_struct20.__reduce_cython__(self)"""

class anon_struct21:
    offset: Incomplete
    pValue: Incomplete
    size: Incomplete
    def __init__(self, void_ptr_ptr) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """anon_struct21.getPtr(self)"""
    def __reduce__(self):
        """anon_struct21.__reduce_cython__(self)"""

class anon_struct22:
    x: Incomplete
    y: Incomplete
    z: Incomplete
    def __init__(self, void_ptr_ptr) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """anon_struct22.getPtr(self)"""
    def __reduce__(self):
        """anon_struct22.__reduce_cython__(self)"""

class anon_struct23:
    event: Incomplete
    flags: Incomplete
    triggerAtBlockStart: Incomplete
    def __init__(self, void_ptr_ptr) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """anon_struct23.getPtr(self)"""
    def __reduce__(self):
        """anon_struct23.__reduce_cython__(self)"""

class anon_struct24:
    x: Incomplete
    y: Incomplete
    z: Incomplete
    def __init__(self, void_ptr_ptr) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """anon_struct24.getPtr(self)"""
    def __reduce__(self):
        """anon_struct24.__reduce_cython__(self)"""

class anon_struct25:
    event: Incomplete
    flags: Incomplete
    def __init__(self, void_ptr_ptr) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """anon_struct25.getPtr(self)"""
    def __reduce__(self):
        """anon_struct25.__reduce_cython__(self)"""

class anon_struct26:
    devNode: Incomplete
    deviceUpdatable: Incomplete
    def __init__(self, void_ptr_ptr) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """anon_struct26.getPtr(self)"""
    def __reduce__(self):
        """anon_struct26.__reduce_cython__(self)"""

class anon_struct27:
    bytesOverBudget: Incomplete
    def __init__(self, void_ptr_ptr) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """anon_struct27.getPtr(self)"""
    def __reduce__(self):
        """anon_struct27.__reduce_cython__(self)"""

class anon_struct3:
    desc: Incomplete
    devPtr: Incomplete
    sizeInBytes: Incomplete
    def __init__(self, void_ptr_ptr) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """anon_struct3.getPtr(self)"""
    def __reduce__(self):
        """anon_struct3.__reduce_cython__(self)"""

class anon_struct4:
    desc: Incomplete
    devPtr: Incomplete
    height: Incomplete
    pitchInBytes: Incomplete
    width: Incomplete
    def __init__(self, void_ptr_ptr) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """anon_struct4.getPtr(self)"""
    def __reduce__(self):
        """anon_struct4.__reduce_cython__(self)"""

class anon_struct5:
    layerHeight: Incomplete
    locHint: Incomplete
    ptr: Incomplete
    rowLength: Incomplete
    def __init__(self, void_ptr_ptr) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """anon_struct5.getPtr(self)"""
    def __reduce__(self):
        """anon_struct5.__reduce_cython__(self)"""

class anon_struct6:
    array: Incomplete
    offset: Incomplete
    def __init__(self, void_ptr_ptr) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """anon_struct6.getPtr(self)"""
    def __reduce__(self):
        """anon_struct6.__reduce_cython__(self)"""

class anon_struct7:
    handle: Incomplete
    name: Incomplete
    def __init__(self, void_ptr_ptr) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """anon_struct7.getPtr(self)"""
    def __reduce__(self):
        """anon_struct7.__reduce_cython__(self)"""

class anon_struct8:
    handle: Incomplete
    name: Incomplete
    def __init__(self, void_ptr_ptr) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """anon_struct8.getPtr(self)"""
    def __reduce__(self):
        """anon_struct8.__reduce_cython__(self)"""

class anon_union0:
    array: Incomplete
    linear: Incomplete
    mipmap: Incomplete
    pitch2D: Incomplete
    def __init__(self, void_ptr_ptr) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """anon_union0.getPtr(self)"""
    def __reduce__(self):
        """anon_union0.__reduce_cython__(self)"""

class anon_union1:
    array: Incomplete
    ptr: Incomplete
    def __init__(self, void_ptr_ptr) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """anon_union1.getPtr(self)"""
    def __reduce__(self):
        """anon_union1.__reduce_cython__(self)"""

class anon_union10:
    overBudget: Incomplete
    def __init__(self, void_ptr_ptr) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """anon_union10.getPtr(self)"""
    def __reduce__(self):
        """anon_union10.__reduce_cython__(self)"""

class anon_union11:
    pArray: Incomplete
    pPitch: Incomplete
    def __init__(self, void_ptr_ptr) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """anon_union11.getPtr(self)"""
    def __reduce__(self):
        """anon_union11.__reduce_cython__(self)"""

class anon_union2:
    fd: Incomplete
    nvSciBufObject: Incomplete
    win32: Incomplete
    def __init__(self, void_ptr_ptr) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """anon_union2.getPtr(self)"""
    def __reduce__(self):
        """anon_union2.__reduce_cython__(self)"""

class anon_union3:
    fd: Incomplete
    nvSciSyncObj: Incomplete
    win32: Incomplete
    def __init__(self, void_ptr_ptr) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """anon_union3.getPtr(self)"""
    def __reduce__(self):
        """anon_union3.__reduce_cython__(self)"""

class anon_union6:
    fence: Incomplete
    reserved: Incomplete
    def __init__(self, void_ptr_ptr) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """anon_union6.getPtr(self)"""
    def __reduce__(self):
        """anon_union6.__reduce_cython__(self)"""

class anon_union7:
    fence: Incomplete
    reserved: Incomplete
    def __init__(self, void_ptr_ptr) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """anon_union7.getPtr(self)"""
    def __reduce__(self):
        """anon_union7.__reduce_cython__(self)"""

class anon_union9:
    gridDim: Incomplete
    isEnabled: Incomplete
    param: Incomplete
    def __init__(self, void_ptr_ptr) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """anon_union9.getPtr(self)"""
    def __reduce__(self):
        """anon_union9.__reduce_cython__(self)"""

class cudaAccessPolicyWindow:
    base_ptr: Incomplete
    hitProp: Incomplete
    hitRatio: Incomplete
    missProp: Incomplete
    num_bytes: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaAccessPolicyWindow.getPtr(self)"""
    def __reduce__(self):
        """cudaAccessPolicyWindow.__reduce_cython__(self)"""

class cudaAccessProperty(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaAccessPropertyNormal: ClassVar[cudaAccessProperty] = ...
    cudaAccessPropertyPersisting: ClassVar[cudaAccessProperty] = ...
    cudaAccessPropertyStreaming: ClassVar[cudaAccessProperty] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaArrayMemoryRequirements:
    alignment: Incomplete
    reserved: Incomplete
    size: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaArrayMemoryRequirements.getPtr(self)"""
    def __reduce__(self):
        """cudaArrayMemoryRequirements.__reduce_cython__(self)"""

class cudaArraySparseProperties:
    flags: Incomplete
    miptailFirstLevel: Incomplete
    miptailSize: Incomplete
    reserved: Incomplete
    tileExtent: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaArraySparseProperties.getPtr(self)"""
    def __reduce__(self):
        """cudaArraySparseProperties.__reduce_cython__(self)"""

class cudaArray_const_t:
    def __init__(self, *args, **kwargs) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaArray_const_t.getPtr(self)"""
    def __index__(self) -> int:
        """Return self converted to an integer, if self is suitable for use as an index into a list."""
    def __int__(self) -> int:
        """int(self)"""
    def __reduce__(self):
        """cudaArray_const_t.__reduce_cython__(self)"""

class cudaArray_t:
    def __init__(self, *args, **kwargs) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaArray_t.getPtr(self)"""
    def __index__(self) -> int:
        """Return self converted to an integer, if self is suitable for use as an index into a list."""
    def __int__(self) -> int:
        """int(self)"""
    def __reduce__(self):
        """cudaArray_t.__reduce_cython__(self)"""

class cudaAsyncCallback:
    def __init__(self, *args, **kwargs) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaAsyncCallback.getPtr(self)"""
    def __index__(self) -> int:
        """Return self converted to an integer, if self is suitable for use as an index into a list."""
    def __int__(self) -> int:
        """int(self)"""
    def __reduce__(self):
        """cudaAsyncCallback.__reduce_cython__(self)"""

class cudaAsyncCallbackHandle_t:
    def __init__(self, *args, **kwargs) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaAsyncCallbackHandle_t.getPtr(self)"""
    def __index__(self) -> int:
        """Return self converted to an integer, if self is suitable for use as an index into a list."""
    def __int__(self) -> int:
        """int(self)"""
    def __reduce__(self):
        """cudaAsyncCallbackHandle_t.__reduce_cython__(self)"""

class cudaAsyncNotificationInfo:
    info: Incomplete
    type: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaAsyncNotificationInfo.getPtr(self)"""
    def __reduce__(self):
        """cudaAsyncNotificationInfo.__reduce_cython__(self)"""

class cudaAsyncNotificationInfo_t(cudaAsyncNotificationInfo):
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""

class cudaAsyncNotificationType(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaAsyncNotificationTypeOverBudget: ClassVar[cudaAsyncNotificationType] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaCGScope(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaCGScopeGrid: ClassVar[cudaCGScope] = ...
    cudaCGScopeInvalid: ClassVar[cudaCGScope] = ...
    cudaCGScopeMultiGrid: ClassVar[cudaCGScope] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaChannelFormatDesc:
    f: Incomplete
    w: Incomplete
    x: Incomplete
    y: Incomplete
    z: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaChannelFormatDesc.getPtr(self)"""
    def __reduce__(self):
        """cudaChannelFormatDesc.__reduce_cython__(self)"""

class cudaChannelFormatKind(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaChannelFormatKindFloat: ClassVar[cudaChannelFormatKind] = ...
    cudaChannelFormatKindNV12: ClassVar[cudaChannelFormatKind] = ...
    cudaChannelFormatKindNone: ClassVar[cudaChannelFormatKind] = ...
    cudaChannelFormatKindSigned: ClassVar[cudaChannelFormatKind] = ...
    cudaChannelFormatKindSignedBlockCompressed4: ClassVar[cudaChannelFormatKind] = ...
    cudaChannelFormatKindSignedBlockCompressed5: ClassVar[cudaChannelFormatKind] = ...
    cudaChannelFormatKindSignedBlockCompressed6H: ClassVar[cudaChannelFormatKind] = ...
    cudaChannelFormatKindSignedNormalized16X1: ClassVar[cudaChannelFormatKind] = ...
    cudaChannelFormatKindSignedNormalized16X2: ClassVar[cudaChannelFormatKind] = ...
    cudaChannelFormatKindSignedNormalized16X4: ClassVar[cudaChannelFormatKind] = ...
    cudaChannelFormatKindSignedNormalized8X1: ClassVar[cudaChannelFormatKind] = ...
    cudaChannelFormatKindSignedNormalized8X2: ClassVar[cudaChannelFormatKind] = ...
    cudaChannelFormatKindSignedNormalized8X4: ClassVar[cudaChannelFormatKind] = ...
    cudaChannelFormatKindUnsigned: ClassVar[cudaChannelFormatKind] = ...
    cudaChannelFormatKindUnsignedBlockCompressed1: ClassVar[cudaChannelFormatKind] = ...
    cudaChannelFormatKindUnsignedBlockCompressed1SRGB: ClassVar[cudaChannelFormatKind] = ...
    cudaChannelFormatKindUnsignedBlockCompressed2: ClassVar[cudaChannelFormatKind] = ...
    cudaChannelFormatKindUnsignedBlockCompressed2SRGB: ClassVar[cudaChannelFormatKind] = ...
    cudaChannelFormatKindUnsignedBlockCompressed3: ClassVar[cudaChannelFormatKind] = ...
    cudaChannelFormatKindUnsignedBlockCompressed3SRGB: ClassVar[cudaChannelFormatKind] = ...
    cudaChannelFormatKindUnsignedBlockCompressed4: ClassVar[cudaChannelFormatKind] = ...
    cudaChannelFormatKindUnsignedBlockCompressed5: ClassVar[cudaChannelFormatKind] = ...
    cudaChannelFormatKindUnsignedBlockCompressed6H: ClassVar[cudaChannelFormatKind] = ...
    cudaChannelFormatKindUnsignedBlockCompressed7: ClassVar[cudaChannelFormatKind] = ...
    cudaChannelFormatKindUnsignedBlockCompressed7SRGB: ClassVar[cudaChannelFormatKind] = ...
    cudaChannelFormatKindUnsignedNormalized1010102: ClassVar[cudaChannelFormatKind] = ...
    cudaChannelFormatKindUnsignedNormalized16X1: ClassVar[cudaChannelFormatKind] = ...
    cudaChannelFormatKindUnsignedNormalized16X2: ClassVar[cudaChannelFormatKind] = ...
    cudaChannelFormatKindUnsignedNormalized16X4: ClassVar[cudaChannelFormatKind] = ...
    cudaChannelFormatKindUnsignedNormalized8X1: ClassVar[cudaChannelFormatKind] = ...
    cudaChannelFormatKindUnsignedNormalized8X2: ClassVar[cudaChannelFormatKind] = ...
    cudaChannelFormatKindUnsignedNormalized8X4: ClassVar[cudaChannelFormatKind] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaChildGraphNodeParams:
    graph: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaChildGraphNodeParams.getPtr(self)"""
    def __reduce__(self):
        """cudaChildGraphNodeParams.__reduce_cython__(self)"""

class cudaClusterSchedulingPolicy(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaClusterSchedulingPolicyDefault: ClassVar[cudaClusterSchedulingPolicy] = ...
    cudaClusterSchedulingPolicyLoadBalancing: ClassVar[cudaClusterSchedulingPolicy] = ...
    cudaClusterSchedulingPolicySpread: ClassVar[cudaClusterSchedulingPolicy] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaComputeMode(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaComputeModeDefault: ClassVar[cudaComputeMode] = ...
    cudaComputeModeExclusive: ClassVar[cudaComputeMode] = ...
    cudaComputeModeExclusiveProcess: ClassVar[cudaComputeMode] = ...
    cudaComputeModeProhibited: ClassVar[cudaComputeMode] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaConditionalNodeParams:
    handle: Incomplete
    phGraph_out: Incomplete
    size: Incomplete
    type: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaConditionalNodeParams.getPtr(self)"""
    def __reduce__(self):
        """cudaConditionalNodeParams.__reduce_cython__(self)"""

class cudaDataType(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    CUDA_C_16BF: ClassVar[cudaDataType] = ...
    CUDA_C_16F: ClassVar[cudaDataType] = ...
    CUDA_C_16I: ClassVar[cudaDataType] = ...
    CUDA_C_16U: ClassVar[cudaDataType] = ...
    CUDA_C_32F: ClassVar[cudaDataType] = ...
    CUDA_C_32I: ClassVar[cudaDataType] = ...
    CUDA_C_32U: ClassVar[cudaDataType] = ...
    CUDA_C_4I: ClassVar[cudaDataType] = ...
    CUDA_C_4U: ClassVar[cudaDataType] = ...
    CUDA_C_64F: ClassVar[cudaDataType] = ...
    CUDA_C_64I: ClassVar[cudaDataType] = ...
    CUDA_C_64U: ClassVar[cudaDataType] = ...
    CUDA_C_8I: ClassVar[cudaDataType] = ...
    CUDA_C_8U: ClassVar[cudaDataType] = ...
    CUDA_R_16BF: ClassVar[cudaDataType] = ...
    CUDA_R_16F: ClassVar[cudaDataType] = ...
    CUDA_R_16I: ClassVar[cudaDataType] = ...
    CUDA_R_16U: ClassVar[cudaDataType] = ...
    CUDA_R_32F: ClassVar[cudaDataType] = ...
    CUDA_R_32I: ClassVar[cudaDataType] = ...
    CUDA_R_32U: ClassVar[cudaDataType] = ...
    CUDA_R_4F_E2M1: ClassVar[cudaDataType] = ...
    CUDA_R_4I: ClassVar[cudaDataType] = ...
    CUDA_R_4U: ClassVar[cudaDataType] = ...
    CUDA_R_64F: ClassVar[cudaDataType] = ...
    CUDA_R_64I: ClassVar[cudaDataType] = ...
    CUDA_R_64U: ClassVar[cudaDataType] = ...
    CUDA_R_6F_E2M3: ClassVar[cudaDataType] = ...
    CUDA_R_6F_E3M2: ClassVar[cudaDataType] = ...
    CUDA_R_8F_E4M3: ClassVar[cudaDataType] = ...
    CUDA_R_8F_E5M2: ClassVar[cudaDataType] = ...
    CUDA_R_8F_UE4M3: ClassVar[cudaDataType] = ...
    CUDA_R_8F_UE8M0: ClassVar[cudaDataType] = ...
    CUDA_R_8I: ClassVar[cudaDataType] = ...
    CUDA_R_8U: ClassVar[cudaDataType] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaDeviceAttr(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaDevAttrAsyncEngineCount: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrCanFlushRemoteWrites: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrCanMapHostMemory: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrCanUseHostPointerForRegisteredMem: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrClockRate: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrClusterLaunch: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrComputeCapabilityMajor: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrComputeCapabilityMinor: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrComputeMode: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrComputePreemptionSupported: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrConcurrentKernels: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrConcurrentManagedAccess: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrCooperativeLaunch: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrCooperativeMultiDeviceLaunch: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrD3D12CigSupported: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrDeferredMappingCudaArraySupported: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrDirectManagedMemAccessFromHost: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrEccEnabled: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrGPUDirectRDMAFlushWritesOptions: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrGPUDirectRDMASupported: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrGPUDirectRDMAWritesOrdering: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrGlobalL1CacheSupported: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrGlobalMemoryBusWidth: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrGpuOverlap: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrGpuPciDeviceId: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrGpuPciSubsystemId: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrHostNativeAtomicSupported: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrHostNumaId: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrHostNumaMultinodeIpcSupported: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrHostRegisterReadOnlySupported: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrHostRegisterSupported: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrIntegrated: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrIpcEventSupport: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrIsMultiGpuBoard: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrKernelExecTimeout: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrL2CacheSize: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrLocalL1CacheSupported: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrManagedMemory: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMax: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxAccessPolicyWindowSize: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxBlockDimX: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxBlockDimY: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxBlockDimZ: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxBlocksPerMultiprocessor: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxGridDimX: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxGridDimY: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxGridDimZ: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxPersistingL2CacheSize: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxPitch: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxRegistersPerBlock: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxRegistersPerMultiprocessor: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxSharedMemoryPerBlock: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxSharedMemoryPerBlockOptin: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxSharedMemoryPerMultiprocessor: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxSurface1DLayeredLayers: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxSurface1DLayeredWidth: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxSurface1DWidth: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxSurface2DHeight: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxSurface2DLayeredHeight: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxSurface2DLayeredLayers: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxSurface2DLayeredWidth: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxSurface2DWidth: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxSurface3DDepth: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxSurface3DHeight: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxSurface3DWidth: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxSurfaceCubemapLayeredLayers: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxSurfaceCubemapLayeredWidth: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxSurfaceCubemapWidth: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxTexture1DLayeredLayers: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxTexture1DLayeredWidth: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxTexture1DLinearWidth: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxTexture1DMipmappedWidth: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxTexture1DWidth: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxTexture2DGatherHeight: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxTexture2DGatherWidth: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxTexture2DHeight: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxTexture2DLayeredHeight: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxTexture2DLayeredLayers: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxTexture2DLayeredWidth: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxTexture2DLinearHeight: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxTexture2DLinearPitch: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxTexture2DLinearWidth: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxTexture2DMipmappedHeight: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxTexture2DMipmappedWidth: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxTexture2DWidth: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxTexture3DDepth: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxTexture3DDepthAlt: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxTexture3DHeight: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxTexture3DHeightAlt: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxTexture3DWidth: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxTexture3DWidthAlt: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxTextureCubemapLayeredLayers: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxTextureCubemapLayeredWidth: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxTextureCubemapWidth: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxThreadsPerBlock: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxThreadsPerMultiProcessor: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMaxTimelineSemaphoreInteropSupported: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMemSyncDomainCount: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMemoryClockRate: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMemoryPoolSupportedHandleTypes: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMemoryPoolsSupported: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMpsEnabled: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMultiGpuBoardGroupID: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrMultiProcessorCount: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrNumaConfig: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrNumaId: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrPageableMemoryAccess: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrPageableMemoryAccessUsesHostPageTables: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrPciBusId: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrPciDeviceId: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrPciDomainId: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrReserved122: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrReserved123: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrReserved124: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrReserved127: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrReserved128: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrReserved129: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrReserved132: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrReserved92: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrReserved93: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrReserved94: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrReservedSharedMemoryPerBlock: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrSingleToDoublePrecisionPerfRatio: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrSparseCudaArraySupported: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrStreamPrioritiesSupported: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrSurfaceAlignment: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrTccDriver: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrTextureAlignment: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrTexturePitchAlignment: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrTimelineSemaphoreInteropSupported: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrTotalConstantMemory: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrUnifiedAddressing: ClassVar[cudaDeviceAttr] = ...
    cudaDevAttrWarpSize: ClassVar[cudaDeviceAttr] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaDeviceNumaConfig(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaDeviceNumaConfigNone: ClassVar[cudaDeviceNumaConfig] = ...
    cudaDeviceNumaConfigNumaNode: ClassVar[cudaDeviceNumaConfig] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaDeviceP2PAttr(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaDevP2PAttrAccessSupported: ClassVar[cudaDeviceP2PAttr] = ...
    cudaDevP2PAttrCudaArrayAccessSupported: ClassVar[cudaDeviceP2PAttr] = ...
    cudaDevP2PAttrNativeAtomicSupported: ClassVar[cudaDeviceP2PAttr] = ...
    cudaDevP2PAttrPerformanceRank: ClassVar[cudaDeviceP2PAttr] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaDeviceProp:
    ECCEnabled: Incomplete
    accessPolicyMaxWindowSize: Incomplete
    asyncEngineCount: Incomplete
    canMapHostMemory: Incomplete
    canUseHostPointerForRegisteredMem: Incomplete
    clockRate: Incomplete
    clusterLaunch: Incomplete
    computeMode: Incomplete
    computePreemptionSupported: Incomplete
    concurrentKernels: Incomplete
    concurrentManagedAccess: Incomplete
    cooperativeLaunch: Incomplete
    cooperativeMultiDeviceLaunch: Incomplete
    deferredMappingCudaArraySupported: Incomplete
    deviceOverlap: Incomplete
    directManagedMemAccessFromHost: Incomplete
    globalL1CacheSupported: Incomplete
    gpuDirectRDMAFlushWritesOptions: Incomplete
    gpuDirectRDMASupported: Incomplete
    gpuDirectRDMAWritesOrdering: Incomplete
    hostNativeAtomicSupported: Incomplete
    hostRegisterReadOnlySupported: Incomplete
    hostRegisterSupported: Incomplete
    integrated: Incomplete
    ipcEventSupported: Incomplete
    isMultiGpuBoard: Incomplete
    kernelExecTimeoutEnabled: Incomplete
    l2CacheSize: Incomplete
    localL1CacheSupported: Incomplete
    luid: Incomplete
    luidDeviceNodeMask: Incomplete
    major: Incomplete
    managedMemory: Incomplete
    maxBlocksPerMultiProcessor: Incomplete
    maxGridSize: Incomplete
    maxSurface1D: Incomplete
    maxSurface1DLayered: Incomplete
    maxSurface2D: Incomplete
    maxSurface2DLayered: Incomplete
    maxSurface3D: Incomplete
    maxSurfaceCubemap: Incomplete
    maxSurfaceCubemapLayered: Incomplete
    maxTexture1D: Incomplete
    maxTexture1DLayered: Incomplete
    maxTexture1DLinear: Incomplete
    maxTexture1DMipmap: Incomplete
    maxTexture2D: Incomplete
    maxTexture2DGather: Incomplete
    maxTexture2DLayered: Incomplete
    maxTexture2DLinear: Incomplete
    maxTexture2DMipmap: Incomplete
    maxTexture3D: Incomplete
    maxTexture3DAlt: Incomplete
    maxTextureCubemap: Incomplete
    maxTextureCubemapLayered: Incomplete
    maxThreadsDim: Incomplete
    maxThreadsPerBlock: Incomplete
    maxThreadsPerMultiProcessor: Incomplete
    memPitch: Incomplete
    memoryBusWidth: Incomplete
    memoryClockRate: Incomplete
    memoryPoolSupportedHandleTypes: Incomplete
    memoryPoolsSupported: Incomplete
    minor: Incomplete
    multiGpuBoardGroupID: Incomplete
    multiProcessorCount: Incomplete
    name: Incomplete
    pageableMemoryAccess: Incomplete
    pageableMemoryAccessUsesHostPageTables: Incomplete
    pciBusID: Incomplete
    pciDeviceID: Incomplete
    pciDomainID: Incomplete
    persistingL2CacheMaxSize: Incomplete
    regsPerBlock: Incomplete
    regsPerMultiprocessor: Incomplete
    reserved: Incomplete
    reservedSharedMemPerBlock: Incomplete
    sharedMemPerBlock: Incomplete
    sharedMemPerBlockOptin: Incomplete
    sharedMemPerMultiprocessor: Incomplete
    singleToDoublePrecisionPerfRatio: Incomplete
    sparseCudaArraySupported: Incomplete
    streamPrioritiesSupported: Incomplete
    surfaceAlignment: Incomplete
    tccDriver: Incomplete
    textureAlignment: Incomplete
    texturePitchAlignment: Incomplete
    timelineSemaphoreInteropSupported: Incomplete
    totalConstMem: Incomplete
    totalGlobalMem: Incomplete
    unifiedAddressing: Incomplete
    unifiedFunctionPointers: Incomplete
    uuid: Incomplete
    warpSize: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaDeviceProp.getPtr(self)"""
    def __reduce__(self):
        """cudaDeviceProp.__reduce_cython__(self)"""

class cudaDriverEntryPointQueryResult(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaDriverEntryPointSuccess: ClassVar[cudaDriverEntryPointQueryResult] = ...
    cudaDriverEntryPointSymbolNotFound: ClassVar[cudaDriverEntryPointQueryResult] = ...
    cudaDriverEntryPointVersionNotSufficent: ClassVar[cudaDriverEntryPointQueryResult] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaEglColorFormat(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaEglColorFormatA: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatABGR: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatARGB: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatAYUV: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatAYUV_ER: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatBGRA: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatBayer10BGGR: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatBayer10CCCC: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatBayer10GBRG: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatBayer10GRBG: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatBayer10RGGB: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatBayer12BCCR: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatBayer12BGGR: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatBayer12CBRC: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatBayer12CCCC: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatBayer12CRBC: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatBayer12GBRG: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatBayer12GRBG: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatBayer12RCCB: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatBayer12RGGB: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatBayer14BGGR: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatBayer14GBRG: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatBayer14GRBG: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatBayer14RGGB: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatBayer20BGGR: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatBayer20GBRG: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatBayer20GRBG: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatBayer20RGGB: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatBayerBCCR: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatBayerBGGR: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatBayerCBRC: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatBayerCRBC: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatBayerGBRG: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatBayerGRBG: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatBayerIspBGGR: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatBayerIspGBRG: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatBayerIspGRBG: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatBayerIspRGGB: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatBayerRCCB: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatBayerRGGB: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatL: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatR: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatRG: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatRGBA: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatUYVY2020: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatUYVY422: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatUYVY709: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatUYVY709_ER: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatUYVY_ER: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatVYUY: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatVYUY_ER: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatY: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatY10V10U10_420SemiPlanar: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatY10V10U10_420SemiPlanar_2020: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatY10V10U10_420SemiPlanar_709: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatY10V10U10_420SemiPlanar_709_ER: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatY10V10U10_420SemiPlanar_ER: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatY10V10U10_422SemiPlanar: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatY10V10U10_422SemiPlanar_2020: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatY10V10U10_422SemiPlanar_709: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatY10V10U10_444SemiPlanar: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatY10V10U10_444SemiPlanar_709_ER: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatY10V10U10_444SemiPlanar_ER: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatY10_709_ER: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatY10_ER: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatY12V12U12_420SemiPlanar: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatY12V12U12_420SemiPlanar_709_ER: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatY12V12U12_420SemiPlanar_ER: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatY12V12U12_444SemiPlanar: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatY12V12U12_444SemiPlanar_709_ER: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatY12V12U12_444SemiPlanar_ER: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatY12_709_ER: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatY12_ER: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatYUV420Planar: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatYUV420Planar_2020: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatYUV420Planar_709: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatYUV420Planar_ER: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatYUV420SemiPlanar: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatYUV420SemiPlanar_2020: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatYUV420SemiPlanar_709: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatYUV420SemiPlanar_ER: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatYUV422Planar: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatYUV422Planar_ER: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatYUV422SemiPlanar: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatYUV422SemiPlanar_ER: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatYUV444Planar: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatYUV444Planar_ER: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatYUV444SemiPlanar: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatYUV444SemiPlanar_ER: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatYUVA: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatYUVA_ER: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatYUYV422: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatYUYV_ER: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatYVU420Planar: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatYVU420Planar_2020: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatYVU420Planar_709: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatYVU420Planar_ER: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatYVU420SemiPlanar: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatYVU420SemiPlanar_2020: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatYVU420SemiPlanar_709: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatYVU420SemiPlanar_ER: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatYVU422Planar: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatYVU422Planar_ER: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatYVU422SemiPlanar: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatYVU422SemiPlanar_ER: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatYVU444Planar: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatYVU444Planar_ER: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatYVU444SemiPlanar: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatYVU444SemiPlanar_ER: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatYVYU: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatYVYU_ER: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatY_709_ER: ClassVar[cudaEglColorFormat] = ...
    cudaEglColorFormatY_ER: ClassVar[cudaEglColorFormat] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaEglFrame(cudaEglFrame_st):
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""

class cudaEglFrameType(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaEglFrameTypeArray: ClassVar[cudaEglFrameType] = ...
    cudaEglFrameTypePitch: ClassVar[cudaEglFrameType] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaEglFrame_st:
    eglColorFormat: Incomplete
    frame: Incomplete
    frameType: Incomplete
    planeCount: Incomplete
    planeDesc: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaEglFrame_st.getPtr(self)"""
    def __reduce__(self):
        """cudaEglFrame_st.__reduce_cython__(self)"""

class cudaEglPlaneDesc(cudaEglPlaneDesc_st):
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""

class cudaEglPlaneDesc_st:
    channelDesc: Incomplete
    depth: Incomplete
    height: Incomplete
    numChannels: Incomplete
    pitch: Incomplete
    reserved: Incomplete
    width: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaEglPlaneDesc_st.getPtr(self)"""
    def __reduce__(self):
        """cudaEglPlaneDesc_st.__reduce_cython__(self)"""

class cudaEglResourceLocationFlags(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaEglResourceLocationSysmem: ClassVar[cudaEglResourceLocationFlags] = ...
    cudaEglResourceLocationVidmem: ClassVar[cudaEglResourceLocationFlags] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaEglStreamConnection(cuda.bindings.driver.CUeglStreamConnection):
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""

class cudaError_t(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaErrorAddressOfConstant: ClassVar[cudaError_t] = ...
    cudaErrorAlreadyAcquired: ClassVar[cudaError_t] = ...
    cudaErrorAlreadyMapped: ClassVar[cudaError_t] = ...
    cudaErrorApiFailureBase: ClassVar[cudaError_t] = ...
    cudaErrorArrayIsMapped: ClassVar[cudaError_t] = ...
    cudaErrorAssert: ClassVar[cudaError_t] = ...
    cudaErrorCallRequiresNewerDriver: ClassVar[cudaError_t] = ...
    cudaErrorCapturedEvent: ClassVar[cudaError_t] = ...
    cudaErrorCdpNotSupported: ClassVar[cudaError_t] = ...
    cudaErrorCdpVersionMismatch: ClassVar[cudaError_t] = ...
    cudaErrorCompatNotSupportedOnDevice: ClassVar[cudaError_t] = ...
    cudaErrorContained: ClassVar[cudaError_t] = ...
    cudaErrorContextIsDestroyed: ClassVar[cudaError_t] = ...
    cudaErrorCooperativeLaunchTooLarge: ClassVar[cudaError_t] = ...
    cudaErrorCudartUnloading: ClassVar[cudaError_t] = ...
    cudaErrorDeviceAlreadyInUse: ClassVar[cudaError_t] = ...
    cudaErrorDeviceNotLicensed: ClassVar[cudaError_t] = ...
    cudaErrorDeviceUninitialized: ClassVar[cudaError_t] = ...
    cudaErrorDevicesUnavailable: ClassVar[cudaError_t] = ...
    cudaErrorDuplicateSurfaceName: ClassVar[cudaError_t] = ...
    cudaErrorDuplicateTextureName: ClassVar[cudaError_t] = ...
    cudaErrorDuplicateVariableName: ClassVar[cudaError_t] = ...
    cudaErrorECCUncorrectable: ClassVar[cudaError_t] = ...
    cudaErrorExternalDevice: ClassVar[cudaError_t] = ...
    cudaErrorFileNotFound: ClassVar[cudaError_t] = ...
    cudaErrorFunctionNotLoaded: ClassVar[cudaError_t] = ...
    cudaErrorGraphExecUpdateFailure: ClassVar[cudaError_t] = ...
    cudaErrorHardwareStackError: ClassVar[cudaError_t] = ...
    cudaErrorHostMemoryAlreadyRegistered: ClassVar[cudaError_t] = ...
    cudaErrorHostMemoryNotRegistered: ClassVar[cudaError_t] = ...
    cudaErrorIllegalAddress: ClassVar[cudaError_t] = ...
    cudaErrorIllegalInstruction: ClassVar[cudaError_t] = ...
    cudaErrorIllegalState: ClassVar[cudaError_t] = ...
    cudaErrorIncompatibleDriverContext: ClassVar[cudaError_t] = ...
    cudaErrorInitializationError: ClassVar[cudaError_t] = ...
    cudaErrorInsufficientDriver: ClassVar[cudaError_t] = ...
    cudaErrorInvalidAddressSpace: ClassVar[cudaError_t] = ...
    cudaErrorInvalidChannelDescriptor: ClassVar[cudaError_t] = ...
    cudaErrorInvalidClusterSize: ClassVar[cudaError_t] = ...
    cudaErrorInvalidConfiguration: ClassVar[cudaError_t] = ...
    cudaErrorInvalidDevice: ClassVar[cudaError_t] = ...
    cudaErrorInvalidDeviceFunction: ClassVar[cudaError_t] = ...
    cudaErrorInvalidDevicePointer: ClassVar[cudaError_t] = ...
    cudaErrorInvalidFilterSetting: ClassVar[cudaError_t] = ...
    cudaErrorInvalidGraphicsContext: ClassVar[cudaError_t] = ...
    cudaErrorInvalidHostPointer: ClassVar[cudaError_t] = ...
    cudaErrorInvalidKernelImage: ClassVar[cudaError_t] = ...
    cudaErrorInvalidMemcpyDirection: ClassVar[cudaError_t] = ...
    cudaErrorInvalidNormSetting: ClassVar[cudaError_t] = ...
    cudaErrorInvalidPc: ClassVar[cudaError_t] = ...
    cudaErrorInvalidPitchValue: ClassVar[cudaError_t] = ...
    cudaErrorInvalidPtx: ClassVar[cudaError_t] = ...
    cudaErrorInvalidResourceConfiguration: ClassVar[cudaError_t] = ...
    cudaErrorInvalidResourceHandle: ClassVar[cudaError_t] = ...
    cudaErrorInvalidResourceType: ClassVar[cudaError_t] = ...
    cudaErrorInvalidSource: ClassVar[cudaError_t] = ...
    cudaErrorInvalidSurface: ClassVar[cudaError_t] = ...
    cudaErrorInvalidSymbol: ClassVar[cudaError_t] = ...
    cudaErrorInvalidTexture: ClassVar[cudaError_t] = ...
    cudaErrorInvalidTextureBinding: ClassVar[cudaError_t] = ...
    cudaErrorInvalidValue: ClassVar[cudaError_t] = ...
    cudaErrorJitCompilationDisabled: ClassVar[cudaError_t] = ...
    cudaErrorJitCompilerNotFound: ClassVar[cudaError_t] = ...
    cudaErrorLaunchFailure: ClassVar[cudaError_t] = ...
    cudaErrorLaunchFileScopedSurf: ClassVar[cudaError_t] = ...
    cudaErrorLaunchFileScopedTex: ClassVar[cudaError_t] = ...
    cudaErrorLaunchIncompatibleTexturing: ClassVar[cudaError_t] = ...
    cudaErrorLaunchMaxDepthExceeded: ClassVar[cudaError_t] = ...
    cudaErrorLaunchOutOfResources: ClassVar[cudaError_t] = ...
    cudaErrorLaunchPendingCountExceeded: ClassVar[cudaError_t] = ...
    cudaErrorLaunchTimeout: ClassVar[cudaError_t] = ...
    cudaErrorLossyQuery: ClassVar[cudaError_t] = ...
    cudaErrorMapBufferObjectFailed: ClassVar[cudaError_t] = ...
    cudaErrorMemoryAllocation: ClassVar[cudaError_t] = ...
    cudaErrorMemoryValueTooLarge: ClassVar[cudaError_t] = ...
    cudaErrorMisalignedAddress: ClassVar[cudaError_t] = ...
    cudaErrorMissingConfiguration: ClassVar[cudaError_t] = ...
    cudaErrorMixedDeviceExecution: ClassVar[cudaError_t] = ...
    cudaErrorMpsClientTerminated: ClassVar[cudaError_t] = ...
    cudaErrorMpsConnectionFailed: ClassVar[cudaError_t] = ...
    cudaErrorMpsMaxClientsReached: ClassVar[cudaError_t] = ...
    cudaErrorMpsMaxConnectionsReached: ClassVar[cudaError_t] = ...
    cudaErrorMpsRpcFailure: ClassVar[cudaError_t] = ...
    cudaErrorMpsServerNotReady: ClassVar[cudaError_t] = ...
    cudaErrorNoDevice: ClassVar[cudaError_t] = ...
    cudaErrorNoKernelImageForDevice: ClassVar[cudaError_t] = ...
    cudaErrorNotMapped: ClassVar[cudaError_t] = ...
    cudaErrorNotMappedAsArray: ClassVar[cudaError_t] = ...
    cudaErrorNotMappedAsPointer: ClassVar[cudaError_t] = ...
    cudaErrorNotPermitted: ClassVar[cudaError_t] = ...
    cudaErrorNotReady: ClassVar[cudaError_t] = ...
    cudaErrorNotSupported: ClassVar[cudaError_t] = ...
    cudaErrorNotYetImplemented: ClassVar[cudaError_t] = ...
    cudaErrorNvlinkUncorrectable: ClassVar[cudaError_t] = ...
    cudaErrorOperatingSystem: ClassVar[cudaError_t] = ...
    cudaErrorPeerAccessAlreadyEnabled: ClassVar[cudaError_t] = ...
    cudaErrorPeerAccessNotEnabled: ClassVar[cudaError_t] = ...
    cudaErrorPeerAccessUnsupported: ClassVar[cudaError_t] = ...
    cudaErrorPriorLaunchFailure: ClassVar[cudaError_t] = ...
    cudaErrorProfilerAlreadyStarted: ClassVar[cudaError_t] = ...
    cudaErrorProfilerAlreadyStopped: ClassVar[cudaError_t] = ...
    cudaErrorProfilerDisabled: ClassVar[cudaError_t] = ...
    cudaErrorProfilerNotInitialized: ClassVar[cudaError_t] = ...
    cudaErrorSetOnActiveProcess: ClassVar[cudaError_t] = ...
    cudaErrorSharedObjectInitFailed: ClassVar[cudaError_t] = ...
    cudaErrorSharedObjectSymbolNotFound: ClassVar[cudaError_t] = ...
    cudaErrorSoftwareValidityNotEstablished: ClassVar[cudaError_t] = ...
    cudaErrorStartupFailure: ClassVar[cudaError_t] = ...
    cudaErrorStreamCaptureImplicit: ClassVar[cudaError_t] = ...
    cudaErrorStreamCaptureInvalidated: ClassVar[cudaError_t] = ...
    cudaErrorStreamCaptureIsolation: ClassVar[cudaError_t] = ...
    cudaErrorStreamCaptureMerge: ClassVar[cudaError_t] = ...
    cudaErrorStreamCaptureUnjoined: ClassVar[cudaError_t] = ...
    cudaErrorStreamCaptureUnmatched: ClassVar[cudaError_t] = ...
    cudaErrorStreamCaptureUnsupported: ClassVar[cudaError_t] = ...
    cudaErrorStreamCaptureWrongThread: ClassVar[cudaError_t] = ...
    cudaErrorStubLibrary: ClassVar[cudaError_t] = ...
    cudaErrorSymbolNotFound: ClassVar[cudaError_t] = ...
    cudaErrorSyncDepthExceeded: ClassVar[cudaError_t] = ...
    cudaErrorSynchronizationError: ClassVar[cudaError_t] = ...
    cudaErrorSystemDriverMismatch: ClassVar[cudaError_t] = ...
    cudaErrorSystemNotReady: ClassVar[cudaError_t] = ...
    cudaErrorTensorMemoryLeak: ClassVar[cudaError_t] = ...
    cudaErrorTextureFetchFailed: ClassVar[cudaError_t] = ...
    cudaErrorTextureNotBound: ClassVar[cudaError_t] = ...
    cudaErrorTimeout: ClassVar[cudaError_t] = ...
    cudaErrorTooManyPeers: ClassVar[cudaError_t] = ...
    cudaErrorUnknown: ClassVar[cudaError_t] = ...
    cudaErrorUnmapBufferObjectFailed: ClassVar[cudaError_t] = ...
    cudaErrorUnsupportedDevSideSync: ClassVar[cudaError_t] = ...
    cudaErrorUnsupportedExecAffinity: ClassVar[cudaError_t] = ...
    cudaErrorUnsupportedLimit: ClassVar[cudaError_t] = ...
    cudaErrorUnsupportedPtxVersion: ClassVar[cudaError_t] = ...
    cudaSuccess: ClassVar[cudaError_t] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaEventRecordNodeParams:
    event: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaEventRecordNodeParams.getPtr(self)"""
    def __reduce__(self):
        """cudaEventRecordNodeParams.__reduce_cython__(self)"""

class cudaEventWaitNodeParams:
    event: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaEventWaitNodeParams.getPtr(self)"""
    def __reduce__(self):
        """cudaEventWaitNodeParams.__reduce_cython__(self)"""

class cudaEvent_t(cuda.bindings.driver.CUevent):
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""

class cudaExtent:
    depth: Incomplete
    height: Incomplete
    width: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaExtent.getPtr(self)"""
    def __reduce__(self):
        """cudaExtent.__reduce_cython__(self)"""

class cudaExternalMemoryBufferDesc:
    flags: Incomplete
    offset: Incomplete
    size: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaExternalMemoryBufferDesc.getPtr(self)"""
    def __reduce__(self):
        """cudaExternalMemoryBufferDesc.__reduce_cython__(self)"""

class cudaExternalMemoryHandleDesc:
    flags: Incomplete
    handle: Incomplete
    size: Incomplete
    type: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaExternalMemoryHandleDesc.getPtr(self)"""
    def __reduce__(self):
        """cudaExternalMemoryHandleDesc.__reduce_cython__(self)"""

class cudaExternalMemoryHandleType(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaExternalMemoryHandleTypeD3D11Resource: ClassVar[cudaExternalMemoryHandleType] = ...
    cudaExternalMemoryHandleTypeD3D11ResourceKmt: ClassVar[cudaExternalMemoryHandleType] = ...
    cudaExternalMemoryHandleTypeD3D12Heap: ClassVar[cudaExternalMemoryHandleType] = ...
    cudaExternalMemoryHandleTypeD3D12Resource: ClassVar[cudaExternalMemoryHandleType] = ...
    cudaExternalMemoryHandleTypeNvSciBuf: ClassVar[cudaExternalMemoryHandleType] = ...
    cudaExternalMemoryHandleTypeOpaqueFd: ClassVar[cudaExternalMemoryHandleType] = ...
    cudaExternalMemoryHandleTypeOpaqueWin32: ClassVar[cudaExternalMemoryHandleType] = ...
    cudaExternalMemoryHandleTypeOpaqueWin32Kmt: ClassVar[cudaExternalMemoryHandleType] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaExternalMemoryMipmappedArrayDesc:
    extent: Incomplete
    flags: Incomplete
    formatDesc: Incomplete
    numLevels: Incomplete
    offset: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaExternalMemoryMipmappedArrayDesc.getPtr(self)"""
    def __reduce__(self):
        """cudaExternalMemoryMipmappedArrayDesc.__reduce_cython__(self)"""

class cudaExternalMemory_t:
    def __init__(self, *args, **kwargs) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaExternalMemory_t.getPtr(self)"""
    def __index__(self) -> int:
        """Return self converted to an integer, if self is suitable for use as an index into a list."""
    def __int__(self) -> int:
        """int(self)"""
    def __reduce__(self):
        """cudaExternalMemory_t.__reduce_cython__(self)"""

class cudaExternalSemaphoreHandleDesc:
    flags: Incomplete
    handle: Incomplete
    type: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaExternalSemaphoreHandleDesc.getPtr(self)"""
    def __reduce__(self):
        """cudaExternalSemaphoreHandleDesc.__reduce_cython__(self)"""

class cudaExternalSemaphoreHandleType(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaExternalSemaphoreHandleTypeD3D11Fence: ClassVar[cudaExternalSemaphoreHandleType] = ...
    cudaExternalSemaphoreHandleTypeD3D12Fence: ClassVar[cudaExternalSemaphoreHandleType] = ...
    cudaExternalSemaphoreHandleTypeKeyedMutex: ClassVar[cudaExternalSemaphoreHandleType] = ...
    cudaExternalSemaphoreHandleTypeKeyedMutexKmt: ClassVar[cudaExternalSemaphoreHandleType] = ...
    cudaExternalSemaphoreHandleTypeNvSciSync: ClassVar[cudaExternalSemaphoreHandleType] = ...
    cudaExternalSemaphoreHandleTypeOpaqueFd: ClassVar[cudaExternalSemaphoreHandleType] = ...
    cudaExternalSemaphoreHandleTypeOpaqueWin32: ClassVar[cudaExternalSemaphoreHandleType] = ...
    cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt: ClassVar[cudaExternalSemaphoreHandleType] = ...
    cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd: ClassVar[cudaExternalSemaphoreHandleType] = ...
    cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32: ClassVar[cudaExternalSemaphoreHandleType] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaExternalSemaphoreSignalNodeParams:
    extSemArray: Incomplete
    numExtSems: Incomplete
    paramsArray: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaExternalSemaphoreSignalNodeParams.getPtr(self)"""
    def __reduce__(self):
        """cudaExternalSemaphoreSignalNodeParams.__reduce_cython__(self)"""

class cudaExternalSemaphoreSignalNodeParamsV2:
    extSemArray: Incomplete
    numExtSems: Incomplete
    paramsArray: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaExternalSemaphoreSignalNodeParamsV2.getPtr(self)"""
    def __reduce__(self):
        """cudaExternalSemaphoreSignalNodeParamsV2.__reduce_cython__(self)"""

class cudaExternalSemaphoreSignalParams:
    flags: Incomplete
    params: Incomplete
    reserved: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaExternalSemaphoreSignalParams.getPtr(self)"""
    def __reduce__(self):
        """cudaExternalSemaphoreSignalParams.__reduce_cython__(self)"""

class cudaExternalSemaphoreWaitNodeParams:
    extSemArray: Incomplete
    numExtSems: Incomplete
    paramsArray: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaExternalSemaphoreWaitNodeParams.getPtr(self)"""
    def __reduce__(self):
        """cudaExternalSemaphoreWaitNodeParams.__reduce_cython__(self)"""

class cudaExternalSemaphoreWaitNodeParamsV2:
    extSemArray: Incomplete
    numExtSems: Incomplete
    paramsArray: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaExternalSemaphoreWaitNodeParamsV2.getPtr(self)"""
    def __reduce__(self):
        """cudaExternalSemaphoreWaitNodeParamsV2.__reduce_cython__(self)"""

class cudaExternalSemaphoreWaitParams:
    flags: Incomplete
    params: Incomplete
    reserved: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaExternalSemaphoreWaitParams.getPtr(self)"""
    def __reduce__(self):
        """cudaExternalSemaphoreWaitParams.__reduce_cython__(self)"""

class cudaExternalSemaphore_t:
    def __init__(self, *args, **kwargs) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaExternalSemaphore_t.getPtr(self)"""
    def __index__(self) -> int:
        """Return self converted to an integer, if self is suitable for use as an index into a list."""
    def __int__(self) -> int:
        """int(self)"""
    def __reduce__(self):
        """cudaExternalSemaphore_t.__reduce_cython__(self)"""

class cudaFlushGPUDirectRDMAWritesOptions(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaFlushGPUDirectRDMAWritesOptionHost: ClassVar[cudaFlushGPUDirectRDMAWritesOptions] = ...
    cudaFlushGPUDirectRDMAWritesOptionMemOps: ClassVar[cudaFlushGPUDirectRDMAWritesOptions] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaFlushGPUDirectRDMAWritesScope(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaFlushGPUDirectRDMAWritesToAllDevices: ClassVar[cudaFlushGPUDirectRDMAWritesScope] = ...
    cudaFlushGPUDirectRDMAWritesToOwner: ClassVar[cudaFlushGPUDirectRDMAWritesScope] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaFlushGPUDirectRDMAWritesTarget(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaFlushGPUDirectRDMAWritesTargetCurrentDevice: ClassVar[cudaFlushGPUDirectRDMAWritesTarget] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaFuncAttribute(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaFuncAttributeClusterDimMustBeSet: ClassVar[cudaFuncAttribute] = ...
    cudaFuncAttributeClusterSchedulingPolicyPreference: ClassVar[cudaFuncAttribute] = ...
    cudaFuncAttributeMax: ClassVar[cudaFuncAttribute] = ...
    cudaFuncAttributeMaxDynamicSharedMemorySize: ClassVar[cudaFuncAttribute] = ...
    cudaFuncAttributeNonPortableClusterSizeAllowed: ClassVar[cudaFuncAttribute] = ...
    cudaFuncAttributePreferredSharedMemoryCarveout: ClassVar[cudaFuncAttribute] = ...
    cudaFuncAttributeRequiredClusterDepth: ClassVar[cudaFuncAttribute] = ...
    cudaFuncAttributeRequiredClusterHeight: ClassVar[cudaFuncAttribute] = ...
    cudaFuncAttributeRequiredClusterWidth: ClassVar[cudaFuncAttribute] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaFuncAttributes:
    binaryVersion: Incomplete
    cacheModeCA: Incomplete
    clusterDimMustBeSet: Incomplete
    clusterSchedulingPolicyPreference: Incomplete
    constSizeBytes: Incomplete
    localSizeBytes: Incomplete
    maxDynamicSharedSizeBytes: Incomplete
    maxThreadsPerBlock: Incomplete
    nonPortableClusterSizeAllowed: Incomplete
    numRegs: Incomplete
    preferredShmemCarveout: Incomplete
    ptxVersion: Incomplete
    requiredClusterDepth: Incomplete
    requiredClusterHeight: Incomplete
    requiredClusterWidth: Incomplete
    reserved: Incomplete
    sharedSizeBytes: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaFuncAttributes.getPtr(self)"""
    def __reduce__(self):
        """cudaFuncAttributes.__reduce_cython__(self)"""

class cudaFuncCache(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaFuncCachePreferEqual: ClassVar[cudaFuncCache] = ...
    cudaFuncCachePreferL1: ClassVar[cudaFuncCache] = ...
    cudaFuncCachePreferNone: ClassVar[cudaFuncCache] = ...
    cudaFuncCachePreferShared: ClassVar[cudaFuncCache] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaFunction_t(cuda.bindings.driver.CUfunction):
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""

class cudaGLDeviceList(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaGLDeviceListAll: ClassVar[cudaGLDeviceList] = ...
    cudaGLDeviceListCurrentFrame: ClassVar[cudaGLDeviceList] = ...
    cudaGLDeviceListNextFrame: ClassVar[cudaGLDeviceList] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaGLMapFlags(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaGLMapFlagsNone: ClassVar[cudaGLMapFlags] = ...
    cudaGLMapFlagsReadOnly: ClassVar[cudaGLMapFlags] = ...
    cudaGLMapFlagsWriteDiscard: ClassVar[cudaGLMapFlags] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaGPUDirectRDMAWritesOrdering(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaGPUDirectRDMAWritesOrderingAllDevices: ClassVar[cudaGPUDirectRDMAWritesOrdering] = ...
    cudaGPUDirectRDMAWritesOrderingNone: ClassVar[cudaGPUDirectRDMAWritesOrdering] = ...
    cudaGPUDirectRDMAWritesOrderingOwner: ClassVar[cudaGPUDirectRDMAWritesOrdering] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaGetDriverEntryPointFlags(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaEnableDefault: ClassVar[cudaGetDriverEntryPointFlags] = ...
    cudaEnableLegacyStream: ClassVar[cudaGetDriverEntryPointFlags] = ...
    cudaEnablePerThreadDefaultStream: ClassVar[cudaGetDriverEntryPointFlags] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaGraphConditionalHandle:
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaGraphConditionalHandle.getPtr(self)"""
    def __int__(self) -> int:
        """int(self)"""
    def __reduce__(self):
        """cudaGraphConditionalHandle.__reduce_cython__(self)"""

class cudaGraphConditionalHandleFlags(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaGraphCondAssignDefault: ClassVar[cudaGraphConditionalHandleFlags] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaGraphConditionalNodeType(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaGraphCondTypeIf: ClassVar[cudaGraphConditionalNodeType] = ...
    cudaGraphCondTypeSwitch: ClassVar[cudaGraphConditionalNodeType] = ...
    cudaGraphCondTypeWhile: ClassVar[cudaGraphConditionalNodeType] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaGraphDebugDotFlags(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaGraphDebugDotFlagsConditionalNodeParams: ClassVar[cudaGraphDebugDotFlags] = ...
    cudaGraphDebugDotFlagsEventNodeParams: ClassVar[cudaGraphDebugDotFlags] = ...
    cudaGraphDebugDotFlagsExtSemasSignalNodeParams: ClassVar[cudaGraphDebugDotFlags] = ...
    cudaGraphDebugDotFlagsExtSemasWaitNodeParams: ClassVar[cudaGraphDebugDotFlags] = ...
    cudaGraphDebugDotFlagsHandles: ClassVar[cudaGraphDebugDotFlags] = ...
    cudaGraphDebugDotFlagsHostNodeParams: ClassVar[cudaGraphDebugDotFlags] = ...
    cudaGraphDebugDotFlagsKernelNodeAttributes: ClassVar[cudaGraphDebugDotFlags] = ...
    cudaGraphDebugDotFlagsKernelNodeParams: ClassVar[cudaGraphDebugDotFlags] = ...
    cudaGraphDebugDotFlagsMemcpyNodeParams: ClassVar[cudaGraphDebugDotFlags] = ...
    cudaGraphDebugDotFlagsMemsetNodeParams: ClassVar[cudaGraphDebugDotFlags] = ...
    cudaGraphDebugDotFlagsVerbose: ClassVar[cudaGraphDebugDotFlags] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaGraphDependencyType(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaGraphDependencyTypeDefault: ClassVar[cudaGraphDependencyType] = ...
    cudaGraphDependencyTypeProgrammatic: ClassVar[cudaGraphDependencyType] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaGraphDeviceNode_t:
    def __init__(self, *args, **kwargs) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaGraphDeviceNode_t.getPtr(self)"""
    def __index__(self) -> int:
        """Return self converted to an integer, if self is suitable for use as an index into a list."""
    def __int__(self) -> int:
        """int(self)"""
    def __reduce__(self):
        """cudaGraphDeviceNode_t.__reduce_cython__(self)"""

class cudaGraphEdgeData(cudaGraphEdgeData_st):
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""

class cudaGraphEdgeData_st:
    from_port: Incomplete
    reserved: Incomplete
    to_port: Incomplete
    type: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaGraphEdgeData_st.getPtr(self)"""
    def __reduce__(self):
        """cudaGraphEdgeData_st.__reduce_cython__(self)"""

class cudaGraphExecUpdateResult(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaGraphExecUpdateError: ClassVar[cudaGraphExecUpdateResult] = ...
    cudaGraphExecUpdateErrorAttributesChanged: ClassVar[cudaGraphExecUpdateResult] = ...
    cudaGraphExecUpdateErrorFunctionChanged: ClassVar[cudaGraphExecUpdateResult] = ...
    cudaGraphExecUpdateErrorNodeTypeChanged: ClassVar[cudaGraphExecUpdateResult] = ...
    cudaGraphExecUpdateErrorNotSupported: ClassVar[cudaGraphExecUpdateResult] = ...
    cudaGraphExecUpdateErrorParametersChanged: ClassVar[cudaGraphExecUpdateResult] = ...
    cudaGraphExecUpdateErrorTopologyChanged: ClassVar[cudaGraphExecUpdateResult] = ...
    cudaGraphExecUpdateErrorUnsupportedFunctionChange: ClassVar[cudaGraphExecUpdateResult] = ...
    cudaGraphExecUpdateSuccess: ClassVar[cudaGraphExecUpdateResult] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaGraphExecUpdateResultInfo(cudaGraphExecUpdateResultInfo_st):
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""

class cudaGraphExecUpdateResultInfo_st:
    errorFromNode: Incomplete
    errorNode: Incomplete
    result: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaGraphExecUpdateResultInfo_st.getPtr(self)"""
    def __reduce__(self):
        """cudaGraphExecUpdateResultInfo_st.__reduce_cython__(self)"""

class cudaGraphExec_t(cuda.bindings.driver.CUgraphExec):
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""

class cudaGraphInstantiateFlags(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaGraphInstantiateFlagAutoFreeOnLaunch: ClassVar[cudaGraphInstantiateFlags] = ...
    cudaGraphInstantiateFlagDeviceLaunch: ClassVar[cudaGraphInstantiateFlags] = ...
    cudaGraphInstantiateFlagUpload: ClassVar[cudaGraphInstantiateFlags] = ...
    cudaGraphInstantiateFlagUseNodePriority: ClassVar[cudaGraphInstantiateFlags] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaGraphInstantiateParams(cudaGraphInstantiateParams_st):
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""

class cudaGraphInstantiateParams_st:
    errNode_out: Incomplete
    flags: Incomplete
    result_out: Incomplete
    uploadStream: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaGraphInstantiateParams_st.getPtr(self)"""
    def __reduce__(self):
        """cudaGraphInstantiateParams_st.__reduce_cython__(self)"""

class cudaGraphInstantiateResult(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaGraphInstantiateConditionalHandleUnused: ClassVar[cudaGraphInstantiateResult] = ...
    cudaGraphInstantiateError: ClassVar[cudaGraphInstantiateResult] = ...
    cudaGraphInstantiateInvalidStructure: ClassVar[cudaGraphInstantiateResult] = ...
    cudaGraphInstantiateMultipleDevicesNotSupported: ClassVar[cudaGraphInstantiateResult] = ...
    cudaGraphInstantiateNodeOperationNotSupported: ClassVar[cudaGraphInstantiateResult] = ...
    cudaGraphInstantiateSuccess: ClassVar[cudaGraphInstantiateResult] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaGraphKernelNodeField(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaGraphKernelNodeFieldEnabled: ClassVar[cudaGraphKernelNodeField] = ...
    cudaGraphKernelNodeFieldGridDim: ClassVar[cudaGraphKernelNodeField] = ...
    cudaGraphKernelNodeFieldInvalid: ClassVar[cudaGraphKernelNodeField] = ...
    cudaGraphKernelNodeFieldParam: ClassVar[cudaGraphKernelNodeField] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaGraphKernelNodeUpdate:
    field: Incomplete
    node: Incomplete
    updateData: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaGraphKernelNodeUpdate.getPtr(self)"""
    def __reduce__(self):
        """cudaGraphKernelNodeUpdate.__reduce_cython__(self)"""

class cudaGraphMemAttributeType(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaGraphMemAttrReservedMemCurrent: ClassVar[cudaGraphMemAttributeType] = ...
    cudaGraphMemAttrReservedMemHigh: ClassVar[cudaGraphMemAttributeType] = ...
    cudaGraphMemAttrUsedMemCurrent: ClassVar[cudaGraphMemAttributeType] = ...
    cudaGraphMemAttrUsedMemHigh: ClassVar[cudaGraphMemAttributeType] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaGraphNodeParams:
    alloc: Incomplete
    conditional: Incomplete
    eventRecord: Incomplete
    eventWait: Incomplete
    extSemSignal: Incomplete
    extSemWait: Incomplete
    free: Incomplete
    graph: Incomplete
    host: Incomplete
    kernel: Incomplete
    memcpy: Incomplete
    memset: Incomplete
    reserved0: Incomplete
    reserved1: Incomplete
    reserved2: Incomplete
    type: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaGraphNodeParams.getPtr(self)"""
    def __reduce__(self):
        """cudaGraphNodeParams.__reduce_cython__(self)"""

class cudaGraphNodeType(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaGraphNodeTypeConditional: ClassVar[cudaGraphNodeType] = ...
    cudaGraphNodeTypeCount: ClassVar[cudaGraphNodeType] = ...
    cudaGraphNodeTypeEmpty: ClassVar[cudaGraphNodeType] = ...
    cudaGraphNodeTypeEventRecord: ClassVar[cudaGraphNodeType] = ...
    cudaGraphNodeTypeExtSemaphoreSignal: ClassVar[cudaGraphNodeType] = ...
    cudaGraphNodeTypeExtSemaphoreWait: ClassVar[cudaGraphNodeType] = ...
    cudaGraphNodeTypeGraph: ClassVar[cudaGraphNodeType] = ...
    cudaGraphNodeTypeHost: ClassVar[cudaGraphNodeType] = ...
    cudaGraphNodeTypeKernel: ClassVar[cudaGraphNodeType] = ...
    cudaGraphNodeTypeMemAlloc: ClassVar[cudaGraphNodeType] = ...
    cudaGraphNodeTypeMemFree: ClassVar[cudaGraphNodeType] = ...
    cudaGraphNodeTypeMemcpy: ClassVar[cudaGraphNodeType] = ...
    cudaGraphNodeTypeMemset: ClassVar[cudaGraphNodeType] = ...
    cudaGraphNodeTypeWaitEvent: ClassVar[cudaGraphNodeType] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaGraphNode_t(cuda.bindings.driver.CUgraphNode):
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""

class cudaGraph_t(cuda.bindings.driver.CUgraph):
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""

class cudaGraphicsCubeFace(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaGraphicsCubeFaceNegativeX: ClassVar[cudaGraphicsCubeFace] = ...
    cudaGraphicsCubeFaceNegativeY: ClassVar[cudaGraphicsCubeFace] = ...
    cudaGraphicsCubeFaceNegativeZ: ClassVar[cudaGraphicsCubeFace] = ...
    cudaGraphicsCubeFacePositiveX: ClassVar[cudaGraphicsCubeFace] = ...
    cudaGraphicsCubeFacePositiveY: ClassVar[cudaGraphicsCubeFace] = ...
    cudaGraphicsCubeFacePositiveZ: ClassVar[cudaGraphicsCubeFace] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaGraphicsMapFlags(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaGraphicsMapFlagsNone: ClassVar[cudaGraphicsMapFlags] = ...
    cudaGraphicsMapFlagsReadOnly: ClassVar[cudaGraphicsMapFlags] = ...
    cudaGraphicsMapFlagsWriteDiscard: ClassVar[cudaGraphicsMapFlags] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaGraphicsRegisterFlags(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaGraphicsRegisterFlagsNone: ClassVar[cudaGraphicsRegisterFlags] = ...
    cudaGraphicsRegisterFlagsReadOnly: ClassVar[cudaGraphicsRegisterFlags] = ...
    cudaGraphicsRegisterFlagsSurfaceLoadStore: ClassVar[cudaGraphicsRegisterFlags] = ...
    cudaGraphicsRegisterFlagsTextureGather: ClassVar[cudaGraphicsRegisterFlags] = ...
    cudaGraphicsRegisterFlagsWriteDiscard: ClassVar[cudaGraphicsRegisterFlags] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaGraphicsResource_t:
    def __init__(self, *args, **kwargs) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaGraphicsResource_t.getPtr(self)"""
    def __index__(self) -> int:
        """Return self converted to an integer, if self is suitable for use as an index into a list."""
    def __int__(self) -> int:
        """int(self)"""
    def __reduce__(self):
        """cudaGraphicsResource_t.__reduce_cython__(self)"""

class cudaHostFn_t:
    def __init__(self, *args, **kwargs) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaHostFn_t.getPtr(self)"""
    def __index__(self) -> int:
        """Return self converted to an integer, if self is suitable for use as an index into a list."""
    def __int__(self) -> int:
        """int(self)"""
    def __reduce__(self):
        """cudaHostFn_t.__reduce_cython__(self)"""

class cudaHostNodeParams:
    fn: Incomplete
    userData: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaHostNodeParams.getPtr(self)"""
    def __reduce__(self):
        """cudaHostNodeParams.__reduce_cython__(self)"""

class cudaHostNodeParamsV2:
    fn: Incomplete
    userData: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaHostNodeParamsV2.getPtr(self)"""
    def __reduce__(self):
        """cudaHostNodeParamsV2.__reduce_cython__(self)"""

class cudaIpcEventHandle_st:
    reserved: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaIpcEventHandle_st.getPtr(self)"""
    def __reduce__(self):
        """cudaIpcEventHandle_st.__reduce_cython__(self)"""

class cudaIpcEventHandle_t(cudaIpcEventHandle_st):
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""

class cudaIpcMemHandle_st:
    reserved: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaIpcMemHandle_st.getPtr(self)"""
    def __reduce__(self):
        """cudaIpcMemHandle_st.__reduce_cython__(self)"""

class cudaIpcMemHandle_t(cudaIpcMemHandle_st):
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""

class cudaJitOption(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaJitCacheMode: ClassVar[cudaJitOption] = ...
    cudaJitErrorLogBuffer: ClassVar[cudaJitOption] = ...
    cudaJitErrorLogBufferSizeBytes: ClassVar[cudaJitOption] = ...
    cudaJitFallbackStrategy: ClassVar[cudaJitOption] = ...
    cudaJitGenerateDebugInfo: ClassVar[cudaJitOption] = ...
    cudaJitGenerateLineInfo: ClassVar[cudaJitOption] = ...
    cudaJitInfoLogBuffer: ClassVar[cudaJitOption] = ...
    cudaJitInfoLogBufferSizeBytes: ClassVar[cudaJitOption] = ...
    cudaJitLogVerbose: ClassVar[cudaJitOption] = ...
    cudaJitMaxRegisters: ClassVar[cudaJitOption] = ...
    cudaJitMaxThreadsPerBlock: ClassVar[cudaJitOption] = ...
    cudaJitMinCtaPerSm: ClassVar[cudaJitOption] = ...
    cudaJitOptimizationLevel: ClassVar[cudaJitOption] = ...
    cudaJitOverrideDirectiveValues: ClassVar[cudaJitOption] = ...
    cudaJitPositionIndependentCode: ClassVar[cudaJitOption] = ...
    cudaJitThreadsPerBlock: ClassVar[cudaJitOption] = ...
    cudaJitWallTime: ClassVar[cudaJitOption] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaJit_CacheMode(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaJitCacheOptionCA: ClassVar[cudaJit_CacheMode] = ...
    cudaJitCacheOptionCG: ClassVar[cudaJit_CacheMode] = ...
    cudaJitCacheOptionNone: ClassVar[cudaJit_CacheMode] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaJit_Fallback(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaPreferBinary: ClassVar[cudaJit_Fallback] = ...
    cudaPreferPtx: ClassVar[cudaJit_Fallback] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaKernelNodeAttrID(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaLaunchAttributeAccessPolicyWindow: ClassVar[cudaKernelNodeAttrID] = ...
    cudaLaunchAttributeClusterDimension: ClassVar[cudaKernelNodeAttrID] = ...
    cudaLaunchAttributeClusterSchedulingPolicyPreference: ClassVar[cudaKernelNodeAttrID] = ...
    cudaLaunchAttributeCooperative: ClassVar[cudaKernelNodeAttrID] = ...
    cudaLaunchAttributeDeviceUpdatableKernelNode: ClassVar[cudaKernelNodeAttrID] = ...
    cudaLaunchAttributeIgnore: ClassVar[cudaKernelNodeAttrID] = ...
    cudaLaunchAttributeLaunchCompletionEvent: ClassVar[cudaKernelNodeAttrID] = ...
    cudaLaunchAttributeMemSyncDomain: ClassVar[cudaKernelNodeAttrID] = ...
    cudaLaunchAttributeMemSyncDomainMap: ClassVar[cudaKernelNodeAttrID] = ...
    cudaLaunchAttributePreferredClusterDimension: ClassVar[cudaKernelNodeAttrID] = ...
    cudaLaunchAttributePreferredSharedMemoryCarveout: ClassVar[cudaKernelNodeAttrID] = ...
    cudaLaunchAttributePriority: ClassVar[cudaKernelNodeAttrID] = ...
    cudaLaunchAttributeProgrammaticEvent: ClassVar[cudaKernelNodeAttrID] = ...
    cudaLaunchAttributeProgrammaticStreamSerialization: ClassVar[cudaKernelNodeAttrID] = ...
    cudaLaunchAttributeSynchronizationPolicy: ClassVar[cudaKernelNodeAttrID] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaKernelNodeAttrValue(cudaLaunchAttributeValue):
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""

class cudaKernelNodeParams:
    blockDim: Incomplete
    extra: Incomplete
    func: Incomplete
    gridDim: Incomplete
    kernelParams: Incomplete
    sharedMemBytes: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaKernelNodeParams.getPtr(self)"""
    def __reduce__(self):
        """cudaKernelNodeParams.__reduce_cython__(self)"""

class cudaKernelNodeParamsV2:
    blockDim: Incomplete
    extra: Incomplete
    func: Incomplete
    gridDim: Incomplete
    kernelParams: Incomplete
    sharedMemBytes: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaKernelNodeParamsV2.getPtr(self)"""
    def __reduce__(self):
        """cudaKernelNodeParamsV2.__reduce_cython__(self)"""

class cudaKernel_t:
    def __init__(self, *args, **kwargs) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaKernel_t.getPtr(self)"""
    def __index__(self) -> int:
        """Return self converted to an integer, if self is suitable for use as an index into a list."""
    def __int__(self) -> int:
        """int(self)"""
    def __reduce__(self):
        """cudaKernel_t.__reduce_cython__(self)"""

class cudaLaunchAttribute(cudaLaunchAttribute_st):
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""

class cudaLaunchAttributeID(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaLaunchAttributeAccessPolicyWindow: ClassVar[cudaLaunchAttributeID] = ...
    cudaLaunchAttributeClusterDimension: ClassVar[cudaLaunchAttributeID] = ...
    cudaLaunchAttributeClusterSchedulingPolicyPreference: ClassVar[cudaLaunchAttributeID] = ...
    cudaLaunchAttributeCooperative: ClassVar[cudaLaunchAttributeID] = ...
    cudaLaunchAttributeDeviceUpdatableKernelNode: ClassVar[cudaLaunchAttributeID] = ...
    cudaLaunchAttributeIgnore: ClassVar[cudaLaunchAttributeID] = ...
    cudaLaunchAttributeLaunchCompletionEvent: ClassVar[cudaLaunchAttributeID] = ...
    cudaLaunchAttributeMemSyncDomain: ClassVar[cudaLaunchAttributeID] = ...
    cudaLaunchAttributeMemSyncDomainMap: ClassVar[cudaLaunchAttributeID] = ...
    cudaLaunchAttributePreferredClusterDimension: ClassVar[cudaLaunchAttributeID] = ...
    cudaLaunchAttributePreferredSharedMemoryCarveout: ClassVar[cudaLaunchAttributeID] = ...
    cudaLaunchAttributePriority: ClassVar[cudaLaunchAttributeID] = ...
    cudaLaunchAttributeProgrammaticEvent: ClassVar[cudaLaunchAttributeID] = ...
    cudaLaunchAttributeProgrammaticStreamSerialization: ClassVar[cudaLaunchAttributeID] = ...
    cudaLaunchAttributeSynchronizationPolicy: ClassVar[cudaLaunchAttributeID] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaLaunchAttributeValue:
    accessPolicyWindow: Incomplete
    clusterDim: Incomplete
    clusterSchedulingPolicyPreference: Incomplete
    cooperative: Incomplete
    deviceUpdatableKernelNode: Incomplete
    launchCompletionEvent: Incomplete
    memSyncDomain: Incomplete
    memSyncDomainMap: Incomplete
    pad: Incomplete
    preferredClusterDim: Incomplete
    priority: Incomplete
    programmaticEvent: Incomplete
    programmaticStreamSerializationAllowed: Incomplete
    sharedMemCarveout: Incomplete
    syncPolicy: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaLaunchAttributeValue.getPtr(self)"""
    def __reduce__(self):
        """cudaLaunchAttributeValue.__reduce_cython__(self)"""

class cudaLaunchAttribute_st:
    id: Incomplete
    val: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaLaunchAttribute_st.getPtr(self)"""
    def __reduce__(self):
        """cudaLaunchAttribute_st.__reduce_cython__(self)"""

class cudaLaunchMemSyncDomain(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaLaunchMemSyncDomainDefault: ClassVar[cudaLaunchMemSyncDomain] = ...
    cudaLaunchMemSyncDomainRemote: ClassVar[cudaLaunchMemSyncDomain] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaLaunchMemSyncDomainMap(cudaLaunchMemSyncDomainMap_st):
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""

class cudaLaunchMemSyncDomainMap_st:
    default_: Incomplete
    remote: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaLaunchMemSyncDomainMap_st.getPtr(self)"""
    def __reduce__(self):
        """cudaLaunchMemSyncDomainMap_st.__reduce_cython__(self)"""

class cudaLibraryOption(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaLibraryBinaryIsPreserved: ClassVar[cudaLibraryOption] = ...
    cudaLibraryHostUniversalFunctionAndDataTable: ClassVar[cudaLibraryOption] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaLibrary_t:
    def __init__(self, *args, **kwargs) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaLibrary_t.getPtr(self)"""
    def __index__(self) -> int:
        """Return self converted to an integer, if self is suitable for use as an index into a list."""
    def __int__(self) -> int:
        """int(self)"""
    def __reduce__(self):
        """cudaLibrary_t.__reduce_cython__(self)"""

class cudaLimit(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaLimitDevRuntimePendingLaunchCount: ClassVar[cudaLimit] = ...
    cudaLimitDevRuntimeSyncDepth: ClassVar[cudaLimit] = ...
    cudaLimitMallocHeapSize: ClassVar[cudaLimit] = ...
    cudaLimitMaxL2FetchGranularity: ClassVar[cudaLimit] = ...
    cudaLimitPersistingL2CacheSize: ClassVar[cudaLimit] = ...
    cudaLimitPrintfFifoSize: ClassVar[cudaLimit] = ...
    cudaLimitStackSize: ClassVar[cudaLimit] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaMemAccessDesc:
    flags: Incomplete
    location: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaMemAccessDesc.getPtr(self)"""
    def __reduce__(self):
        """cudaMemAccessDesc.__reduce_cython__(self)"""

class cudaMemAccessFlags(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaMemAccessFlagsProtNone: ClassVar[cudaMemAccessFlags] = ...
    cudaMemAccessFlagsProtRead: ClassVar[cudaMemAccessFlags] = ...
    cudaMemAccessFlagsProtReadWrite: ClassVar[cudaMemAccessFlags] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaMemAllocNodeParams:
    accessDescCount: Incomplete
    accessDescs: Incomplete
    bytesize: Incomplete
    dptr: Incomplete
    poolProps: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaMemAllocNodeParams.getPtr(self)"""
    def __reduce__(self):
        """cudaMemAllocNodeParams.__reduce_cython__(self)"""

class cudaMemAllocNodeParamsV2:
    accessDescCount: Incomplete
    accessDescs: Incomplete
    bytesize: Incomplete
    dptr: Incomplete
    poolProps: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaMemAllocNodeParamsV2.getPtr(self)"""
    def __reduce__(self):
        """cudaMemAllocNodeParamsV2.__reduce_cython__(self)"""

class cudaMemAllocationHandleType(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaMemHandleTypeFabric: ClassVar[cudaMemAllocationHandleType] = ...
    cudaMemHandleTypeNone: ClassVar[cudaMemAllocationHandleType] = ...
    cudaMemHandleTypePosixFileDescriptor: ClassVar[cudaMemAllocationHandleType] = ...
    cudaMemHandleTypeWin32: ClassVar[cudaMemAllocationHandleType] = ...
    cudaMemHandleTypeWin32Kmt: ClassVar[cudaMemAllocationHandleType] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaMemAllocationType(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaMemAllocationTypeInvalid: ClassVar[cudaMemAllocationType] = ...
    cudaMemAllocationTypeMax: ClassVar[cudaMemAllocationType] = ...
    cudaMemAllocationTypePinned: ClassVar[cudaMemAllocationType] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaMemFabricHandle_st:
    reserved: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaMemFabricHandle_st.getPtr(self)"""
    def __reduce__(self):
        """cudaMemFabricHandle_st.__reduce_cython__(self)"""

class cudaMemFabricHandle_t(cudaMemFabricHandle_st):
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""

class cudaMemFreeNodeParams:
    dptr: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaMemFreeNodeParams.getPtr(self)"""
    def __reduce__(self):
        """cudaMemFreeNodeParams.__reduce_cython__(self)"""

class cudaMemLocation:
    id: Incomplete
    type: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaMemLocation.getPtr(self)"""
    def __reduce__(self):
        """cudaMemLocation.__reduce_cython__(self)"""

class cudaMemLocationType(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaMemLocationTypeDevice: ClassVar[cudaMemLocationType] = ...
    cudaMemLocationTypeHost: ClassVar[cudaMemLocationType] = ...
    cudaMemLocationTypeHostNuma: ClassVar[cudaMemLocationType] = ...
    cudaMemLocationTypeHostNumaCurrent: ClassVar[cudaMemLocationType] = ...
    cudaMemLocationTypeInvalid: ClassVar[cudaMemLocationType] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaMemPoolAttr(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaMemPoolAttrReleaseThreshold: ClassVar[cudaMemPoolAttr] = ...
    cudaMemPoolAttrReservedMemCurrent: ClassVar[cudaMemPoolAttr] = ...
    cudaMemPoolAttrReservedMemHigh: ClassVar[cudaMemPoolAttr] = ...
    cudaMemPoolAttrUsedMemCurrent: ClassVar[cudaMemPoolAttr] = ...
    cudaMemPoolAttrUsedMemHigh: ClassVar[cudaMemPoolAttr] = ...
    cudaMemPoolReuseAllowInternalDependencies: ClassVar[cudaMemPoolAttr] = ...
    cudaMemPoolReuseAllowOpportunistic: ClassVar[cudaMemPoolAttr] = ...
    cudaMemPoolReuseFollowEventDependencies: ClassVar[cudaMemPoolAttr] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaMemPoolProps:
    allocType: Incomplete
    handleTypes: Incomplete
    location: Incomplete
    maxSize: Incomplete
    reserved: Incomplete
    usage: Incomplete
    win32SecurityAttributes: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaMemPoolProps.getPtr(self)"""
    def __reduce__(self):
        """cudaMemPoolProps.__reduce_cython__(self)"""

class cudaMemPoolPtrExportData:
    reserved: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaMemPoolPtrExportData.getPtr(self)"""
    def __reduce__(self):
        """cudaMemPoolPtrExportData.__reduce_cython__(self)"""

class cudaMemPool_t(cuda.bindings.driver.CUmemoryPool):
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""

class cudaMemRangeAttribute(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaMemRangeAttributeAccessedBy: ClassVar[cudaMemRangeAttribute] = ...
    cudaMemRangeAttributeLastPrefetchLocation: ClassVar[cudaMemRangeAttribute] = ...
    cudaMemRangeAttributeLastPrefetchLocationId: ClassVar[cudaMemRangeAttribute] = ...
    cudaMemRangeAttributeLastPrefetchLocationType: ClassVar[cudaMemRangeAttribute] = ...
    cudaMemRangeAttributePreferredLocation: ClassVar[cudaMemRangeAttribute] = ...
    cudaMemRangeAttributePreferredLocationId: ClassVar[cudaMemRangeAttribute] = ...
    cudaMemRangeAttributePreferredLocationType: ClassVar[cudaMemRangeAttribute] = ...
    cudaMemRangeAttributeReadMostly: ClassVar[cudaMemRangeAttribute] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaMemcpy3DBatchOp:
    dst: Incomplete
    extent: Incomplete
    flags: Incomplete
    src: Incomplete
    srcAccessOrder: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaMemcpy3DBatchOp.getPtr(self)"""
    def __reduce__(self):
        """cudaMemcpy3DBatchOp.__reduce_cython__(self)"""

class cudaMemcpy3DOperand:
    op: Incomplete
    type: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaMemcpy3DOperand.getPtr(self)"""
    def __reduce__(self):
        """cudaMemcpy3DOperand.__reduce_cython__(self)"""

class cudaMemcpy3DOperandType(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaMemcpyOperandTypeArray: ClassVar[cudaMemcpy3DOperandType] = ...
    cudaMemcpyOperandTypeMax: ClassVar[cudaMemcpy3DOperandType] = ...
    cudaMemcpyOperandTypePointer: ClassVar[cudaMemcpy3DOperandType] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaMemcpy3DParms:
    dstArray: Incomplete
    dstPos: Incomplete
    dstPtr: Incomplete
    extent: Incomplete
    kind: Incomplete
    srcArray: Incomplete
    srcPos: Incomplete
    srcPtr: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaMemcpy3DParms.getPtr(self)"""
    def __reduce__(self):
        """cudaMemcpy3DParms.__reduce_cython__(self)"""

class cudaMemcpy3DPeerParms:
    dstArray: Incomplete
    dstDevice: Incomplete
    dstPos: Incomplete
    dstPtr: Incomplete
    extent: Incomplete
    srcArray: Incomplete
    srcDevice: Incomplete
    srcPos: Incomplete
    srcPtr: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaMemcpy3DPeerParms.getPtr(self)"""
    def __reduce__(self):
        """cudaMemcpy3DPeerParms.__reduce_cython__(self)"""

class cudaMemcpyAttributes:
    dstLocHint: Incomplete
    flags: Incomplete
    srcAccessOrder: Incomplete
    srcLocHint: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaMemcpyAttributes.getPtr(self)"""
    def __reduce__(self):
        """cudaMemcpyAttributes.__reduce_cython__(self)"""

class cudaMemcpyFlags(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaMemcpyFlagDefault: ClassVar[cudaMemcpyFlags] = ...
    cudaMemcpyFlagPreferOverlapWithCompute: ClassVar[cudaMemcpyFlags] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaMemcpyKind(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaMemcpyDefault: ClassVar[cudaMemcpyKind] = ...
    cudaMemcpyDeviceToDevice: ClassVar[cudaMemcpyKind] = ...
    cudaMemcpyDeviceToHost: ClassVar[cudaMemcpyKind] = ...
    cudaMemcpyHostToDevice: ClassVar[cudaMemcpyKind] = ...
    cudaMemcpyHostToHost: ClassVar[cudaMemcpyKind] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaMemcpyNodeParams:
    copyParams: Incomplete
    flags: Incomplete
    reserved: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaMemcpyNodeParams.getPtr(self)"""
    def __reduce__(self):
        """cudaMemcpyNodeParams.__reduce_cython__(self)"""

class cudaMemcpySrcAccessOrder(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaMemcpySrcAccessOrderAny: ClassVar[cudaMemcpySrcAccessOrder] = ...
    cudaMemcpySrcAccessOrderDuringApiCall: ClassVar[cudaMemcpySrcAccessOrder] = ...
    cudaMemcpySrcAccessOrderInvalid: ClassVar[cudaMemcpySrcAccessOrder] = ...
    cudaMemcpySrcAccessOrderMax: ClassVar[cudaMemcpySrcAccessOrder] = ...
    cudaMemcpySrcAccessOrderStream: ClassVar[cudaMemcpySrcAccessOrder] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaMemoryAdvise(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaMemAdviseSetAccessedBy: ClassVar[cudaMemoryAdvise] = ...
    cudaMemAdviseSetPreferredLocation: ClassVar[cudaMemoryAdvise] = ...
    cudaMemAdviseSetReadMostly: ClassVar[cudaMemoryAdvise] = ...
    cudaMemAdviseUnsetAccessedBy: ClassVar[cudaMemoryAdvise] = ...
    cudaMemAdviseUnsetPreferredLocation: ClassVar[cudaMemoryAdvise] = ...
    cudaMemAdviseUnsetReadMostly: ClassVar[cudaMemoryAdvise] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaMemoryType(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaMemoryTypeDevice: ClassVar[cudaMemoryType] = ...
    cudaMemoryTypeHost: ClassVar[cudaMemoryType] = ...
    cudaMemoryTypeManaged: ClassVar[cudaMemoryType] = ...
    cudaMemoryTypeUnregistered: ClassVar[cudaMemoryType] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaMemsetParams:
    dst: Incomplete
    elementSize: Incomplete
    height: Incomplete
    pitch: Incomplete
    value: Incomplete
    width: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaMemsetParams.getPtr(self)"""
    def __reduce__(self):
        """cudaMemsetParams.__reduce_cython__(self)"""

class cudaMemsetParamsV2:
    dst: Incomplete
    elementSize: Incomplete
    height: Incomplete
    pitch: Incomplete
    value: Incomplete
    width: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaMemsetParamsV2.getPtr(self)"""
    def __reduce__(self):
        """cudaMemsetParamsV2.__reduce_cython__(self)"""

class cudaMipmappedArray_const_t:
    def __init__(self, *args, **kwargs) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaMipmappedArray_const_t.getPtr(self)"""
    def __index__(self) -> int:
        """Return self converted to an integer, if self is suitable for use as an index into a list."""
    def __int__(self) -> int:
        """int(self)"""
    def __reduce__(self):
        """cudaMipmappedArray_const_t.__reduce_cython__(self)"""

class cudaMipmappedArray_t:
    def __init__(self, *args, **kwargs) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaMipmappedArray_t.getPtr(self)"""
    def __index__(self) -> int:
        """Return self converted to an integer, if self is suitable for use as an index into a list."""
    def __int__(self) -> int:
        """int(self)"""
    def __reduce__(self):
        """cudaMipmappedArray_t.__reduce_cython__(self)"""

class cudaOffset3D:
    x: Incomplete
    y: Incomplete
    z: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaOffset3D.getPtr(self)"""
    def __reduce__(self):
        """cudaOffset3D.__reduce_cython__(self)"""

class cudaPitchedPtr:
    pitch: Incomplete
    ptr: Incomplete
    xsize: Incomplete
    ysize: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaPitchedPtr.getPtr(self)"""
    def __reduce__(self):
        """cudaPitchedPtr.__reduce_cython__(self)"""

class cudaPointerAttributes:
    device: Incomplete
    devicePointer: Incomplete
    hostPointer: Incomplete
    type: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaPointerAttributes.getPtr(self)"""
    def __reduce__(self):
        """cudaPointerAttributes.__reduce_cython__(self)"""

class cudaPos:
    x: Incomplete
    y: Incomplete
    z: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaPos.getPtr(self)"""
    def __reduce__(self):
        """cudaPos.__reduce_cython__(self)"""

class cudaResourceDesc:
    res: Incomplete
    resType: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaResourceDesc.getPtr(self)"""
    def __reduce__(self):
        """cudaResourceDesc.__reduce_cython__(self)"""

class cudaResourceType(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaResourceTypeArray: ClassVar[cudaResourceType] = ...
    cudaResourceTypeLinear: ClassVar[cudaResourceType] = ...
    cudaResourceTypeMipmappedArray: ClassVar[cudaResourceType] = ...
    cudaResourceTypePitch2D: ClassVar[cudaResourceType] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaResourceViewDesc:
    depth: Incomplete
    firstLayer: Incomplete
    firstMipmapLevel: Incomplete
    format: Incomplete
    height: Incomplete
    lastLayer: Incomplete
    lastMipmapLevel: Incomplete
    width: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaResourceViewDesc.getPtr(self)"""
    def __reduce__(self):
        """cudaResourceViewDesc.__reduce_cython__(self)"""

class cudaResourceViewFormat(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaResViewFormatFloat1: ClassVar[cudaResourceViewFormat] = ...
    cudaResViewFormatFloat2: ClassVar[cudaResourceViewFormat] = ...
    cudaResViewFormatFloat4: ClassVar[cudaResourceViewFormat] = ...
    cudaResViewFormatHalf1: ClassVar[cudaResourceViewFormat] = ...
    cudaResViewFormatHalf2: ClassVar[cudaResourceViewFormat] = ...
    cudaResViewFormatHalf4: ClassVar[cudaResourceViewFormat] = ...
    cudaResViewFormatNone: ClassVar[cudaResourceViewFormat] = ...
    cudaResViewFormatSignedBlockCompressed4: ClassVar[cudaResourceViewFormat] = ...
    cudaResViewFormatSignedBlockCompressed5: ClassVar[cudaResourceViewFormat] = ...
    cudaResViewFormatSignedBlockCompressed6H: ClassVar[cudaResourceViewFormat] = ...
    cudaResViewFormatSignedChar1: ClassVar[cudaResourceViewFormat] = ...
    cudaResViewFormatSignedChar2: ClassVar[cudaResourceViewFormat] = ...
    cudaResViewFormatSignedChar4: ClassVar[cudaResourceViewFormat] = ...
    cudaResViewFormatSignedInt1: ClassVar[cudaResourceViewFormat] = ...
    cudaResViewFormatSignedInt2: ClassVar[cudaResourceViewFormat] = ...
    cudaResViewFormatSignedInt4: ClassVar[cudaResourceViewFormat] = ...
    cudaResViewFormatSignedShort1: ClassVar[cudaResourceViewFormat] = ...
    cudaResViewFormatSignedShort2: ClassVar[cudaResourceViewFormat] = ...
    cudaResViewFormatSignedShort4: ClassVar[cudaResourceViewFormat] = ...
    cudaResViewFormatUnsignedBlockCompressed1: ClassVar[cudaResourceViewFormat] = ...
    cudaResViewFormatUnsignedBlockCompressed2: ClassVar[cudaResourceViewFormat] = ...
    cudaResViewFormatUnsignedBlockCompressed3: ClassVar[cudaResourceViewFormat] = ...
    cudaResViewFormatUnsignedBlockCompressed4: ClassVar[cudaResourceViewFormat] = ...
    cudaResViewFormatUnsignedBlockCompressed5: ClassVar[cudaResourceViewFormat] = ...
    cudaResViewFormatUnsignedBlockCompressed6H: ClassVar[cudaResourceViewFormat] = ...
    cudaResViewFormatUnsignedBlockCompressed7: ClassVar[cudaResourceViewFormat] = ...
    cudaResViewFormatUnsignedChar1: ClassVar[cudaResourceViewFormat] = ...
    cudaResViewFormatUnsignedChar2: ClassVar[cudaResourceViewFormat] = ...
    cudaResViewFormatUnsignedChar4: ClassVar[cudaResourceViewFormat] = ...
    cudaResViewFormatUnsignedInt1: ClassVar[cudaResourceViewFormat] = ...
    cudaResViewFormatUnsignedInt2: ClassVar[cudaResourceViewFormat] = ...
    cudaResViewFormatUnsignedInt4: ClassVar[cudaResourceViewFormat] = ...
    cudaResViewFormatUnsignedShort1: ClassVar[cudaResourceViewFormat] = ...
    cudaResViewFormatUnsignedShort2: ClassVar[cudaResourceViewFormat] = ...
    cudaResViewFormatUnsignedShort4: ClassVar[cudaResourceViewFormat] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaRoundMode(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaRoundMinInf: ClassVar[cudaRoundMode] = ...
    cudaRoundNearest: ClassVar[cudaRoundMode] = ...
    cudaRoundPosInf: ClassVar[cudaRoundMode] = ...
    cudaRoundZero: ClassVar[cudaRoundMode] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaSharedCarveout(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaSharedmemCarveoutDefault: ClassVar[cudaSharedCarveout] = ...
    cudaSharedmemCarveoutMaxL1: ClassVar[cudaSharedCarveout] = ...
    cudaSharedmemCarveoutMaxShared: ClassVar[cudaSharedCarveout] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaSharedMemConfig(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaSharedMemBankSizeDefault: ClassVar[cudaSharedMemConfig] = ...
    cudaSharedMemBankSizeEightByte: ClassVar[cudaSharedMemConfig] = ...
    cudaSharedMemBankSizeFourByte: ClassVar[cudaSharedMemConfig] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaStreamAttrID(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaLaunchAttributeAccessPolicyWindow: ClassVar[cudaStreamAttrID] = ...
    cudaLaunchAttributeClusterDimension: ClassVar[cudaStreamAttrID] = ...
    cudaLaunchAttributeClusterSchedulingPolicyPreference: ClassVar[cudaStreamAttrID] = ...
    cudaLaunchAttributeCooperative: ClassVar[cudaStreamAttrID] = ...
    cudaLaunchAttributeDeviceUpdatableKernelNode: ClassVar[cudaStreamAttrID] = ...
    cudaLaunchAttributeIgnore: ClassVar[cudaStreamAttrID] = ...
    cudaLaunchAttributeLaunchCompletionEvent: ClassVar[cudaStreamAttrID] = ...
    cudaLaunchAttributeMemSyncDomain: ClassVar[cudaStreamAttrID] = ...
    cudaLaunchAttributeMemSyncDomainMap: ClassVar[cudaStreamAttrID] = ...
    cudaLaunchAttributePreferredClusterDimension: ClassVar[cudaStreamAttrID] = ...
    cudaLaunchAttributePreferredSharedMemoryCarveout: ClassVar[cudaStreamAttrID] = ...
    cudaLaunchAttributePriority: ClassVar[cudaStreamAttrID] = ...
    cudaLaunchAttributeProgrammaticEvent: ClassVar[cudaStreamAttrID] = ...
    cudaLaunchAttributeProgrammaticStreamSerialization: ClassVar[cudaStreamAttrID] = ...
    cudaLaunchAttributeSynchronizationPolicy: ClassVar[cudaStreamAttrID] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaStreamAttrValue(cudaLaunchAttributeValue):
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""

class cudaStreamCallback_t:
    def __init__(self, *args, **kwargs) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaStreamCallback_t.getPtr(self)"""
    def __index__(self) -> int:
        """Return self converted to an integer, if self is suitable for use as an index into a list."""
    def __int__(self) -> int:
        """int(self)"""
    def __reduce__(self):
        """cudaStreamCallback_t.__reduce_cython__(self)"""

class cudaStreamCaptureMode(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaStreamCaptureModeGlobal: ClassVar[cudaStreamCaptureMode] = ...
    cudaStreamCaptureModeRelaxed: ClassVar[cudaStreamCaptureMode] = ...
    cudaStreamCaptureModeThreadLocal: ClassVar[cudaStreamCaptureMode] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaStreamCaptureStatus(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaStreamCaptureStatusActive: ClassVar[cudaStreamCaptureStatus] = ...
    cudaStreamCaptureStatusInvalidated: ClassVar[cudaStreamCaptureStatus] = ...
    cudaStreamCaptureStatusNone: ClassVar[cudaStreamCaptureStatus] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaStreamUpdateCaptureDependenciesFlags(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaStreamAddCaptureDependencies: ClassVar[cudaStreamUpdateCaptureDependenciesFlags] = ...
    cudaStreamSetCaptureDependencies: ClassVar[cudaStreamUpdateCaptureDependenciesFlags] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaStream_t(cuda.bindings.driver.CUstream):
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""

class cudaSurfaceBoundaryMode(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaBoundaryModeClamp: ClassVar[cudaSurfaceBoundaryMode] = ...
    cudaBoundaryModeTrap: ClassVar[cudaSurfaceBoundaryMode] = ...
    cudaBoundaryModeZero: ClassVar[cudaSurfaceBoundaryMode] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaSurfaceFormatMode(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaFormatModeAuto: ClassVar[cudaSurfaceFormatMode] = ...
    cudaFormatModeForced: ClassVar[cudaSurfaceFormatMode] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaSurfaceObject_t:
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaSurfaceObject_t.getPtr(self)"""
    def __int__(self) -> int:
        """int(self)"""
    def __reduce__(self):
        """cudaSurfaceObject_t.__reduce_cython__(self)"""

class cudaSynchronizationPolicy(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaSyncPolicyAuto: ClassVar[cudaSynchronizationPolicy] = ...
    cudaSyncPolicyBlockingSync: ClassVar[cudaSynchronizationPolicy] = ...
    cudaSyncPolicySpin: ClassVar[cudaSynchronizationPolicy] = ...
    cudaSyncPolicyYield: ClassVar[cudaSynchronizationPolicy] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaTextureAddressMode(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaAddressModeBorder: ClassVar[cudaTextureAddressMode] = ...
    cudaAddressModeClamp: ClassVar[cudaTextureAddressMode] = ...
    cudaAddressModeMirror: ClassVar[cudaTextureAddressMode] = ...
    cudaAddressModeWrap: ClassVar[cudaTextureAddressMode] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaTextureDesc:
    addressMode: Incomplete
    borderColor: Incomplete
    disableTrilinearOptimization: Incomplete
    filterMode: Incomplete
    maxAnisotropy: Incomplete
    maxMipmapLevelClamp: Incomplete
    minMipmapLevelClamp: Incomplete
    mipmapFilterMode: Incomplete
    mipmapLevelBias: Incomplete
    normalizedCoords: Incomplete
    readMode: Incomplete
    sRGB: Incomplete
    seamlessCubemap: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaTextureDesc.getPtr(self)"""
    def __reduce__(self):
        """cudaTextureDesc.__reduce_cython__(self)"""

class cudaTextureFilterMode(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaFilterModeLinear: ClassVar[cudaTextureFilterMode] = ...
    cudaFilterModePoint: ClassVar[cudaTextureFilterMode] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaTextureObject_t:
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def getPtr(self) -> Any:
        """cudaTextureObject_t.getPtr(self)"""
    def __int__(self) -> int:
        """int(self)"""
    def __reduce__(self):
        """cudaTextureObject_t.__reduce_cython__(self)"""

class cudaTextureReadMode(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaReadModeElementType: ClassVar[cudaTextureReadMode] = ...
    cudaReadModeNormalizedFloat: ClassVar[cudaTextureReadMode] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaUUID_t(CUuuid_st):
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""

class cudaUserObjectFlags(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaUserObjectNoDestructorSync: ClassVar[cudaUserObjectFlags] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaUserObjectRetainFlags(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    cudaGraphUserObjectMove: ClassVar[cudaUserObjectRetainFlags] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class cudaUserObject_t(cuda.bindings.driver.CUuserObject):
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""

class cudalibraryHostUniversalFunctionAndDataTable:
    dataTable: Incomplete
    dataWindowSize: Incomplete
    functionTable: Incomplete
    functionWindowSize: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """cudalibraryHostUniversalFunctionAndDataTable.getPtr(self)"""
    def __reduce__(self):
        """cudalibraryHostUniversalFunctionAndDataTable.__reduce_cython__(self)"""

class dim3:
    x: Incomplete
    y: Incomplete
    z: Incomplete
    def __init__(self, void_ptr_ptr=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getPtr(self) -> Any:
        """dim3.getPtr(self)"""
    def __reduce__(self):
        """dim3.__reduce_cython__(self)"""

class libraryPropertyType(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    MAJOR_VERSION: ClassVar[libraryPropertyType] = ...
    MINOR_VERSION: ClassVar[libraryPropertyType] = ...
    PATCH_LEVEL: ClassVar[libraryPropertyType] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""