from typing import *
import enum
from typing import Any, Callable, ClassVar

__pyx_capi__: dict
__test__: dict

def add_data(handle, input_type, data, size, name):
    """
    add_data(intptr_t handle, int input_type, data, size_t size, name)
    nvJitLinkAddData adds data image to the link.

        Args:
            handle (intptr_t): nvJitLink handle.
            input_type (InputType): kind of input.
            data (bytes): pointer to data image in memory.
            size (size_t): size of the data.
            name (str): name of input object.

        .. seealso:: `nvJitLinkAddData`
    """


def add_file(handle, input_type, file_name):
    """
    add_file(intptr_t handle, int input_type, file_name)
    nvJitLinkAddFile reads data from file and links it in.

        Args:
            handle (intptr_t): nvJitLink handle.
            input_type (InputType): kind of input.
            file_name (str): name of file.

        .. seealso:: `nvJitLinkAddFile`
    """


def complete(handle):
    """
    complete(intptr_t handle)
    nvJitLinkComplete does the actual link.

        Args:
            handle (intptr_t): nvJitLink handle.

        .. seealso:: `nvJitLinkComplete`
    """


def create(num_options, options):
    """
    create(uint32_t num_options, options) -> intptr_t
    nvJitLinkCreate creates an instance of nvJitLinkHandle with the given input options, and sets the output parameter ``handle``.

        Args:
            num_options (uint32_t): Number of options passed.
            options (object): Array of size ``num_options`` of option strings. It can be:

                - an :class:`int` as the pointer address to the nested sequence, or
                - a Python sequence of :class:`int`\s, each of which is a pointer address
                  to a valid sequence of 'char', or
                - a nested Python sequence of ``str``.


        Returns:
            intptr_t: Address of nvJitLink handle.

        .. seealso:: `nvJitLinkCreate`
    """


def destroy(handle):
    """
    destroy(intptr_t handle)
    nvJitLinkDestroy frees the memory associated with the given handle.

        Args:
            handle (intptr_t): nvJitLink handle.

        .. seealso:: `nvJitLinkDestroy`
    """


def get_error_log(handle, log):
    """
    get_error_log(intptr_t handle, log)
    nvJitLinkGetErrorLog puts any error messages in the log.

        Args:
            handle (intptr_t): nvJitLink handle.
            log (bytes): The error log.

        .. seealso:: `nvJitLinkGetErrorLog`
    """


def get_error_log_size(handle):
    """
    get_error_log_size(intptr_t handle) -> size_t
    nvJitLinkGetErrorLogSize gets the size of the error log.

        Args:
            handle (intptr_t): nvJitLink handle.

        Returns:
            size_t: Size of the error log.

        .. seealso:: `nvJitLinkGetErrorLogSize`
    """


def get_info_log(handle, log):
    """
    get_info_log(intptr_t handle, log)
    nvJitLinkGetInfoLog puts any info messages in the log.

        Args:
            handle (intptr_t): nvJitLink handle.
            log (bytes): The info log.

        .. seealso:: `nvJitLinkGetInfoLog`
    """


def get_info_log_size(handle):
    """
    get_info_log_size(intptr_t handle) -> size_t
    nvJitLinkGetInfoLogSize gets the size of the info log.

        Args:
            handle (intptr_t): nvJitLink handle.

        Returns:
            size_t: Size of the info log.

        .. seealso:: `nvJitLinkGetInfoLogSize`
    """


def get_linked_cubin(handle, cubin):
    """
    get_linked_cubin(intptr_t handle, cubin)
    nvJitLinkGetLinkedCubin gets the linked cubin.

        Args:
            handle (intptr_t): nvJitLink handle.
            cubin (bytes): The linked cubin.

        .. seealso:: `nvJitLinkGetLinkedCubin`
    """


def get_linked_cubin_size(handle):
    """
    get_linked_cubin_size(intptr_t handle) -> size_t
    nvJitLinkGetLinkedCubinSize gets the size of the linked cubin.

        Args:
            handle (intptr_t): nvJitLink handle.

        Returns:
            size_t: Size of the linked cubin.

        .. seealso:: `nvJitLinkGetLinkedCubinSize`
    """


def get_linked_ptx(handle, ptx):
    """
    get_linked_ptx(intptr_t handle, ptx)
    nvJitLinkGetLinkedPtx gets the linked ptx.

        Args:
            handle (intptr_t): nvJitLink handle.
            ptx (bytes): The linked PTX.

        .. seealso:: `nvJitLinkGetLinkedPtx`
    """


def get_linked_ptx_size(handle):
    """
    get_linked_ptx_size(intptr_t handle) -> size_t
    nvJitLinkGetLinkedPtxSize gets the size of the linked ptx.

        Args:
            handle (intptr_t): nvJitLink handle.

        Returns:
            size_t: Size of the linked PTX.

        .. seealso:: `nvJitLinkGetLinkedPtxSize`
    """


def version():
    """
    version() -> tuple
    nvJitLinkVersion returns the current version of nvJitLink.

        Returns:
            A 2-tuple containing:

            - unsigned int: The major version.
            - unsigned int: The minor version.

        .. seealso:: `nvJitLinkVersion`
    """


class InputType(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    ANY: ClassVar[InputType] = ...
    CUBIN: ClassVar[InputType] = ...
    FATBIN: ClassVar[InputType] = ...
    INDEX: ClassVar[InputType] = ...
    LIBRARY: ClassVar[InputType] = ...
    LTOIR: ClassVar[InputType] = ...
    NONE: ClassVar[InputType] = ...
    OBJECT: ClassVar[InputType] = ...
    PTX: ClassVar[InputType] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class Result(enum.IntEnum):
    __new__: ClassVar[Callable] = ...
    ERROR_FINALIZE: ClassVar[Result] = ...
    ERROR_INTERNAL: ClassVar[Result] = ...
    ERROR_INVALID_INPUT: ClassVar[Result] = ...
    ERROR_MISSING_ARCH: ClassVar[Result] = ...
    ERROR_NVVM_COMPILE: ClassVar[Result] = ...
    ERROR_PTX_COMPILE: ClassVar[Result] = ...
    ERROR_THREADPOOL: ClassVar[Result] = ...
    ERROR_UNRECOGNIZED_INPUT: ClassVar[Result] = ...
    ERROR_UNRECOGNIZED_OPTION: ClassVar[Result] = ...
    SUCCESS: ClassVar[Result] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""

class nvJitLinkError(Exception):
    def __init__(self, status) -> Any:
        """nvJitLinkError.__init__(self, status)"""
    def __reduce__(self) -> Any:
        """nvJitLinkError.__reduce__(self)"""