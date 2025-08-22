# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import dataclasses
import inspect
import os
import threading
import typing as tp
from abc import ABCMeta
from copy import deepcopy

from flax.nnx import variablelib
import jax
import numpy as np
import treescope  # type: ignore[import-untyped]
from treescope import rendering_parts

from flax import errors, nnx
from flax.nnx import (
  graph,
  reprlib,
  tracers,
  visualization,
)
from flax import config
from flax.nnx.variablelib import Variable
from flax.typing import SizeBytes

BUILDING_DOCS = 'FLAX_DOC_BUILD' in os.environ

A = tp.TypeVar('A')
O = tp.TypeVar('O', bound='Object')

DataAnnotation = '__data__'
Data = tp.Annotated[A, DataAnnotation]
Data.__doc__ = """Data marks attributes of a class as pytree data using type annotations.

Data annotations must be used at the class level and will apply to all instances.
The usage of Data is recommended when type annotations are used already present
or required e.g. for dataclasses.

Example::

  from flax import nnx
  import jax
  import dataclasses

  @dataclasses.dataclass
  class Foo(nnx.Pytree):
    a: nnx.Data[int]  # Annotates `a` as pytree data
    b: str            # `b` is not pytree data

  foo = Foo(a=42, b='hello')

  assert jax.tree.leaves(foo) == [42]
"""
DATA_REGISTRY: set[type] = set()


@dataclasses.dataclass(frozen=True, slots=True)
class DataAttr:
  value: tp.Any


def data(value: A, /) -> A:
  """Annotates a an attribute as pytree data.

  The return value from `data` must be directly assigned to an Object attribute
  which will be registered as a pytree data attribute.

  Example::

    from flax import nnx
    import jax

    class Foo(nnx.Pytree):
      def __init__(self):
        self.data_attr = nnx.data(42)  # pytree data
        self.static_attr = "hello"     # static attribute

    foo = Foo()

    assert jax.tree.leaves(foo) == [42]

  Args:
    value: The value to annotate as data.

  Returns:
    A value which will register the attribute as data on assignment.

  """
  return DataAttr(value)  # type: ignore[return-value]


def register_data_type(type_: type, /) -> None:
  """Registers a type as pytree data type recognized by Object.

  Custom types registered as data will be automatically recognized
  as data attributes when assigned to an Object attribute. This means
  that values of this type do not need to be wrapped in `nnx.data(...)`
  for Object to mark the attribute its being assigned to as data.

  Example::

    from flax import nnx
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class MyType:
      value: int

    nnx.register_data_type(MyType)

    class Foo(nnx.Pytree):
      def __init__(self, a):
        self.a = MyType(a)  # Automatically registered as data
        self.b = "hello"     # str not registered as data

    foo = Foo(42)

    assert nnx.is_data(foo.a)  # True
    assert jax.tree.leaves(foo) == [MyType(value=42)]

  """
  DATA_REGISTRY.add(type_)

def _leaf_is_data(value: tp.Any, /) -> bool:
  return (
    graph.is_node_leaf(value)
    or graph.is_graph_node(value)
    or type(value) in DATA_REGISTRY
  )

UNDEFINED_TYPES: set[type] = {
  int,
  float,
  complex,
  bool,
  jax.ShapeDtypeStruct,
  jax.sharding.PartitionSpec,
  jax.sharding.NamedSharding,
}

def register_undefined_type(t: type, /) -> None:
  """Registers a type as an undefined type recognized by nnx.Pytree.
  """
  UNDEFINED_TYPES.add(t)

def _leaf_is_undefined(value: tp.Any, /) -> bool:
  return type(value) in UNDEFINED_TYPES

def is_data(value: tp.Any, /) -> bool | None:
  """Checks if a value is a registered data type.

  This function checks a the value is registered data type, which means it is
  automatically recognized as data when assigned a nnx.Pytree attribute.

  Data types are:
  - jax.Arrays
  - np.ndarrays
  - ArrayRefs
  - Variables (Param, BatchStat, RngState, etc.)
  - All graph nodes (Object, Module, Rngs, etc.)
  - Any type registered with `nnx.register_data_type`
  - Any pytree that contains at least one node or leaf element of the above

  Example::

    >>> from flax import nnx
    >>> import jax.numpy as jnp
    ...
    >>> module = nnx.Linear(1, 1, rngs=nnx.Rngs(0))
    >>> blocks = [module, module, module]
    >>> shape = [32, 16, 512]
    >>> stuff = [1, "hello", jnp.array(42)]
    ...
    >>> assert nnx.is_data(jnp.array(42)) == True  # Arrays are data
    >>> assert nnx.is_data(nnx.Param(1)) == True   # Variables are data
    >>> assert nnx.is_data(nnx.Rngs(0)) == True    # nnx.Pytrees are data
    >>> assert nnx.is_data(module) == True         # nnx.Modules are data
    >>> assert nnx.is_data(blocks) == True         # pytrees with data are data
    ...
    >>> assert nnx.is_data(stuff) == True          # pytrees with data and static are mixed
    >>> assert nnx.is_data(0.) is None             # float is a weak type
    >>> assert nnx.is_data(1) is None              # int is a weak type
    >>> assert nnx.is_data("hello") == False       # str is static
    >>> assert nnx.is_data(shape) is None          # pytrees with weak types are undefined


  Args:
    value: The value to check.

  Returns:
    A string representing the attribute status.
  """
  is_static = False

  for leaf in jax.tree.leaves(value, is_leaf=_leaf_is_data):
    if _leaf_is_data(leaf):
      return True
    elif not _leaf_is_undefined(leaf):
      is_static = True

  if is_static:
    return False
  else:
    return None

StaticAnnotation = '__static__'
Static = tp.Annotated[A, StaticAnnotation]
Static.__doc__ = """Static marks attributes of a class as static using type annotations.
Static annotations must be used at the class level and will apply to all instances.
The usage of Static is recommended when type annotations are used already present
or required e.g. for dataclasses.
"""

@dataclasses.dataclass(frozen=True, slots=True)
class StaticAttr:
  value: tp.Any

def static(value: A, /) -> A:
  """Annotates a an attribute as static.

  The return value from `static` must be directly assigned to an Object attribute
  which will be registered as static attribute.

  Example::

    from flax import nnx

    class Foo(nnx.Pytree):
      def __init__(self, a, b):
        self.a = nnx.static(a)  # pytree metadata
        self.b = nnx.data(b)    # pytree data

    foo = Foo("one", "two")

    assert jax.tree.leaves(foo) == ["two"]

  By default ``nnx.Pytree`` will ...
  """
  return StaticAttr(value)  # type: ignore[return-value]


def _collect_stats(
  node: tp.Any, node_stats: dict[int, dict[type[Variable], SizeBytes]]
):
  if not graph.is_node(node) and not isinstance(node, Variable):
    raise ValueError(f'Expected a graph node or Variable, got {type(node)!r}.')

  if id(node) in node_stats:
    return

  stats: dict[type[Variable], SizeBytes] = {}
  node_stats[id(node)] = stats

  if isinstance(node, Variable):
    var_type = type(node)
    if issubclass(var_type, nnx.RngState):
      var_type = nnx.RngState
    size_bytes = SizeBytes.from_any(node.raw_value)
    if size_bytes:
      stats[var_type] = size_bytes

  else:
    node_impl = graph.get_node_impl(node)
    assert node_impl is not None
    node_dict = node_impl.node_dict(node)
    for key, value in node_dict.items():
      if id(value) in node_stats:
        continue
      if graph.is_node(value) or isinstance(value, Variable):
        _collect_stats(value, node_stats)
        child_stats = node_stats[id(value)]
        for var_type, size_bytes in child_stats.items():
          if var_type in stats:
            stats[var_type] += size_bytes
          else:
            stats[var_type] = size_bytes


@dataclasses.dataclass
class ObjectContext(threading.local):
  seen_modules_repr: set[int] | None = None
  node_stats: dict[int, dict[type[Variable], SizeBytes]] | None = None


OBJECT_CONTEXT = ObjectContext()


class PytreeState(reprlib.Representable):
  __slots__ = ('_trace_state', '_initializing', '_is_setup')

  def __init__(self, initializing: bool = False, is_setup: bool = False):
    self._trace_state = tracers.TraceState()
    self._initializing = initializing
    self._is_setup = is_setup

  @property
  def trace_state(self) -> tracers.TraceState:
    return self._trace_state

  @property
  def initializing(self) -> bool:
    return self._initializing

  @property
  def is_setup(self) -> bool:
    return self._is_setup

  def __nnx_repr__(self):
    yield reprlib.Object(type(self))
    yield reprlib.Attr('trace_state', self._trace_state)

  def __treescope_repr__(self, path, subtree_renderer):
    return visualization.render_object_constructor(
      object_type=type(self),
      attributes={'trace_state': self._trace_state},
      path=path,
      subtree_renderer=subtree_renderer,
    )


def _flatten_pytree_state(state: PytreeState):
  return (), (state.initializing, state.is_setup)


def _unflatten_pytree_state(static: tuple[bool, bool], _):
  initializing, setup = static
  return PytreeState(initializing, setup)


jax.tree_util.register_pytree_node(
  PytreeState,
  _flatten_pytree_state,
  _unflatten_pytree_state,
)


def check_pytree(pytree):
  """Checks if a pytree is valid."""
  if not isinstance(pytree, Pytree):
    raise TypeError(f"Expected a Pytree, got {type(pytree)}.")

  for name, value in vars(pytree).items():
    is_static = name not in pytree._pytree__data_mapping or not pytree._pytree__data_mapping[name]
    is_annotated = name in type(pytree)._pytree__class_nodes
    _check_value(name, value, is_static, incomming=False, annotated=is_annotated)

def _check_value(name: str, value, is_static: bool | None, incomming: bool, annotated: bool):
  def _has_arrays(leaves):
    return any(
      isinstance(leaf, (np.ndarray, jax.Array)) or variablelib.is_array_ref(leaf)
      for leaf in leaves
    )
  def _get_tags(leaves):
    return {
      leaf for leaf in leaves
      if type(leaf) is DataAttr or type(leaf) is StaticAttr
    }
  visited = set()
  def _has_visited(x):
    if id(x) in visited:
      return True
    visited.add(id(x))
    return False
  leaves = jax.tree.leaves(value, is_leaf=_has_visited)
  if is_static and _has_arrays(leaves):
    if annotated:
      if incomming:
        raise ValueError(
          f"Found arrays in value annotated with nnx.static(...) when setting "
          f"attribute '{name}'."
        )
      raise ValueError(
        f"Found arrays in attribute '{name}' which was annotated with nnx.Static."
      )
    raise ValueError(
      f"Found unexpected arrays on attribute '{name}'. Consider using "
      "nnx.data on assignment:\n\n"
      f"  self.{name} = nnx.data(...)\n"
    )
  if (tags := _get_tags(leaves)):
    raise ValueError(
      f"Found unexpected tags {tags} on attribute '{name}'. Values from nnx.data(...) and "
      f"nnx.static(...) can only be assigned to Pytree attributes directly, they should not "
      f"be stored inside lists, dicts, tuples, or other pytree types."
    )


class PytreeMeta(ABCMeta):
  if not tp.TYPE_CHECKING:

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
      return _graph_node_meta_call(cls, *args, **kwargs)

  def _pytree_meta_construct(cls, self, *args, **kwargs):
    self.__init__(*args, **kwargs)

ObjectMeta = PytreeMeta

def _graph_node_meta_call(cls: tp.Type[O], *args, **kwargs) -> O:
  node = cls.__new__(cls, *args, **kwargs)
  vars_obj = vars(node)
  vars_obj['_pytree__state'] = PytreeState()
  vars_obj['_pytree__nodes'] = graph.HashableMapping({}, copy=False)
  cls._pytree_meta_construct(node, *args, **kwargs)
  if cls._pytree__is_pytree:
    check_pytree(node)

  return node


@dataclasses.dataclass(frozen=True, repr=False)
class ArrayRepr(reprlib.Representable):
  shape: tp.Tuple[int, ...]
  dtype: tp.Any

  @staticmethod
  def from_array(array: jax.Array | np.ndarray) -> ArrayRepr:
    return ArrayRepr(array.shape, array.dtype)

  def __nnx_repr__(self):
    yield reprlib.Object(type='Array', same_line=True)
    yield reprlib.Attr('shape', self.shape)
    yield reprlib.Attr('dtype', self.dtype)


@dataclasses.dataclass(frozen=True, repr=False)
class MutableArrayRepr(reprlib.Representable):
  shape: tp.Tuple[int, ...]
  dtype: tp.Any

  @staticmethod
  def from_array(array: jax.Array | np.ndarray) -> MutableArrayRepr:
    return MutableArrayRepr(array.shape, array.dtype)

  def __nnx_repr__(self):
    yield reprlib.Object(type='ArrayRef', same_line=True)
    yield reprlib.Attr('shape', self.shape)
    yield reprlib.Attr('dtype', self.dtype)

@dataclasses.dataclass(repr=False)
class NodeDataMapping(reprlib.MappingReprMixin):
  pytree: Pytree

  @property
  def class_nodes(self):
    return type(self.pytree)._pytree__class_nodes

  @property
  def nodes(self):
    return self.pytree._pytree__nodes

  @nodes.setter
  def nodes(self, value: graph.HashableMapping[str, bool]):
    vars(self.pytree)['_pytree__nodes'] = value

  def __getitem__(self, key: str):
    if key in self.class_nodes:
      return self.class_nodes[key]
    elif key in self.nodes:
      return self.nodes[key]
    raise KeyError(f"key '{key}' not found")

  @staticmethod
  def _value_repr(value: bool) -> str:
    return "data" if value else "static"

  def __setitem__(self, key: str, value: bool):
    if key in self.class_nodes:
      if value != self.class_nodes[key]:
        raise ValueError(
          f"Cannot update status for attribute '{key}' to '{self._value_repr(value)}' "
          f"since its defined as '{self._value_repr(self.class_nodes[key]).capitalize()}' "
          f"by {type(self.pytree)}"
        )
      return

    self.nodes = self.nodes.update({key: value})

  def update(self, other: tp.Mapping[str, bool]):
    # check class_nodes not modified
    updates = {}
    for key, value in other.items():
      if key in self.class_nodes:
        if value != self.class_nodes[key]:
          raise ValueError(
            f"Cannot update status for attribute '{key}' to '{self._value_repr(value)}' "
            f"since its defined as '{self._value_repr(self.class_nodes[key]).capitalize()}' "
            f"by {type(self.pytree)}"
          )
        continue

      updates[key] = value

    self.nodes = self.nodes.update(updates)

  def __contains__(self, key: str):
    return key in self.class_nodes or key in self.nodes

  def items(self):
    yield from self.class_nodes.items()
    yield from self.nodes.items()

class Pytree(reprlib.Representable, metaclass=PytreeMeta):
  """Base class for all NNX objects."""

  if tp.TYPE_CHECKING:
    _pytree__class_nodes: graph.HashableMapping[str, bool]
    _pytree__nodes: graph.HashableMapping[str, bool]
    _pytree__state: PytreeState
    _pytree__is_pytree: bool

  @property
  def _pytree__data_mapping(self):
    return NodeDataMapping(self)

  def __init_subclass__(
    cls,
    *,
    pytree: bool = config.flax_pytree_module,
    **kwargs,
  ) -> None:
    super().__init_subclass__(**kwargs)
    cls._pytree__is_pytree = pytree

    graph.register_graph_node_type(
      type=cls,
      flatten=cls._graph_node_flatten,
      set_key=cls._graph_node_set_key,  # type: ignore
      pop_key=cls._graph_node_pop_key,  # type: ignore
      create_empty=cls._graph_node_create_empty,
      clear=cls._graph_node_clear,
      init=cls._graph_node_init,  # type: ignore
    )

    cls_nodes: dict[str, bool] = dict(getattr(cls, '_pytree__class_nodes', ()))
    cls_nodes['_pytree__state'] = True
    # add annotation attributes
    type_: type
    for name, type_ in cls.__annotations__.items():
      type_metadata = getattr(type_, '__metadata__', ())
      if type_ == tp.ClassVar:
        continue
      elif DataAnnotation in type_metadata:
        cls_nodes[name] = True
      elif StaticAnnotation in type_metadata:
        cls_nodes[name] = False

    cls._pytree__class_nodes = graph.HashableMapping(cls_nodes, copy=False)

    if pytree:
      jax.tree_util.register_pytree_with_keys(
        cls,
        flatten_with_keys=cls._pytree__flatten_with_paths,
        unflatten_func=cls._pytree__unflatten,
        flatten_func=cls._pytree__flatten,
      )

    if BUILDING_DOCS:
      # set correct signature for sphinx
      cls.__signature__ = inspect.signature(cls.__init__)

  # Backward compatibility with PR #4863
  @property
  def _object__nodes(self):
    return self._pytree__nodes
  @property
  def _object__state(self):
    return self._pytree__state

  if not tp.TYPE_CHECKING:

    def __setattr__(self, name: str, value: Any) -> None:
      self._setattr(name, value)

  def _setattr(self, name: str, value: tp.Any) -> None:
    self._check_valid_context(
      lambda: f"Cannot mutate '{type(self).__name__}' from different trace level"
    )
    if type(value) is DataAttr:
      value = value.value
      if self._pytree__is_pytree:
        _check_value(name, value, is_static=False, incomming=True, annotated=True)
      object.__setattr__(self, name, value)
      if self._pytree__is_pytree:
        self._pytree__data_mapping[name] = True
    elif type(value) is StaticAttr:
      value = value.value
      if self._pytree__is_pytree:
        _check_value(name, value, is_static=True, incomming=True, annotated=True)
      object.__setattr__(self, name, value)
      if self._pytree__is_pytree:
        self._pytree__data_mapping[name] = False
    elif self._pytree__is_pytree:
      object.__setattr__(self, name, value)
      _check_value(name, value, is_static=None, incomming=True, annotated=False)
      match is_data(value):
        case True:
          self._pytree__data_mapping[name] = True
        case False:
          self._pytree__data_mapping[name] = False
    else:
      object.__setattr__(self, name, value)

  def _check_valid_context(self, error_msg: tp.Callable[[], str]) -> None:
    if not self._pytree__state.trace_state.is_valid():
      raise errors.TraceContextError(error_msg())

  def __deepcopy__(self: O, memo=None) -> O:
    graphdef, state = graph.split(self)
    graphdef = deepcopy(graphdef)
    state = deepcopy(state)
    return graph.merge(graphdef, state)

  def __nnx_repr__(self):
    if OBJECT_CONTEXT.node_stats is None or id(self) not in OBJECT_CONTEXT.node_stats:
      node_stats: dict[int, dict[type[Variable], SizeBytes]] = {}
      _collect_stats(self, node_stats)
      OBJECT_CONTEXT.node_stats = node_stats
      stats = node_stats[id(self)]
      clear_node_stats = True
    else:
      stats = OBJECT_CONTEXT.node_stats[id(self)]
      clear_node_stats = False

    if OBJECT_CONTEXT.seen_modules_repr is None:
      OBJECT_CONTEXT.seen_modules_repr = set()
      clear_seen = True
    else:
      clear_seen = False

    if id(self) in OBJECT_CONTEXT.seen_modules_repr:
      yield reprlib.Object(type=type(self), empty_repr='...')
      return

    try:
      if stats:
        stats_repr = ' # ' + ', '.join(
          f'{var_type.__name__}: {size_bytes}'
          for var_type, size_bytes in stats.items()
        )
        if len(stats) > 1:
          total_bytes = sum(stats.values(), SizeBytes(0, 0))
          stats_repr += f', Total: {total_bytes}'
      else:
        stats_repr = ''

      yield reprlib.Object(type=type(self), comment=stats_repr)
      OBJECT_CONTEXT.seen_modules_repr.add(id(self))

      for name, value in vars(self).items():
        if name.startswith('_'):
          continue

        def to_shape_dtype(value):
          if isinstance(value, Variable):
            return value.replace(
              raw_value=jax.tree.map(to_shape_dtype, value.raw_value)
            )
          elif variablelib.is_array_ref(value) and np.prod(value.shape) > 1:
            return MutableArrayRepr(value.shape, value.dtype)
          elif (
            isinstance(value, (np.ndarray, jax.Array))
            and np.prod(value.shape) > 1
          ):
            return ArrayRepr(value.shape, value.dtype)
          return value

        value = jax.tree.map(to_shape_dtype, value, is_leaf=graph.is_graph_node)
        yield reprlib.Attr(name, value)
    finally:
      if clear_seen:
        OBJECT_CONTEXT.seen_modules_repr = None
      if clear_node_stats:
        OBJECT_CONTEXT.node_stats = None

  def __treescope_repr__(self, path, subtree_renderer):
    from flax import nnx

    if OBJECT_CONTEXT.node_stats is None:
      node_stats: dict[int, dict[type[Variable], SizeBytes]] = {}
      _collect_stats(self, node_stats)
      OBJECT_CONTEXT.node_stats = node_stats
      stats = node_stats[id(self)]
      clear_node_stats = True
    else:
      stats = OBJECT_CONTEXT.node_stats[id(self)]
      clear_node_stats = False

    try:
      if stats:
        stats_repr = ' # ' + ', '.join(
          f'{var_type.__name__}: {size_bytes}'
          for var_type, size_bytes in stats.items()
        )
        if len(stats) > 1:
          total_bytes = sum(stats.values(), SizeBytes(0, 0))
          stats_repr += f', Total: {total_bytes}'

        first_line_annotation = rendering_parts.comment_color(
          rendering_parts.text(f'{stats_repr}')
        )
      else:
        first_line_annotation = None
      children = {}
      for name, value in vars(self).items():
        if name.startswith('_'):
          continue
        children[name] = value

      if isinstance(self, nnx.Module):
        color = treescope.formatting_util.color_from_string(
          type(self).__qualname__
        )
      else:
        color = None
      return visualization.render_object_constructor(
        object_type=type(self),
        attributes=children,
        path=path,
        subtree_renderer=subtree_renderer,
        first_line_annotation=first_line_annotation,
        color=color,
      )
    finally:
      if clear_node_stats:
        OBJECT_CONTEXT.node_stats = None

  # pickle support
  def __getstate__(self):
    return vars(self).copy()

  def __setstate__(self, state):
    vars(self).update(state)

  # -------------------------
  # Pytree Definition
  # -------------------------
  def _pytree__flatten_with_paths(self):
    obj_vars = vars(self)
    node_attributes = self._pytree__data_mapping
    node_names: list[str] = []
    node_attrs: list[tuple[tp.Any, tp.Any]] = []
    static_attrs: list[tuple[str, tp.Any]] = []
    for name, value in sorted(obj_vars.items()):
      if name in node_attributes and node_attributes[name]:
        node_names.append(name)
        node_attrs.append((jax.tree_util.GetAttrKey(name), value))
      else:
        static_attrs.append((name, value))

    return node_attrs, (tuple(node_names), tuple(static_attrs))

  def _pytree__flatten(self):
    obj_vars = vars(self)
    node_attributes = self._pytree__data_mapping
    node_names: list[str] = []
    node_attrs: list[tp.Any] = []
    static_attrs: list[tuple[str, tp.Any]] = []
    for name, value in sorted(obj_vars.items()):
      if name in node_attributes and node_attributes[name]:
        node_names.append(name)
        node_attrs.append(value)
      else:
        static_attrs.append((name, value))

    return node_attrs, (tuple(node_names), tuple(static_attrs))

  @classmethod
  def _pytree__unflatten(
    cls,
    static: tuple[tuple[str, ...], tuple[tuple[str, tp.Any], ...]],
    node_attrs: tp.Iterable[tp.Any],
  ):
    node_names, static_attrs = static
    obj = object.__new__(cls)
    vars_obj = vars(obj)
    vars_obj.update(zip(node_names, node_attrs, strict=True))
    vars_obj.update(static_attrs)
    return obj

  # -------------------------
  # Graph Definition
  # -------------------------
  def _graph_node_flatten(self):
    nodes = vars(self).copy()
    nodes = sorted(nodes.items())
    return nodes, type(self)

  def _graph_node_set_key(self, key: str, value: tp.Any):
    if not isinstance(key, str):
      raise KeyError(f'Invalid key: {key!r}')
    elif (
      hasattr(self, key)
      and isinstance(variable := getattr(self, key), Variable)
      and isinstance(value, Variable)
    ):
      variable.update_from_state(value)
    else:
      setattr(self, key, value)

  def _graph_node_pop_key(self, key: str):
    if not isinstance(key, str):
      raise KeyError(f'Invalid key: {key!r}')
    return vars(self).pop(key)

  @staticmethod
  def _graph_node_create_empty(node_type: tp.Type[O]) -> O:
    node = object.__new__(node_type)
    return node

  def _graph_node_clear(self):
    vars(self).clear()

  def _graph_node_init(self, attributes: tp.Iterable[tuple[str, tp.Any]]):
    vars(self).update(attributes)

Object = Pytree