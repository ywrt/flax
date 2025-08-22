---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
---

```{code-cell} ipython3
from flax import nnx
import jax
import jax.numpy as jnp
import dataclasses

#----------------------
# helper functions
#----------------------
def pytree_structure(pytree, title='pytree structure'):
  print(f"{title}:")
  for path, value in jax.tree.leaves_with_path(pytree):
    print(f"- pytree{jax.tree_util.keystr(path)} = {value!r}")
```

## Pytree

```{code-cell} ipython3
class Linear(nnx.Pytree):
  def __init__(self, din: int, dout: int):
    self.din, self.dout = din, dout
    self.w = jnp.ones((din, dout))
    self.b = jnp.zeros((dout,))

class MLP(nnx.Pytree):
  def __init__(self, num_layers, dim):
    self.num_layers = num_layers
    self.layers = nnx.List([Linear(dim, dim) for _ in range(num_layers)])

pytree = MLP(num_layers=2, dim=1)
pytree_structure(pytree)
```

### Attribute Annotations

```{code-cell} ipython3
class Foo(nnx.Pytree):
  def __init__(self, i: int):
    self.i = nnx.data(i)  # explicit data
    self.f = nnx.static(i * 0.5)  # explicit static
    self.x = jnp.array(42 * i)  # arrays are data
    self.s = "Hi" + "!" * i  # strings are static
    self.h = hash(i)  # weak types are initially static
    self.u = None  # empty pytrees are initially static

class Bar(nnx.Pytree):
  def __init__(self):
    self.ls = nnx.List([Foo(i) for i in range(2)])  # pytree of data are data
    self.shapes = (8, 16, 32)  # pytree of weak types are initially static

pytree = Bar()
pytree_structure(pytree)
```

```{code-cell} ipython3
print(f"""
{nnx.is_data(jnp.array(0)) = }                  # Arrays are data
{nnx.is_data('hello') = }                      # Strings are static
{nnx.is_data(42) = }                            # weak types are undefined
{nnx.is_data(nnx.Rngs(0)) = }                   # nnx.Pytrees are data
{nnx.is_data([jnp.array(0), jnp.array(1)]) = }  # pytrees of data are data
{nnx.is_data((1, 2.0, 3j, False)) = }           # pytrees of weak types are undefined
{nnx.is_data([jnp.array(0), 5]) = }             # pytrees of data and weak types are data
{nnx.is_data([]) = }                            # empty pytrees are undefined
{nnx.is_data({'msg': 'goodbye'}) = }           # pytrees of static are static
{nnx.is_data(['oh no!', jnp.array(3)]) = }      # pytrees with any data are data
""")
```

* remove mixed
* error on nnx.data/nnx.static in pytrees

+++

### Class Type Hints

```{code-cell} ipython3
@dataclasses.dataclass
class Foo(nnx.Pytree):
  i: nnx.Data[int]
  s: nnx.Static[str]
  x: jax.Array
  a: int

@dataclasses.dataclass
class Bar(nnx.Pytree):
  ls: list[Foo]
  shapes: list[int]

pytree = Bar(
  ls=[Foo(i, "Hi" + "!" * i, jnp.array(42 * i), hash(i)) for i in range(2)],
  shapes=[8, 16, 32]
)
pytree_structure(pytree)

pytree.ls[0]._pytree__data_mapping
```

#### When to use explicit annotations?

```{code-cell} ipython3
class Bar(nnx.Pytree):
  def __init__(self, x, num_layers: int, use_bias: bool):
    self.x = nnx.data(x)  # force inputs (e.g. user could pass Array or ShapeDtypeStruct)
    self.ls = nnx.data([]) # on empty pytrees
    for i in range(num_layers):
      self.ls.append(jnp.array(i))
    if use_bias:
      self.bias = nnx.Param(jnp.array(0.0))
    else:
      self.bias = nnx.data(None)  # on branches that cause mismatch

pytree = Bar(1.0, 3, True)
pytree_structure(pytree)
```

### Attribute Updates

```{code-cell} ipython3
class Foo(nnx.Pytree):
  def __init__(self):
    self.a = jnp.array(1.0)  # data
    self.b = "Hello, world!"  # static

pytree = Foo()
pytree_structure(pytree, "original")

pytree.a = 10  # undefined type, doesn't change attribute status
pytree.b = False  # undefined type, doesn't change attribute status
pytree_structure(pytree, "update with undefined types, no strucutre change")

pytree.a = object()  # static type, attribute is now static
pytree.b = jnp.array(42)  # data type, attribute is now data
pytree_structure(pytree, "update with strong types")
```

### Post `__init__` attribute checks

* `nnx.check_pytree`

```{code-cell} ipython3
class Foo(nnx.Pytree):
  def __init__(self):
    self.ls = []  # should be nnx.data([]), treated as static
    for i in range(5):
      self.ls.append(jnp.array(i))  # error: inserting arrays into static attribute
    self.a = {}
    self.a['x'] = nnx.data(42)  # error: inserting nnx.data/static in non-nnx.Pytree types

try:
  foo = Foo()  # nnx.check_pytree ran after __init__
except Exception as e:
  print("Error:", e)
```

### Trace-level awareness

```{code-cell} ipython3
class Foo(nnx.Pytree):
  def __init__(self):
    self.count = nnx.data(0)

foo = Foo()

@jax.vmap  # or jit, grad, shard_map, pmap, scan, etc.
def increment(n):
  foo.count += 1

try:
  increment(jnp.arange(5))
except Exception as e:
  print(f"Error: {e}")
```

## Module

+++

### set_attributes

```{code-cell} ipython3
class Block(nnx.Module):
  def __init__(self, din: int, dout: int, rngs: nnx.Rngs):
    self.mode = 1
    self.linear = nnx.Linear(din, dout, rngs=rngs)
    self.bn = nnx.BatchNorm(dout, rngs=rngs)
    self.dropout = nnx.Dropout(0.1, rngs=rngs)

  def __call__(self, x):
    return nnx.relu(self.dropout(self.bn(self.linear(x))))
  
model = Block(din=1, dout=2, rngs=nnx.Rngs(0))

print("train:")
print(f"- {model.mode = }")
print(f"- {model.bn.use_running_average = }")
print(f"- {model.dropout.deterministic = }")

# Set attributes for evaluation
model.set_attributes(deterministic=True, use_running_average=True, mode=2)

print("eval:")
print(f"- {model.mode = }")
print(f"- {model.bn.use_running_average = }")
print(f"- {model.dropout.deterministic = }")
```

```{code-cell} ipython3
model = Block(din=1, dout=2, rngs=nnx.Rngs(0))

model.eval(mode=2)  # .set_attributes(deterministic=True, use_running_average=True, mode=2)
print("eval:")
print(f"- {model.mode = }")
print(f"- {model.bn.use_running_average = }")
print(f"- {model.dropout.deterministic = }")

model.train(mode=1)  # .set_attributes(deterministic=False, use_running_average=False, mode=1)
print("train:")
print(f"- {model.mode = }")
print(f"- {model.bn.use_running_average = }")
print(f"- {model.dropout.deterministic = }")
```

### sow

```{code-cell} ipython3
class Block(nnx.Module):
  def __init__(self, din: int, dout: int, rngs: nnx.Rngs):
    self.linear = nnx.Linear(din, dout, rngs=rngs)
    self.bn = nnx.BatchNorm(dout, rngs=rngs)
    self.dropout = nnx.Dropout(0.1, rngs=rngs)

  def __call__(self, x):
    y = nnx.relu(self.dropout(self.bn(self.linear(x))))
    self.sow(nnx.Intermediate, "y_mean", jnp.mean(y))
    return y

class MLP(nnx.Module):
  def __init__(self, num_layers, dim, rngs: nnx.Rngs):
    self.blocks = [Block(dim, dim, rngs) for _ in range(num_layers)]

  def __call__(self, x):
    for block in self.blocks:
      x = block(x)
    return x


model = MLP(num_layers=3, dim=20, rngs=nnx.Rngs(0))
x = jnp.ones((10, 20))
y = model(x)
intermediates = nnx.pop(model, nnx.Intermediate) # extract intermediate values
print(intermediates)
```
