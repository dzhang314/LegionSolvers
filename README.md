# Legion Solver Library

This repository contains work in progress on an experimental linear solver library for Legion (and, by extension, Regent and Pygion) applications. The key goals of this library are:

* Providing portable, high-performance implementations of standard direct and iterative linear solver algorithms that work out of the box
* Taking advantage of Legion's data model to operate on application data in place (as opposed to requiring explicit vector/matrix assembly)
* Taking advantage of Legion's execution model to interleave solver computation with communication and other application tasks
* Providing a foundation for research into novel solver algorithms that take advantage of Legion's support for irregular task parallelism

## Library Components

The library is divided into the following components:

* A framework for defining generalized linear systems
* Solver algorithms that operate on a specified linear system
* A planner that determines how the solve of a given system will be decomposed into a task graph
* A mapper that determines where the solver tasks should be executed
* Bindings for higher-level languages that sit on top of Legion (i.e. Regent, Pygion)
* Utilities (e.g. for reading/writing standard matrix formats in a distributed fashion)

### Generalized Linear Systems

A linear system is typically specified as `A*x == b`, where `A` is a _linear operator_, `b` is a _right-hand side_ (_RHS_), and `x` is the desired _solution_ to the system. We cover more use cases by generalizing in a number of ways:

* A system can contain more that one right-hand-side term, and each can have its own dimensionality. For example, a 2D RHS term might describe the source term at every point in
a 2D grid, while an additional 1D RHS term might capture the known fluxes on the boundary of the grid. (Note that this should not be confused with solvers that can solve the same system for "multiple right hand sides" concurrently.)
* A system can contain more than one solution term, again with variable dimensions.
* A system can contain more than one operator. Each operator relates one RHS term to one solution term, and has a logical dimension equal to the sum of the corresponding RHS and
solution term dimensions.
* Each operator may be materialized in either sparse or dense form that associates linear factors with some (sparse) or all (dense) pairs of RHS and solution elements. Alternatively, an operator may be "matrix-free", relying on arbitrarily simple or complex data to implicitly define those factors. In either case, an operator has a "kernel space" of arbitrary dimensionality and uses "mappings" (which may be materialized in a Legion field or described via an affine expression) between the kernel space and the index spaces of the RHS and solution terms.
* Each operator may include a second operator that acts as an approximate (and hopefully cheap) inverse of the original for use in preconditioning.

### Solver Algorithms
There's no shortage of known good algorithms to implement, but the first few will probably be

* Congujate Gradient - for SPD operators
* BiCGStab - for asymmetric operators
* LSQR - a CG-like approach to finding approximate solutions to over/under-determined systems

Iterative algorithms are hopefully very simple to implement, as they mostly rely on performing "forward" and "backward" computations on each of the operators in the system along with computing norms over "vectors" in either the RHS or solution spaces. There are no direct solvers listed right now, but that is artifact of prioritization rather than absolute interest.

### Planner

For both performance and capacity reasons, it is critical that a solver's computation be parallelized
and distributed across the computational resources that are available to the application. Both
because it maximizes affinity with the rest of the (presumably-parallelized-and-distributed)
application and because there no obvious general-purpose alternative, the planner relies on
application-provided partitions of one or more of the RHS terms, solution terms, and/or operators in
a linear system. Using the mappings between RHS<->operator<->solution terms and Legion's
dependent partitioning support, the mapper can derive partitions of of an operator's logical space
(i.e. the cartesian product of the RHS and solution spaces) the application-specified partition(s).
Using these partitions of operators' logical spaces directly can guarantee that no individual task will
need more than a subset of existing instances for application fields. However, complex systems will
require (and even simpler systems may benefit from) optimizations made in the mapper to disregard
some of the the application-provided partitions as being either redundant or "more trouble than
they are worth" (e.g. because they create more parallelism than is needed for performance or
capacity reasons).


### Mapper

It is unreasonable to expect an application's mapper to take on the responsibility of mapping the
solver's tasks, so the solver library must include mapper(s) that perform this role. It remains to be
seen whether a single mapper will work across multiple solver algorithms or whether each algorithm
will generally provide its own mapper. Although the planner and the mapper(s) are listed as different
components, the number and dependence pattern of the tasks that will be mapped were determined
by the planner. As a result, it is expected that the mapper and planner will share a fair amount of
"back channel" information that avoids the need for the mapper to reverse engineer the reasoning
performed in the planning stage.

Legion's mapping API makes it straightforward for the combination of multiple mappers to always
yield the correct results, but getting good performance is another matter. The mapper(s) provided
must generate mappings that respect the existing application's data distribution at least to some
extent. For example, an iterative solver might choose to distribute residuals differently than the RHS
terms if it judges that the communication savings of using the different distribution on multiple
iterations outweighs the cost of the initial data movement, but that should be a conscious choice
made by the mapper rather than the result of being oblivious to where the application had placed its
data.

### Bindings

A large amount of Legion functionality is made available to Regent and Pygion through the use of a
C language wrapper API that is then accessed via FFI tools. Although that can be made to work here,
the complexity of the data structure describing a linear system would make it painful to move back
and forth across a pure C interface, and given that programmer productivity is one of the goals of
this project, it seems worth the effort to provide native Regent and Python interfaces that can track
some of the state and perform some of the reasoning "above" the interface to the C++ core of the
solver library.

### Utilities

This will be a grab-bag of helper routines that are outside of the "core" of the Legion solver library,
but are likely to get used more than once and therefore benefit from only being written once. This
will include things like reading in matrices from file formats like MATLAB or Matrix Market and
storing them in standard formats (e.g. COO, CSR, ELL, ...), but also helper functions for building
simple linear systems (e.g. Ax = b). As a rule of thumb, it'd be great to have each of the examples
provided with the library include no more than 30-50 lines of "interesting" (i.e. not comments or
boilerplate or argument parsing) code. If there's more than that, there's probably something that can
be pulled into the bag of utilities.


## Requirements

Required software components include:

* CMake (version TBD)
* Legion (version TBD), built and installed by CMake - the solver is intended to work either with or without the use of control replication
* Kokkos (version TBD), built and installed by CMake

Optional software or hardware includes:

* GASNet-EX (version TBD) and networking infrastructure supported by a GASNet conduit for distributed execution
* CUDA (version TBD) and recentish (TBD) NVIDIA GPUs
* maybe various BLAS libraries?
