module ImplicitAD

using ForwardDiff
using ReverseDiff
using ChainRulesCore
using LinearAlgebra: factorize, ldiv!, diag

import LinearAlgebra

include("internals.jl")

export implicit
include("nonlinear.jl")
include("nonlinear_svd.jl")

export explicit_unsteady, implicit_unsteady, explicit_unsteady_cache, implicit_unsteady_cache
include("unsteady.jl")

export implicit_linear, apply_factorization
include("linear.jl")

export implicit_eigval
include("eigenvalues.jl")

export provide_rule
include("external.jl")


end
