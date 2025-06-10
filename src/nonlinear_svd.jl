######## SVD Compression Stuff ########

struct SVDVector{T} <: AbstractVector{T}
    m::Int # Matrix dimension rows
    n::Int # Matrix dimension cols
    nsv::Int
    singular_values::Vector{T}
    U::Matrix{T}
    Vt::Matrix{T}
end

function _SVDVector(u, m, n, other)

    @assert(m > 0 && n > 0)
    @assert(m*n == length(u))

    umat = reshape(u, m, n)
    usvd = LinearAlgebra.svd(umat)
    (nsv, idx) = _make_index(usvd.S, other)

    @assert(nsv <= min(m,n))

    sv = usvd.S[idx]
    uv = usvd.U[:, idx]
    vv = usvd.Vt[idx, :]

    # @show nsv
    # @show LinearAlgebra.norm(u - (uv*LinearAlgebra.diagm(sv)*vv)[:], Inf)

    return SVDVector(m, n, nsv, sv, uv, vv)
end

function _SVDVector(u, other)
    nsq = length(u)
    n = Int(sqrt(nsq))
    return _SVDVector(u, n, n, other)
end

function SVDVector(u::AbstractVector, tol::Real)
    @assert(tol >= 0.0)
    return _SVDVector(u, tol)
end

function SVDVector(u::AbstractVector, m::Integer, n::Integer, tol::Real)
    @assert(tol >= 0)
    return _SVDVector(u, m, n, tol)
end

function SVDVector(u::AbstractVector, nsv::Integer)
    @assert(nsv >= 0)
    return _SVDVector(u, nsv)
end

function SVDVector(u::AbstractVector, m::Integer, n::Integer, nsv::Integer)
    @assert(nsv >= 0)
    return _SVDVector(u, m, n, nsv)
end

#### AbstractArray Interface Functions ####

function Base.size(u::SVDVector)
    return (u.m * u.n,)
end

function Base.getindex(sv::SVDVector, idx::Integer)
    # println("SVD getindex")
    (i, j) = matrix_index(idx, sv.m, sv.n)
    # @show (i,j)
    ui = @view sv.U[i,:]
    vti = @view sv.Vt[:,j]
    # return sv.singular_values[i] * LinearAlgebra.dot(ui, vti)
    return LinearAlgebra.dot(sv.singular_values .* ui, vti)
end

function Base.IndexStyle(::SVDVector)
    return Base.IndexLinear()
end

#### Helper Functions ####

function matrix_index(idx::I, m::I, n::I) where {I<:Integer}
    (j, i) = divrem(idx, m)
    if i == zero(I)
        i = m
    else
        j += one(I)
    end
    return (i, j)
end

function _make_index(svs::Vector, tol::Real)
    idx = svs .>= tol
    return (sum(idx), idx)
end

function _make_index(::Vector, nsv::Integer)
    idx = 1:nsv
    return (nsv, idx)
end



"""
    implicit_svd(solve, residual, x, p=(); drdy=drdy_forward, lsolve=linear_solve)

Make implicit function AD compatible (specifically with ForwardDiff and ReverseDiff).

# Arguments
- `solve::function`: y = solve(x, p). Solve implicit function returning state variable y, for input variables x, and fixed parameters p.
- `residual::function`: Either r = residual(y, x, p) or in-place residual!(r, y, x, p). Set residual r (scalar or vector), given state y (scalar or vector), variables x (vector) and fixed parameters p (tuple).
- `x::vector{float}`: evaluation point.
- `p::tuple`: fixed parameters. default is empty tuple.
- `drdy::function`: drdy(residual, y, x, p).  Provide (or compute yourself): ∂r_i/∂y_j.  Default is forward mode AD.
- `lsolve::function`: lsolve(A, b).  Linear solve A x = b  (where A is computed in drdy and b is computed in jvp, or it solves A^T x = c where c is computed in vjp).  Default is backslash operator.
"""
function implicit_svd(solve, residual, x, p=(); drdy=drdy_forward, lsolve=linear_solve)

    # ---- check for in-place version and wrap as needed -------
    new_residual = residual
    if applicable(residual, 1.0, 1.0, 1.0, 1.0)  # in-place

        function residual_wrap(yw, xw, pw)  # wrap residual function in a explicit form for convenience and ensure type of r is appropriate
            T = promote_type(eltype(xw), eltype(yw))
            rw = zeros(T, length(yw))  # match type of input variables
            residual(rw, yw, xw, pw)
            return rw
        end
        new_residual = residual_wrap
    end

    return _implicit_svd(solve, new_residual, x, p, drdy, lsolve)
end

# If no AD, just solve normally.
_implicit_svd(solve, residual, x, p, drdy, lsolve) = solve(x, p)

# Overloaded for ForwardDiff inputs, providing exact derivatives using
# Jacobian vector product.
function _implicit_svd(solve, residual, x::AbstractVector{<:ForwardDiff.Dual{T}}, p, drdy, lsolve) where {T}

    # evaluate solver
    xv = fd_value(x)
    yv = solve(xv, p)

    if get(p, :forward_svd, false)
        tol = get(p, :tol, 0.0)
        nsv = get(p, :nsv, 3)
        (m,n) = get(p, :matdim, (-1,-1))
        yv = SVDVector(yv, m, n, tol > 0.0 ? tol : nsv)
    end

    # solve for Jacobian-vector product
    b = jvp(residual, yv, x, p)

    # compute partial derivatives
    A = drdy(residual, yv, xv, p)

    # linear solve
    ydot = lsolve(A, b)

    # repack in ForwardDiff Dual
    return pack_dual(yv, ydot, T)
end


# Provide a ChainRule rule for reverse mode
function ChainRulesCore.rrule(::typeof(_implicit_svd), solve, residual, x, p, drdy, lsolve)

    # evaluate solver (create local copy of the output to guard against `y` getting overwritten)
    # y = copy(solve(x, p))
    tol = get(p, :tol, 0.0)
    nsv = get(p, :nsv, 3)
    (m,n) = get(p, :matdim, (-1,-1))
    y = SVDVector(solve(x, p), m, n, tol > 0.0 ? tol : nsv)

    function pullback(ybar)
        A = drdy(residual, y, x, p)
        u = lsolve(A', ybar)
        xbar = vjp(residual, y, x, p, -u)
        return NoTangent(), NoTangent(), NoTangent(), xbar, NoTangent(), NoTangent(), NoTangent()
    end

    return y, pullback
end


# register above rule for ReverseDiff
ReverseDiff.@grad_from_chainrules _implicit_svd(solve, residual, x::ReverseDiff.TrackedArray, p, drdy, lsolve)
ReverseDiff.@grad_from_chainrules _implicit_svd(solve, residual, x::AbstractVector{<:ReverseDiff.TrackedReal}, p, drdy, lsolve)
