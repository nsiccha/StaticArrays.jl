@inline exp(A::StaticMatrix) = _exp(Size(A), A)

# Keep the numerical kernels shared with compilers that stage lazy control flow.
# Only the choice and iteration operations need backend-specific implementations.
@inline _exp_choose(condition, yes, no, args) = condition ? yes(args...) : no(args...)
@inline _exp_solve(A, B) = lu(A) \ B

@inline function _exp_square(A, count)
    for _ in 1:count
        A = A*A
    end
    return A
end

@inline function _exp(::Size{(0,0)}, A::StaticMatrix)
    T = typeof(exp(zero(eltype(A))))
    newtype = similar_type(A,T)

    (newtype)()
end

@inline function _exp(::Size{(1,1)}, A::StaticMatrix)
    T = typeof(exp(zero(eltype(A))))
    newtype = similar_type(A,T)

    (newtype)((exp(A[1]), ))
end

# Bernstein, D. S. & So, W. 1993. "Some Explicit Formulas for the Matrix Exponential"
@inline function _exp(::Size{(2,2)}, A::StaticMatrix{<:Any,<:Any,<:Real})
    return _exp_real_2x2(A)
end

@inline function _exp_real_2x2(A)
    T = typeof(exp(zero(eltype(A))))
    newtype = similar_type(A,T)

    @inbounds a = T(A[1])
    @inbounds c = T(A[2])
    @inbounds b = T(A[3])
    @inbounds d = T(A[4])

    v = (a-d)^2 + 4*b*c

    m = (a + d) / 2

    entries = _exp_choose(abs(v) <= cbrt(eps(T)), _exp_2x2_near_zero, _exp_2x2_nonzero,
                          (a, b, c, d, v, m, T))
    return (newtype)(entries)
end

@inline _exp_2x2_nonzero(a, b, c, d, v, m, T) =
    _exp_choose(v > 0, _exp_2x2_positive, _exp_2x2_negative, (a, b, c, d, v, m, T))

@inline function _exp_2x2_positive(a, b, c, d, v, m, T)
    # In this case the formulas in the entries of the matrix
    # are a function of cosh and sinh, and could (in theory)
    # follow the same code pattern of the other branches (v ≤ 0).
    # However, cosh and sinh explode with large arguments and
    # we use the following identity to avoid numerical issues:
    #
    # exp(m) * [c₁ * cohs(δ) + c₂ * sinh(δ)] =
    #   c₁ * (e₊ + e₋) / 2 + c₂ * (e₊ - e₋) / 2
    #
    # where e₊ = exp(m + δ) and e₋ = exp(m - δ).
    #
    # See https://github.com/JuliaArrays/StaticArrays.jl/issues/1295
    δ = sqrt(v) / 2
    e₊ = exp(m + δ)
    e₋ = exp(m - δ)
    e₁ = (e₊ + e₋) / 2
    e₂ = (e₊ - e₋) / 2
    c₂ = (a - d) / 2δ
    m11 = (e₁ + c₂ * e₂)
    m12 = (b / δ) * e₂
    m21 = (c / δ) * e₂
    m22 = (e₁ - c₂ * e₂)
    return (m11, m21, m12, m22)
end

@inline function _exp_2x2_negative(a, b, c, d, v, m, T)
    z = sqrt(-v)
    r = exp(m)
    z1 = cos(z / 2)
    z2 = sin(z / 2) / z
    m11 = r * (z1 + (a - d) * z2)
    m12 = r * 2b * z2
    m21 = r * 2c * z2
    m22 = r * (z1 - (a - d) * z2)
    return (m11, m21, m12, m22)
end

@inline function _exp_2x2_near_zero(a, b, c, d, v, m, T)
    # cosh(sqrt(v)/2) and sinh(sqrt(v)/2)/sqrt(v) are entire in v.
    # Their Taylor forms avoid cancellation and the removable singularity at
    # repeated eigenvalues, preserving the ordinary differential there too.
    # At |v| <= cbrt(eps(T)) the first omitted terms are below roundoff.
    z1 = @evalpoly(v, T(1), T(1)/T(8), T(1)/T(384), T(1)/T(46080))
    z2 = @evalpoly(v, T(1)/T(2), T(1)/T(48), T(1)/T(3840), T(1)/T(645120))
    r = exp(m)
    return (r*(z1 + (a-d)*z2), r*2c*z2, r*2b*z2, r*(z1 - (a-d)*z2))
end

@inline function _exp(::Size{(2,2)}, A::StaticMatrix{<:Any,<:Any,<:Complex})
    T = typeof(exp(zero(eltype(A))))
    newtype = similar_type(A,T)

    @inbounds a = A[1]
    @inbounds c = A[2]
    @inbounds b = A[3]
    @inbounds d = A[4]

    z = sqrt((a - d)*(a - d) + 4*b*c )
    e = expm1((a + d - z) / 2)
    f = expm1((a + d + z) / 2)
    ϵ = eps()
    g = abs2(z) < ϵ^2 ? exp((a + d) / 2) * (1 + z^2 / 24) : (f - e) / z

    m11 = (g * (a - d) + f + e) / 2 + 1
    m12 = g * b
    m21 = g * c
    m22 = (-g * (a - d) + f + e) / 2 + 1

    (newtype)((m11, m21, m12, m22))
end

# Adapted from implementation in Base; algorithm from
# Higham, "Functions of Matrices: Theory and Computation", SIAM, 2008
@inline _exp(::Size, A::StaticMatrix) = _exp_pade(A)

function _exp_pade(_A::StaticMatrix{<:Any,<:Any,T}) where T
    S = typeof((zero(T)*zero(T) + zero(T)*zero(T))/one(T))
    A = S.(_A)
    # omitted: matrix balancing, i.e., LAPACK.gebal!
    nA = maximum(sum(abs.(A); dims=Val(1)))    # marginally more performant than norm(A, 1)
    ## For sufficiently small nA, use lower order Padé-Approximations
    U, V, s = _exp_choose(nA <= 2.1, _exp_pade_small, _exp_pade_large, (A, nA, S))
    expA = _exp_solve(V - U, V + U)
    count = _exp_choose(s > 0, _exp_square_count, _exp_zero_count, (s,))
    return _exp_square(expA, count)
end

@inline function _exp_pade_small(A, nA, S)
    A2 = A*A
    U, V = _exp_choose(nA > 0.95, _exp_pade9, _exp_pade_smaller, (A, A2, nA, S))
    return U, V, zero(nA/5.4)
end

@inline _exp_pade_smaller(A, A2, nA, S) =
    _exp_choose(nA > 0.25, _exp_pade7, _exp_pade_smallest, (A, A2, nA, S))
@inline _exp_pade_smallest(A, A2, nA, S) =
    _exp_choose(nA > 0.015, _exp_pade5, _exp_pade3, (A, A2, nA, S))

@inline function _exp_pade9(A, A2, nA, S)
    U = @evalpoly(A2, S(8821612800)*I, S(302702400)*I, S(2162160)*I, S(3960)*I, S(1)*I)
    U = A*U
    V = @evalpoly(A2, S(17643225600)*I, S(2075673600)*I, S(30270240)*I, S(110880)*I, S(90)*I)
    return U, V
end

@inline function _exp_pade7(A, A2, nA, S)
    U = @evalpoly(A2, S(8648640)*I, S(277200)*I, S(1512)*I, S(1)*I)
    U = A*U
    V = @evalpoly(A2, S(17297280)*I, S(1995840)*I, S(25200)*I, S(56)*I)
    return U, V
end

@inline function _exp_pade5(A, A2, nA, S)
    U = @evalpoly(A2, S(15120)*I, S(420)*I, S(1)*I)
    U = A*U
    V = @evalpoly(A2, S(30240)*I, S(3360)*I, S(30)*I)
    return U, V
end

@inline function _exp_pade3(A, A2, nA, S)
    U = @evalpoly(A2, S(60)*I, S(1)*I)
    U = A*U
    V = @evalpoly(A2, S(120)*I, S(12)*I)
    return U, V
end

@inline _exp_unscaled(A, s, S) = A
@inline _exp_scaled(A, s, S) = A / (S(2)^ceil(Int,s))
@inline _exp_zero_count(s) = zero(Int)
@inline _exp_square_count(s) = ceil(Int, s)

function _exp_pade_large(A, nA, S)
    s  = log2(nA/5.4)               # power of 2 later reversed by squaring
    A = _exp_choose(s > 0, _exp_scaled, _exp_unscaled, (A, s, S))

    A2 = A*A
    A4 = A2*A2
    A6 = A2*A4

    U = A6*(S(1)*A6 + S(16380)*A4 + S(40840800)*A2) +
    (S(33522128640)*A6 + S(10559470521600)*A4 + S(1187353796428800)*A2) +
    S(32382376266240000)*I
    U = A*U
    V = A6*(S(182)*A6 + S(960960)*A4 + S(1323241920)*A2) +
    (S(670442572800)*A6 + S(129060195264000)*A4 + S(7771770303897600)*A2) +
    S(64764752532480000)*I
    return U, V, s
end
