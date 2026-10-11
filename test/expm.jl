using StaticArrays, Test, LinearAlgebra

@testset "Matrix exponential" begin
    @test exp(SMatrix{0,0,Int}())::SMatrix === SMatrix{0,0,Float64}()
    @test exp(@SMatrix [2])::SMatrix ≈ SMatrix{1,1}(exp(2))
    @test exp(@SMatrix [5 2; -2 1])::SMatrix ≈ exp([5 2; -2 1])
    @test exp(@SMatrix [4 2; -2 1])::SMatrix ≈ exp([4 2; -2 1])
    @test exp(@SMatrix [4 2; 2 1])::SMatrix ≈ exp([4 2; 2 1])
    @test exp(@SMatrix [ -3+1im  -1+2im;-4-5im   5-2im])::SMatrix ≈ exp(Complex{Float64}[ -3+1im  -1+2im;-4-5im   5-2im])
    # test for case `(a[1,1] - a[2,2])^2 + 4*a[2,1]*a[1,2] == 0`
    @test exp(@SMatrix [2+2im  1-1im; -2-2im  6+2im])::SMatrix ≈ exp(Complex{Float64}[2+2im  1-1im; -2-2im  6+2im])
    # test for zeros matrix
    @test exp(@SMatrix zeros(Complex{Float64}, 2, 2))::SMatrix ≈ Complex{Float64}[1 0; 0 1]
    @test exp(@SMatrix [1 2 0; 2 1 0; 0 0 1]) ≈ exp([1 2 0; 2 1 0; 0 0 1])

    # https://github.com/JuliaArrays/StaticArrays.jl/issues/1295
    @test exp(@SMatrix [-800.0 800.0; 800.0 -800.0])::SMatrix ≈ [0.5 0.5; 0.5 0.5]
    @test exp(@SMatrix [-800.0 800.0; 800.0 -800.0]) ≈ exp([-800.0 800.0; 800.0 -800.0])

    for sz in (3,4), typ in (Float64, Complex{Float64})
        A = rand(typ, sz, sz)
        nA = norm(A, 1)
        for nB in (0.005, 0.1, 0.5, 3.0, 20.0)
            B = A*nB/nA
            SB = SMatrix{sz,sz,typ}(B)
            if Sys.ARCH in (:x86_64, :i686) && VERSION >= v"1.12" && VERSION < v"1.13" && typ === ComplexF64
                # Julia 1.12 has a bug that makes this test fail on some systems,
                # see https://github.com/JuliaLang/julia/issues/62368
            else
                @test exp(B) ≈ exp(SB)
            end
        end
    end
end
@testset "Real Padé approximation and Float32 factorization" begin
    # Large Padé coefficients overflow a determinant-based 3×3 solve in Float32.
    A = Float32[3 0.1 0.2; 0.2 2 0.1; 0.1 0.2 1]
    actual = exp(SMatrix{3,3}(A))
    @test all(isfinite, actual)
    @test actual ≈ exp(A) rtol=1e-5

    for T in (Float32, Float64), n in (0, 1, 2, 3, 4, 6)
        A = T.(reshape(sin.(1:(n*n)), n, n))
        n == 0 || (A ./= opnorm(A, 1))
        # Exercise each Padé degree and both sides of the scaling threshold.
        for magnitude in (0.005, 0.1, 0.5, 1.5, 3.0, 20.0)
            B = A .* T(magnitude)
            rtol = T === Float32 ? 1e-4 : 1e-10
            atol = T === Float32 ? 1e-6 : 1e-12
            @test exp(SMatrix{n,n}(B)) ≈ exp(B) rtol=rtol atol=atol
        end
    end

    @test exp(@SMatrix [0.3 1.2; 0.0 0.3]) ≈ exp([0.3 1.2; 0.0 0.3])
end

@testset "Real 2x2 repeated eigenvalue limit" begin
    for T in (Float32, Float64), v in (-1.1e-4, -1e-4, -1e-10, 0.0, 1e-10, 1e-4, 1.1e-4)
        A = @SMatrix T[0.3 1.2; v/(4*1.2) 0.3]
        @test exp(A) ≈ exp(Matrix(A)) rtol=(T === Float32 ? 2e-6 : 1e-12)
    end
end

@testset "Scaling beyond machine integer exponent range" begin
    for T in (Float32, Float64)
        A = @SMatrix T[0 1e22 0; 0 0 0; 0 0 0]
        @test exp(A) ≈ one(A) + A rtol=1e-6
    end
end

@testset "Real 2x2 integer promotion precedes discriminant arithmetic" begin
    for I in (Int32, Int64)
        A = I[-100000 1; 0 0]
        @test Matrix(exp(SMatrix{2,2}(A))) ≈ [0.0 1e-5; 0.0 1.0] rtol=1e-12
    end
    A = Int64[typemin(Int64) 0; 0 0]
    @test Matrix(exp(SMatrix{2,2}(A))) == [0.0 0.0; 0.0 1.0]
end
