### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# ╔═╡ b6cede50-fe79-11ea-10c2-13b9e4851b06
begin
	import Pkg
	Pkg.activate(mktempdir())
	Pkg.Registry.update()
	
	Pkg.add("Plots")
	Pkg.add("PlutoUI")
	
	using LinearAlgebra
	using Plots
	using PlutoUI
end

# ╔═╡ 4e5ac200-fdff-11ea-3d80-d56549011cac
md"
## Solving the TopologicalComputer N point interpolation problem in 3 dimensions
The TopologicalComputer N point interpolation problem in 3 dimensions involves the generation of a continuous surface across a region defined by 4 points. This problem becomes more difficult in higher dimensions, as the number of permutations of {0, 1} for each dimension grows with the function 2^d, whereas the number of points required to make a linear function is equal to d.

The solution I present is to use a similar method to the tracing above. Given 4 points, draw two lines between the points such that all 4 points have one line intersecting them. Then start from one end drawing traces between them such that a surface is drawn in the region bounded by the four points.
"

# ╔═╡ 63395150-fdff-11ea-317a-956ff6431475
P = [
	0 0 1 # A
	1 0 1 # B
	0 1 1 # C
	1 1 -1 # D
]

# ╔═╡ 0ff87db0-fe7a-11ea-3afb-ddca5f27acab
begin
	A = P[1, :]
	B = P[2, :]
	C = P[3, :]
	D = P[4, :]
end

# ╔═╡ 7e3e4710-fe79-11ea-3d4b-4fc54cc58d07
vecMag(P1, P2) = √((P2 .- P1) ⋅ (P2 .- P1))

# ╔═╡ 00d26220-fe83-11ea-3cf6-492a6e1f3339
proj(u, v) = (u ⋅ v) / (v ⋅ v) * v

# ╔═╡ b19ba752-3dc2-11eb-0b48-2bb17d047b06
md"
$D_1(P)=||\overrightarrow{AP}-proj_{\overrightarrow{AB}}\overrightarrow{AP}||$
$D_2(P)=||\overrightarrow{CP}-proj_{\overrightarrow{CD}}\overrightarrow{CP}||$
"

# ╔═╡ 74c51212-fe81-11ea-2b20-45511a32be82
D1(P) = vecMag((P .- A[1:2]) - proj((P .- A[1:2]), (B[1:2] .- A[1:2])), [0, 0])

# ╔═╡ 453a9720-fe83-11ea-2ef2-fb2f40fb9394
D2(P) = vecMag((P .- C[1:2]) - proj((P .- C[1:2]), (D[1:2] .- C[1:2])), [0, 0])

# ╔═╡ fd028552-3dc3-11eb-2538-31025cd4a16c
md"
$P_{ab}(P)=\frac{P\circ\overrightarrow{AB}}{||\overrightarrow{AB}||\frac{D_2(P)}{D_1(P) + D_2(P)}+||\overrightarrow{CD}||\frac{D_1(P)}{D_1(P)+D_2(P)}} + A_{xy}$
$P_{cd}(P)=\frac{P\circ\overrightarrow{CD}}{||\overrightarrow{AB}||\frac{D_2(P)}{D_1(P) + D_2(P)}+||\overrightarrow{CD}||\frac{D_1(P)}{D_1(P)+D_2(P)}} + C_{xy}$
"

# ╔═╡ 246e2a50-fe80-11ea-1181-594649e2ce2e
Pab(P) = (P .* (B[1:2] .- A[1:2])) / ((vecMag(A[1:2], B[1:2]) * (D2(P) / (D1(P) + D2(P)))) + (vecMag(C[1:2], D[1:2]) * (D1(P) / (D1(P) + D2(P))))) + A[1:2]

# ╔═╡ 209fcaf0-fe85-11ea-0d9e-794a77c715a4
Pcd(P) = (P .* (D[1:2] .- C[1:2])) / ((vecMag(A[1:2], B[1:2]) * (D2(P) / (D1(P) + D2(P)))) + (vecMag(C[1:2], D[1:2]) * (D1(P) / (D1(P) + D2(P))))) + C[1:2]

# ╔═╡ a015f1e0-3dc0-11eb-3cef-1ddcb8c1f656
md"
$Z_{P_{ab}}(P)=(B_z-A_z)\frac{||\overrightarrow{AP_{ab}(P)}||}{||\overrightarrow{AB}||}+A_z$
$Z_{P_{cd}}(P)=(D_z-C_z)\frac{||\overrightarrow{CP_{ab}(P)}||}{||\overrightarrow{CD}||}+C_z$
"

# ╔═╡ 252d9b00-fe86-11ea-269c-959635d8db35
ZPab(P) = ((B[3] - A[3]) / vecMag(A[1:2], B[1:2])) * vecMag(Pab(P), A[1:2]) + A[3]

# ╔═╡ 7eb90970-fe86-11ea-346f-2b9cf9f8ea22
ZPcd(P) = ((D[3] - C[3]) / vecMag(C[1:2], D[1:2])) * vecMag(Pcd(P), C[1:2]) + C[3]

# ╔═╡ 5bc2b230-3dc5-11eb-3dd7-a51cce70eebc
md"
$Z(P)=\left(Z_{P_{ab}}(P)-Z_{P_{cd}}(P)\right)\frac{||\overrightarrow{P_{cd}(P)P}||}{||\overrightarrow{P_{ab}(P)P_{cd}(P)}||}+Z_{P_{cd}}(P)$
"

# ╔═╡ ba242d60-fe85-11ea-3397-8da61bd274ca
Z(P) = ((ZPab(P) - ZPcd(P)) / vecMag(Pab(P), Pcd(P))) * vecMag(Pcd(P), P) + ZPcd(P)

# ╔═╡ de52a1d0-3dbe-11eb-3860-3b72f74e299d
Z(X, Y) = Z([X, Y])

# ╔═╡ b4b47ce0-3dbe-11eb-230a-a7eaa71cc91c
x = min(P[:, 1]...):0.1:max(P[:, 1]...)

# ╔═╡ b775bfc0-3dbe-11eb-3ec1-453899565e99
y = min(P[:, 2]...):0.1:max(P[:, 2]...)

# ╔═╡ 015cae7e-fe88-11ea-353a-a9907ace0173
plot(x, y, Z, st=:surface)

# ╔═╡ 601d1460-fe87-11ea-1066-0d86847a1ad9
begin
	p = plot(x, y, Z, st=:contour)
	plot!(p, P[:, 1], P[:, 2], st=:scatter)
end

# ╔═╡ 6562e540-3dbf-11eb-358e-ed02cee52f6d
P[:, 1:2]

# ╔═╡ Cell order:
# ╟─4e5ac200-fdff-11ea-3d80-d56549011cac
# ╠═b6cede50-fe79-11ea-10c2-13b9e4851b06
# ╠═63395150-fdff-11ea-317a-956ff6431475
# ╠═0ff87db0-fe7a-11ea-3afb-ddca5f27acab
# ╠═7e3e4710-fe79-11ea-3d4b-4fc54cc58d07
# ╠═00d26220-fe83-11ea-3cf6-492a6e1f3339
# ╟─b19ba752-3dc2-11eb-0b48-2bb17d047b06
# ╠═74c51212-fe81-11ea-2b20-45511a32be82
# ╠═453a9720-fe83-11ea-2ef2-fb2f40fb9394
# ╟─fd028552-3dc3-11eb-2538-31025cd4a16c
# ╠═246e2a50-fe80-11ea-1181-594649e2ce2e
# ╠═209fcaf0-fe85-11ea-0d9e-794a77c715a4
# ╟─a015f1e0-3dc0-11eb-3cef-1ddcb8c1f656
# ╠═252d9b00-fe86-11ea-269c-959635d8db35
# ╠═7eb90970-fe86-11ea-346f-2b9cf9f8ea22
# ╟─5bc2b230-3dc5-11eb-3dd7-a51cce70eebc
# ╠═ba242d60-fe85-11ea-3397-8da61bd274ca
# ╠═de52a1d0-3dbe-11eb-3860-3b72f74e299d
# ╠═b4b47ce0-3dbe-11eb-230a-a7eaa71cc91c
# ╠═b775bfc0-3dbe-11eb-3ec1-453899565e99
# ╠═015cae7e-fe88-11ea-353a-a9907ace0173
# ╠═601d1460-fe87-11ea-1066-0d86847a1ad9
# ╠═6562e540-3dbf-11eb-358e-ed02cee52f6d
