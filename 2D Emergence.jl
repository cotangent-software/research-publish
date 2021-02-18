### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# ╔═╡ 72a931f2-7177-11eb-38e0-c50ea724abb7
begin
	import Pkg
	Pkg.activate(mktempdir())
	Pkg.Registry.update()
	
	Pkg.add("Plots")
	
	using Plots
end

# ╔═╡ 54ce7910-0985-11eb-329f-290d043c9fdf
nodes = 10

# ╔═╡ ee6ff5e0-0984-11eb-28c0-0160e9d508b5
R = [
	0 0
	0 0
]

# ╔═╡ fb1e6ba2-0984-11eb-1733-ed758f0facf0
E = [
	0 1
	-1 0
]

# ╔═╡ 249d6fd0-0985-11eb-028c-43bade37885f
T = zeros(nodes, nodes)

# ╔═╡ 2c14e7c0-0985-11eb-3919-3fa9e77c64fd
for r ∈ 1:2:nodes-1
	for c ∈ 1:2:nodes-1
		if c == r
			T[r:r+1, c:c+1] = R
		else
			T[r:r+1, c:c+1] = E
		end
	end
end

# ╔═╡ 31d7d1e0-0985-11eb-267e-e304ef44476d
T

# ╔═╡ 056442a0-0981-11eb-1465-13903cca5862
dS(S, T, Δt) = T * S * Δt

# ╔═╡ 46c15a80-0981-11eb-1e1c-11baed14f1ba
iterations = 10000

# ╔═╡ 506cc970-0981-11eb-0af4-d709202a7bdb
Δt = 0.0007

# ╔═╡ 62585780-0981-11eb-25c3-55fbef2b599a
function iterate(angle)
	s = [1; 0; -1; 0; 0; 1; 0; -1; cos(angle); sin(angle)]
	sList = zeros(nodes, iterations)
	
	for i ∈ 1:iterations
		s += dS(s, T, Δt)
		sList[:, i] = s
	end
	
	return sList
end

# ╔═╡ d7a8d2a0-0993-11eb-2b42-d1ca39d1989f
sList = iterate(π/2)

# ╔═╡ fc8849ee-0986-11eb-3257-8d05c1aef75d
plot(sList[1, :], sList[2, :])

# ╔═╡ 2de4b0a0-0983-11eb-1896-6385d44cfb65
plot(permutedims(sList[1:2:end, :], [2, 1]), permutedims(sList[2:2:end, :], [2, 1]))

# ╔═╡ 33b8c910-0994-11eb-2daf-2b93f3635ad7
anim = @animate for i ∈ 0:π/180:2π
	p = iterate(i)
	plot(permutedims(p[1:2:end, :], [2, 1]), permutedims(p[2:2:end, :], [2, 1]),
		xlims=(-1.75, 1.75),
		ylims=(-1.75, 1.75)
	)
end

# ╔═╡ 98421710-0994-11eb-3fe9-b7a0b2492aa7
gif(anim, "scratch.gif", fps = 30)

# ╔═╡ Cell order:
# ╠═72a931f2-7177-11eb-38e0-c50ea724abb7
# ╠═54ce7910-0985-11eb-329f-290d043c9fdf
# ╠═ee6ff5e0-0984-11eb-28c0-0160e9d508b5
# ╠═fb1e6ba2-0984-11eb-1733-ed758f0facf0
# ╠═249d6fd0-0985-11eb-028c-43bade37885f
# ╠═2c14e7c0-0985-11eb-3919-3fa9e77c64fd
# ╠═31d7d1e0-0985-11eb-267e-e304ef44476d
# ╠═056442a0-0981-11eb-1465-13903cca5862
# ╠═46c15a80-0981-11eb-1e1c-11baed14f1ba
# ╠═506cc970-0981-11eb-0af4-d709202a7bdb
# ╠═62585780-0981-11eb-25c3-55fbef2b599a
# ╠═d7a8d2a0-0993-11eb-2b42-d1ca39d1989f
# ╠═fc8849ee-0986-11eb-3257-8d05c1aef75d
# ╠═2de4b0a0-0983-11eb-1896-6385d44cfb65
# ╠═33b8c910-0994-11eb-2daf-2b93f3635ad7
# ╠═98421710-0994-11eb-3fe9-b7a0b2492aa7
