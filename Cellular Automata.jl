### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# ╔═╡ a29c2ea0-6ff4-11eb-2954-372c6727bee2
begin
	import Pkg
	Pkg.activate(mktempdir())
	Pkg.Registry.update()
	
	Pkg.add("Plots")
	using Plots
end

# ╔═╡ 724acb50-6db3-11eb-07a8-7d048b8e4b3e
md"# Introduction to Elementary Cellular Automata"

# ╔═╡ 9464f530-6db3-11eb-0629-9f05df1979bd
md"
Cellular automata are fascination \"mathematical machines\" which, despite normally being associated with having relatively simple rules, can still produce highly complex output. This point is stressed over and over in Stephen Wolfram's *A New Kind of Science*. This is how Wikipedia defines them:
> A cellular automaton is a discrete model of computation studied in automata theory.
In this notebook we will explore some simple cellular automata, called the *Elementary Cellular Automata*.
"

# ╔═╡ f3f9f3b0-6db3-11eb-1446-cfe467e42fee
md"## Defining Rules"

# ╔═╡ fdc546b0-6db3-11eb-1a4c-514e32ae4a27
md"Put simply, a cellular automata takes some simple rule and applies it over and over on an infinite string of numbers. We will be taking a look at 1 dimensional cellular automata, which is to say there is one spatial dimension and one temporal dimension. This can easily be visualized in a 2D image unlike higher-dimensional automata. Chances are you have heard of Conway's *Game of Life*, which is an example of a two dimensional cellular automaton."

# ╔═╡ 87bd67d0-6db4-11eb-0931-f9a3ec94cd70
md"Let's define some rules as a dictionary of associations"

# ╔═╡ b3fdfd00-6db4-11eb-3fa3-79d5911c567d
rule = Dict(
	0b000 => 0,
	0b001 => 1,
	0b010 => 1,
	0b011 => 1,
	0b100 => 1,
	0b101 => 0,
	0b110 => 0,
	0b111 => 0
)

# ╔═╡ f21ba14e-6db4-11eb-382a-87bc07250f65
md"For our rule, each cell can have a value of either a $0$ or a $1$. To determine what the cell's next value will be, our rule looks at the value of its left and right neighbors, where the left neighbor is the most significant digit and the right neighbor is the least significant digit, and associates that configuration, which has $2^3=8$ different combinations, to a new value of either $1$ or $0$."

# ╔═╡ 7895f0e0-6db6-11eb-1d6e-b12880ebd5b9
md"## Executing Rules

To start out we can define a recursive function which will return a list of lists, where each list represents the state of the automaton at a particular timestep."

# ╔═╡ 0ab21310-6db6-11eb-00b5-1f2bd2203c79
function value_at(state, pos)
	if pos < 1 || pos > length(state)
		return 0
	end
	return state[pos]
end

# ╔═╡ 656f4030-6db5-11eb-0360-bd9ac3ac6b45
function automaton(initial_state, steps, rule)
	# Base case
	if steps <= 1
		return [initial_state]
	end
	
	# Otherwise...
	previous_states = automaton(initial_state, steps - 1, rule)
	ps = previous_states[end]
	next_state = [rule[value_at(ps, i-1) << 2 | value_at(ps, i) << 1 | value_at(ps, i+1)] for i = 1:length(initial_state)]
	
	[previous_states..., next_state]
end

# ╔═╡ 232031c0-6db6-11eb-2f48-81afe98ccac3
md"Despite not being the most efficient it will do for smaller step counts. Let's try it out for our rule for 50 steps"

# ╔═╡ 71d85310-6db6-11eb-0728-f34323017fe1
history = automaton([i == 51 ? 1 : 0 for i = 1:101], 50, rule)

# ╔═╡ aefbdff0-6db6-11eb-0441-a924d0f567c1
history_mat = [history[i][j] for i=1:length(history), j=1:length(history[1])]

# ╔═╡ ea3c54f0-6db6-11eb-1312-3daa47f3fa91
md"## Rendering the Results"

# ╔═╡ f018e190-6db6-11eb-3eef-c984453d36d9
md"Now that we have a matrix of values we can make use of the Plots package to make a plot displaying it."

# ╔═╡ 06c775f0-6db7-11eb-2603-4942353b201e
heatmap(1 .- history_mat, color = :greys, yflip = true, size=(640, 320))

# ╔═╡ 8bb7e6a0-6db7-11eb-3491-039358de17d7
md"Wow! Interpretting this plot is fairly easy. Black is a $1$ and white is a $0$, and time flows downwards. It's rather remarkable how starting from just a single $1$ and the rest $0$'s at $t=1$ the behavior of this cellular automata still appears quite complex! In fact, the center column of this particular rule has been shown to exhibit many of the qualities of a pseudo-random number generator.

Just for fun let's run this automaton for 500 steps just to see what happens if we let it run for a little longer."

# ╔═╡ dfe4d210-6db7-11eb-1425-732f8753c0f5
longer_history = automaton([i == 501 ? 1 : 0 for i = 1:1001], 500, rule)

# ╔═╡ f4bd6df0-6db7-11eb-15a0-c52c636db094
longer_history_mat = [longer_history[i][j] for i=1:length(longer_history), j=1:length(longer_history[1])]

# ╔═╡ f9fe56d0-6db7-11eb-1a02-6133a37e6426
heatmap(1 .- longer_history_mat, color = :greys, yflip = true, size=(640, 320))

# ╔═╡ 065d25f2-6db8-11eb-0ab4-95151bc3554f
md"It's a little tough to see the details since our image isn't high enough resolution to see each and every state value, but it is still fairly clear that the pseudo-random behavior does not settle down in the first 500 steps. However when viewing this many steps it begins to become clear that patterns, or rather regularities, begin to appear moving through space and time at the speed of causality on the left side."

# ╔═╡ 4117f030-6db8-11eb-0df5-4d606d3faf6b
md"## So what??

Beyond being an interesting thing to look at, what else are these automata useful for? The Wikipedia definition of what a cellular automata are mentioned that they are a **model of computation**. Put simply this means that fundamentally cellular automata are performing computations over time. In fact it has been shown that 1 particular rule using only nearest-neighbors and two colors in one dimension (just like our rule) is **Turing-complete**, which means it can do any computation that the computer you're viewing this on can!

I encourage you now to go back and experiment with some of the rule values on the right-hand side. Don't change the left since all 8 of the possible 3 bit combinations have to be present for the automata to be guarenteed to have a valid next state. Also, try using the rule below, which has been proven to be Turing-complete!"

# ╔═╡ caf3afb0-6db8-11eb-39de-852bb7609cb3
turing_complete_rule = Dict(
	0b000 => 0,
	0b001 => 1,
	0b010 => 1,
	0b011 => 1,
	0b100 => 0,
	0b101 => 1,
	0b110 => 1,
	0b111 => 0
)

# ╔═╡ Cell order:
# ╟─724acb50-6db3-11eb-07a8-7d048b8e4b3e
# ╟─9464f530-6db3-11eb-0629-9f05df1979bd
# ╟─f3f9f3b0-6db3-11eb-1446-cfe467e42fee
# ╟─fdc546b0-6db3-11eb-1a4c-514e32ae4a27
# ╟─87bd67d0-6db4-11eb-0931-f9a3ec94cd70
# ╠═b3fdfd00-6db4-11eb-3fa3-79d5911c567d
# ╟─f21ba14e-6db4-11eb-382a-87bc07250f65
# ╟─7895f0e0-6db6-11eb-1d6e-b12880ebd5b9
# ╠═0ab21310-6db6-11eb-00b5-1f2bd2203c79
# ╠═656f4030-6db5-11eb-0360-bd9ac3ac6b45
# ╟─232031c0-6db6-11eb-2f48-81afe98ccac3
# ╠═71d85310-6db6-11eb-0728-f34323017fe1
# ╠═aefbdff0-6db6-11eb-0441-a924d0f567c1
# ╟─ea3c54f0-6db6-11eb-1312-3daa47f3fa91
# ╟─f018e190-6db6-11eb-3eef-c984453d36d9
# ╠═a29c2ea0-6ff4-11eb-2954-372c6727bee2
# ╠═06c775f0-6db7-11eb-2603-4942353b201e
# ╟─8bb7e6a0-6db7-11eb-3491-039358de17d7
# ╠═dfe4d210-6db7-11eb-1425-732f8753c0f5
# ╠═f4bd6df0-6db7-11eb-15a0-c52c636db094
# ╠═f9fe56d0-6db7-11eb-1a02-6133a37e6426
# ╟─065d25f2-6db8-11eb-0ab4-95151bc3554f
# ╟─4117f030-6db8-11eb-0df5-4d606d3faf6b
# ╠═caf3afb0-6db8-11eb-39de-852bb7609cb3
