### A Pluto.jl notebook ###
# v0.14.0

using Markdown
using InteractiveUtils

# ╔═╡ 958d41ca-1b5e-405b-8767-d4bebb3aac66
begin
	import Pkg
	Pkg.activate(mktempdir())
	Pkg.add(["MLDatasets", "Images", "Flux", "ImageIO"])
	
	using MLDatasets
	using Images
	using Flux
	using Flux: onehotbatch, onecold, unsqueeze, Data.DataLoader
	using Statistics
end

# ╔═╡ 2b038257-a1a4-40d3-bbba-6dc0ff1e2474
using Serialization

# ╔═╡ b3b34e7a-3831-4e70-a4e2-8499108c3d60
md"## Training notebook for \"PlutoCon 2021 WYSIWYR Demo (MNIST)\"
Once again I shoud stress that the focus of this talk is *not* on this model, but rather on \"building\" an API around it with [upcoming Pluto features](https://github.com/fonsp/Pluto.jl/pull/1052)!"

# ╔═╡ ac3dfbe8-5298-43f5-b639-6f4599cf6ced
ENV["DATADEPS_ALWAYS_ACCEPT"] = "true";

# ╔═╡ aeccd705-966f-44da-888b-dd025ef82185
MNIST.download(; i_accept_the_terms_of_use=true)

# ╔═╡ eb78c825-da2b-4b90-bf3e-967191b23aca
md"Loads the MNIST dataset from downloaded files"

# ╔═╡ 4f336d93-5dd8-4412-af6f-aa1749fb5d61
train_raw_x, train_cold_y = MNIST.traindata();

# ╔═╡ 5cf9e100-4112-48b5-8235-e6bacc44f102
test_raw_x,  test_cold_y  = MNIST.testdata();

# ╔═╡ a0230805-9fc9-4a88-884d-ed6d4aae5129
md"Reshapes the data to work with `Flux.jl` model"

# ╔═╡ da7feb20-812b-47e1-88a4-430d9f87630d
train_x, train_y = unsqueeze(train_raw_x, 3), onehotbatch(train_cold_y, 0:9);

# ╔═╡ b637e559-88c7-472f-9985-39a1d0541dd4
test_x, test_y = unsqueeze(test_raw_x, 3), onehotbatch(test_cold_y, 0:9);

# ╔═╡ 92dde47d-53d1-4f88-9616-67c1e07f22c7
train_data = DataLoader(train_x, train_y; batchsize=128);

# ╔═╡ de8da9ba-3604-4616-ad02-fa7a3f0ff5f3
size(train_x)

# ╔═╡ fd5f98ca-e8f1-4ded-a5e8-40930aa06163
md"The first image in the test set is shown below"

# ╔═╡ ceb5c86a-6c3e-4d6e-bcff-26c5216a9731
Gray.(permutedims(train_x[:, :, 1, 1], (2, 1)))

# ╔═╡ 12ed781c-ceb5-4064-ac95-2e497cbf6082
md"Now we define a simple model involving several convolutional layers, max pooling layers, and a dense layer"

# ╔═╡ f260f73e-9504-410b-afea-4160351e04c3
model = Chain(
	Conv((3, 3), 1=>8, relu),
	MaxPool((2, 2)),
	Conv((3, 3), 8=>16, relu),
	MaxPool((2, 2)),
	Conv((3, 3), 16=>32, relu),
	MaxPool((2, 2)),
	flatten,
	
	Dense(32, 10),
	softmax
)

# ╔═╡ 8726661b-9321-4daa-85e8-ce208a0bbf2d
size(model(train_x[:, :, :, 1:2]))

# ╔═╡ ad95785a-1aa1-4eab-9acb-f46757fc69b8
md"Functions for calculating our model's loss and accuracy"

# ╔═╡ 6fcf9495-bad3-4e94-8674-526266c8b645
accuracy(ŷ, y) = mean(onecold(ŷ) .== onecold(y))

# ╔═╡ e34d2bc6-c4c1-40d3-b3f6-356cfecac442
loss(x, y) = Flux.crossentropy(model(x), y)

# ╔═╡ 674c2b37-e4c0-4271-8b47-147ce1b777d0
loss(train_x, train_y)

# ╔═╡ 8bb2e369-5019-409d-a3fc-93d74b32e9f0
md"Now we define some basic hyperparameters of our model"

# ╔═╡ 5ddd3c96-37fc-4eff-be70-f48200f5db6b
α = 0.1

# ╔═╡ e9367fdd-a77c-43dd-b7a5-ffacfc940fff
opt = Descent(α)

# ╔═╡ 292d2dea-8d9e-40d1-b7ba-9893cecdd9f9
epochs = 10

# ╔═╡ 5b44fe18-7b33-4e26-9ae7-3b9c83aa422f
params = Flux.params(model)

# ╔═╡ 1ae3ed92-fe72-408e-96e4-ef0f4b7c496d
md"We are ready to train our model now. Training this model takes about 10 minutes"

# ╔═╡ 4a355a94-0f90-4b89-a35c-a325b0249886
for epoch ∈ 1:epochs
	@info epoch accuracy(model(train_x), train_y) accuracy(model(test_x), test_y)
	
	for batch ∈ train_data
		∇J = Flux.gradient(params) do
			loss(batch...)
		end

		Flux.update!(opt, params, ∇J)
	end
end

# ╔═╡ 7cf65377-c66e-44ec-bba0-4ae46fbc2d0d
md"And now our model should have around a 98% accuracy"

# ╔═╡ ae4f8ebf-e844-4e0a-a6db-8f8d8dc61df4
accuracy(model(train_x), train_y), accuracy(model(test_x), test_y)

# ╔═╡ 4fbad422-a37c-4588-a864-f591df732775
md"Now we save our model to a `jls` file (stands for julia serialization) which will allow us to load it in other notebooks later."

# ╔═╡ 9505711a-f20c-43f0-87c1-4532007bb0af
open(io -> serialize(io, model), "mnist_conv.jls", "w")

# ╔═╡ 0394483b-d85e-496f-b439-ff9c3aa506aa
md"And now that the model is trained we can hop back on over to the test notebook!"

# ╔═╡ Cell order:
# ╟─b3b34e7a-3831-4e70-a4e2-8499108c3d60
# ╟─958d41ca-1b5e-405b-8767-d4bebb3aac66
# ╟─ac3dfbe8-5298-43f5-b639-6f4599cf6ced
# ╠═aeccd705-966f-44da-888b-dd025ef82185
# ╟─eb78c825-da2b-4b90-bf3e-967191b23aca
# ╠═4f336d93-5dd8-4412-af6f-aa1749fb5d61
# ╠═5cf9e100-4112-48b5-8235-e6bacc44f102
# ╟─a0230805-9fc9-4a88-884d-ed6d4aae5129
# ╠═da7feb20-812b-47e1-88a4-430d9f87630d
# ╠═b637e559-88c7-472f-9985-39a1d0541dd4
# ╠═92dde47d-53d1-4f88-9616-67c1e07f22c7
# ╠═de8da9ba-3604-4616-ad02-fa7a3f0ff5f3
# ╟─fd5f98ca-e8f1-4ded-a5e8-40930aa06163
# ╠═ceb5c86a-6c3e-4d6e-bcff-26c5216a9731
# ╟─12ed781c-ceb5-4064-ac95-2e497cbf6082
# ╠═f260f73e-9504-410b-afea-4160351e04c3
# ╠═8726661b-9321-4daa-85e8-ce208a0bbf2d
# ╟─ad95785a-1aa1-4eab-9acb-f46757fc69b8
# ╠═6fcf9495-bad3-4e94-8674-526266c8b645
# ╠═e34d2bc6-c4c1-40d3-b3f6-356cfecac442
# ╠═674c2b37-e4c0-4271-8b47-147ce1b777d0
# ╟─8bb2e369-5019-409d-a3fc-93d74b32e9f0
# ╠═5ddd3c96-37fc-4eff-be70-f48200f5db6b
# ╠═e9367fdd-a77c-43dd-b7a5-ffacfc940fff
# ╠═292d2dea-8d9e-40d1-b7ba-9893cecdd9f9
# ╠═5b44fe18-7b33-4e26-9ae7-3b9c83aa422f
# ╟─1ae3ed92-fe72-408e-96e4-ef0f4b7c496d
# ╠═4a355a94-0f90-4b89-a35c-a325b0249886
# ╟─7cf65377-c66e-44ec-bba0-4ae46fbc2d0d
# ╠═ae4f8ebf-e844-4e0a-a6db-8f8d8dc61df4
# ╟─4fbad422-a37c-4588-a864-f591df732775
# ╠═2b038257-a1a4-40d3-bbba-6dc0ff1e2474
# ╠═9505711a-f20c-43f0-87c1-4532007bb0af
# ╟─0394483b-d85e-496f-b439-ff9c3aa506aa
