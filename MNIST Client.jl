### A Pluto.jl notebook ###
# v0.14.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ a721d7c0-980a-11eb-1b4e-23bc8dbc5211
begin
	using Pluto, PlutoUI, HypertextLiteral
end

# ╔═╡ 71bbf1ec-c8c4-4e19-b544-0fd5752e2869
using ImageTransformations

# ╔═╡ 6a4b75d5-03f7-43cf-a9dc-e1a4781397da
using Images

# ╔═╡ cd8c0ee9-9c9e-4099-83a1-e208f569a455
md"# Classifying handwritten digits!"

# ╔═╡ 0e600c58-6156-41da-b0cb-c638c1ef6798
@htl("""
	
<div>
	$(@bind classify_canvas Button("Classify"))
	$(@bind clear_canvas Button("Clear"))
</div>

""")

# ╔═╡ 115b959e-4e44-42ca-a97d-73be8c3826de
@htl("""

<!-- Update cell if we want to clear the canvas -->
<!-- $(clear_canvas) -->

Try writing a digit in the box!<br><br>
<canvas id="canvas" width="140" height="140" style="border: 1px solid grey">HTML5 is required to run this notebook</canvas>
<script>
	const canvas = document.getElementById('canvas');
	const ctx = canvas.getContext('2d');
	
	let mouseDown = false;
	let paths = [];
	
	canvas.onmousedown = () => {
		mouseDown = true;
		paths.push([]);
	};
	window.onmouseup = () => {mouseDown=false;};
	
	canvas.onmousemove = function(e) {
		const rect = canvas.getBoundingClientRect();
		const x = e.clientX - rect.left;
		const y = e.clientY - rect.top;
	
		// The mouse is being dragged
		if(mouseDown) {
			paths[paths.length - 1].push([x, y]);
			render();
		}
	}
	
	
	// Render our list of points
	function render() {
		if(paths.length === 0) return;
		ctx.clearRect(0, 0, canvas.width, canvas.height);
	
		for(let p=0; p<paths.length; p++) {
			const points = paths[p];
			ctx.moveTo(points[0][0], points[0][1]);
			for(let i=1; i<points.length; i++) {
				ctx.lineTo(points[i][0], points[i][1]);
			}
			ctx.strokeStyle = 'black';
			ctx.lineWidth = 10;
			ctx.stroke();
		}
	}
</script>
	
""")

# ╔═╡ 33652a07-96ef-4388-94f6-3b60e33df6ff
@htl("""
	
<!-- $(classify_canvas) -->
	
<script>
	const canvas = document.getElementById('canvas');
	const ctx = canvas.getContext('2d');
	
	const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
	
	const hiddenText = document.getElementById('hidden_imgdata');
	hiddenText.addEventListener('change', console.log);
	hiddenText.value = imageData.data.join(',');
	hiddenText.dispatchEvent(new Event("input"));
	
</script>
	
""")

# ╔═╡ 42743644-4984-4b84-a924-bb6e28f85fc8
@bind pixel_data_raw html"""<input type="hidden" id="hidden_imgdata"/>"""

# ╔═╡ 5e73ad1c-b2a5-40ee-ad6c-0a633f8974b1
pixel_data_flat = parse.(Int, split(pixel_data_raw, ","))

# ╔═╡ 68fd63ec-6181-44d0-a6bf-fa3684b12744
function get_pixel_data(x, y)
	return pixel_data_flat[((x - 1) * 140 + y) * 4]
end

# ╔═╡ 8f9c01ef-03c5-451b-a9c2-1a672b003f91
pixel_data = [get_pixel_data(x, y) for x ∈ 1:140, y ∈ 1:140]

# ╔═╡ fce4dd05-8b31-4d0e-9d4a-5dde003b90d0
image = Gray.(pixel_data ./ 255)

# ╔═╡ bb579671-5685-405a-a0b9-57cf367ef27d
small_image = imresize(image, 28, 28)

# ╔═╡ de39700a-7b6d-44bf-9296-c053fd997aea
input_images = permutedims(reshape(collect(channelview(small_image)), (28, 28, 1, 1)), (2, 1, 3, 4));

# ╔═╡ 636054d4-8c33-4995-8e59-e0c1cce1c4fe
classifier_notebook = PlutoNotebook("Connor Burns — MNIST classifier for WYSIWYR.jl")

# ╔═╡ 699f3821-41e3-4fdc-814a-f906be3fe2ce
output_labels = classifier_notebook(; input_images=input_images).output_labels

# ╔═╡ 27f58f6f-01d9-4d45-a876-04dc6d5e9d72
md"### Prediction: $(output_labels |> first)"

# ╔═╡ 2465eae0-f58e-4f6c-8571-ac5e816b71c6
md"### Guess: **$(output_labels |> first)**"

# ╔═╡ Cell order:
# ╟─cd8c0ee9-9c9e-4099-83a1-e208f569a455
# ╟─a721d7c0-980a-11eb-1b4e-23bc8dbc5211
# ╟─115b959e-4e44-42ca-a97d-73be8c3826de
# ╟─0e600c58-6156-41da-b0cb-c638c1ef6798
# ╟─27f58f6f-01d9-4d45-a876-04dc6d5e9d72
# ╟─33652a07-96ef-4388-94f6-3b60e33df6ff
# ╟─42743644-4984-4b84-a924-bb6e28f85fc8
# ╠═5e73ad1c-b2a5-40ee-ad6c-0a633f8974b1
# ╠═68fd63ec-6181-44d0-a6bf-fa3684b12744
# ╠═8f9c01ef-03c5-451b-a9c2-1a672b003f91
# ╠═71bbf1ec-c8c4-4e19-b544-0fd5752e2869
# ╠═6a4b75d5-03f7-43cf-a9dc-e1a4781397da
# ╠═fce4dd05-8b31-4d0e-9d4a-5dde003b90d0
# ╠═bb579671-5685-405a-a0b9-57cf367ef27d
# ╠═de39700a-7b6d-44bf-9296-c053fd997aea
# ╠═636054d4-8c33-4995-8e59-e0c1cce1c4fe
# ╠═699f3821-41e3-4fdc-814a-f906be3fe2ce
# ╟─2465eae0-f58e-4f6c-8571-ac5e816b71c6
