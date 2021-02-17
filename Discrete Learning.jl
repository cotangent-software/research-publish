### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# ╔═╡ a9d68520-7176-11eb-3696-f1f8fa05ab9d
md"""
# Discrete Learning
### Setup


```julia
function print_set(set_in, set_out)
    for i = 1:size(set_in, 1)
        println(string(set_in[i, :], " -> ", set_out[i]))
    end
end
function index(arr, val)
    indices = findall(x->x==val, arr)
    if length(indices) === 0
        return -1
    end
    return indices[1]
end
```




    index (generic function with 1 method)



## Simple Case (Unconditional, one-to-one, top-constant)
For this first case, we will be learning a discrete square function. Below is the set of values which will be used.


```julia
square_in = Float64[1, 2, 3, 4]
square_out = Float64[1, 4, 9, 16]
print_set(square_in, square_out)
```

    [1.0] -> 1.0
    [2.0] -> 4.0
    [3.0] -> 9.0
    [4.0] -> 16.0
    

The objective, so to speak, is to find some operation and value which will translate the left-hand operand to the right hand operand. We will allow addition and multiplication. To solve this analytically, the given operations must also have inverse operations, in this case being subtraction and division.


```julia
ops = [+, *]
inv_ops = [-, /]
```




    2-element Array{Function,1}:
     -
     /



From here, a translation can be found in most cases for each pair. A notable exception is division by zero. To find solutions, one can create a stack to store structs containing the entire traversal of the operation tree. The structure can be defined as follows


```julia
mutable struct OperationPath
    constants::Array{Float64, 1}
    operations::Array{Function, 1}
end
```

The constants variable in the structure denotes the Nth inverse operation between some input and the previous constants. In the intial condition, the constants are considered to be the output vector, as can be seen on line 2 of the algorithm below. The operations vector is a method of storing which operations, in proper order, were performed on the input set to result in the output set. Each individual operation path can fully reproduce the output set given the input set in a unique way.

The algorithm below is a naive implementation which traverses the tree of all possible paths from the input to the output set down to a depth of D. It performs no checks for certain favorable conditions which allow for optimization. These conditions will be discussed in depth later.


```julia
all_paths = [ ]
path_stack = [ OperationPath(copy(square_out), []) ]
D = 3
for i in 1:D
    next_path_stack = OperationPath[]

    for path in path_stack
        next_paths = (
            inv_op->OperationPath(
                inv_op.(path.constants, square_in), 
                push!(copy(path.operations), inv_op)
            )
        ).(inv_ops)

        push!(next_path_stack, next_paths...)
    end
    
    path_stack = next_path_stack
    push!(all_paths, next_path_stack...)
end
println("Total paths: ", length(all_paths))
path_stack
```

    Total paths: 14
    




    8-element Array{OperationPath,1}:
     OperationPath([-2.0, -2.0, 0.0, 4.0], Function[-, -, -])              
     OperationPath([-1.0, 0.0, 1.0, 2.0], Function[-, -, /])               
     OperationPath([-1.0, -1.0, -1.0, -1.0], Function[-, /, -])            
     OperationPath([0.0, 0.5, 0.6666666666666666, 0.75], Function[-, /, /])
     OperationPath([-1.0, -2.0, -3.0, -4.0], Function[/, -, -])            
     OperationPath([0.0, 0.0, 0.0, 0.0], Function[/, -, /])                
     OperationPath([0.0, -1.0, -2.0, -3.0], Function[/, /, -])             
     OperationPath([1.0, 0.5, 0.3333333333333333, 0.25], Function[/, /, /])



One can call a given path a perfect solution if all the constants are equal. This equates to each number in the input set following an identical set of operations to perfectly produce the output set.


```julia
struct SolutionPath
    constant::Float64
    operations::Array{Function, 1}
end
function inv(op)
    i = index(ops, op)
    if i === -1
        return ops[index(inv_ops, op)]
    end
    return inv_ops[i]
end
solutions = (x->SolutionPath(x.constants[1], inv.(x.operations))).(
    filter(path->path.constants==fill(path.constants[1], length(path.constants)), all_paths)
)
solutions
```




    4-element Array{SolutionPath,1}:
     SolutionPath(0.0, Function[*, +])    
     SolutionPath(1.0, Function[*, *])    
     SolutionPath(-1.0, Function[+, *, +])
     SolutionPath(0.0, Function[*, +, *]) 



To ensure that these are indeed solutions, we can compare each solution path's reconstructed output set and compare it to the desired output set. 


```julia
function resolve_solution_path(solution)
    constants = fill(solution.constant, length(square_in))
    for op in reverse(solution.operations)
        constants = op.(square_in, constants)
    end
    return constants
end
reproduced_square_outs = resolve_solution_path.(solutions)

for reproduced_square_out in reproduced_square_outs
    println(square_out, " == ", reproduced_square_out, " -> ", reproduced_square_out == square_out)
end
```

    [1.0, 4.0, 9.0, 16.0] == [1.0, 4.0, 9.0, 16.0] -> true
    [1.0, 4.0, 9.0, 16.0] == [1.0, 4.0, 9.0, 16.0] -> true
    [1.0, 4.0, 9.0, 16.0] == [1.0, 4.0, 9.0, 16.0] -> true
    [1.0, 4.0, 9.0, 16.0] == [1.0, 4.0, 9.0, 16.0] -> true
    

## Simple Multi-output Case (Unconditional, one-to-many, top-constant)
In this case, a single variable is related to multiple outputs. This is a near-identical problem to the simple single output case, as each output can be treated individually as multiple single output input pairs.

For sake of simplicity, I will leave out the implementation which is nearly identical.

## Simple Multi-variable Case (Unconditional, many-to-one, top-constant)
Unlike the last case, this is a significantly more difficult problem to solve. By relating two variables to a single output, the number of paths per layer effectively is multiplied by the number of variables.


```julia
mutable struct MultiOperationPath
    constants::Array{Float64, 1}
    operations::Array{Function, 1}
    variables::Array{Int64, 1}
end
```

Unlike the previous structure defined for operation paths, this one contains a variables array. This contains the list of which variables the given operation at the same index should be performed on.

The algorithm below is similar to the simple single variable case, with a few slight modifications to enable multiple independent variables.

First we will define the multi-varaible input and the single-variable output. We will also use the same operations as previous.


```julia
add_in = Float64[1 2; 2 1; 3 5; 4 2]
add_out = Float64[3; 3; 8; 6]
print_set(add_in, add_out)
```

    [1.0, 2.0] -> 3.0
    [2.0, 1.0] -> 3.0
    [3.0, 5.0] -> 8.0
    [4.0, 2.0] -> 6.0
    

Next we define the algorithm which computes the operation paths.


```julia
all_paths = [ ]
path_stack = [ MultiOperationPath(copy(add_out), [], []) ]
D = 3
for i in 1:D
    next_path_stack = MultiOperationPath[]

    for path in path_stack
        next_paths = MultiOperationPath[]
        for idx in 1:size(add_in, 2)
            for inv_op in inv_ops
                push!(next_paths, MultiOperationPath(
                        inv_op.(path.constants, add_in[:, idx]), 
                        push!(copy(path.operations), inv_op), 
                        push!(copy(path.variables), idx)))
            end
        end
        
        push!(next_path_stack, next_paths...)
    end
    
    path_stack = next_path_stack
    push!(all_paths, next_path_stack...)
end
println("Total paths: ", length(all_paths))
path_stack
```

    Total paths: 84
    




    64-element Array{MultiOperationPath,1}:
     MultiOperationPath([0.0, -3.0, -1.0, -6.0], Function[-, -, -], [1, 1, 1])                 
     MultiOperationPath([1.0, -0.5, 0.6666666666666666, -0.5], Function[-, -, /], [1, 1, 1])   
     MultiOperationPath([-1.0, -2.0, -3.0, -4.0], Function[-, -, -], [1, 1, 2])                
     MultiOperationPath([0.5, -1.0, 0.4, -1.0], Function[-, -, /], [1, 1, 2])                  
     MultiOperationPath([1.0, -1.5, -1.3333333333333333, -3.5], Function[-, /, -], [1, 1, 1])  
     MultiOperationPath([2.0, 0.25, 0.5555555555555556, 0.125], Function[-, /, /], [1, 1, 1])  
     MultiOperationPath([0.0, -0.5, -3.333333333333333, -1.5], Function[-, /, -], [1, 1, 2])   
     MultiOperationPath([1.0, 0.5, 0.33333333333333337, 0.25], Function[-, /, /], [1, 1, 2])   
     MultiOperationPath([-1.0, -2.0, -3.0, -4.0], Function[-, -, -], [1, 2, 1])                
     MultiOperationPath([0.0, 0.0, 0.0, 0.0], Function[-, -, /], [1, 2, 1])                    
     MultiOperationPath([-2.0, -1.0, -5.0, -2.0], Function[-, -, -], [1, 2, 2])                
     MultiOperationPath([0.0, 0.0, 0.0, 0.0], Function[-, -, /], [1, 2, 2])                    
     MultiOperationPath([0.0, -1.0, -2.0, -3.0], Function[-, /, -], [1, 2, 1])                 
     ⋮                                                                                         
     MultiOperationPath([0.5, -0.5, -2.466666666666667, -3.25], Function[/, /, -], [2, 1, 1])  
     MultiOperationPath([1.5, 0.75, 0.17777777777777778, 0.1875], Function[/, /, /], [2, 1, 1])
     MultiOperationPath([-0.5, 0.5, -4.466666666666667, -1.25], Function[/, /, -], [2, 1, 2])  
     MultiOperationPath([0.75, 1.5, 0.10666666666666666, 0.375], Function[/, /, /], [2, 1, 2]) 
     MultiOperationPath([-1.5, 0.0, -6.4, -3.0], Function[/, -, -], [2, 2, 1])                 
     MultiOperationPath([-0.5, 1.0, -1.1333333333333333, 0.25], Function[/, -, /], [2, 2, 1])  
     MultiOperationPath([-2.5, 1.0, -8.4, -1.0], Function[/, -, -], [2, 2, 2])                 
     MultiOperationPath([-0.25, 2.0, -0.6799999999999999, 0.5], Function[/, -, /], [2, 2, 2])  
     MultiOperationPath([-0.25, 1.0, -2.68, -2.5], Function[/, /, -], [2, 2, 1])               
     MultiOperationPath([0.75, 1.5, 0.10666666666666667, 0.375], Function[/, /, /], [2, 2, 1]) 
     MultiOperationPath([-1.25, 2.0, -4.68, -0.5], Function[/, /, -], [2, 2, 2])               
     MultiOperationPath([0.375, 3.0, 0.064, 0.75], Function[/, /, /], [2, 2, 2])               



At this point, we need to perform another similar process to find and transform the solution operation paths into multi solutions paths.


```julia
struct MultiSolutionPath
    constant::Float64
    operations::Array{Function, 1}
    variables::Array{Int64, 1}
end
solutions = (x->MultiSolutionPath(x.constants[1], inv.(x.operations), x.variables)).(
    filter(path->path.constants==fill(path.constants[1], length(path.constants)), all_paths)
)
solutions
```




    8-element Array{MultiSolutionPath,1}:
     MultiSolutionPath(0.0, Function[+, +], [1, 2])      
     MultiSolutionPath(1.0, Function[+, *], [1, 2])      
     MultiSolutionPath(0.0, Function[+, +], [2, 1])      
     MultiSolutionPath(1.0, Function[+, *], [2, 1])      
     MultiSolutionPath(0.0, Function[+, +, *], [1, 2, 1])
     MultiSolutionPath(0.0, Function[+, +, *], [1, 2, 2])
     MultiSolutionPath(0.0, Function[+, +, *], [2, 1, 1])
     MultiSolutionPath(0.0, Function[+, +, *], [2, 1, 2])



Once again, employing a similar tactic to previously, we will convert all these solution paths into their corresponding output sets.


```julia
function resolve_multi_solution_path(solution)
    constants = fill(solution.constant, size(add_in, 1))
    i = 1
    rev_vars = reverse(solution.variables)
    for op in reverse(solution.operations)
        constants = op.(add_in[:, rev_vars[i]], constants)
        i+=1
    end
    return constants
end
reproduced_add_outs = resolve_multi_solution_path.(solutions)

for reproduced_add_out in reproduced_add_outs
    println(add_out, " == ", reproduced_add_out, " -> ", reproduced_add_out == add_out)
end
```

    [3.0, 3.0, 8.0, 6.0] == [3.0, 3.0, 8.0, 6.0] -> true
    [3.0, 3.0, 8.0, 6.0] == [3.0, 3.0, 8.0, 6.0] -> true
    [3.0, 3.0, 8.0, 6.0] == [3.0, 3.0, 8.0, 6.0] -> true
    [3.0, 3.0, 8.0, 6.0] == [3.0, 3.0, 8.0, 6.0] -> true
    [3.0, 3.0, 8.0, 6.0] == [3.0, 3.0, 8.0, 6.0] -> true
    [3.0, 3.0, 8.0, 6.0] == [3.0, 3.0, 8.0, 6.0] -> true
    [3.0, 3.0, 8.0, 6.0] == [3.0, 3.0, 8.0, 6.0] -> true
    [3.0, 3.0, 8.0, 6.0] == [3.0, 3.0, 8.0, 6.0] -> true
    
"""

# ╔═╡ Cell order:
# ╟─a9d68520-7176-11eb-3696-f1f8fa05ab9d
