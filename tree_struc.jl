##############
## Tree class
##############

# Height is the value of the random density. Nothing for the root.
# rinter is the value (linter, rinter]
mutable struct Node
    height::Float
    parent::Union{Int, Nothing}
    children::Vector{Int}
    level::Int
    rinter::Float
    # Constructor with default value, new instansiates a node
    function Node(height::Float, parent::Union{Int, Nothing}, level::Int, rinter::Float)
        new(height, parent, Vector{Int}(), level, rinter)
    end
end
 

mutable struct Tree
    nodes::Vector{Node}


"""
    Tree(quants::Vector{Float64})

Constructor for a posterior tree. probs is a vector containig the sampled probabilities for the sapmle quantiles
"""
function Tree(probs::Vector{Float64}, inter::Vector{Float64})
        tree = new(Vector{Node}()) #first split in 2 and then in another 2
        add_node!(tree, 1, nothing, inter[2]) #prob 1 to be in the root
        add_node!(tree, probs[1], 1, inter[1])
        add_node!(tree, probs[2], 1, inter[3])
    end
end

# i: the node you want to attach a child to
function add_node!(tree::Tree, height::Float64, i::Union{Int, Nothing}, rinter::Float)
    i > length(tree.nodes) || throw(BoundsError(tree, i)) # this throws the exception when the condition is met

    # If adding a root
    if isnothing(i)
        push!(tree.nodes, Node(height, i, 0, rinter))
    else
        # Add the node to the tree. 
        push!(tree.nodes, Node(height, i, tree.nodes[i].level + 1, rinter))

        # Indicate the parent now has a new kid
        push!(tree.nodes[i].children, length(tree.nodes))
    end
end

## to add which level each node belongs to
#= function distance_to_root(tree, node_index)
    if isnothing(tree[node_index].parent)  # root node
        return 0
    else
        parent_index = tree[node_index].parent
        return 1 + distance_to_root(tree, parent_index)
    end
end =#

## getter for children
function get_children(tree::Tree, id::Int)
    tree.nodes[id].children
end

function get_parent(tree::Tree, id::Int)::Node
    tree[tree.nodes[id].parent]
end


