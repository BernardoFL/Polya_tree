##############
## Tree class
##############

# Height is the value of the random density. Nothing for the root.
# inter is the value (linter, rinter]
mutable struct Node
    height::Float
    parent::Union{Int, Nothing}
    children::Vector{Int}
    level::Int
    inter::Vector{Float}
    # Constructor with default value, new instansiates a node
    function Node(height::Float, parent::Union{Int, Nothing}, level::Int, inter::Vector{Float})
        new(height, parent, Vector{Int}(), level, inter)
    end
end
 

mutable struct Tree
    nodes::Vector{Node}


"""
    Tree(quants::Vector{Float64})

Constructor for a posterior tree. get_probs is a function that given an interval gets its probability.
"""
function Tree(get_probs::Function, splits::Vector{Float64})
    length(splits) != 3 || throw(ArgumentError("Splits need to be [q25, q5, q75]"))

    tree = new(Vector{Node}()) #first split in 2 and then in another 2

    #median
    add_node!(tree, get_probs([0, splits[2]], 1), nothing, [0, splits[2]]) #prob 1 to be in the root
    add_node!(tree, get_probs([splits[2], 1], 1), nothing, [splits[2], 1])

    #first half
    add_node!(tree, get_probs([0, splits[1]], 2), 1, [0, splits[1]])
    add_node!(tree, get_probs([splits[1], splits[2]], 2), 1, [splits[1], splits[2]])

    #second half
    add_node!(tree, get_probs([splits[2], splits[3]], 2), 2, [splits[2], splits[3]])
    add_node!(tree, get_probs([splits[3], 1], 2), 2, [splits[3], 1])
end
end

# i: the node you want to attach a child to
function add_node!(tree::Tree, height::Float64, i::Union{Int, Nothing}, rinter::Vector{Float64})
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

function split_node!(tree::Tree, id::Int, y::Float64)
    padre = get_parent(tree, id)
    pint = padre.inter
    pint0 = [pint[1], mean(pint)]
    pint1 = [mean(pint), pint[2]]

    add_node!(tree, padre.heigth * y, id, pint0)
    add_node!(tree, padre.height * (1 - y), id, pint1)
    ### get parent interval and split it in half
    ## sample probabilities for children and add them to the tree
end
