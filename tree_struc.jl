##############
## Tree class
##############

# Height is the value of the random density. Nothing for the root.
mutable struct Node
    height::Float64
    parent::Union{Int, Nothing}
    children::Vector{Int}
    level::Int

    # Constructor with default value, new instansiates a node
    function Node(height::Float64, parent::Union{Int, Nothing}, level::Int)
        new(height, parent, Vector{Int}(), level)
    end
end


mutable struct Tree
    nodes::Vector{Node}


    """
    Tree(quants::Vector{Float64})

Constructor for a posterior tree. quants is a vector containig [median, q_25, q_75].
"""
function Tree(quants::Vector{Float64})
        tree = new(Vector{Node}())
        add_node!(tree, quants[1], nothing, 1)
        add_node!(tree, quants[2], 1)
        add_node!(tree, quants[3], 1)
    end
end

# i: the node you want to attach a child to
function add_node!(tree::Tree, height::Float64, i::Union{Int, Nothing})
    i > length(tree.nodes) || throw(BoundsError(tree, i)) # this throws the exception when the condition is met

    # Add the node to the tree. 
    push!(tree.nodes, Node(height, i, distance_to_root(tree, i)))

    # Indicate the parent now has a new kid
    push!(tree.nodes[i].children, length(tree.nodes))
end

## to add which level each node belongs to
function distance_to_root(tree, node_index)
    if tree[node_index].parent == 0  # root node
        return 0
    else
        parent_index = tree[node_index].parent
        return 1 + distance_to_root(tree, parent_index)
    end
end

## getter for children
function get_children(tree::Tree, id::Int)
    tree.nodes[id].children
end

function get_parent(tree::Tree, id::Int)::Node
    tree[tree.nodes[id].parent]
end


