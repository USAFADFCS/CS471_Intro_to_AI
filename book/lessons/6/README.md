**Mathematical Notation for Informed Search**
* $n$: The current node or state being evaluated.
* $g(n)$: The exact, historical path cost from the start node to node $n$ (e.g., fuel expended so far).
* $h(n)$: The heuristic estimated cost from node $n$ to the goal (e.g., straight-line distance to target).
* $f(n)$: The total estimated cost of the path through node $n$. Calculated as $f(n) = g(n) + h(n)$.