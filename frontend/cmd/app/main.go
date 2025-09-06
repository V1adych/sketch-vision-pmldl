package main

import (
	"fmt"
	"github.com/example/sketch-vision/frontend/internal/algo"
	"github.com/example/sketch-vision/frontend/internal/model"
	"github.com/example/sketch-vision/frontend/internal/util"
)

func main() {
	fmt.Println("Frontend sandbox running.")

	fmt.Println("IsBlank(\"  \"):", util.IsBlank("  "))
	fmt.Println("Repeat(\"Go\", 3):", util.Repeat("Go", 3))
	fmt.Println("Join([1,2,3], -):", util.Join([]any{1, 2, 3}, "-"))

	p := model.Pair[int, string]{First: 7, Second: "seven"}
	fmt.Println("Pair:", p)

	m := algo.Identity(3)
	n := algo.From([][]float64{{1,2,3},{4,5,6},{7,8,9}})
	fmt.Println("I*X == X:", m.Multiply(n))
}
