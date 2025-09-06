package algo

import (
	"fmt"
	"strings"
)

type Matrix struct {
	Rows int
	Cols int
	Data [][]float64
}

func New(rows, cols int) Matrix {
	m := Matrix{Rows: rows, Cols: cols, Data: make([][]float64, rows)}
	for i := 0; i < rows; i++ {
		m.Data[i] = make([]float64, cols)
	}
	return m
}

func Identity(n int) Matrix {
	m := New(n, n)
	for i := 0; i < n; i++ { m.Data[i][i] = 1 }
	return m
}

func From(d [][]float64) Matrix {
	r := len(d)
	c := 0
	if r > 0 { c = len(d[0]) }
	m := New(r, c)
	for i := 0; i < r; i++ {
		copy(m.Data[i], d[i])
	}
	return m
}

func (m Matrix) Multiply(o Matrix) Matrix {
	if m.Cols != o.Rows {
		panic("shape mismatch")
	}
	out := New(m.Rows, o.Cols)
	for i := 0; i < m.Rows; i++ {
		for k := 0; k < m.Cols; k++ {
			a := m.Data[i][k]
			for j := 0; j < o.Cols; j++ {
				out.Data[i][j] += a * o.Data[k][j]
			}
		}
	}
	return out
}

func (m Matrix) String() string {
	rows := make([]string, m.Rows)
	for i := 0; i < m.Rows; i++ {
		rows[i] = fmt.Sprint(m.Data[i])
	}
	return "[" + strings.Join(rows, ", ") + "]"
}
