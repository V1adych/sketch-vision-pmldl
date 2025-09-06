package com.sketchvision.fronted.algo;

import java.util.Arrays;

public class Matrix {
    private final int rows;
    private final int cols;
    private final double[][] data;

    public Matrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = new double[rows][cols];
    }

    public static Matrix identity(int n) {
        Matrix m = new Matrix(n, n);
        for (int i = 0; i < n; i++) m.data[i][i] = 1.0;
        return m;
    }

    public static Matrix from(double[][] d) {
        int r = d.length;
        int c = r == 0 ? 0 : d[0].length;
        Matrix m = new Matrix(r, c);
        for (int i = 0; i < r; i++) m.data[i] = Arrays.copyOf(d[i], c);
        return m;
    }

    public Matrix multiply(Matrix other) {
        if (this.cols != other.rows) throw new IllegalArgumentException("shape mismatch");
        Matrix out = new Matrix(this.rows, other.cols);
        for (int i = 0; i < rows; i++) {
            for (int k = 0; k < cols; k++) {
                double a = this.data[i][k];
                for (int j = 0; j < other.cols; j++) {
                    out.data[i][j] += a * other.data[k][j];
                }
            }
        }
        return out;
    }

    @Override public String toString() { return Arrays.deepToString(data); }
}
