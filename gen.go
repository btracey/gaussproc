// gaussproc implements a lot of things to help with Gaussian processes.
package gaussproc

import (
	"github.com/btracey/kernel"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

// MeanStdMat returns the mean and standard deviations of the columns of the
// data matrix. If all of the elements of the column have the same value, a
// standard deviation of 1 is returned. If x == nil, MeanStd panics.
func MeanStdMat(x mat.Matrix) (mean, std []float64) {
	if x == nil {
		panic(nilInput)
	}
	samp, dim := x.Dims()
	mean = make([]float64, dim)
	std = make([]float64, dim)
	col := make([]float64, samp)
	for j := 0; j < dim; j++ {
		mat.Col(col, j, x)
		m, s := stat.MeanStdDev(col, nil)
		mean[j] = m
		if s == 0 {
			s = 1
		}
		std[j] = s
	}
	return mean, std
}

// kernelMatrix computes the kernel matrix between the elements of x and xp,
// scaling the rows if necessary
func kernelMatrix(k *mat.Dense, x, xp mat.Matrix, mean, std []float64, ker kernel.Kerneler) *mat.Dense {
	m, p := x.Dims()
	n, p2 := xp.Dims()
	if p != p2 {
		panic(badInputDim)
	}
	if k == nil {
		k = mat.NewDense(m, n, nil)
	}
	m2, n2 := k.Dims()
	if m2 != m || n2 != n {
		panic(badStorageDim)
	}
	xi := make([]float64, p)
	xj := make([]float64, p)
	for i := 0; i < m; i++ {
		rowScaled(xi, i, x, mean, std)
		for j := 0; j < n; j++ {
			rowScaled(xj, j, xp, mean, std)
			v := ker.Kernel(xi, xj)
			k.Set(i, j, v)
		}
	}
	return k
}

// kernelMatrixSym computes the kernel matrix between the elements of x and
// themselves, scaling the rows if necessary.
func kernelMatrixSym(k *mat.SymDense, x mat.Matrix, mean, std []float64, ker kernel.Kerneler, noise float64) *mat.SymDense {
	m, p := x.Dims()
	if k == nil {
		k = mat.NewSymDense(m, nil)
	}
	mk := k.Symmetric()
	if m != mk {
		panic(badStorageDim)
	}
	xi := make([]float64, p)
	xj := make([]float64, p)
	for i := 0; i < m; i++ {
		rowScaled(xi, i, x, mean, std)
		for j := i; j < m; j++ {
			rowScaled(xj, j, x, mean, std)
			v := ker.Kernel(xi, xj)
			if i == j {
				v += noise * noise // so the noise is the std not the covariance
			}
			k.SetSym(i, j, v)
		}
	}
	return k
}

func scaleY(dst, y []float64, mean, std float64) []float64 {
	if dst == nil {
		dst = make([]float64, len(y))
	}
	if len(dst) != len(y) {
		panic(badInputDim)
	}
	if mean == 0 && std == 0 {
		copy(dst, y)
		return dst
	}
	for i, v := range y {
		dst[i] = (v - mean) / std
	}
	return dst
}

func unscaleY(dst, y []float64, mean, std float64) []float64 {
	if dst == nil {
		dst = make([]float64, len(y))
	}
	if len(dst) != len(y) {
		panic(badInputDim)
	}
	if mean == 0 && std == 0 {
		copy(dst, y)
		return dst
	}
	for i, v := range y {
		dst[i] = v*std + mean
	}
	return dst
}

// rowScaled returns the i'th row of the matrix a, scaling the row according to
// the scaling provided if necessary.
func rowScaled(row []float64, i int, a mat.Matrix, mean, std []float64) {
	mat.Row(row, i, a)
	if mean == nil {
		return
	}
	if len(row) != len(mean) {
		panic("gaussproc: bad size")
	}
	if len(mean) != len(std) {
		panic("gaussproc: bad size")
	}
	for i, v := range row {
		row[i] = (v - mean[i]) / std[i]
	}
	return
}
