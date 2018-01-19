package gaussproc

import (
	"errors"
	"fmt"
	"math"

	"gonum.org/v1/gonum/stat/distmv"

	"github.com/btracey/kernel"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/gonum/stat/distuv"
)

var (
	badInputDim        = "gaussproc: input dimension mismatch"
	badStorageDim      = "gaussproc: storage dimension mismatch"
	nilInput           = "gaussproc: nil input not allowed"
	dataLengthMismatch = "gaussproc: data length mismatch"
)

var (
	NotPosDef = errors.New("gaussproc: error not positive definite")
)

type GP struct {
	ker   kernel.Kerneler
	noise float64

	x *mat.Dense
	y []float64

	kInv  *mat.Cholesky
	kInvY *mat.VecDense

	meanX []float64
	stdX  []float64
	meanY float64
	stdY  float64
}

// NewGP constructs a new Gaussian process with the given input and output data.
// If normalized is true, the data is scaled to have mean 0 variance 1 before
// being passed to the kernel. If x and y are both nil, normalized has no effect
func NewGP(ker kernel.Kerneler, x mat.Matrix, y []float64, noise float64, normalized bool) (*GP, error) {
	if x == nil {
		if y != nil {
			panic(badInputDim)
		}
		return &GP{
			ker:   ker,
			noise: noise,
			meanY: 0,
			stdY:  1,
		}, nil
	}

	samp, dim := x.Dims()
	if len(y) != samp {
		panic(badInputDim)
	}

	xCopy := mat.NewDense(samp, dim, nil)
	xCopy.Copy(x)
	yCopy := make([]float64, len(y))
	copy(yCopy, y)

	// Need to put scaling in here.
	var meanX, stdX []float64
	meanY, stdY := 0.0, 1.0
	if normalized {
		meanX, stdX = MeanStdMat(xCopy)
		meanY, stdY = stat.MeanStdDev(yCopy, nil)
	}

	gp := &GP{
		ker:   ker,
		noise: noise,
		x:     xCopy,
		y:     yCopy,

		meanX: meanX,
		stdX:  stdX,
		meanY: meanY,
		stdY:  stdY,
	}

	k := kernelMatrixSym(nil, xCopy, meanX, stdX, ker, gp.noise)

	var chol mat.Cholesky
	ok := chol.Factorize(k)
	if !ok {
		return nil, NotPosDef
	}

	yScaled := scaleY(nil, yCopy, meanY, stdY)
	yScaledVec := mat.NewVecDense(len(yScaled), yScaled)
	kInvY := mat.NewVecDense(len(yScaled), nil)

	chol.SolveVec(kInvY, yScaledVec)

	gp.kInv = &chol
	gp.kInvY = kInvY
	return gp, nil
}

// TODO(btracey): This should really be a Marginal.

func (gp *GP) InputScaling() (mean, std []float64) {
	meanX := make([]float64, len(gp.meanX))
	copy(meanX, gp.meanX)
	stdX := make([]float64, len(gp.stdX))
	copy(stdX, gp.stdX)
	return meanX, stdX
}

func (gp *GP) OutputScaling() (mean, std float64) {
	return gp.meanY, gp.stdY
}

func (gp *GP) MeanStd(x []float64) (mean, std float64) {
	xMat := mat.NewDense(1, len(x), x)
	means, stds := gp.MeansStds(xMat)
	return means[0], stds[0]
}

// MeanStd returns the mean and standard deviation for the input locations.
func (gp *GP) MeansStds(x mat.Matrix) (mean, std []float64) {
	samp, dim := x.Dims()
	meanDst := make([]float64, samp)
	stdDst := make([]float64, samp)
	if gp.x == nil {
		// There are no data points yet, so the prediction is just the base
		// mean and variance.
		for i := range meanDst {
			meanDst[i] = gp.meanY
			stdDst[i] = gp.stdY
		}
	}

	// Compute the kernel between the new data and the existing data.
	kxd := kernelMatrix(nil, x, gp.x, gp.meanX, gp.stdX, gp.ker)

	// Mean prediction is
	//  k_{x,d}*k_{d,d}^-1 y
	meanVec := mat.NewVecDense(samp, meanDst)
	meanVec.MulVec(kxd, gp.kInvY)

	// For each point, the variance is
	//  k_{x,x} - k_{x,d}*k_{d,d}^-1 * k_{x,d}'
	// Compute these terms one at a time since otherwise there is a huge explosion
	// in the memory requirements.
	row := make([]float64, dim)
	rowMat := mat.NewDense(1, len(row), row)

	nData, _ := gp.x.Dims()
	kxdData := make([]float64, nData)
	kxdMat := mat.NewDense(1, nData, kxdData)
	kxdVec := mat.NewVecDense(nData, kxdData)

	kxx := mat.NewSymDense(1, nil)
	tmp := mat.NewVecDense(nData, nil)
	for i := 0; i < samp; i++ {
		mat.Row(row, i, x)
		kernelMatrix(kxdMat, rowMat, gp.x, gp.meanX, gp.stdX, gp.ker)
		kernelMatrixSym(kxx, rowMat, gp.meanX, gp.stdX, gp.ker, 0) // This isn't supposed to have noise
		gp.kInv.SolveVec(tmp, kxdVec)
		stdDst[i] = kxx.At(0, 0) - mat.Dot(tmp, kxdVec)
	}

	// Above is the variance, NOT the standard deviation.
	for i, v := range stdDst {
		stdDst[i] = math.Sqrt(v)
	}

	// Need to unscale the mean and std.
	mean = make([]float64, samp)
	std = make([]float64, samp)
	unscaleY(mean, meanDst, gp.meanY, gp.stdY)
	unscaleY(std, stdDst, 0, gp.stdY) // mean doesn't shift, just the scale.
	return mean, std
}

func (gp *GP) MeanCov(x mat.Matrix) (means []float64, cov *mat.SymDense) {
	samp, _ := x.Dims()
	meanDst := make([]float64, samp)
	covDst := mat.NewSymDense(samp, nil)

	// Compute the kernel between the new locations and themselves.
	kdd := kernelMatrixSym(nil, x, gp.meanX, gp.meanX, gp.ker, gp.noise)

	if gp.x == nil {
		// There are no data points yet, so the prediction is just the base
		// mean and variance.
		for i := range meanDst {
			meanDst[i] = gp.meanY
		}
		covDst.CopySym(kdd)
		if gp.meanY != 0 || gp.stdY != 1 {
			panic("not coded for scaled")
		}
		return meanDst, covDst
	}

	// Compute the kernel between the new data and the existing data.
	kxd := kernelMatrix(nil, x, gp.x, gp.meanX, gp.stdX, gp.ker)

	// Mean prediction is
	//  k_{x,d}*k_{d,d}^-1 y
	meanVec := mat.NewVecDense(samp, meanDst)
	meanVec.MulVec(kxd, gp.kInvY)

	// Cov matrix is:
	//  k_{x,x} - k_{x,d}*k_{d,d}^-1 * k_{x,d}'
	var tmp mat.Dense
	gp.kInv.Solve(&tmp, kxd.T())
	var tmp2 mat.Dense
	tmp2.Mul(kxd, &tmp)
	for i := 0; i < samp; i++ {
		for j := 0; j < samp; j++ {
			v := kdd.At(i, j) - tmp2.At(i, j)
			covDst.SetSym(i, j, v)
		}
	}
	if gp.meanY != 0 || gp.stdY != 1 {
		panic("not coded for scaled")
	}
	return meanDst, covDst
}

// Observe updates the Gaussian process with the observation that f(x) = y. This
// does not update the variable scaling or anything involving the Kernel.
func (gp *GP) Observe(x []float64, y float64) error {
	nData, dim := gp.x.Dims()
	if len(x) != dim {
		panic(badInputDim)
	}
	// Update the Cholesky decomposition of k_{d,d}
	kData := make([]float64, nData+1) // ndata + 1 because we also need the kernel with itself
	// First, compute the kernel between the new points and the old locations.
	kxdMat := mat.NewDense(1, nData, kData[:nData])
	xmat := mat.NewDense(1, dim, x)
	kernelMatrix(kxdMat, xmat, gp.x, gp.meanX, gp.stdX, gp.ker)
	// Add the kernel with itself at the end
	kxxMat := mat.NewSymDense(1, kData[nData:nData+1])
	kernelMatrixSym(kxxMat, xmat, gp.meanX, gp.stdX, gp.ker, gp.noise)

	// Now, update the Cholesky decomposition with the new kernel data.
	kVec := mat.NewVecDense(len(kData), kData)
	ok := gp.kInv.ExtendVecSym(gp.kInv, kVec)
	if !ok {
		return errors.New("not pos def")
	}
	// Extend the existing data with the new data.
	// TODO(btracey): Be smarter about growing the matrix, so dont' need to copy
	// everything every time.
	newx := mat.NewDense(nData+1, dim, nil)
	newx.Copy(gp.x)
	for j := 0; j < dim; j++ {
		newx.Set(nData, j, x[j])
	}
	gp.x = newx
	gp.y = append(gp.y, y)

	yScaled := scaleY(nil, gp.y, gp.meanY, gp.stdY)
	yScaledVec := mat.NewVecDense(len(yScaled), yScaled)
	kInvY := mat.NewVecDense(len(yScaled), nil)
	gp.kInv.SolveVec(kInvY, yScaledVec)
	gp.kInvY = kInvY
	return nil
}

func (gp *GP) ExpectedImprovement(x []float64, best float64) float64 {
	mean, std := gp.MeanStd(x)
	n := distuv.Normal{Mu: mean, Sigma: std}
	return ExpectedImprovementGaussian(best, n)
}

// ExpectedImprovementGaussian returns the expected improvement over the
// current best. This assumes the function is being minimized. Higher
// expected improvement is better. If math.IsInf(best,1), then it assumes
// no samples have been observed and so the expected improvement is just the
// negative mean.
func ExpectedImprovementGaussian(best float64, n distuv.Normal) float64 {
	mean := n.Mean()
	std := n.StdDev()
	if math.IsInf(best, 1) {
		return -mean
	}
	d := distuv.UnitNormal
	z := (best - mean) / std
	ei := (best-mean)*d.CDF(z) + std*d.Prob(z)
	return ei
}

// MarginalLikelihoodGP is a type for computing the marginal likelihood as a function
// of Kernel hyperparameters for a Gaussian Process.
//
// TODO(btracey): Have an OptNoise term.
type LikelihoodGP struct {
	Kernel kernel.LogKernelHyperer
	Noise  float64
	X      mat.Matrix
	Y      []float64
	MeanX  []float64
	StdX   []float64
	MeanY  float64
	StdY   float64
}

// Normalize normalizes the X and Y data, overwriting the existing X and Y data.
func (m LikelihoodGP) Normalize() {
	meanX, stdX := MeanStdMat(m.X)
	m.MeanX = meanX
	m.StdX = stdX
	meanY, stdY := stat.MeanStdDev(m.Y, nil)
	m.MeanY = meanY
	m.StdY = stdY
}

// NegativeLikelihood computes the negative marginal likelihood of the data with
// the given hyperparameters. The negative likelihood is returned so the best
// value is the minimum of the function.
func (m LikelihoodGP) NegativeLikelihood(hyper []float64) float64 {
	// The marginal likelihood is.
	// log[p(y|x,theta)] =
	//      -1/2 y^T * K_y^-1 * y - 1/2*log |K_y| - n/2 * log(2*pi)
	// which is the same thing as the log probability of the y vector for a
	// normal distribution with mean 0 and covariance K_y.

	r, _ := m.X.Dims()
	if r != len(m.Y) {
		panic(dataLengthMismatch)
	}
	ker := kernel.LogKernelWrapper{
		Hyper:       hyper,
		LogKerneler: m.Kernel,
	}
	ky := kernelMatrixSym(nil, m.X, m.MeanX, m.StdX, ker, m.Noise)
	mu := make([]float64, r)
	norm, ok := distmv.NewNormal(mu, ky, nil)
	if !ok {
		fmt.Println("not pos def")
		return math.Inf(1)
	}

	yScale := scaleY(nil, m.Y, m.MeanY, m.StdY)
	likeNorm := norm.LogProb(yScale)
	return -likeNorm
}

// Have a TrainGP function

/*
// Mean returns the mean predictions for the locations at xnew given the cholesky
// decomposition of the kernel matrix. Stores the result into yNew.
func Mean(y []float64, k mat.Matrix, kInvY []float64) []float64 {
	// Mean is k xx' * K^-1 * y
	m, n := k.Dims()
	n2 := len(kInvY)
	if n != n2 {
		panic(badInputDim)
	}
	if y == nil {
		y = make([]float64, m)
	}
	if len(y) != m {
		panic(badStorageDim)
	}
	yVec := mat.NewVecDense(m, y)
	kInvYVec := mat.NewVecDense(m, y)
	yVec.MulVec(k, kInvYVec)
	return y
}

// KernelMatrix computes the kernel matrix between the samples in x and xprime.
// The i,jth entry in the kernel mat is the kernel between x_i and x_j
func KernelMatrix(k *mat.Dense, x, xprime mat.Matrix, ker kernel.Kerneler) *mat.Dense {
	m, p := x.Dims()
	n, p2 := xprime.Dims()
	if p != p2 {
		panic(badInputDim)
	}
	if k == nil {
		k = mat.NewDense(m, n, nil)
	}
	mk, nk := k.Dims()
	if mk != m || nk != n {
		panic(badStorageDim)
	}
	xi := make([]float64, p)
	xj := make([]float64, p)
	for i := 0; i < m; i++ {
		mat.Row(xi, i, x)
		for j := 0; j < n; j++ {
			mat.Row(xj, j, xprime)
			v := kernel.Kernel(xi, xj)
			k.Set(i, j, v)
		}
	}
	return k
}
*/
