// gaussproc is a package for using gaussian processes.
package gaussproc

import (
	"errors"
	"fmt"
	"math"

	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"
	"github.com/gonum/optimize"
	"github.com/gonum/stat"
)

// TODO: Check if mat64 does the right thing when zero

var (
	minLogNoise = math.Log(1e-6)
	maxLogNoise = math.Log(1.0)
)

const (
	initGpSize = 0

	barrierPow = 4 // what power on breaking the barrier
	//minGrowSize = 100
	//maxGrowSize = 1000
)

const (
	badInputLength  = "gp: input length mismatch"
	badOutputLength = "gp: output length mismatch"
	badInOut        = "gp: inequal number of input and output samples"
	badStorage      = "gp: bad storage length"
)

var (
	ErrSingular = errors.New("gp: kernel matrix singular or near singular")
)

// TODO (btracey): Should the inputs be scaled automatically? Would help guess
// the hyperparameters better. Could otherwise just use the data as an initial guess
// and don't scale

// TODO(btracey): Need to add in noise parameter. Only multiplied along diagonal.
// TODO(btracey): Think about adding weights

// Gp contains a Gaussian Process fit.
type GP struct {
	kernel Kernel  // Kernel function of the Gaussian process
	noise  float64 // noise added to the diagonal of the covariance matrix

	inputDim int

	inputs  *mat64.Dense // matrix of the actual input data
	outputs []float64    // output data stored scaled

	mean float64 // The mean of the output data
	std  float64 // standard deviation of the output data

	k *mat64.SymDense // kernel matrix between inputs
	//cholK   *mat64.TriDense
	cholK   *mat64.Cholesky
	sigInvY *mat64.Vector
}

// New creates a new GP with the given input dimension, the given
// kernel function, and output noise parameter. Output dim must be one.
func New(inputDim int, kernel Kernel, noise float64) *GP {
	if inputDim <= 0 {
		panic("gp: non-positive inputDim")
	}
	if kernel == nil {
		panic("gp: nil kernel")
	}
	if !(noise >= 0) {
		panic("gp: negative noise") // also handles NaN.
	}

	return &GP{
		kernel:   kernel,
		noise:    noise,
		inputDim: inputDim,
		mean:     0,
		std:      1,
		inputs:   &mat64.Dense{},
		outputs:  make([]float64, 0),
		k:        mat64.NewSymDense(0, nil),
		sigInvY:  &mat64.Vector{},
		cholK:    &mat64.Cholesky{},
	}
}

// AddBatch adds a set training points to the Gp. This call updates internal
// values needed for prediction, so it is more efficient to add samples
// as a batch.
func (g *GP) AddBatch(x mat64.Matrix, y []float64) error {
	// Note: The outputs are stored scaled to have a mean of zero and a variance
	// of 1.

	// Verify input parameters
	rx, cx := x.Dims()
	ry := len(y)
	if rx != ry {
		panic(badInOut)
	}
	if cx != g.inputDim {
		panic(badInputLength)
	}
	nSamples := len(g.outputs)

	// Append the new data to the list of stored data.
	inputs := mat64.NewDense(rx+nSamples, g.inputDim, nil)
	inputs.Copy(g.inputs)
	inputs.View(nSamples, 0, rx, g.inputDim).(*mat64.Dense).Copy(x)
	g.inputs = inputs
	// Rescale the output data to its original value, append the new data, and
	// then rescale to have mean 0 and variance of 1.
	for i, v := range g.outputs {
		g.outputs[i] = v*g.std + g.mean
	}
	g.outputs = append(g.outputs, y...)
	g.mean = stat.Mean(g.outputs, nil)
	g.std = stat.StdDev(g.outputs, nil)
	for i, v := range g.outputs {
		g.outputs[i] = (v - g.mean) / g.std
	}

	// Add to the kernel matrix.
	k := mat64.NewSymDense(rx+nSamples, nil)
	k.CopySym(g.k)
	g.k = k
	// Compute the kernel with the new points and the old points
	for i := 0; i < nSamples; i++ {
		for j := nSamples; j < rx+nSamples; j++ {
			v := g.kernel.Distance(g.inputs.RawRowView(i), g.inputs.RawRowView(j))
			g.k.SetSym(i, j, v)
		}
	}

	// Compute the kernel with the new points and themselves
	for i := nSamples; i < rx+nSamples; i++ {
		for j := i; j < nSamples+rx; j++ {
			v := g.kernel.Distance(g.inputs.RawRowView(i), g.inputs.RawRowView(j))
			if i == j {
				v += g.noise
			}
			g.k.SetSym(i, j, v)
		}
	}
	// Cache necessary matrix results for computing predictions.
	var chol mat64.Cholesky
	ok := chol.Factorize(g.k)
	if !ok {
		return ErrSingular
	}
	g.cholK = &chol
	g.sigInvY.Reset()
	v := mat64.NewVector(len(g.outputs), g.outputs)
	g.sigInvY.SolveCholeskyVec(g.cholK, v)
	return nil
}

// Mean returns the gaussian process prediction of the mean at the location x.
func (g *GP) Mean(x []float64) float64 {
	// y_mean = k_*^T K^-1 y
	// where k_* is the vector of the kernel between the new location and all
	// of the data points
	// y are the outputs at all the data points
	// K^-1 is the full covariance of the data points
	// (K^-1y is stored)

	if len(x) != g.inputDim {
		panic(badInputLength)
	}
	nSamples, _ := g.inputs.Dims()

	covariance := make([]float64, nSamples)
	for i := range covariance {
		covariance[i] = g.kernel.Distance(x, g.inputs.RawRowView(i))
	}
	y := floats.Dot(g.sigInvY.RawVector().Data, covariance)
	return y*g.std + g.mean
}

// MeanBatch predicts the mean at the set of locations specified by x. Stores in-place into yPred
// If yPred is nil new memory is allocated.
func (g *GP) MeanBatch(yPred []float64, x mat64.Matrix) []float64 {
	rx, cx := x.Dims()
	if cx != g.inputDim {
		panic(badInputLength)
	}
	if yPred == nil {
		yPred = make([]float64, rx)
	}
	ry := len(yPred)
	if rx != ry {
		panic(badOutputLength)
	}
	nSamples, _ := g.inputs.Dims()

	covariance := mat64.NewDense(nSamples, rx, nil)
	row := make([]float64, g.inputDim)
	for j := 0; j < rx; j++ {
		for k := 0; k < g.inputDim; k++ {
			row[k] = x.At(j, k)
		}
		for i := 0; i < nSamples; i++ {
			v := g.kernel.Distance(g.inputs.RawRowView(i), row)
			covariance.Set(i, j, v)
		}
	}
	yPredVec := mat64.NewVector(len(yPred), yPred)
	yPredVec.MulVec(covariance.T(), g.sigInvY)
	// Rescale the outputs
	for i, v := range yPred {
		yPred[i] = v*g.std + g.mean
	}
	return yPred
}

// StdDev predicts the standard deviation of the function at x.
func (g *GP) StdDev(x []float64) float64 {
	if len(x) != g.inputDim {
		panic(badInputLength)
	}
	// nu_* = k(x_*, k_*) - k_*^T * K^-1 * k_*
	n := len(g.outputs)
	kstar := mat64.NewVector(n, nil)
	for i := 0; i < n; i++ {
		v := g.kernel.Distance(g.inputs.RawRowView(i), x)
		kstar.SetVec(i, v)
	}
	self := g.kernel.Distance(x, x)
	var tmp mat64.Vector
	tmp.SolveCholeskyVec(g.cholK, kstar)
	var tmp2 mat64.Vector
	tmp2.MulVec(kstar.T(), &tmp)
	rt, ct := tmp2.Dims()
	if rt != 1 || ct != 1 {
		panic("bad size")
	}
	return math.Sqrt(self-tmp2.At(0, 0)) * g.std
}

// StdDevBatch predicts the standard deviation at a set of locations of x.
func (g *GP) StdDevBatch(std []float64, x mat64.Matrix) []float64 {
	r, c := x.Dims()
	if c != g.inputDim {
		panic(badInputLength)
	}
	if std == nil {
		std = make([]float64, r)
	}
	if len(std) != r {
		panic(badStorage)
	}
	// For a single point, the stddev is
	// 		sigma = k(x,x) - k_*^T * K^-1 * k_*
	// where k is the vector of kernels between the input points and the output points
	// For many points, the formula is:
	// 		nu_* = k(x_*, k_*) - k_*^T * K^-1 * k_*
	// This creates the full covariance matrix which is an rxr matrix. However,
	// the standard deviations are just the diagonal of this matrix. Instead, be
	// smart about it and compute the diagonal terms one at a time.
	kStar := g.formKStar(x)
	var tmp mat64.Dense
	tmp.SolveCholesky(g.cholK, kStar)

	// set k(x_*, x_*) into std then subtract k_*^T K^-1 k_* , computed one row at a time
	var tmp2 mat64.Vector
	row := make([]float64, c)
	for i := range std {
		for k := 0; k < c; k++ {
			row[k] = x.At(i, k)
		}
		std[i] = g.kernel.Distance(row, row)
		tmp2.MulVec(kStar.ColView(i).T(), tmp.ColView(i))
		rt, ct := tmp2.Dims()
		if rt != 1 && ct != 1 {
			panic("bad size")
		}
		std[i] -= tmp2.At(0, 0)
		std[i] = math.Sqrt(std[i])
	}
	// Need to scale the standard deviation to be in the same units as y.
	floats.Scale(g.std, std)
	return std
}

// Cov returns the covariance between a set of data points based on the current
// GP fit.
func (g *GP) Cov(m *mat64.SymDense, x mat64.Matrix) *mat64.SymDense {
	if m != nil {
		// TODO(btracey): Make this k**
		panic("resuing m not coded")
	}
	// The joint covariance matrix is
	// K(x_*, k_*) - k(x_*, x) k(x,x)^-1 k(x, x*)
	nSamp, nDim := x.Dims()
	if nDim != g.inputDim {
		panic(badInputLength)
	}

	// Compute K(x_*, x) K(x, x)^-1 K(x, x_*)
	kstar := g.formKStar(x)
	var tmp mat64.Dense
	tmp.SolveCholesky(g.cholK, kstar)
	var tmp2 mat64.Dense
	tmp2.Mul(kstar.T(), &tmp)

	// Compute k(x_*, x_*) and perform the subtraction.
	kstarstar := mat64.NewSymDense(nSamp, nil)
	for i := 0; i < nSamp; i++ {
		for j := i; j < nSamp; j++ {
			v := g.kernel.Distance(mat64.Row(nil, i, x), mat64.Row(nil, j, x))
			if i == j {
				v += g.noise
			}
			kstarstar.SetSym(i, j, v-tmp2.At(i, j))
		}
	}
	return kstarstar
}

// formKStar forms the covariance matrix between the inputs and new points.
func (g *GP) formKStar(x mat64.Matrix) *mat64.Dense {
	// TODO(btracey): Parallelize
	r, c := x.Dims()
	n := len(g.outputs)
	kStar := mat64.NewDense(n, r, nil)
	data := make([]float64, c)
	for j := 0; j < r; j++ {
		for k := 0; k < c; k++ {
			data[k] = x.At(j, k)
		}
		for i := 0; i < n; i++ {
			row := g.inputs.RawRowView(i)
			v := g.kernel.Distance(row, data)
			kStar.Set(i, j, v)
		}
	}
	return kStar
}

// Train sets the paramters of the gaussian process. If noise == true,
// the noise parameter is adjusted, otherwise it is not.
// TODO(btracey): Need to implement barrier method for parameters. Steps get crazy.
func (g *GP) Train(trainNoise bool) error {
	// TODO(btracey): Implement a memory struct that can be passed around with
	// all of this data.

	initHyper := g.kernel.Hyper(nil)
	nKerHyper := len(initHyper)
	if trainNoise {
		initHyper = append(initHyper, math.Log(g.noise))
	}

	mem := newMargLikeMemory(len(initHyper), len(g.outputs))

	f := func(x []float64) float64 {
		fmt.Println("x =", x)
		obj := g.marginalLikelihood(x, trainNoise, mem)
		fmt.Println("obj =", obj)
		return obj
	}
	df := func(x, grad []float64) {
		g.marginalLikelihoodDerivative(x, grad, trainNoise, mem)
		fmt.Println("x = ", x)
		fmt.Println("grad = ", grad)
	}

	//	grad =  [0.4500442759224154 -3.074041876494095 0.42568788880060204]
	/*
		x := []float64{0.7287793210009457, -0.9371471942974932, -14.017213937483529}
		fofx := f(x)
		fmt.Println("fofx", fofx)

		set := fd.DefaultSettings()
		set.Method.Step = 1e-4
		fdGrad := fd.Gradient(nil, f, x, nil)
		fmt.Println("fd grad = ", fdGrad)
		grad := make([]float64, len(fdGrad))
		df(x, grad)
		fmt.Println("real grad = ", grad)
		os.Exit(1)
	*/

	problem := optimize.Problem{
		Func: f,
		Grad: df,
	}
	settings := optimize.DefaultSettings()
	settings.GradientThreshold = 1e-4
	result, err := optimize.Local(problem, initHyper, settings, nil)
	// set noise
	g.noise = math.Exp(result.X[len(result.X)-1])
	g.kernel.SetHyper(result.X[:nKerHyper])
	g.setKernelMat(g.k, g.noise)
	ok := g.cholK.Factorize(g.k)
	if !ok {
		return errors.New("gp: final kernel matrix is not positive definite")
	}
	v := mat64.NewVector(len(g.outputs), g.outputs)
	g.sigInvY.SolveCholeskyVec(g.cholK, v)
	return err
}

type margLikeMemory struct {
	lastX []float64
	// likelihood only
	k     *mat64.SymDense
	chol  *mat64.Cholesky
	alpha *mat64.Vector
	tmp   *mat64.Vector
	// For derivative
	dKdTheta []*mat64.SymDense
	kInvDK   *mat64.Dense
}

func newMargLikeMemory(hyper, outputs int) *margLikeMemory {
	m := &margLikeMemory{
		lastX:    make([]float64, hyper),
		k:        mat64.NewSymDense(outputs, nil),
		chol:     &mat64.Cholesky{},
		alpha:    mat64.NewVector(outputs, nil),
		tmp:      mat64.NewVector(1, nil),
		dKdTheta: make([]*mat64.SymDense, hyper),
		kInvDK:   mat64.NewDense(outputs, outputs, nil),
	}
	for i := 0; i < hyper; i++ {
		m.dKdTheta[i] = mat64.NewSymDense(outputs, nil)
	}
	return m
}

// marginalLikelihood computes the negative marginal likelihood of the data with the
// given hyperparameters.
func (g *GP) marginalLikelihood(x []float64, trainNoise bool, mem *margLikeMemory) float64 {

	// TODO(btracey): Find a less hack-y way to introduce bounds.
	nHyper := g.kernel.NumHyper()

	// If the parameters are outside the bounds introduce a quadratic penalty method
	var barrier float64
	bounds := g.kernel.Bounds()
	if trainNoise {
		bounds = append(bounds, Bound{minLogNoise, maxLogNoise})
	}

	for i, v := range x {
		if v < bounds[i].Min {
			barrier += math.Pow(v-bounds[i].Min, barrierPow)
		}
		if v > bounds[i].Max {
			barrier += math.Pow(v-bounds[i].Max, barrierPow)
		}
	}
	//fmt.Println("barrier = ", barrier)

	// log[p(y|x,theta)] =
	//      -1/2 y^T * K_y^-1 * y -1/2 log |K_y| - n/2 * log(2*pi)
	// Want to maximize probability. Multiply by -2 to minimize and simplify,
	// and ignore constant
	// 		y^T * K_y^-1 * y + log |K_y|
	// alpha = K_y^-1 * y
	n := len(g.outputs)
	copy(mem.lastX, x)
	k := mem.k
	chol := mem.chol
	alpha := mem.alpha
	tmp := mem.tmp

	y := mat64.NewVector(n, g.outputs)
	var noise float64
	if trainNoise {
		noise = math.Exp(x[len(x)-1])
	} else {
		noise = g.noise
	}
	g.kernel.SetHyper(x[:nHyper])
	g.setKernelMat(k, noise)
	ok := chol.Factorize(k)
	if !ok {
		// The kernel matrix is singular. Don't let it be
		return math.Inf(1)
	}
	alpha.SolveCholeskyVec(chol, y)
	// TODO(btracey): add mat64.Dot(*Vector, *Vector)
	tmp.MulVec(y.T(), alpha)
	r, c := tmp.Dims()
	if r != 1 || c != 1 {
		panic("tmp bad size")
	}
	/*
		var logdet float64
		for i := 0; i < n; i++ {
			logdet += 2 * math.Log(chol.At(i, i))
		}
	*/
	logdet := chol.LogDet()
	// This is proportional to negative likelihood.
	// Divide by the number of samples to make the barrier penalty
	// the same regardless of data size.
	negLogLike := (tmp.At(0, 0) + logdet) / float64(n)
	return negLogLike + barrier
}

func (g *GP) marginalLikelihoodDerivative(x, grad []float64, trainNoise bool, mem *margLikeMemory) {
	// d/dTheta_j log[(p|X,theta)] =
	//		1/2 * y^T * K^-1 dK/dTheta_j * K^-1 * y - 1/2 * tr(K^-1 * dK/dTheta_j)
	//		1/2 * α^T * dK/dTheta_j * α - 1/2 * tr(K^-1 dK/dTheta_j)
	// Multiply by the same -2
	//		-α^T * K^-1 * α + tr(K^-1 dK/dTheta_j)
	// This first computation is an inner product.
	n := len(g.outputs)
	nHyper := g.kernel.NumHyper()
	k := mem.k
	chol := mem.chol
	alpha := mem.alpha
	dKdTheta := mem.dKdTheta
	kInvDK := mem.kInvDK

	y := mat64.NewVector(n, g.outputs)

	var noise float64
	if trainNoise {
		noise = math.Exp(x[len(x)-1])
	} else {
		noise = g.noise
	}

	// If x is the same, then reuse what has been computed in the function.
	if !floats.Equal(mem.lastX, x) {
		copy(mem.lastX, x)
		g.kernel.SetHyper(x[:nHyper])
		g.setKernelMat(k, noise)
		//chol.Cholesky(k, false)
		chol.Factorize(k)
		alpha.SolveCholeskyVec(chol, y)
	}
	g.setKernelMatDeriv(dKdTheta, trainNoise, noise)
	for i := range dKdTheta {
		kInvDK.SolveCholesky(chol, dKdTheta[i])
		inner := mat64.Inner(alpha, dKdTheta[i], alpha)
		grad[i] = -inner + mat64.Trace(kInvDK)
	}
	floats.Scale(1/float64(n), grad)

	bounds := g.kernel.Bounds()
	if trainNoise {
		bounds = append(bounds, Bound{minLogNoise, maxLogNoise})
	}
	barrierGrad := make([]float64, len(grad))
	for i, v := range x {
		// Quadratic barrier penalty.
		if v < bounds[i].Min {
			diff := bounds[i].Min - v
			barrierGrad[i] = -(barrierPow) * math.Pow(diff, barrierPow-1)
		}
		if v > bounds[i].Max {
			diff := v - bounds[i].Max
			barrierGrad[i] = (barrierPow) * math.Pow(diff, barrierPow-1)
		}
	}
	fmt.Println("noise, minNoise", x[len(x)-1], bounds[len(x)-1].Min)
	fmt.Println("barrier Grad", barrierGrad)
	floats.Add(grad, barrierGrad)
	//copy(grad, barrierGrad)
}

func (gp *GP) setKernelMat(s *mat64.SymDense, noise float64) {
	n := s.Symmetric()
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			v := gp.kernel.Distance(
				gp.inputs.RawRowView(i),
				gp.inputs.RawRowView(j),
			)
			if i == j {
				v += noise
			}
			s.SetSym(i, j, v)
		}
	}
}

func (gp *GP) setKernelMatDeriv(dKdTheta []*mat64.SymDense, trainNoise bool, noise float64) {
	n := len(gp.outputs)
	nHyper := gp.kernel.NumHyper()
	dk := make([]float64, nHyper)
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			one := gp.inputs.RawRowView(i)
			two := gp.inputs.RawRowView(j)
			gp.kernel.DistanceDHyper(one, two, dk)
			for k := 0; k < nHyper; k++ {
				dKdTheta[k].SetSym(i, j, dk[k])
			}
			if trainNoise {
				if i != j {
					// Noise is only added on the diagonal, so zero derivative elsewhere.
					dKdTheta[nHyper].SetSym(i, j, 0)
				} else {
					// The derivative of the actual noise is one, but the optimization
					// is on the log of the noise.
					// df/dlogx = df/dx * dx/dlogx
					//			= df/dx * (1/(dlogx/dx))
					//			= df/dx * (1/(1/x)) = x * df/dx = x
					dKdTheta[nHyper].SetSym(i, j, noise)
				}
			}
		}
	}
}

/*
// PredictCov predicts the covariance of the outputs at a single point x.
// y is outputDim x outputDim. If y == nil, a new matrix is allocated.
func (gp *GP) PredictCov(y *mat64.SymDense, x []float64) *mat64.SymDense {

}
*/

/*
// Update updates the values of the kernel hyperparameters
func (g *Gp) TrainHyper() {
	// Maximum likelihood estimation. See
	// https://github.com/aerialhedgehog/VyPy/blob/master/trunk/VyPy/regression/gpr/learning/Likelihood.py

	// Need to do initialization. Possibily computing distances.
	// Definitely need to rescale the y points for a new mean and variance.

	// Optimize and then set the noise parameter
}
*/

// need predict with stddev

/*

// LogKernelDist returns the log of the distance
func (kernel SqExpIso) LogDistance(dist float64) float64 {
	logDist := math.Log(dist)
	distOverVariance := math.Exp(logDist - kernel.LogVariance)
	return -0.5*distOverVariance*distOverVariance + 2*kernel.LogLength
}

func (kernel SqExpIso) Distance(dist )
*/
