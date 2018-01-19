package gaussproc

import (
	"fmt"
	"log"
	"math"

	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/gonum/stat/distmv"
	"gonum.org/v1/gonum/stat/distuv"

	"github.com/btracey/kernel"
	"gonum.org/v1/gonum/mat"
)

// StudentsT implements a Student's T process
type StudentsT struct {
	gp *GP

	nu   float64
	beta float64
}

func NewStudentsT(nu float64, ker kernel.Kerneler, x mat.Matrix, y []float64, noise float64, normalized bool) (*StudentsT, error) {
	s := &StudentsT{}
	gp, err := NewGP(ker, x, y, noise, normalized)
	if err != nil {
		return nil, err
	}

	// The important part is that the kernel function sets the covariance, which
	// isn't the same as the matrix input to the normal multi-variate students's T
	// distribution. To get the matrix input in Student's T need to multiply by
	// nu / (nu-2).

	// Need to compute \beta which is (y-mu)\Sigma_{x,x}^-1 * (y-mu)
	// y - mu is yScaled
	beta := s.calculateBeta(y, gp.kInv, gp.meanY, gp.stdY)

	s.gp = gp
	s.nu = nu
	s.beta = beta
	return s, nil
}

func (s *StudentsT) InputScaling() (mean, std []float64) {
	return s.gp.InputScaling()
}

func (s *StudentsT) OutputScaling() (mean, std float64) {
	return s.gp.OutputScaling()
}

func (s *StudentsT) MeanStd(x []float64) (float64, float64) {
	marg := s.MarginalStudentsT(x)
	return marg.Mean(), marg.StdDev()
}

func (s *StudentsT) MarginalStudentsT(x []float64) distuv.StudentsT {
	n1 := float64(len(s.gp.y))

	// Mean is the same as the GP mean.
	//  K_{x,d} * K_{d,d}^-1 * y_1
	// TODO(btracey): replace this with a better call
	// Note there is no difference in the nu factor, since it cancels with the inverse.
	mean, std := s.gp.MeanStd(x)

	// http://mlg.eng.cam.ac.uk/amar/papers/workshop.pdf
	// https://pdfs.semanticscholar.org/4fe1/f40dcb6d393499061c3316b8448c966a2a87.pdf
	// The variance is.
	//  var_hat = K_{x,x} - K_{x,d} K^-1_{d,d} K_{d,x}
	//  var = (nu + beta1 - 2)/(nu + n1 -2)
	// where
	//  beta1 = y^T K_11^-1 y_1
	// Note that cov_hat is the same as the GP standard deviation, and that
	// beta1 is the current beta in s.
	// Again, this is an update for the covariance, scale below for the extra
	// factor. If we plugged in Sigma to the parameter,

	variance := ((s.nu + s.beta - 2) / (s.nu + n1 - 2)) * std * std

	// Convert this into the parameter for the Student's T.
	// Student's T takes in sigma, which is
	//  std = Sigma * math.Sqrt(nu/(nu-2))

	sigma := math.Sqrt(variance * (s.nu - 2) / s.nu)

	// The Nu updated is  nu + |N_1|
	nu := s.nu + n1

	return distuv.StudentsT{
		Mu:    mean,
		Sigma: sigma,
		Nu:    nu,
	}
}

func (s *StudentsT) ExpectedImprovement(x []float64, best float64) float64 {
	marginal := s.MarginalStudentsT(x)
	return ExpectedImprovementStudentsT(best, marginal)
}

func ExpectedImprovementStudentsT(best float64, s distuv.StudentsT) float64 {
	// (f_b - mu)*Phi(z) + (nu/(nu-1))*sigma*(1+z^2/nu) * phi(z)
	mean := s.Mu
	if math.IsInf(best, 1) {
		return -mean
	}
	// (f_b - mu)*Phi(z) + (nu/(nu-1))*sigma*(1+z^2/nu) * phi(z)
	sigma := s.Sigma
	nu := s.Nu
	z := (best - mean) / sigma
	d := distuv.StudentsT{Nu: nu, Mu: 0, Sigma: 1}
	ei := (best-mean)*d.CDF(z) + nu/(nu-1)*sigma*(1+z*z/nu)*d.Prob(z)
	return ei
}

func (s *StudentsT) Observe(x []float64, y float64) error {
	// Observe the data in the GP to update Kinv and KinvY
	err := s.gp.Observe(x, y)
	if err != nil {
		log.Fatal(err)
	}

	// Update beta. gp.Observe() has updated the data.
	s.beta = s.calculateBeta(s.gp.y, s.gp.kInv, s.gp.meanY, s.gp.stdY)
	return nil
}

func (s *StudentsT) calculateBeta(y []float64, kInv *mat.Cholesky, meanY, stdY float64) float64 {
	yScaled := scaleY(nil, y, meanY, stdY)
	yScaledVec := mat.NewVecDense(len(yScaled), yScaled)
	tmp := mat.NewVecDense(len(yScaled), nil)
	kInv.SolveVec(tmp, yScaledVec)
	return mat.Dot(tmp, yScaledVec)
}

// LikelihoodStudentsT is a type for computing the marginal likelihood as a function
// of Kernel hyperparameters for a Student's T Process.
//
// TODO(btracey): Have an OptNoise term and a nu term.
type LikelihoodStudentsT struct {
	Kernel kernel.LogKernelHyperer
	Noise  float64
	Nu     float64
	X      mat.Matrix
	Y      []float64
	MeanX  []float64
	StdX   []float64
	MeanY  float64
	StdY   float64
}

// Normalize normalizes the X and Y data, overwriting the existing X and Y data.
func (m *LikelihoodStudentsT) Normalize() {
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
func (m *LikelihoodStudentsT) NegativeLikelihood(hyper []float64) float64 {
	// The likelihood is log probability of the y vector for a
	// normal distribution with mean 0 and covariance based on K_y. Specifically,
	// need to rescale the covariance matrix to get the right value.

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
	nu := m.Nu

	// K_y specifies the covariance of the Student's T distribution, but we need
	// the shape parameter. The covariance is
	//  cov = ν/(ν-2) Σ
	// So scale K_y to use properly
	ky.ScaleSym((nu-2)/nu, ky)

	st, ok := distmv.NewStudentsT(mu, ky, nu, nil)
	if !ok {
		fmt.Println("not pos def")
		return math.Inf(1)
	}
	yScale := scaleY(nil, m.Y, m.MeanY, m.StdY)
	like := st.LogProb(yScale)
	return -like

	/*
		norm, ok := distmv.NewNormal(mu, ky, nil)
		if !ok {
			fmt.Println("not pos def")
			return math.Inf(1)
		}

		yScale := scaleY(nil, m.Y, m.MeanY, m.StdY)
		likeNorm := norm.LogProb(yScale)
		return -likeNorm
	*/
}
