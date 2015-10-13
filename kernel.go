package gaussproc

import (
	"math"

	"github.com/gonum/floats"
)

type Bound struct {
	Min float64
	Max float64
}

type Kernel interface {
	//LogDistance(x, y []float64) float64
	Distance(x, y []float64) float64
	DistanceDHyper(x, y, deriv []float64) float64
	// StationaryDistance(dist float64) (function of just the distance)
	Hyper([]float64) []float64
	SetHyper(x []float64)
	NumHyper() int
	// Bounds on the parameters
	Bounds() []Bound
}

var _ Kernel = &SqExpIso{}

// TODO: Hyperparameters need to be an input. Otherwise hard for training in parallel.
// Maybe add a Duplicate function where it just returns a copy of itself
// SqExpIso represents an isotropic squared exponential kernel
// Logs are used for improved numerical conditioning
type SqExpIso struct {
	LogVariance float64 // Log of the variance of the kernel
	LogLength   float64 // Log of the length scale of the kernel function
}

func (k *SqExpIso) NumHyper() int {
	return 2
}

func (k SqExpIso) Distance(x, y []float64) float64 {
	return math.Exp(k.LogDistance(x, y))
}

// DistanceDeriv computes the distance between x and y and the derivative of
// the distance with respect to the hyperparameters
// Kernel is
//  variance^2 * math.Exp(- r^2 / (2 * l^2))
func (k SqExpIso) DistanceDHyper(x, y, deriv []float64) float64 {
	if len(deriv) != k.NumHyper() {
		panic("gp: deriv length mismatch")
	}

	norm := floats.Distance(x, y, 2)
	logNorm := math.Log(norm)
	logExp := -math.Exp(2*logNorm - 2*k.LogLength - math.Ln2)
	logdist := 2*k.LogVariance + logExp
	dist := math.Exp(logdist)

	logDDistDVar := math.Ln2 + k.LogVariance + logExp
	logDDistDLogVar := logDDistDVar + k.LogVariance
	deriv[0] = math.Exp(logDDistDLogVar)

	logDDistDLength := logdist + 2*logNorm - 3*k.LogLength
	logDDistDLogLength := k.LogLength + logDDistDLength
	deriv[1] = math.Exp(logDDistDLogLength)
	return dist
}

func (k SqExpIso) LogDistance(x, y []float64) float64 {
	if len(x) != len(y) {
		panic("length mismatch")
	}
	norm := floats.Distance(x, y, 2)
	logNorm := math.Log(norm)
	logExp := -math.Exp(2*logNorm - 2*k.LogLength - math.Ln2)
	return 2*k.LogVariance + logExp
}

func (k SqExpIso) Hyper(h []float64) []float64 {
	if h == nil {
		h = make([]float64, k.NumHyper())
	}
	if len(h) != k.NumHyper() {
		panic("gp: hyperparameter length mismatch")
	}
	h[0] = k.LogVariance
	h[1] = k.LogLength
	return h
}

func (k *SqExpIso) SetHyper(h []float64) {
	if len(h) != k.NumHyper() {
		panic("gp: hyperparameter length mismatch")
	}
	k.LogVariance = h[0]
	k.LogLength = h[1]
}

func (k *SqExpIso) Bounds() []Bound {
	return []Bound{
		{math.Log(0.1), math.Log(10)},
		{math.Log(0.001), math.Log(1)},
	}
}

var _ Kernel = &UniSqExpIso{}

// TODO: Hyperparameters need to be an input. Otherwise hard for training in parallel.
// Maybe add a Duplicate function where it just returns a copy of itself
// SqExpIso represents an isotropic squared exponential kernel
// Logs are used for improved numerical conditioning
type UniSqExpIso struct {
	LogLength float64 // Log of the length scale of the kernel function
}

func (k UniSqExpIso) Bounds() []Bound {
	return []Bound{
		{math.Log(0.001), math.Log(1)},
	}
}

func (k UniSqExpIso) NumHyper() int {
	return 1
}

func (k UniSqExpIso) Distance(x, y []float64) float64 {
	return math.Exp(k.LogDistance(x, y))
}

// DistanceDeriv computes the distance between x and y and the derivative of
// the distance with respect to the hyperparameters
// Kernel is
//  variance^2 * math.Exp(- r^2 / (2 * l^2))
func (k UniSqExpIso) DistanceDHyper(x, y, deriv []float64) float64 {
	if len(deriv) != k.NumHyper() {
		panic("gp: deriv length mismatch")
	}

	norm := floats.Distance(x, y, 2)
	logNorm := math.Log(norm)
	logExp := -math.Exp(2*logNorm - 2*k.LogLength - math.Ln2)
	logdist := logExp
	dist := math.Exp(logdist)

	logDDistDLength := logdist + 2*logNorm - 3*k.LogLength
	logDDistDLogLength := k.LogLength + logDDistDLength
	deriv[0] = math.Exp(logDDistDLogLength)
	return dist
}

func (k UniSqExpIso) LogDistance(x, y []float64) float64 {
	if len(x) != len(y) {
		panic("length mismatch")
	}
	norm := floats.Distance(x, y, 2)
	logNorm := math.Log(norm)
	logExp := -math.Exp(2*logNorm - 2*k.LogLength - math.Ln2)
	return logExp
}

func (k UniSqExpIso) Hyper(h []float64) []float64 {
	if h == nil {
		h = make([]float64, k.NumHyper())
	}
	if len(h) != k.NumHyper() {
		panic("gp: hyperparameter length mismatch")
	}
	h[0] = k.LogLength
	return h
}

func (k *UniSqExpIso) SetHyper(h []float64) {
	if len(h) != k.NumHyper() {
		panic("gp: hyperparameter length mismatch")
	}
	k.LogLength = h[0]
}
