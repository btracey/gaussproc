package gaussproc

import (
	"fmt"
	"log"
	"testing"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"

	"github.com/btracey/btutil"
	"github.com/btracey/kernel"
)

func TestNewGP(t *testing.T) {
	ker := kernel.LogKernelWrapper{Hyper: []float64{0}, LogKerneler: kernel.SqExpIsoUnit{}}
	noise := 1e-3
	// Try NewGP with no data
	_, err := NewGP(ker, nil, nil, noise, true)
	if err != nil {
		t.Errorf("error with nil data: %s", err)
	}
	_, err = NewGP(ker, nil, nil, noise, false)
	if err != nil {
		t.Errorf("error with nil data: %s", err)
	}

	x := mat.NewDense(4, 2, []float64{
		1, -1,
		2, 1.6,
		2, 1,
		-1, 1,
	})
	y := []float64{6, 4, 3, 1}

	_, err = NewGP(ker, x, y, noise, false)
	if err != nil {
		t.Errorf("error with non-nil data: %s", err)
	}
	_, err = NewGP(ker, x, y, noise, true)
	if err != nil {
		t.Errorf("error with non-nil data normalized: %s", err)
	}
}

func TestObserve(t *testing.T) {
	ker := kernel.LogKernelWrapper{Hyper: []float64{0}, LogKerneler: kernel.SqExpIsoUnit{}}
	noise := 1e-3
	x := mat.NewDense(4, 2, []float64{
		1, -1,
		2, 1.6,
		2, 1,
		-1, 1,
	})
	y := []float64{6, 4, 3, 1}

	m, n := x.Dims()

	xs := x.Slice(0, m-1, 0, n)
	ys := y[:m-1]

	// Create a GP with the first xs, and then observe the last one.
	gp, err := NewGP(ker, xs, ys, noise, false)
	if err != nil {
		t.Errorf("new gp failed")
	}
	gp.Observe(x.RawRowView(m-1), y[m-1])

	// Create the GP with all the xs.
	gpAll, err := NewGP(ker, x, y, noise, false)
	if err != nil {
		t.Errorf("new gp failed")
	}

	// Check that the things match.
	if str := gpsEqualApprox(gp, gpAll, 1e-14); str != "" {
		t.Errorf("gp mismatch: %v", str)
	}
}

func gpsEqualApprox(gp1, gp2 *GP, tol float64) string {
	if !mat.Equal(gp1.x, gp2.x) {
		btutil.PrintMat("x1", gp1.x)
		btutil.PrintMat("x2", gp2.x)
		return "x mismatch"
	}
	if !floats.Equal(gp1.y, gp2.y) {
		return "y mismatch"
	}
	s1 := gp1.kInv.ToSym(nil)
	s2 := gp2.kInv.ToSym(nil)
	if !mat.EqualApprox(s1, s2, tol) {
		return "kInv mismatch"
	}
	if !mat.EqualApprox(gp1.kInvY, gp2.kInvY, tol) {
		return "kInvY mismatch"
	}
	if !floats.Equal(gp1.meanX, gp2.meanX) {
		return "meanX mismatch"
	}
	if !floats.Equal(gp1.stdX, gp2.stdX) {
		return "stdX mismatch"
	}
	if gp1.meanY != gp2.meanY {
		return "meanY mismatch"
	}
	if gp1.stdY != gp2.stdY {
		return "stdY mismatch"
	}
	return ""
}

func TestMeansStds(t *testing.T) {
	ker := kernel.LogKernelWrapper{Hyper: []float64{-1}, LogKerneler: kernel.SqExpIsoUnit{}}
	noise := 1e-3

	x := mat.NewDense(4, 1, []float64{
		1,
		1.1,
		1.3,
		2.1,
	})
	y := []float64{6, 4, 3, 1}

	gp, err := NewGP(ker, x, y, noise, false)
	if err != nil {
		log.Fatal(err)
	}

	nEval := 11
	xData := make([]float64, nEval)
	floats.Span(xData, 0.5, 2.5)
	xTest := mat.NewDense(nEval, 1, xData)

	sym := gp.kInv.ToSym(nil)
	btutil.PrintMat("sym", sym)

	means, stds := gp.MeansStds(xTest)
	fmt.Println("x  ", xData)
	fmt.Println("mean", means)
	fmt.Println("std ", stds)
}
