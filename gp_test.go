package gaussproc

import (
	"math"
	"math/rand"
	"testing"

	"gonum.org/v1/gonum/diff/fd"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distmv"
)

type Rastrigin struct{}

func (Rastrigin) F(x []float64) float64 {
	A := 10.0
	f := A * float64(len(x))
	for _, v := range x {
		f += v*v - A*math.Cos(2*math.Pi*v)
	}
	return f
}

func (Rastrigin) FDf(x []float64, grad []float64) float64 {
	A := 10.0
	f := A * float64(len(x))
	for i, v := range x {
		f += v*v - A*math.Cos(2*math.Pi*v)
		grad[i] = 2*v + 2*math.Pi*A*math.Sin(2*math.Pi*v)
	}
	return f
}

func TestAddBatch(t *testing.T) {
	inputDim := 2
	kernel := &SqExpIso{
		LogLength:   -2,
		LogVariance: -2,
	}
	noise := 1e-10
	g := New(inputDim, kernel, noise)
	xs := make([][]float64, 0)
	for i := 0; i < 150; i++ {
		x := make([]float64, inputDim)
		for j := range x {
			x[j] = rand.Float64()
		}
		xs = append(xs, x)
		y := []float64{Rastrigin{}.F(x)}
		// Add a point
		err := g.AddBatch(mat.NewDense(1, inputDim, x), y)
		if err != nil {
			t.Fatalf("Error adding data: ", err.Error())
		}
		xr, xc := g.inputs.Dims()
		yr := len(g.outputs)
		if xr != i+1 {
			t.Errorf("Wrong number of input rows. Want %v, got %v.", i, xr)
		}
		if xc != inputDim {
			t.Errorf("Wrong number of input columns. Want %v, got %v.", inputDim, xc)
		}
		if yr != i+1 {
			t.Errorf("Wrong number of output rows. Want %v, got %v.", i, yr)
		}
		for j := 0; j < inputDim; j++ {
			if g.inputs.At(i, j) != x[j] {
				t.Errorf("%v,%v does not match", i, j)
			}
		}
		if math.Abs(g.outputs[i]-(y[0]-g.mean)*g.std) > 1e-12 {
			t.Errorf("%v, output does not match", i)
		}
		for j := 0; j < i; j++ {
			for k := 0; k < i; k++ {
				v := g.kernel.Distance(xs[j], xs[k])
				if j == k {
					v += g.noise
				}
				if g.k.At(j, k) != v {
					t.Errorf("%v, %v, %v mismatch. Want %v, got %v.", i, j, k)
				}
			}
		}
	}
}

func TestPredict(t *testing.T) {
	nSamples := 1000
	inputDim := 1
	min, max := 0.0, 10.0
	x := mat.NewDense(nSamples, inputDim, nil)
	for i := 0; i < nSamples; i++ {
		for j := 0; j < inputDim; j++ {
			x.Set(i, j, rand.Float64()*(max-min)+min)
		}
	}
	y := make([]float64, nSamples)
	for i := 0; i < nSamples; i++ {
		y[i] = Rastrigin{}.F(x.Row(nil, i))
	}

	kernel := &UniSqExpIso{
		LogLength: -2,
	}
	noise := 1e-10
	g := New(inputDim, kernel, noise)
	err := g.AddBatch(x, y)
	if err != nil {
		t.Fatalf(err.Error())
	}
	nTest := 11
	xs := make([]float64, nTest)
	floats.Span(xs, min, max)
	ys := make([]float64, nTest)
	for i, x := range xs {
		yPred := g.Mean([]float64{x})
		yTrue := Rastrigin{}.F([]float64{x})
		//if math.Abs(yPred-yTrue) > 1e-3 {
		if !floats.EqualWithinAbsOrRel(yPred, yTrue, 1e-3, 1e-3) {
			t.Errorf("Predict mismatch at x =%v. Want %v, got %v.", x, yTrue, yPred)
		}
		ys[i] = yPred
	}

	// Test that predict batch and predict return roughly the same answer.
	xMat := mat.NewDense(nTest, 1, xs)
	yBatch := g.MeanBatch(nil, xMat)
	if !floats.EqualApprox(yBatch, ys, 1e-8) {
		t.Errorf("Predict batch mismatch. Want %v, got %v.", ys, yBatch)
	}
}

/*
type Bound struct {
	Min float64
	Max float64
}
*/

// Test that the barrier gives the correct gradients.
func TestMLGrad(t *testing.T) {
	for _, trainNoise := range []bool{true} {
		for _, test := range []struct {
			f        func([]float64) float64
			n        int // number of training points
			inputDim int
			bounds   []distmv.Bound
			kernel   Kernel
			noise    float64
		}{
			{
				f:        Rastrigin{}.F,
				n:        10,
				inputDim: 1,
				bounds:   []distmv.Bound{{-3, 3}},
				kernel:   &UniSqExpIso{-2},
				noise:    1e-5,
			},
			{
				f:        Rastrigin{}.F,
				n:        10,
				inputDim: 1,
				bounds:   []distmv.Bound{{-3, 3}},
				kernel:   &SqExpIso{0.3, -2},
				noise:    1e-5,
			},
		} {
			noise := test.noise
			n := test.n
			inputDim := test.inputDim
			// Create fake data.
			xs := mat.NewDense(n, inputDim, nil)
			ys := make([]float64, n)
			unif := distmv.NewUniform(test.bounds, nil)
			for i := 0; i < n; i++ {
				unif.Rand(xs.RawRowView(i))
				ys[i] = test.f(xs.RawRowView(i))
			}
			gp := New(inputDim, test.kernel, noise)
			gp.AddBatch(xs, ys)

			// Test the gradient with the current data
			hyper := test.kernel.Hyper(nil)
			if trainNoise {
				hyper = append(hyper, math.Log(test.noise))
			}
			mem := newMargLikeMemory(len(hyper), len(gp.outputs))
			ml := func(x []float64) float64 {
				return gp.marginalLikelihood(x, trainNoise, mem)
			}
			fdGrad := fd.Gradient(nil, ml, hyper, nil)
			grad := make([]float64, len(hyper))
			mem = newMargLikeMemory(len(hyper), len(gp.outputs))
			gp.marginalLikelihoodDerivative(hyper, grad, trainNoise, mem)

			if !floats.EqualApprox(grad, fdGrad, 1e-6) {
				t.Errorf("Grad mismatch in bounds. Want %v, got %v", fdGrad, grad)
			}

			// Try all of the bounds too low
			bounds := test.kernel.Bounds()
			for i, bound := range bounds {
				hyper[i] = bound.Min - 3*rand.Float64()
			}
			if trainNoise {
				hyper[len(hyper)-1] = minLogNoise - 3*rand.Float64()
			}
			fdGrad = fd.Gradient(nil, ml, hyper, nil)
			mem = newMargLikeMemory(len(hyper), len(gp.outputs))
			gp.marginalLikelihoodDerivative(hyper, grad, trainNoise, mem)

			if !floats.EqualApprox(grad, fdGrad, 1e-6) {
				t.Errorf("Grad mismatch low bounds. Want %v, got %v", fdGrad, grad)
			}

			// Try all of the bounds too high
			for i, bound := range bounds {
				hyper[i] = bound.Max + 3*rand.Float64()
			}
			if trainNoise {
				hyper[len(hyper)-1] = maxLogNoise + 3*rand.Float64()
			}
			fdGrad = fd.Gradient(nil, ml, hyper, nil)
			mem = newMargLikeMemory(len(hyper), len(gp.outputs))
			gp.marginalLikelihoodDerivative(hyper, grad, trainNoise, mem)

			if !floats.EqualApprox(grad, fdGrad, 1e-4) {
				t.Errorf("Grad mismatch low bounds. Want %v, got %v", fdGrad, grad)
			}
		}
	}
}

/*
func TestTrain(t *testing.T) {
	for _, test := range []struct {
		f          func([]float64) float64
		n          int // number of training points
		inputDim   int
		bounds     []distmv.Bound
		kernel     Kernel
		noise      float64
		trainNoise bool
		testGrad   bool
	}{
		{
			f:          Rastrigin{}.F,
			n:          10,
			inputDim:   1,
			bounds:     []distmv.Bound{{-3, 3}},
			kernel:     &UniSqExpIso{-2},
			noise:      1e-10,
			trainNoise: false,
			testGrad:   true,
		},
		{
			f:          Rastrigin{}.F,
			n:          10,
			inputDim:   1,
			bounds:     []distmv.Bound{{-3, 3}},
			kernel:     &SqExpIso{0, -2},
			noise:      1e-10,
			trainNoise: false,
			testGrad:   true,
		},
		{
			f:          Rastrigin{}.F,
			n:          10,
			inputDim:   1,
			bounds:     []distmv.Bound{{-3, 3}},
			kernel:     &UniSqExpIso{1},
			noise:      1e-3,
			trainNoise: true,
			testGrad:   true,
		},
		{
			f:          Rastrigin{}.F,
			n:          10,
			inputDim:   1,
			bounds:     []distmv.Bound{{-3, 3}},
			kernel:     &SqExpIso{0, -2},
			noise:      1e-10,
			trainNoise: true,
			testGrad:   true,
		},
	} {
		trainNoise := test.trainNoise
		noise := test.noise
		n := test.n
		inputDim := test.inputDim
		xs := .NewDense(n, inputDim, nil)
		ys := make([]float64, n)
		unif := distmv.NewUniform(test.bounds, nil)
		for i := 0; i < n; i++ {
			unif.Rand(xs.RawRowView(i))
			ys[i] = test.f(xs.RawRowView(i))
		}
		gp := New(inputDim, test.kernel, noise)
		gp.AddBatch(xs, ys)

		nKerHyper := test.kernel.NumHyper()
		nHyper := nKerHyper
		if trainNoise {
			nHyper++
		}

		hyper := gp.kernel.Hyper(nil)
		if trainNoise {
			hyper = append(hyper, math.Log(noise))
		}

		// Test that the kernel function matches the kernel function derivatives.
		k := .NewSymDense(n, nil)
		dk := make([]*.SymDense, nHyper)
		for i := range dk {
			dk[i] = .NewSymDense(n, nil)
		}
		gp.setKernelMatDeriv(dk, trainNoise, noise)
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				f := func(x []float64) float64 {
					gp.kernel.SetHyper(x[:nKerHyper])
					noise := gp.noise
					if trainNoise {
						noise = math.Exp(x[len(x)-1])
					}
					gp.setKernelMat(k, noise)
					return k.At(i, j)
				}
				fdGrad := fd.Gradient(nil, f, hyper, nil)
				grad := make([]float64, nHyper)
				for k := 0; k < nHyper; k++ {
					grad[k] = dk[k].At(i, j)
				}
				if !floats.EqualApprox(grad, fdGrad, 1e-6) {
					t.Errorf("Kernel deriv mismatch at %v, %v. Want %v, got %v.", i, j, fdGrad, grad)
				}
			}
		}

		if test.testGrad {
			mem := newMargLikeMemory(len(hyper), len(gp.outputs))
			f := func(x []float64) float64 {
				return gp.marginalLikelihood(x, trainNoise, mem)
			}
			fdGrad := fd.Gradient(nil, f, hyper, nil)
			grad := make([]float64, len(hyper))
			gp.marginalLikelihoodDerivative(hyper, grad, trainNoise, mem)
			if !floats.EqualApprox(grad, fdGrad, 1e-6) {
				t.Errorf("Gradient mismatch. Want %v, got %v.", fdGrad, grad)
				continue
			}
		}

		err := gp.Train(trainNoise)
		if err != nil {
			t.Errorf("training errror: %v", err)
		}
	}
}
*/

/*
func TestMarginalLikelihood(t *testing.T) {
	xData := []float64{
		2.083970427750732,
		-0.821018066101379,
		-0.617870699182597,
		-1.183822608860694,
		0.274087442277144,
		0.599441729295593,
		1.768897919204435,
		-0.465645549031928,
		0.588852784375935,
		-0.832982214438054,
		-0.512106527960363,
		0.277883144210116,
		-0.065870426922211,
		-0.821412363806325,
		0.185399443778088,
		-0.858296174995998,
		0.370786630037059,
		-1.409869162416639,
		-0.144668412325022,
		-0.553299615220374,
	}
	yData := []float64{
		4.549203746331698,
		0.371985574437271,
		0.711307965514790,
		-0.013212893618430,
		2.255473255338191,
		1.009915749295733,
		3.744675937965029,
		0.424592771793202,
		1.322833652295811,
		0.278298293510020,
		0.267229130945574,
		2.200112286723834,
		1.200609983308979,
		0.439971697236063,
		2.628580433511271,
		0.503774817335562,
		1.942525313820552,
		0.579133950013607,
		0.670874423968597,
		0.377353755101082,
	}
	x := .NewDense(len(xData), 1, xData)
	y := .NewDense(len(yData), 1, yData)

	kernel := &SqExpIso{
		LogLength:   0, // -0.993396872857613,
		LogVariance: 0, //0.685943458077873,
	}

	g := New(1, 1, kernel, 0.1) //0.149188186046916)
	//g.noise = math.Log(0.1)
	g.Add(x, y)

	//fmt.Println(g.kernelMat)

	kerHyper := g.Kernel.Hyper(nil)
	initHyper := make([]float64, g.Kernel.NumHyper()+1)
	initHyper[0] = math.Log(g.noise)
	copy(initHyper[1:], kerHyper)

	m := &MarginalLikelihood{Gp: g}
	gradient := make([]float64, g.Kernel.NumHyper()+1)
	m.FDf(initHyper, gradient)

	fmt.Println(gradient)

	result, err := optimize.Local(m, initHyper, nil, nil)
	if err != nil {
		t.Errorf("Error optimizing: " + err.Error())
	}
	fmt.Println(result.X)
}
*/

/*
func TestLeaveOneOut(t *testing.T) {
	// xData and yData from GPML code.
	xData := []float64{
		2.083970427750732,
		-0.821018066101379,
		-0.617870699182597,
		-1.183822608860694,
		0.274087442277144,
		0.599441729295593,
		1.768897919204435,
		-0.465645549031928,
		0.588852784375935,
		-0.832982214438054,
		-0.512106527960363,
		0.277883144210116,
		-0.065870426922211,
		-0.821412363806325,
		0.185399443778088,
		-0.858296174995998,
		0.370786630037059,
		-1.409869162416639,
		-0.144668412325022,
		-0.553299615220374,
	}
	yData := []float64{
		4.549203746331698,
		0.371985574437271,
		0.711307965514790,
		-0.013212893618430,
		2.255473255338191,
		1.009915749295733,
		3.744675937965029,
		0.424592771793202,
		1.322833652295811,
		0.278298293510020,
		0.267229130945574,
		2.200112286723834,
		1.200609983308979,
		0.439971697236063,
		2.628580433511271,
		0.503774817335562,
		1.942525313820552,
		0.579133950013607,
		0.670874423968597,
		0.377353755101082,
	}
	x := .NewDense(len(xData), 1, xData)
	y := .NewDense(len(yData), 1, yData)

	nSamples := 1000
	inputDim := 1
	min, max := -1.0, 1.0
	x := .NewDense(nSamples, inputDim, nil)
	for i := 0; i < nSamples; i++ {
		for j := 0; j < inputDim; j++ {
			x.Set(i, j, rand.Float64()*(max-min)+min)
		}
	}
	y := .NewDense(nSamples, 1, nil)
	for i := 0; i < nSamples; i++ {
		y.SetRow(i, []float64{Rastrigin{}.F(x.Row(nil, i))})
	}

	kernel := &UniSqExpIso{
		LogLength: 0.01,
	}

	g := New(1, 1, kernel, 0.1) // 0.149188186046916)
	//g.noise = math.Log(0.1)
	g.AddBatch(x, y)

	//fmt.Println(g.kernelMat)

	kerHyper := g.kernel.Hyper(nil)
	initHyper := make([]float64, g.kernel.NumHyper()+1)
	initHyper[0] = math.Log(g.noise)
	copy(initHyper[1:], kerHyper)

	m := &LeaveOneOut{Gp: g}
	gradient := make([]float64, g.Kernel.NumHyper()+1)
	m.FDf(initHyper, gradient)
	fdGrad := fd.Gradient(nil, m.F, initHyper, nil)
	if !floats.EqualApprox(gradient, fdGrad, 1e-6) {
		fmt.Println(gradient)
		fmt.Println(fdGrad)
		t.Fatalf("gradient mismatch")
	}

	fmt.Println(gradient)
	fmt.Println(fdGrad)

	settings := optimize.DefaultSettings()
	settings.GradientAbsTol = 1e-4

	result, err := optimize.Local(m, initHyper, settings, nil)
	if err != nil {
		t.Errorf("Error optimizing: " + err.Error())
	}
	fmt.Println(result.X)
}
*/

/*
func TestMarginalLikelihood2(t *testing.T) {
	nSamples := 50
	inputDim := 1
	min, max := -1.0, 1.0
	x := .NewDense(nSamples, inputDim, nil)
	for i := 0; i < nSamples; i++ {
		for j := 0; j < inputDim; j++ {
			x.Set(i, j, rand.Float64()*(max-min)+min)
		}
	}
	y := .NewDense(nSamples, 1, nil)
	for i := 0; i < nSamples; i++ {
		y.SetRow(i, []float64{Rastrigin{}.F(x.Row(nil, i))})
	}

	fmt.Println("x =", x)
	fmt.Println("y = ", y)
	//panic("here")
	/*
		meanY := stat.Mean(y.Col(nil, 0), nil)
		for i := 0; i < nSamples; i++ {
			y.Set(i, 0, y.At(i, 0)-meanY)
		}
*/

// TODO: Add one where Variance is fixed at 1 -- that seems to be problematic
/*
	outputDim := 1
	kernel := &UniSqExpIso{
		LogLength: 0,
	}
	noise := 0.01
	g := New(inputDim, outputDim, kernel, noise)
	err := g.Add(x, y)
	if err != nil {
		t.Fatalf(err.Error())
	}

	kerHyper := g.Kernel.Hyper(nil)
	initHyper := make([]float64, g.Kernel.NumHyper()+1)
	initHyper[0] = math.Log(g.noise)
	copy(initHyper[1:], kerHyper)

	m := &MarginalLikelihood{Gp: g}
	gradient := make([]float64, g.Kernel.NumHyper()+1)
	m.FDf(initHyper, gradient)

	//fmt.Println("gradient = ", gradient)

	//fmt.Println("Starting fd")
	fdGradient := make([]float64, len(gradient))
	fd.Gradient(fdGradient, m.F, initHyper, nil)
	fmt.Println("fd Grad =", fdGradient)
	if !floats.EqualApprox(gradient, fdGradient, 1e-4) {
		t.Fatalf("Gradient mismatch: Want \n %v, got \n %v.", fdGradient, gradient)
	}

	fmt.Println()
	// Run an optimization
	result, err := optimize.Local(m, initHyper, nil, nil)
	if err != nil {
		t.Errorf("Error optimizing: " + err.Error())
	}
	fmt.Println(result.X)
	fmt.Println(result.Gradient)

}
*/
/*

func TestSqExpIso(t *testing.T) {
	for _, test := range []struct {
		s     *SqExpIso
		hyper []float64
		dim   int
	}{
		{
			s:     &SqExpIso{},
			hyper: []float64{-2, -4},
			dim:   4,
		},
		{
			s:     &SqExpIso{},
			hyper: []float64{2, -4},
			dim:   4,
		},
		{
			s:     &SqExpIso{},
			hyper: []float64{-2, 4},
			dim:   4,
		},
		{
			s:     &SqExpIso{},
			hyper: []float64{2, 4},
			dim:   4,
		},
	} {
		n := test.s.NumHyper()
		test.s.SetHyper(test.hyper)
		x := make([]float64, test.dim)
		y := make([]float64, test.dim)
		for k := range x {
			x[k] = rand.Float64() * 0.1
			y[k] = rand.Float64() * 0.1
		}
		gradient := make([]float64, n)
		d1 := test.s.Distance(x, y)
		d2 := test.s.DistanceDHyper(x, y, gradient)
		if math.Abs(d1-d2) > 1e-14 {
			t.Errorf("Distance doesn't match")
		}
		f := func(hyper []float64) float64 {
			test.s.SetHyper(hyper)
			return test.s.Distance(x, y)
		}
		fdGradient := fd.Gradient(nil, f, test.hyper, nil)
		if !floats.EqualApprox(gradient, fdGradient, 1e-6) {
			t.Errorf("Gradient mismatch. Want %v, got %v.", fdGradient, gradient)
		}
	}
}
*/

// TODO: Need to test distance and derivative
// TODO: Need to test trainer
