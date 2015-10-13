package gaussproc

/*
const minLogNoise = -15

type LeaveOneOut struct {
	Gp *Gp
}

func (l *LeaveOneOut) F(x []float64) float64 {
	deriv := make([]float64, l.Gp.Kernel.NumHyper()+1)
	return l.FDf(x, deriv)
}

func (l *LeaveOneOut) FDf(x, deriv []float64) float64 {
	// 0th input is the noise parametr
	if len(deriv) != l.Gp.Kernel.NumHyper()+1 {
		panic("deriv length mismatch")
	}
	for i := range deriv {
		deriv[i] = 0
	}
	g := l.Gp
	rx, _ := g.inputs.Dims()
	_, cy := g.outputs.Dims()
	if cy != 1 {
		panic("not coded for multiple outputs")
	}
	g.Kernel.SetHyper(x[1:])

	// We will use a barrier method for the noise. Evaluate this first so we save the
	// extra cost
	if x[0] < minLogNoise {
		for i := range deriv {
			deriv[i] = math.Inf(1)
		}
		return math.NaN()
	}

	k, derivk := covAndDeriv(g, x, deriv)

	chol := mat64.Cholesky(k)
	// TODO: Replace this with triangular inverse
	linv, err := mat64.Inverse(chol.L)
	if err != nil {
		panic(err)
	}

	kinv := mat64.NewDense(0, 0, nil)
	kinv.MulTrans(linv, true, linv, false)

	//	var tmp mat64.Dense
	//	tmp.Mul(kinv, k)
	//	fmt.Println(tmp)

	//	kinv, err := mat64.Inverse(k)
	//	if err != nil {
			//fmt.Println(k)
			//panic(err)
	//		for i := range deriv {
	//			deriv[i] = math.NaN()
	//		}
	//		return math.NaN()
	//	}

	alpha, err := mat64.Solve(k, g.outputs)
	if err != nil {
		//panic(err)
		for i := range deriv {
			deriv[i] = math.NaN()
		}
		return math.NaN()
	}

	var logLikelihood float64
	for i := 0; i < rx; i++ {
		ki := kinv.At(i, i)
		mu := alpha.At(i, 0) / ki
		sigmaSq := 1 / ki
		logLikelihood += -0.5*math.Log(sigmaSq) - 0.5*(mu*mu)/sigmaSq - 0.5*math.Log(2*math.Pi)
		//fmt.Println("real", g.outputs.At(i, 0), "mu =", mu, "sigma = ", sigma)
	}

	for j := range deriv {
		z, err := mat64.Solve(k, derivk[j])
		if err != nil {
			panic(err)
		}
		// TODO(btracey): preallocate
		var zja mat64.Dense
		zja.Mul(z, alpha) // this could be a vec
		var zjk mat64.Dense
		zjk.Mul(z, kinv)
		var sum float64
		for i := 0; i < rx; i++ {
			ai := alpha.At(i, 0)
			ki := kinv.At(i, i)
			sum += (ai*zja.At(i, 0) - 0.5*(1+ai*ai/ki)*zjk.At(i, i)) / ki
		}
		deriv[j] = sum
	}
	// Add a barrier method to the noise so that it can't drop below a certain value
	logLikelihood += math.Log(x[0] - minLogNoise)
	deriv[0] += 1 / (x[0] - minLogNoise)
	floats.Scale(-1, deriv)
	fmt.Println("x = ", x)
	fmt.Println("deriv = ", deriv)
	fmt.Println(logLikelihood)
	return -logLikelihood
}
*/
