package gaussproc

/*
func covAndDeriv(g *Gp, x, deriv []float64) (*mat64.Dense, []*mat64.Dense) {
	// Compute the covariance matrix and derivative matrices
	rx, cx := g.inputs.Dims()
	_ = cx
	k := mat64.NewDense(rx, rx, nil)

	derivK := make([]*mat64.Dense, len(deriv))
	for i := range derivK {
		derivK[i] = mat64.NewDense(rx, rx, nil)
	}
	tmp := make([]float64, len(deriv)-1)
	logNoise := x[0]
	noise := math.Exp(2 * logNoise)
	for i := 0; i < rx; i++ {
		for j := i; j < rx; j++ {
			v := g.Kernel.DistanceDHyper(g.inputs.RawRowView(i), g.inputs.RawRowView(j), tmp)
			if i == j {
				v += noise
			}
			k.Set(i, j, v)
			k.Set(j, i, v)
			for l := range derivK {
				if l == 0 {
					if i == j {
						derivK[l].Set(i, j, 2*noise)
					}
					continue
				}
				derivK[l].Set(i, j, tmp[l-1])
				derivK[l].Set(j, i, tmp[l-1])
			}
		}
	}
	return k, derivK
}

type MarginalLikelihood struct {
	Gp *Gp

	Err error
}

func (t MarginalLikelihood) F(x []float64) float64 {
	// extra 1 is for the noise parameter
	deriv := make([]float64, t.Gp.Kernel.NumHyper()+1)
	return t.FDf(x, deriv)
}

func (t MarginalLikelihood) FDf(x, deriv []float64) float64 {
	// 0th input is the noise parametr
	if len(deriv) != t.Gp.Kernel.NumHyper()+1 {
		panic("deriv length mismatch")
	}
	for i := range deriv {
		deriv[i] = 0
	}
	g := t.Gp
	rx, _ := g.inputs.Dims()
	g.Kernel.SetHyper(x[1:])

	// TODO (btracey): Need to figure out how to get this to work with outside
	// loss functions maybe.
	// TODO(btracey): Figure out caching and stuff. Probably relates to future
	// implementation of Global. Need to fix zeroing if that's true
	// TODO(btracey): Do some sort of distance caching if a distance matrix.
	// inputs don't change (maybe have this be part of Gp) to avoid race condition stuff.
	// TODO(btracey): There's all sorts of stuff that can be done to speed this up
	// using the cholesky decomposition.
	// TODO(btracey): Update when have symmetric matrices

	// Compute the covariance matrix and derivative matrices
	//	rx, cx := g.inputs.Dims()
	//	_ = cx
	//	k := mat64.NewDense(rx, rx, nil)
	//
	//	derivK := make([]*mat64.Dense, len(deriv))
	//	for i := range derivK {
	//		derivK[i] = mat64.NewDense(rx, rx, nil)
	//	}
	//	tmp := make([]float64, nKernelHyper)
	//	logNoise := x[0]
	//	noise := math.Exp(2 * logNoise)
	//	for i := 0; i < rx; i++ {
	//		for j := i; j < rx; j++ {
	//			v := g.Kernel.DistanceDHyper(g.inputs.RawRowView(i), g.inputs.RawRowView(j), tmp)
	//			if i == j {
	//				v += noise
	//			}
	//			k.Set(i, j, v)
	//			k.Set(j, i, v)
	//			for l := range derivK {
	//				if l == 0 {
	//					if i == j {
	//						derivK[l].Set(i, j, 2*noise)
	//					}
	//					continue
	//				}
	//				derivK[l].Set(i, j, tmp[l-1])
	//				derivK[l].Set(j, i, tmp[l-1])
	//			}
	//		}
	//	}

	k, derivK := covAndDeriv(g, x, deriv)

	//	fmt.Println(k)

	alpha, err := mat64.Solve(k, g.outputs)
	fmt.Println("y", g.outputs)
	fmt.Println("alpha", alpha)
	if err != nil {
		//	fmt.Println("alpha solve nan")
		for i := range deriv {
			deriv[i] = math.NaN()
		}
		return math.NaN()
	}

	var tmpMat mat64.Dense
	tmpMat.MulTrans(g.outputs, true, alpha, false)
	//fmt.Println(k)
	//fmt.Println("det k = ", mat64.Det(k))
	//fmt.Println("tmp", -0.5*tmpMat.At(0, 0))
	cholK := mat64.Cholesky(k)

	fmt.Println("CholK = ", cholK.L)

	var logDet float64
	for i := 0; i < rx; i++ {
		//fmt.Println("chol", i, cholK.L.At(i, i))
		logDet += 2 * math.Log(cholK.L.At(i, i))
	}

	fmt.Println("logdet1 =", logDet)
	//fmt.Println("logdet2 = ", math.Log(mat64.Det(k)))

	logLikelihood := -0.5*tmpMat.At(0, 0) - 0.5*logDet - (float64(rx)/2)*math.Log(2*math.Pi)
	fmt.Println("term 1 = ", -0.5*tmpMat.At(0, 0))
	fmt.Println("neg log det =", -0.5*logDet)
	fmt.Println("term 3 =", -(float64(rx)/2)*math.Log(2*math.Pi))
	//fmt.Println("outputs =", g.outputs)
	fmt.Println("log likelihood = ", logLikelihood)

	var alphaMul mat64.Dense
	alphaMul.MulTrans(alpha, false, alpha, true)

	for i := range deriv {
		dk := derivK[i]
		var tmp1 mat64.Dense
		tmp1.Clone(dk)
		tmp1.Mul(&alphaMul, dk)

		tmp2, err := mat64.Solve(k, dk)
		if err != nil {
			//	fmt.Println("k dk solve nan")
			for i := range deriv {
				deriv[i] = math.NaN()
			}
			return math.NaN()
		}
		var trace float64
		for j := 0; j < rx; j++ {
			trace += tmp1.At(j, j)
			trace -= tmp2.At(j, j)
		}
		// fmt.Println("trace = ", trace)
		deriv[i] = 0.5 * trace
	}
	// want to maximize log likelihood, so flip the sign
	logLikelihood *= -1
	floats.Scale(-1, deriv)

	fmt.Println()
	fmt.Println("x = ", x)
	fmt.Println("loglike = ", logLikelihood)
	fmt.Println("deriv = ", deriv)
	return logLikelihood
}
*/
