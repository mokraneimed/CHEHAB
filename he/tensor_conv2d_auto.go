package main

import (
	"flag"
	"fmt"
	"math"
	"time"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

func getRotationSteps() []int {
	return []int{4, 2, 1}
}

func tensor_conv2d(
	encryptedInputs map[string]*rlwe.Ciphertext,
	encodedInputs map[string]*rlwe.Plaintext,
	encryptedOutputs map[string]*rlwe.Ciphertext,
	encodedOutputs map[string]*rlwe.Plaintext,
	encoder *hefloat.Encoder,
	enc *rlwe.Encryptor,
	eval *hefloat.Evaluator,
	params hefloat.Parameters,
) {
	c567 := encryptedInputs["c12"]
	_ = c567
	c550 := encryptedInputs["c3"]
	_ = c550
	p549 := encodedInputs["p2"]
	_ = p549
	p548 := encodedInputs["p2"]
	_ = p548
	p547 := encodedInputs["p11"]
	_ = p547
	p546 := encodedInputs["p11"]
	_ = p546
	c545 := encryptedInputs["c1"]
	_ = c545
	c544 := encryptedInputs["c1"]
	_ = c544
	p543 := encodedInputs["p0"]
	_ = p543
	p542 := encodedInputs["p0"]
	_ = p542
	p541 := encodedInputs["p13"]
	_ = p541
	p540 := encodedInputs["p13"]
	_ = p540
	c539 := encryptedInputs["c14"]
	_ = c539
	c538 := encryptedInputs["c14"]
	_ = c538
	c551 := encryptedInputs["c3"]
	_ = c551
	c552 := encryptedInputs["c5"]
	_ = c552
	c553 := encryptedInputs["c5"]
	_ = c553
	c554 := encryptedInputs["c7"]
	_ = c554
	c555 := encryptedInputs["c7"]
	_ = c555
	p556 := encodedInputs["p6"]
	_ = p556
	p557 := encodedInputs["p6"]
	_ = p557
	p558 := encodedInputs["p8"]
	_ = p558
	p559 := encodedInputs["p8"]
	_ = p559
	c560 := encryptedInputs["c9"]
	_ = c560
	c561 := encryptedInputs["c9"]
	_ = c561
	p562 := encodedInputs["p4"]
	_ = p562
	p563 := encodedInputs["p4"]
	_ = p563
	c564 := encryptedInputs["c10"]
	_ = c564
	c565 := encryptedInputs["c10"]
	_ = c565
	c566 := encryptedInputs["c12"]
	_ = c566

	// FHE Operations
	c538, _ = eval.MulNew(c538, p540)
	_ = eval.Rescale(c538, c538)
	c566, _ = eval.MulNew(c566, p546)
	_ = eval.Rescale(c566, c566)
	c538, _ = eval.AddNew(c538, c566)
	c561, _ = eval.AddNew(c561, c565)
	c561, _ = eval.MulNew(c561, p559)
	_ = eval.Rescale(c561, c561)
	c538, _ = eval.AddNew(c538, c561)
	c555, _ = eval.MulNew(c555, p557)
	_ = eval.Rescale(c555, c555)
	c553, _ = eval.MulNew(c553, p563)
	_ = eval.Rescale(c553, c553)
	c555, _ = eval.AddNew(c555, c553)
	c551, _ = eval.MulNew(c551, p548)
	_ = eval.Rescale(c551, c551)
	c544, _ = eval.MulNew(c544, p542)
	_ = eval.Rescale(c544, c544)
	c551, _ = eval.AddNew(c551, c544)
	c555, _ = eval.AddNew(c555, c551)
	c538, _ = eval.AddNew(c538, c555)
	c551, _ = eval.RotateNew(c538, 4)
	c538, _ = eval.AddNew(c538, c551)
	c553, _ = eval.RotateNew(c538, 2)
	c561, _ = eval.AddNew(c538, c553)
	c561, _ = eval.RotateNew(c561, 1)
	c553, _ = eval.AddNew(c553, c561)
	c538, _ = eval.AddNew(c538, c553)

	// Store outputs
	encryptedOutputs["c36"] = c538
}


func main() {
	// CKKS Parameters (generated from CKKSParamSelector)
	// LogN=14 (n=16384, slots=8192)
	// MaxLevel=3, LogScale=40
	params, err := hefloat.NewParametersFromLiteral(hefloat.ParametersLiteral{
		LogN:            14,
		LogQ:            []int{55, 40, 40, 40},
		LogP:            []int{45, 45},
		LogDefaultScale: 40,
	})
	if err != nil {
		panic(err)
	}

	// Key Generation
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk)
	rlk := kgen.GenRelinearizationKeyNew(sk)

	// Galois keys for rotations
	rotations := getRotationSteps()
	galoisElements := make([]uint64, len(rotations))
	for i, r := range rotations {
		galoisElements[i] = params.GaloisElement(r)
	}

	gks := kgen.GenGaloisKeysNew(galoisElements, sk)

	var galois_keys_total_size int
	for _, gk := range gks {
		if b, err := gk.MarshalBinary(); err == nil {
			galois_keys_total_size += len(b)
		}
	}
	fmt.Printf("rotation_keys_size_(MB): %f\n", float64(galois_keys_total_size)/(1024.0*1024.0))

	evk := rlwe.NewMemEvaluationKeySet(rlk, gks...)

	// Encoder, Encryptor, Decryptor, Evaluator
	encoder := hefloat.NewEncoder(params)
	enc := rlwe.NewEncryptor(params, pk)
	dec := rlwe.NewDecryptor(params, sk)
	eval := hefloat.NewEvaluator(params, evk)

	// Deterministic inputs and timing setup.
	repetitions := flag.Int("repetitions", 1, "number of encrypted inference repetitions")
	flag.Parse()
	if *repetitions < 1 { panic("repetitions must be positive") }
	encryptedInputs := make(map[string]*rlwe.Ciphertext)
	encodedInputs := make(map[string]*rlwe.Plaintext)
	encryptedOutputs := make(map[string]*rlwe.Ciphertext)
	encodedOutputs := make(map[string]*rlwe.Plaintext)
	encryptStart := time.Now()
	values := make([]float64, params.MaxSlots())
	for i := range values { values[i] = float64((i % 17) + 1) / 17.0 }
	{
		pt := hefloat.NewPlaintext(params, params.MaxLevel())
		if err := encoder.Encode(values, pt); err != nil { panic(err) }
		ct, err := enc.EncryptNew(pt)
		if err != nil { panic(err) }
		encryptedInputs["c1"] = ct
	}
	{
		pt := hefloat.NewPlaintext(params, params.MaxLevel())
		if err := encoder.Encode(values, pt); err != nil { panic(err) }
		ct, err := enc.EncryptNew(pt)
		if err != nil { panic(err) }
		encryptedInputs["c10"] = ct
	}
	{
		pt := hefloat.NewPlaintext(params, params.MaxLevel())
		if err := encoder.Encode(values, pt); err != nil { panic(err) }
		ct, err := enc.EncryptNew(pt)
		if err != nil { panic(err) }
		encryptedInputs["c12"] = ct
	}
	{
		pt := hefloat.NewPlaintext(params, params.MaxLevel())
		if err := encoder.Encode(values, pt); err != nil { panic(err) }
		ct, err := enc.EncryptNew(pt)
		if err != nil { panic(err) }
		encryptedInputs["c14"] = ct
	}
	{
		pt := hefloat.NewPlaintext(params, params.MaxLevel())
		if err := encoder.Encode(values, pt); err != nil { panic(err) }
		ct, err := enc.EncryptNew(pt)
		if err != nil { panic(err) }
		encryptedInputs["c3"] = ct
	}
	{
		pt := hefloat.NewPlaintext(params, params.MaxLevel())
		if err := encoder.Encode(values, pt); err != nil { panic(err) }
		ct, err := enc.EncryptNew(pt)
		if err != nil { panic(err) }
		encryptedInputs["c5"] = ct
	}
	{
		pt := hefloat.NewPlaintext(params, params.MaxLevel())
		if err := encoder.Encode(values, pt); err != nil { panic(err) }
		ct, err := enc.EncryptNew(pt)
		if err != nil { panic(err) }
		encryptedInputs["c7"] = ct
	}
	{
		pt := hefloat.NewPlaintext(params, params.MaxLevel())
		if err := encoder.Encode(values, pt); err != nil { panic(err) }
		ct, err := enc.EncryptNew(pt)
		if err != nil { panic(err) }
		encryptedInputs["c9"] = ct
	}

	encryptMillis := float64(time.Since(encryptStart).Microseconds()) / 1000.0
	precisionChecked := false
	expectedOutputs := make(map[string][]float64)

	computeStart := time.Now()
	for i := 0; i < *repetitions; i++ {
			tensor_conv2d(encryptedInputs, encodedInputs, encryptedOutputs, encodedOutputs, encoder, enc, eval, params)
	}
	computeMillis := float64(time.Since(computeStart).Microseconds()) / 1000.0 / float64(*repetitions)
	decryptStart := time.Now()
	maxAbsError := 0.0
	for name, ct := range encryptedOutputs {
		pt := dec.DecryptNew(ct)
		values := make([]float64, params.MaxSlots())
		if err := encoder.Decode(pt, values); err != nil { panic(err) }
		if precisionChecked {
			for i, expected := range expectedOutputs[name] {
				maxAbsError = math.Max(maxAbsError, math.Abs(values[i]-expected))
			}
		}
		fmt.Printf("%s: [%.4f, %.4f, %.4f, ...]\n", name, values[0], values[1], values[2])
	}
	decryptMillis := float64(time.Since(decryptStart).Microseconds()) / 1000.0
	levelsEnd := -1
	for _, ct := range encryptedOutputs { levelsEnd = ct.Level(); break }
	maxAbsErrorJSON := "null"
	if precisionChecked { maxAbsErrorJSON = fmt.Sprintf("%.12g", maxAbsError) }
	fmt.Printf("BENCHMARK_METRICS {\"repetitions\":%d,\"encrypt_ms\":%.3f,\"compute_ms\":%.3f,\"decrypt_ms\":%.3f,\"galois_keys\":%d,\"bootstrap_count\":0,\"level_start\":%d,\"level_end\":%d,\"max_abs_error\":%s,\"precision_checked\":%t}\n",
		*repetitions, encryptMillis, computeMillis, decryptMillis, len(galoisElements), params.MaxLevel(), levelsEnd, maxAbsErrorJSON, precisionChecked)
	_ = encodedOutputs
}
