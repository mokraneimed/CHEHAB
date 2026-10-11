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
	return []int{}
}

func fhe(
	encryptedInputs map[string]*rlwe.Ciphertext,
	encodedInputs map[string]*rlwe.Plaintext,
	encryptedOutputs map[string]*rlwe.Ciphertext,
	encodedOutputs map[string]*rlwe.Plaintext,
	encoder *hefloat.Encoder,
	enc *rlwe.Encryptor,
	eval *hefloat.Evaluator,
	params hefloat.Parameters,
) {
	c3 := encryptedInputs["v2_0"]
	_ = c3
	c2 := encryptedInputs["v1_0"]
	_ = c2

	// FHE Operations
	c2, _ = eval.MulRelinNew(c2, c3)
	_ = eval.Rescale(c2, c2)

	// Store outputs
	encryptedOutputs["output"] = c2
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
		encryptedInputs["v1_0"] = ct
	}
	{
		pt := hefloat.NewPlaintext(params, params.MaxLevel())
		if err := encoder.Encode(values, pt); err != nil { panic(err) }
		ct, err := enc.EncryptNew(pt)
		if err != nil { panic(err) }
		encryptedInputs["v2_0"] = ct
	}

	encryptMillis := float64(time.Since(encryptStart).Microseconds()) / 1000.0
	precisionChecked := false
	ref_3 := append([]float64(nil), values...)
	ref_2 := append([]float64(nil), values...)
	ref_4 := make([]float64, len(ref_2))
	for i := range ref_4 { ref_4[i] = ref_2[i] * ref_3[i] }
	expectedOutputs := make(map[string][]float64)
	expectedOutputs["output"] = ref_4
	precisionChecked = true

	computeStart := time.Now()
	for i := 0; i < *repetitions; i++ {
			fhe(encryptedInputs, encodedInputs, encryptedOutputs, encodedOutputs, encoder, enc, eval, params)
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
