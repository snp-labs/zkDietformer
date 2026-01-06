/*use ark_ff::Field;
use ark_r1cs_std::{alloc::AllocVar, eq::EqGadget, fields::fp::FpVar};
use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystemRef, SynthesisError};
use crate::gadgets::matmul::constraints::{mat_vec_mul, enforce_freivalds};



#[derive(Clone)]
struct FullCircuitNystrom {
    // witness
    pub x: Vec<u8>,
    pub q: Vec<u8>,
    pub k: Vec<u8>,
    pub v: Vec<u8>,
    pub y: Vec<u8>,


    pub x_open: KZGRandomness,
    pub x_com: KZGCommitment,
    pub q_open: KZGRandomness,
    pub q_com: KZGCommitment,
    pub k_open: KZGRandomness,
    pub k_com: KZGCommitment,
    pub v_open: KZGRandomness,
    pub v_com: KZGCommitment,
    pub y_open: KZGRandomness,
    pub y_com: KZGCommitment,


    // 제로 포인트가 witness가 아니었음....!
    pub x_0: u8,
    pub qk_0: u8,
    pub v_0: u8,
    pub qtkt_0: u8,
    pub z_0 : u8,
    pub qt_0 : u8,
    pub kt_0 : u8,
    pub y_0 : u8,
}


impl ConstraintSynthesizer<Fq> for FullCircuitNystrom {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        #[cfg(debug_assertion)]
        println!("FullCircuitU8 is setup mode: {}", cs.is_in_setup_mode());
        //because we assume that weights are public constants, prover does not need to commit the weights.
        // x commitment
        let x_com_circuit = KZGComCircuit {
            param: self.params.clone(),
            input: self.x.clone(),
            open: self.x_open,
            commit: self.x_com,
        };
        x_com_circuit.generate_constraints(cs.clone())?;
        let mut _cir_number = cs.num_constraints();
        // #[cfg(debug_assertion)]
        println!("Number of constraints for x commitment {}", _cir_number);

        // z commitment
        let z_com_circuit = KZGComCircuit {
            param: self.params.clone(),
            input: self.z.clone(),
            open: self.z_open,
            commit: self.z_com,
        };
        z_com_circuit.generate_constraints(cs.clone())?;
        // #[cfg(debug_assertion)]
        println!(
            "Number of constraints for z commitment {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        // layer 1
        let mut y = vec![0u8; M];
        let l1_mat_ref: Vec<&[u8]> = self.l1.iter().map(|x| x.as_ref()).collect();
        // vec_mat_mul_u8(&self.x, l1_mat_ref[..].as_ref(), &mut y,
        //                 self.x_0, self.l1_mat_0, self.y_0, self.multiplier_l1);

        let (remainder1, div1) = vec_mat_mul_with_remainder_u8(
            &self.x,
            l1_mat_ref[..].as_ref(),
            &mut y,
            self.x_0,
            self.l1_mat_0,
            self.y_0,
            &self.multiplier_l1,
        );
        let mut y_out = y.clone();
        relu_u8(&mut y_out, self.y_0);

        let l1_circuit = FreivaldsCircuit {
            x: self.x,
            l1_mat: self.l1,
            y: y.clone(),
            remainder: remainder1.clone(),
            div: div1.clone(),

            x_0: self.x_0,
            l1_mat_0: self.l1_mat_0,
            y_0: self.y_0,

            multiplier: self.multiplier_l1,
            knit_encoding: self.knit_encoding,
        };
        l1_circuit.generate_constraints(cs.clone())?;
        println!(
            "Number of constraints for FC1 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        let relu_circuit = SoftmaxCircuit {
            y_in: y,
            y_out: y_out.clone(),

            y_zeropoint: self.y_0,
        };
        relu_circuit.generate_constraints(cs.clone())?;

        println!(
            "Number of constraints for ReLU1 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();
        let l2_mat_ref: Vec<&[u8]> = self.l2.iter().map(|x| x.as_ref()).collect();
        let mut zz = self.z.clone();
        let (remainder2, div2) = vec_mat_mul_with_remainder_u8(
            &y_out,
            l2_mat_ref[..].as_ref(),
            &mut zz,
            self.y_0,
            self.l2_mat_0,
            self.z_0,
            &self.multiplier_l2,
        );
        let l2_circuit = zkDietCircuit {
        };
        l2_circuit.generate_constraints(cs.clone())?;
        println!(
            "Number of constraints for FC2 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );

        let argmax_circuit = ArgmaxCircuitU8 {
            input: self.z.clone(),
            argmax_res: self.argmax_res.clone(),
        };

        argmax_circuit
            .clone()
            .generate_constraints(cs.clone())
            .unwrap();

        _cir_number = cs.num_constraints();

        println!(
            "Total number of FullCircuit inference constraints {}",
            cs.num_constraints()
        );
        Ok(())
    }
}
*/