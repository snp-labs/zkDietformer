use crate::{
    cc_snark::cc_gro::{Proof, ProvingKey, VerifyingKey},
    harisa::Membership,
    linker::Linker,
};

use ark_ec::pairing::Pairing;
use std::marker::Copy;

#[derive(Default, Debug, PartialEq)]
pub struct LookupPP<E: Pairing, M: Membership<E, LNK>, LNK: Linker<E>> {
    pub m_pp: M::Parameters,

    pub ctt_ek: ProvingKey<E>,
    pub ctt_vk: VerifyingKey<E>,
    pub ctt_lnk_pp: LNK::PP,
    pub ctt_lnk_ek: LNK::EK,
    pub ctt_lnk_vk: LNK::VK,

    pub wt_ek: ProvingKey<E>,
    pub wt_vk: VerifyingKey<E>,
    pub wt_lnk_pp: LNK::PP,
    pub wt_lnk_ek: LNK::EK,
    pub wt_lnk_vk: LNK::VK,

    pub ck: Vec<E::G1Affine>,
}

impl<E: Pairing, M: Membership<E, LNK>, LNK: Linker<E>> Clone for LookupPP<E, M, LNK> {
    fn clone(&self) -> Self {
        Self {
            m_pp: self.m_pp.clone(),
            ctt_ek: self.ctt_ek.clone(),
            ctt_vk: self.ctt_vk.clone(),
            ctt_lnk_pp: self.ctt_lnk_pp.clone(),
            ctt_lnk_ek: self.ctt_lnk_ek.clone(),
            ctt_lnk_vk: self.ctt_lnk_vk.clone(),
            wt_ek: self.wt_ek.clone(),
            wt_vk: self.wt_vk.clone(),
            wt_lnk_pp: self.wt_lnk_pp.clone(),
            wt_lnk_ek: self.wt_lnk_ek.clone(),
            wt_lnk_vk: self.wt_lnk_vk.clone(),
            ck: self.ck.clone(),
        }
    }
}

#[derive(Clone, Default, Debug, PartialEq)]
pub struct LookupProof<E: Pairing, M: Membership<E, LNK>, LNK: Linker<E>> {
    pub m_prf: M::Proof,

    pub ctt_prf: Proof<E>,
    pub ctt_lnk_prf: LNK::Proof,
    pub ctt_lnk_cm_aux: LNK::CM,

    pub wt_prf: Proof<E>,
    pub wt_lnk_prf: LNK::Proof,
    pub wt_lnk_cm_aux: LNK::CM,

    pub cm_f_prime: E::G1Affine,
    pub cm_f_hat: E::G1Affine,
    pub cm_f: E::G1Affine,
    pub cm_z: E::G1Affine,
}

