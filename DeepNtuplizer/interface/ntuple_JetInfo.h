/*
 * ntuple_JetInfo.h
 *
 *  Created on: 13 Feb 2017
 *      Author: jkiesele
 */

#ifndef DEEPNTUPLES_DEEPNTUPLIZER_INTERFACE_NTUPLE_JETINFO_H_
#define DEEPNTUPLES_DEEPNTUPLIZER_INTERFACE_NTUPLE_JETINFO_H_

#include "ntuple_content.h"
#include "TRandom3.h"
#include <map>
#include <string>
#include <TFile.h>
#include <TH1.h>
#include <TH2.h>
#include <TGraphAsymmErrors.h>

#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"


/*
 * For global jet info such as eta, pt, gen info
 */
class ntuple_JetInfo: public ntuple_content{
public:
    ntuple_JetInfo():ntuple_content(),
    gluonReduction_(0),
    useherwcompat_matching_(false),
    isherwig_(false)
{}

    void getInput(const edm::ParameterSet& iConfig);
    void initBranches(TTree* );
    void readEvent(const edm::Event& iEvent);

    //use either of these functions

    bool fillBranches(const pat::Jet &, const size_t& jetidx, const edm::Event& iEvent, const  edm::View<pat::Jet> * coll=0);

    void setAxis2Token(edm::EDGetTokenT<edm::ValueMap<float> > axis2Token) {
        axis2Token_ = axis2Token;
    }

    void setMultToken(edm::EDGetTokenT<edm::ValueMap<int> > multToken) {
        multToken_ = multToken;
    }

    void setPtDToken(edm::EDGetTokenT<edm::ValueMap<float> > ptDToken) {
        ptDToken_ = ptDToken;
    }

    void setQglToken(edm::EDGetTokenT<edm::ValueMap<float> > qglToken) {
        qglToken_ = qglToken;
    }

    void setGenJetMatchReclusterToken(
            edm::EDGetTokenT<edm::Association<reco::GenJetCollection> > genJetMatchReclusterToken) {
        genJetMatchReclusterToken_ = genJetMatchReclusterToken;
    }

    void setGenJetMatchWithNuToken(
            edm::EDGetTokenT<edm::Association<reco::GenJetCollection> > genJetMatchWithNuToken) {
        genJetMatchWithNuToken_ = genJetMatchWithNuToken;
    }

    void setGenParticlesToken(edm::EDGetTokenT<reco::GenParticleCollection> genParticlesToken) {
        genParticlesToken_ = genParticlesToken;
    }

    void setMuonsToken(edm::EDGetTokenT<pat::MuonCollection> muonsToken) {
        muonsToken_ = muonsToken;
    }

    void setElectronsToken(edm::EDGetTokenT<pat::ElectronCollection> electronsToken) {
        electronsToken_ = electronsToken;
    }
    void setLHEToken(edm::EDGetTokenT<LHEEventProduct> lheToken){
        lheToken_ = lheToken;
        }

    void setUseHerwigCompatibleMatching(const bool use){
        useherwcompat_matching_=use;
    }
    void setIsHerwig(const bool use){
        isherwig_=use;
    }


    //private:

    double                    jetPtMin_;
    double                    jetPtMax_;
    double                    jetAbsEtaMin_;
    double                    jetAbsEtaMax_;

    //Quark gluon likelihood
    edm::EDGetTokenT<edm::ValueMap<float>>   qglToken_;
    edm::EDGetTokenT<edm::ValueMap<float>>   ptDToken_;
    edm::EDGetTokenT<edm::ValueMap<float>>   axis2Token_;
    edm::EDGetTokenT<edm::ValueMap<int>>     multToken_;

    edm::Handle<edm::ValueMap<float>> qglHandle;
    edm::Handle<edm::ValueMap<float>> ptDHandle;
    edm::Handle<edm::ValueMap<float>> axis2Handle;
    edm::Handle<edm::ValueMap<int>> multHandle;


    edm::EDGetTokenT<edm::Association<reco::GenJetCollection> > genJetMatchReclusterToken_;
    edm::EDGetTokenT<edm::Association<reco::GenJetCollection> > genJetMatchWithNuToken_;

    edm::EDGetTokenT<reco::GenParticleCollection> genParticlesToken_;

    edm::EDGetTokenT<pat::MuonCollection> muonsToken_;
    edm::EDGetTokenT<LHEEventProduct>  lheToken_;

    edm::EDGetTokenT<pat::ElectronCollection> electronsToken_;

    edm::Handle<edm::Association<reco::GenJetCollection> > genJetMatchRecluster;
    edm::Handle<edm::Association<reco::GenJetCollection> > genJetMatchWithNu;

    edm::Handle<reco::GenParticleCollection> genParticlesHandle;

    edm::Handle<pat::MuonCollection> muonsHandle;
    edm::Handle<pat::ElectronCollection> electronsHandle;

    edm::Handle<LHEEventProduct> lheInfo;



    TRandom3 TRandom_;
    float gluonReduction_;

    std::vector <reco::GenParticle> neutrinosLepB;
    std::vector <reco::GenParticle> neutrinosLepB_C;

    std::vector<reco::GenParticle> gToBB;
    std::vector<reco::GenParticle> gToCC;
    std::vector<reco::GenParticle> alltaus_;

    std::vector<reco::GenParticle> Bhadron_;
    std::vector<reco::GenParticle> Bhadron_daughter_;



    bool useherwcompat_matching_;
    bool isherwig_;

    /////////branches

    // labels (MC truth)
    // regressions pt, Deta, Dphi
    float gen_pt_;
    float Delta_gen_pt_;
    //classification
    int isB_;
    int isGBB_;
    int isBB_;
    int isC_;
    int isGCC_;
    int isCC_;
    int isUD_;
    int isS_;
    int isG_;
    int isUndefined_;
    float genDecay_;
    int isLeptonicB_;
    int isLeptonicB_C_;
    int isTau_;
    int isRealData_;

    //truth labeling with fallback to physics definition for light/gluon/undefined of standard flavor definition
    int isPhysB_;
    int isPhysGBB_;
    int isPhysBB_;
    int isPhysC_;
    int isPhysGCC_;
    int isPhysCC_;
    int isPhysUD_;
    int isPhysS_;
    int isPhysG_;
    int isPhysUndefined_;
    int isPhysLeptonicB_;
    int isPhysLeptonicB_C_;
    int isPhysTau_;

    // global variables
    float npv_;
    float ntrueInt_;
    float rho_;
    unsigned int event_no_;
    unsigned int jet_no_;

    // jet variables
    float jet_pt_;
    float jet_corr_pt_;
    float  jet_eta_;
    float  jet_phi_;
    float  jet_mass_;
    float  jet_energy_;

    //Weight variables
    bool useLHEWeights_;
    std::string pupDataDir_;
    std::string pupMCDir_;

    std::vector<double> pupWeights;
    float crossSection_;
    float luminosity_;
    float efficiency_;
    float event_weight_;        //including lepton scalefactors, pup reweighting, cross section, lumi, efficiency

    //scale factor variables
    std::string sfMuonIdDir_;
    std::string sfMuonIsoDir_;
    std::string sfMuonTrackingDir_;
    std::string sfElIdandIsoDir_;
    std::string sfMuonIdName_;
    std::string sfMuonIsoName_;
    std::string sfMuonTrackingName_;
    std::string sfElIdandIsoName_;

    TH2F * sfMuonIdHist;
    TH2F * sfMuonIsoHist;
    TH2F * sfElIdandIsoHist;
    TH1D * sfMuonTrackingHist;

    TAxis *sfMuonIdHist_xaxis;
    TAxis *sfMuonIdHist_yaxis;
    TAxis *sfMuonIsoHist_xaxis;
    TAxis *sfMuonIsoHist_yaxis;
    TAxis *sfElIdandIsoHist_xaxis;
    TAxis *sfElIdandIsoHist_yaxis;
    TAxis *sfMuonTrackingHist_axis;


    float jet_looseId_;

    // quark/gluon
    float jet_qgl_;
    float QG_ptD_;
    float QG_axis2_;
    float QG_mult_;


    float y_multiplicity_;
    float y_charged_multiplicity_;
    float y_neutral_multiplicity_;
    float y_ptD_;
    float y_axis1_;
    float y_axis2_;
    float y_pt_dr_log_;

    static constexpr std::size_t max_num_lept = 5;
    int muons_isLooseMuon_[max_num_lept];
    int muons_isTightMuon_[max_num_lept];
    int muons_isSoftMuon_[max_num_lept];
    int muons_isHighPtMuon_[max_num_lept]; 
    float muons_pt_[max_num_lept]; 
    float muons_relEta_[max_num_lept]; 
    float muons_relPhi_[max_num_lept]; 
    float muons_energy_[max_num_lept]; 
    float electrons_pt_[max_num_lept]; 
    float electrons_relEta_[max_num_lept]; 
    float electrons_relPhi_[max_num_lept]; 
    float electrons_energy_[max_num_lept];

    int muons_number_ = 0;
    int electrons_number_ = 0;

    float gen_pt_Recluster_;
    float gen_pt_WithNu_;
    float Delta_gen_pt_Recluster_;
    float Delta_gen_pt_WithNu_;
    std::map<std::string, float> discriminators_;
};


#endif /* DEEPNTUPLES_DEEPNTUPLIZER_INTERFACE_NTUPLE_JETINFO_H_ */
