import FWCore.ParameterSet.Config as cms

readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring()
source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
for i in range(1,301):
    readFiles.extend( [
        '/store/user/dwalter/TT_Dilepton_v4/WJetsToLNu_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8/WJets/180115_123952/0000/output_0_'+str(i)+'.root'
    ])

deepntuplizer = cms.EDAnalyzer('DeepNtuplizer',
                               vertices   = cms.InputTag("offlineSlimmedPrimaryVertices"),
                               secVertices = cms.InputTag("slimmedSecondaryVertices"),
                               jets       = cms.InputTag("slimmedJets"),
                               jetR       = cms.double(0.4),
                               runFatJet = cms.bool(False),
                               pupInfo = cms.InputTag("slimmedAddPileupInfo"),
                               lheInfo = cms.InputTag("externalLHEProducer"),
                               rhoInfo = cms.InputTag("fixedGridRhoFastjetAll"),
                               SVs  = cms.InputTag("slimmedSecondaryVertices"),
                               LooseSVs = cms.InputTag("inclusiveCandidateSecondaryVertices"),
                               genJetMatchWithNu = cms.InputTag("patGenJetMatchWithNu"),
                               genJetMatchRecluster = cms.InputTag("patGenJetMatchRecluster"),
                               pruned = cms.InputTag("prunedGenParticles"),
                               fatjets = cms.InputTag('slimmedJetsAK8'),
                               muons = cms.InputTag("slimmedMuons"),
                               electrons = cms.InputTag("slimmedElectrons"),
                               jetPtMin     = cms.double(20.0),
                               jetPtMax     = cms.double(1000),
                               jetAbsEtaMin = cms.double(0.0),
                               jetAbsEtaMax = cms.double(2.5),
                               gluonReduction = cms.double(0.0),
                               tagInfoName = cms.string('deepNN'),
                               tagInfoFName = cms.string('pfBoostedDoubleSVAK8'),
                               bDiscriminators = cms.vstring(),
                               qgtagger        = cms.string("QGTagger"),
                               candidates      = cms.InputTag("packedPFCandidates"),


                               useHerwigCompatible=cms.bool(False),
                               isHerwig=cms.bool(False),
                               useOffsets=cms.bool(True),
                               applySelection=cms.bool(True),
                               isData=cms.bool(False),

                               #for computation of event weights
                               pupDataDir=cms.string(   #Directory of the data pileup distribution root file for pileup reweighting
                                   "/afs/desy.de/user/d/dwalter/CMSSW_8_0_25/src/DeepNTuples/DeepNtuplizer/data/pileupData.root"),

                               pupMCDir=cms.string(     # Directory of the data pileup distribution root file for pileup reweighting
                                   "/afs/desy.de/user/d/dwalter/CMSSW_8_0_25/src/DeepNTuples/DeepNtuplizer/data/pileupMC.root"),

                                #scalefactor histograms
                               sfMuonIDFile=cms.string("/afs/desy.de/user/d/dwalter/CMSSW_8_0_25/src/DeepNTuples/DeepNtuplizer/data/EfficienciesAndSF_ID_GH.root"),
                               sfMuonIDName=cms.string("MC_NUM_TightID_DEN_genTracks_PAR_pt_eta/abseta_pt_ratio"),
                               sfMuonISOFile=cms.string("/afs/desy.de/user/d/dwalter/CMSSW_8_0_25/src/DeepNTuples/DeepNtuplizer/data/EfficienciesAndSF_ISO_GH.root"),
                               sfMuonISOName=cms.string("TightISO_TightID_pt_eta/abseta_pt_ratio"),
                               sfMuonTrackingFile=cms.string("/afs/desy.de/user/d/dwalter/CMSSW_8_0_25/src/DeepNTuples/DeepNtuplizer/data/Tracking_EfficienciesAndSF_BCDEFGH.root"),
                               sfMuonTrackingName=cms.string("ratio_eff_aeta_dr030e030_corr"),
                               sfElIDandISOFile=cms.string("/afs/desy.de/user/d/dwalter/CMSSW_8_0_25/src/DeepNTuples/DeepNtuplizer/data/egammaEffi.txt_EGM2D.root"),
                               sfElIDandISOName=cms.string("EGamma_SF2D"),

                               eventIDs=cms.string(""), #directory of csv file with the sorted event ids to not process (avoid double counting)

                               useLHEWeights=cms.bool(True),
                               crossSection=cms.double(61526700),
                               luminosity = cms.double(35.9),
                               efficiency = cms.double(1./16497031)  #1/((effective) number of events
                               )
