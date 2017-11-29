import FWCore.ParameterSet.Config as cms

deepntuplizer = cms.EDAnalyzer('DeepNtuplizer',
                                vertices   = cms.InputTag("offlineSlimmedPrimaryVertices"),
                                secVertices = cms.InputTag("slimmedSecondaryVertices"),
                                jets       = cms.InputTag("slimmedJets"),
                                jetR       = cms.double(0.4),
				                runFatJet = cms.bool(False),
                                pupInfo = cms.InputTag("slimmedAddPileupInfo"),
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
                                runonData=cms.bool(True),

                                #for computation of event weights
                                crossSection=cms.double(1.0),
                                luminosity = cms.double(1.0),
                                efficiency = cms.double(1.0)    #1/((effective) number of events
                                )
