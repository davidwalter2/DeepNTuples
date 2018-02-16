import FWCore.ParameterSet.Config as cms



readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring()
source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
for i in range(1,983):
    readFiles.extend( [
        '/store/user/dwalter/TT_Dilepton_v3/TT_TuneCUETP8M2T4_13TeV-powheg-pythia8/tt/171127_104128/0000/output_0_'+str(i)+'.root'
    ])
for i in range(1,1000):
    readFiles.extend( [
        '/store/user/dwalter/TT_Dilepton_v3/TT_TuneCUETP8M2T4_13TeV-powheg-pythia8/tt_backup/171127_104314/0000/output_0_'+str(i)+'.root'
    ])
for i in range(1,191):
    readFiles.extend( [
        '/store/user/dwalter/TT_Dilepton_v3/TT_TuneCUETP8M2T4_13TeV-powheg-pythia8/tt_backup/171127_104314/0001/output_0_1'+str(i)+'.root'
    ])